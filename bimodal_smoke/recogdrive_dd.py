"""ReCogDrive frozen VLM + explicit latent adapter + unchanged DiffusionDrive head.

This is a hybrid, NOT the native ReCogDrive baseline. No sensor is added.
All 2800 VLM tokens condition a learned 16x16 latent grid, not measured LiDAR BEV.
"""
import os,sys,json,time,math,random,hashlib,gc,argparse
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.insert(0,str(BASE/'repo'));sys.path.insert(0,str(BASE/'bimodal_smoke'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vendor.diffusion_decoder import TrajectoryHead
ROOT=BASE/'demo10';SOURCE=ROOT/'native_recogdrive';OUT=ROOT/'recogdrive_dd'
ANCHOR=BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy'

def save(name,value):
    OUT.mkdir(exist_ok=True);p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def checkpoint(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');torch.save(value,t);t.replace(p)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024

class ReCogDriveDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(1536,256),nn.LayerNorm(256))
        self.latent_queries=nn.Embedding(256,256)
        self.latent_attention=nn.MultiheadAttention(256,8,dropout=0.,batch_first=True)
        self.latent_norm=nn.LayerNorm(256)
        self.latent_ffn=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))
        self.status_encoder=nn.Linear(20,256)
        self.scene_queries=nn.Embedding(31,256)
        self.scene_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,8,1024,dropout=0.,batch_first=True),3)
        self.bev_adapter=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())
        cfg=SimpleNamespace(tf_d_model=256,tf_num_head=8,tf_d_ffn=1024,tf_dropout=0.,lidar_max_x=32,lidar_max_y=32,trajectory_cls_weight=10.,trajectory_reg_weight=8.)
        self.trajectory_head=TrajectoryHead(8,1024,256,str(ANCHOR),cfg)

    def forward(self,features,target=None):
        memory=self.projection(features['last_hidden_state'])
        b=len(memory);q=self.latent_queries.weight[None].expand(b,-1,-1)
        latent=self.latent_norm(q+self.latent_attention(q,memory,memory,need_weights=False)[0])
        latent=latent+self.latent_ffn(latent)
        state=torch.cat([features['status_feature'].flatten(1),features['history_trajectory'].flatten(1)],dim=1)
        assert state.shape[1]==20,state.shape
        state=self.status_encoder(state.float())[:,None]
        scene=self.scene_decoder(self.scene_queries.weight[None].expand(b,-1,-1),torch.cat([latent,state],1))
        bev=self.bev_adapter(latent.transpose(1,2).reshape(b,256,16,16))
        bev=F.interpolate(bev,size=(64,64),mode='bilinear',align_corners=False)
        return self.trajectory_head(scene[:,:1],scene[:,1:],bev,bev.shape[2:],state,targets=None if target is None else {'trajectory':target})

def benchmark():
    torch.set_num_threads(2);seed(42);OUT.mkdir(exist_ok=True)
    sample=torch.load(next((SOURCE/'features').glob('*.pt')),map_location='cpu',weights_only=False)
    print('FEATURE_SHAPES',{k:list(v.shape) for k,v in sample['features'].items()},flush=True)
    results=[]
    for kind in ('native','dd'):
        if kind=='native':
            from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            net=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),cache_hidden_state=True)
        else:net=ReCogDriveDD().cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
        for bs in ([128] if kind=='native' else [32,64,128]):
            f={k:v[None].repeat(bs,*([1]*v.ndim)).cuda() for k,v in sample['features'].items()};y=sample['target'][None].repeat(bs,1,1).cuda()
            if kind=='native':f={k:v.float() for k,v in f.items()}
            torch.cuda.reset_peak_memory_stats();times=[]
            try:
                net.train()
                for i in range(3):
                    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        if kind=='native':p=net(f,{'trajectory':y});loss=net.compute_loss(f,{'trajectory':y},p)
                        else:loss=net(f,y)['trajectory_loss']
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
                net.eval();torch.cuda.synchronize();start=time.perf_counter()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=net(f)['pred_traj' if kind=='native' else 'trajectory']
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                assert p.shape==(bs,8,3) and torch.isfinite(p).all()
                row=dict(model=kind,batch=bs,train_step_seconds=float(np.mean(times[1:])),inference_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(loss),shape=list(p.shape))
            except torch.cuda.OutOfMemoryError:
                row=dict(model=kind,batch=bs,oom=True)
            results.append(row);save('benchmark.json',results);print('BENCH',json.dumps(row),flush=True)
            opt.zero_grad(set_to_none=True);del f,y;gc.collect();torch.cuda.empty_cache()
            if row.get('oom'):break
        del net,opt;gc.collect();torch.cuda.empty_cache()
    # Read a real batch separately; do not attribute host I/O time to diffusion.
    files=sorted((SOURCE/'features').glob('*.pt'))[:128];start=time.perf_counter()
    loaded=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
    io=dict(samples=len(loaded),bytes=sum(p.stat().st_size for p in files),seconds=time.perf_counter()-start)
    save('io_benchmark.json',io);print('IO_BENCH',json.dumps(io),flush=True)

def pilot(epochs,resume=False):
    assert 1<=epochs<=200
    assert epochs<=3 or resume,'Full run must resume the verified pilot.'
    torch.set_num_threads(2);seed(42)
    manifest=json.loads((ROOT/'manifest.json').read_text());digest=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    results=json.loads((OUT/'benchmark.json').read_text());valid=[r['batch'] for r in results if r['model']=='dd' and not r.get('oom') and r['peak_allocated_gib']<19]
    micro=max(valid);assert 128%micro==0
    config=dict(model='ReCogDrive-VLM + latent adapter + existing DiffusionDrive head',native_recogdrive=False,epochs=epochs,scheduler_horizon=200,micro_batch=micro,effective_batch=128,seed=42,manifest_sha256=digest,train=8688,val=1930,input='unchanged native frozen VLM full 2800x1536, ego/history; no LiDAR',latent_grid='learned 16x16, not measured BEV',selection='minimum validation ADE',cache_loading='preload CPU RAM; separate loading time',lr=1e-4,wd=1e-4,betas=[.9,.95],clip=1.,precision='bf16')
    state=None
    if resume:
        state=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False)
        assert {k:v for k,v in state['config'].items() if k!='epochs'}=={k:v for k,v in config.items() if k!='epochs'}
        assert 0<state['epoch']<=epochs
        for name in ('summary.json','config.json','history.json','loading.json'):
            p=OUT/name;backup=OUT/('pilot_'+name)
            if p.exists() and not backup.exists():backup.write_bytes(p.read_bytes())
    save('config.json',config)
    total_bytes=sum((SOURCE/'features'/(r['token']+'.pt')).stat().st_size for r in manifest)
    assert available()>total_bytes+32*2**30,'RAM guard: require full cache plus 32 GiB reserve'
    save('status.json',dict(stage='loading_cpu_cache',samples=0,total=len(manifest)))
    rows=[];start=time.perf_counter()
    for i,r in enumerate(manifest):
        item=torch.load(SOURCE/'features'/(r['token']+'.pt'),map_location='cpu',weights_only=False)
        assert item['token']==r['token'] and item['split']==r['split'] and item['manifest_sha256']==digest
        rows.append(item)
        if i%250==0:
            assert available()>24*2**30,'Host memory reserve reached'
            save('status.json',dict(stage='loading_cpu_cache',samples=i+1,total=len(manifest),seconds=time.perf_counter()-start));print('LOADED',i+1,len(manifest),flush=True)
    loading=time.perf_counter()-start;save('loading.json',dict(seconds=loading,samples=len(rows),cache_bytes=total_bytes))
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val'];assert (len(train),len(val))==(8688,1930)
    def batch(ids):
        return {k:torch.stack([rows[i]['features'][k] for i in ids]).cuda() for k in rows[0]['features']},torch.stack([rows[i]['target'] for i in ids]).cuda()
    net=ReCogDriveDD().cuda();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
    def evaluate():
        net.eval();seed(123);ade=fde=0.;started=time.perf_counter()
        with torch.no_grad():
            for offset in range(0,len(val),micro):
                f,y=batch(val[offset:offset+micro])
                with torch.autocast('cuda',dtype=torch.bfloat16):p=net(f)['trajectory']
                assert torch.isfinite(p).all();d=(p.float()[...,:2]-y[...,:2]).norm(dim=-1);ade+=float(d.sum());fde+=float(d[:,-1].sum())
        return dict(ade=ade/(8*len(val)),fde=fde/len(val),validation_seconds=time.perf_counter()-started)
    if state is not None:
        net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);start_epoch=state['epoch'];steps=state['step']
        history=json.loads((OUT/'history.json').read_text());assert history[-1]['epoch']==start_epoch
        best=min(r['ade'] for r in history if r['epoch']>0)
        save('resume.json',dict(from_epoch=start_epoch,from_step=steps,target_epoch=epochs,optimizer_restored=True,manifest_sha256=digest))
        del state;gc.collect();print('RESUMED',start_epoch,steps,flush=True)
    else:
        history=[dict(epoch=0,**evaluate())];save('history.json',history);best=float('inf');steps=0;start_epoch=0
    began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch+1,epochs+1):
        seed(42+epoch);order=[train[i] for i in torch.randperm(len(train)).tolist()];net.train();loss_sum=0.;started=time.perf_counter();transfer=compute=0.
        lr=1e-4*epoch/3 if epoch<=3 else 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*(epoch-4)/197))
        for group in opt.param_groups:group['lr']=lr
        for offset in range(0,len(order),128):
            ids=order[offset:offset+128];opt.zero_grad(set_to_none=True)
            for j in range(0,len(ids),micro):
                sub=ids[j:j+micro];t=time.perf_counter();f,y=batch(sub);torch.cuda.synchronize();transfer+=time.perf_counter()-t;t=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(f,y)['trajectory_loss']
                assert torch.isfinite(loss);(loss*len(sub)/len(ids)).backward();loss_sum+=float(loss)*len(sub);torch.cuda.synchronize();compute+=time.perf_counter()-t
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
            if steps%10==0:
                save('status.json',dict(stage='training',epoch=epoch,target_epochs=epochs,step=steps,epoch_samples=min(offset+128,len(order)),train_samples=len(train),seconds=time.perf_counter()-started,last_loss=float(loss)))
        train_seconds=time.perf_counter()-started;metrics=evaluate();seconds=time.perf_counter()-started
        row=dict(epoch=epoch,step=steps,lr=lr,train_loss=loss_sum/len(train),train_seconds=train_seconds,transfer_seconds=transfer,forward_backward_seconds=compute,seconds=seconds,**metrics);history.append(row);save('history.json',history)
        state=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config=config,step=steps)
        if metrics['ade']<best:
            best=metrics['ade'];checkpoint('best_val_ade.pt',state)
        checkpoint('latest.pt',state)
        save('status.json',dict(stage='training',target_epochs=epochs,**row,best_val_ade=best,projected_200_epoch_hours=np.mean([r['seconds'] for r in history[1:]])*200/3600,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30));print('EPOCH',json.dumps(row),flush=True)
    save('summary.json',dict(config=config,history=history,loading_seconds=loading,pilot_seconds=time.perf_counter()-began,best_val_ade=best,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,projected_200_epoch_hours=np.mean([r['seconds'] for r in history[1:]])*200/3600))
    save('status.json',dict(stage='training_complete' if epochs>3 else 'pilot_complete',epochs=epochs,best_val_ade=best))

def pipeline():
    import subprocess,fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':'2'}
        tasks=[('training',['recogdrive_dd.py','train','--epochs','200'],'recogdrive_dd_train.log'),('evaluating',['recogdrive_dd_score.py'],'recogdrive_dd_score.log')]
        for stage,arguments,log in tasks:
            save('pipeline_status.json',dict(stage=stage,gpu=2,auto_pdms=True))
            with (ROOT/'logs'/log).open('a') as output:
                subprocess.run([str(BASE/'env/bin/python'),'-u',str(BASE/'bimodal_smoke'/arguments[0]),*arguments[1:]],env=environment,cwd=BASE,stdout=output,stderr=subprocess.STDOUT,check=True)
        result=json.loads((OUT/'evaluation.json').read_text());assert result['successful_scenarios']==200 and result['failed_scenarios']==0
        save('pipeline_status.json',dict(stage='complete',epochs=200,successful_scenarios=200,pdms=result['score']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['benchmark','pilot','train','pipeline']);parser.add_argument('--epochs',type=int,default=3);args=parser.parse_args()
    try:
        if args.mode=='benchmark':benchmark()
        elif args.mode=='pipeline':pipeline()
        else:pilot(args.epochs,resume=args.mode=='train')
    except Exception as exc:
        import traceback
        failure=dict(stage='failed',error=str(exc),traceback=traceback.format_exc())
        save('status.json',failure)
        if args.mode=='pipeline':save('pipeline_status.json',failure)
        raise
