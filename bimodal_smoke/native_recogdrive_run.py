"""Same-manifest native Stage-2 IL; upstream feature builder/agent/head unchanged."""
import os,sys,json,time,random,math,hashlib,subprocess,traceback,tarfile,shutil
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');CODE=BASE/'bimodal_smoke';ROOT=BASE/'demo10';OUT=ROOT/'native_recogdrive'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(BASE/'repo'))
import numpy as np,torch
from native_recogdrive import save_json
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
MANIFEST=json.loads((ROOT/'manifest.json').read_text());HASH=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
CACHE=OUT/'features';CACHE.mkdir(exist_ok=True)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def prepare():
    meta=OUT/'metadata';meta.mkdir(exist_ok=True);wanted={r['log_name'] for r in MANIFEST}
    with tarfile.open(BASE/'data/metadata_trainval.tgz','r|gz') as archive:
        for member in archive:
            name=Path(member.name).name
            if member.isfile() and Path(name).stem in wanted and not (meta/name).exists():(meta/name).write_bytes(archive.extractfile(member).read())
            if all((meta/(name+'.pkl')).exists() for name in wanted):break
    assert all((meta/(name+'.pkl')).exists() for name in wanted)
def cache(rank):
    rows=MANIFEST[rank::2]
    loader=SceneLoader(OUT/'metadata',BASE/'data/sensor_blobs/trainval',SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=sorted({r['log_name'] for r in rows}),tokens=[r['token'] for r in rows]),SensorConfig(cam_f0=[3],cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False),load_image_path=True)
    builder=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(BASE/'models/ReCogDrive-VLM-2B'),device='cuda:0');start=time.perf_counter()
    for i,row in enumerate(rows):
        dest=CACHE/(row['token']+'.pt')
        if not dest.exists():
            with torch.inference_mode():features=builder.compute_features(loader.get_agent_input_from_token(row['token']))
            hidden=features['last_hidden_state'];assert torch.equal(hidden,hidden.bfloat16().float());features['last_hidden_state']=hidden.bfloat16()
            item=dict(features=features,target=torch.tensor(row['target'],dtype=torch.float32),token=row['token'],split=row['split'],manifest_sha256=HASH)
            tmp=dest.with_suffix('.tmp');torch.save(item,tmp);tmp.replace(dest)
        if i%25==0:save_json(f'cache_{rank}_progress.json',dict(done=i+1,total=len(rows),seconds=time.perf_counter()-start));print('CACHE',rank,i+1,len(rows),flush=True)
    save_json(f'cache_{rank}_complete.json',dict(samples=len(rows),seconds=time.perf_counter()-start,manifest_sha256=HASH))
class Cached(torch.utils.data.Dataset):
    def __init__(self,split):self.rows=[r for r in MANIFEST if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];item=torch.load(CACHE/(row['token']+'.pt'),map_location='cpu',weights_only=False)
        assert item['token']==row['token'] and item['manifest_sha256']==HASH
        return item['features'],{'trajectory':item['target']}
def train():
    seed(42);torch.set_num_threads(2)
    net=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=str(BASE/'models/ReCogDrive-VLM-2B'),checkpoint_path=None,cache_hidden_state=True)
    probe=json.loads((OUT/'probe.json').read_text());valid=[r['batch'] for r in probe['train_benchmark'] if not r.get('oom') and r['peak_allocated_gib']<19];micro=max(valid)
    assert 128%micro==0
    config=dict(model='native ReCogDrive 2B small IL',seed=42,micro_batch=micro,effective_batch=128,epochs=200,manifest_sha256=HASH,planner_initialization='random',vlm='official pretrained frozen',native_feature_shape=[2800,1536],precision='bf16',selection='minimum internal validation ADE',training_samples=8688,validation_samples=1930)
    save_json('config.json',config)
    opt=torch.optim.AdamW(net.action_head.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
    trainset=Cached('train');valset=Cached('val');assert len(trainset)==8688 and len(valset)==1930
    def loader(ds,shuffle=False):return torch.utils.data.DataLoader(ds,batch_size=micro,shuffle=shuffle,num_workers=2,pin_memory=True,prefetch_factor=1)
    def evaluate():
        seed(123);net.eval();ade=fde=0.;n=0
        with torch.no_grad():
            for f,t in loader(valset):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(f)['pred_traj']
                y=t['trajectory'].cuda();assert torch.isfinite(pred).all();d=(pred.float()[...,:2]-y[...,:2]).norm(dim=-1);ade+=d.sum().item();fde+=d[:,-1].sum().item();n+=len(y)
        return dict(ade=ade/(8*n),fde=fde/n)
    start=0;steps=0;best=float('inf');history=[];elapsed=0.
    if (OUT/'latest.pt').exists():
        state=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);assert state['config']==config
        net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);start=state['epoch'];steps=state['step'];best=state['best'];elapsed=state['elapsed'];history=json.loads((OUT/'history.json').read_text())
    else:
        torch.save({'state_dict':{'agent.'+k:v for k,v in net.state_dict().items()}},OUT/'initial.ckpt');history=[dict(epoch=0,**evaluate())];save_json('history.json',history)
    began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start+1,201):
        seed(42+epoch);net.train();epoch_start=time.perf_counter();loss_sum=0.;norms=[];consumed=0;effective_count=0;opt.zero_grad(set_to_none=True)
        lr=1e-4*epoch/3 if epoch<=3 else 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*(epoch-4)/197))
        for group in opt.param_groups:group['lr']=lr
        for f,t in loader(trainset,True):
            t={k:v.cuda(non_blocking=True) for k,v in t.items()}
            b=len(t['trajectory']);denominator=min(128,len(trainset)-(consumed//128)*128)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(f,t);loss=net.compute_loss(f,t,pred)
            assert torch.isfinite(loss);(loss*b/denominator).backward();loss_sum+=loss.item()*b;consumed+=b;effective_count+=b
            if effective_count==128 or consumed==len(trainset):
                norms.append(float(torch.nn.utils.clip_grad_norm_(net.action_head.parameters(),1.,error_if_nonfinite=True)));opt.step();opt.zero_grad(set_to_none=True);steps+=1;effective_count=0
        metrics=evaluate();improved=metrics['ade']<best;best=min(best,metrics['ade'])
        row=dict(epoch=epoch,step=steps,lr=lr,train_loss=loss_sum/len(trainset),seconds=time.perf_counter()-epoch_start,grad_norm_mean=float(np.mean(norms)),**metrics);history.append(row);save_json('history.json',history)
        state=dict(model=net.state_dict(),optimizer=opt.state_dict(),config=config,epoch=epoch,step=steps,best=best,elapsed=elapsed+time.perf_counter()-began)
        if improved:torch.save({'state_dict':{'agent.'+k:v for k,v in net.state_dict().items()},'epoch':epoch,'config':config},OUT/'best_val_ade.ckpt')
        tmp=OUT/'latest.tmp';torch.save(state,tmp);tmp.replace(OUT/'latest.pt');save_json('progress.json',dict(**row,best_val_ade=best,elapsed_seconds=state['elapsed'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30));print(json.dumps(row),flush=True)
    save_json('summary.json',dict(config=config,seconds=state['elapsed'],steps=steps,best_val_ade=best,initial=history[0],final=history[-1],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
def run(args,gpu,log):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(BASE/'repo'),'HF_HOME':str(BASE/'hf_cache'),'TMPDIR':str(BASE/'tmp'),'MPLCONFIGDIR':str(BASE/'tmp/matplotlib')}
    f=(ROOT/'logs'/log).open('a');return subprocess.Popen([str(BASE/'env/bin/python'),'-u',*args],cwd=BASE,env=env,stdout=f,stderr=subprocess.STDOUT)
def pipeline():
    probe=json.loads((OUT/'probe.json').read_text())
    assert shutil.disk_usage(BASE).free>probe['estimated_cache_gib']*2**30+20*2**30,'Insufficient disk space including 20 GiB reserve'
    save_json('pipeline_status.json',dict(stage='metadata',manifest_sha256=HASH));prepare()
    save_json('pipeline_status.json',dict(stage='encoding',gpus=[0,2]));workers=[run(['bimodal_smoke/native_recogdrive_run.py','cache',str(rank)],gpu,f'native_cache_{rank}.log') for rank,gpu in enumerate([0,2])]
    codes=[p.wait() for p in workers];assert codes==[0,0],codes
    assert all((CACHE/(r['token']+'.pt')).exists() for r in MANIFEST)
    save_json('pipeline_status.json',dict(stage='training',gpu=2,epochs=200));p=run(['bimodal_smoke/native_recogdrive_run.py','train'],2,'native_train.log');assert p.wait()==0
    save_json('pipeline_status.json',dict(stage='evaluating',gpu=2));p=run(['bimodal_smoke/native_recogdrive_score.py'],2,'native_score.log');assert p.wait()==0
    save_json('pipeline_status.json',dict(stage='complete'))
if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        mode=sys.argv[1]
        if mode=='cache':cache(int(sys.argv[2]))
        elif mode=='train':train()
        elif mode=='pipeline':pipeline()
    except Exception as e:
        save_json('pipeline_status.json',dict(stage='failed',error=str(e),traceback=traceback.format_exc()));raise
