"""Single-GPU IL: official ReCogDrive conventional budget, shared demo10 data."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import numpy as np,torch
from model import BimodalPlanner
ROOT=BASE/'demo10'

def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def checkpoint(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def lr_at(epoch):
    return 1e-4*(epoch+1)/3 if epoch<3 else 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*(epoch-3)/197))

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['cross','mf','mf_plus','mf_plus_small'],required=True);p.add_argument('--micro-batch',type=int,default=128);p.add_argument('--epochs',type=int,default=200);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert 128%a.micro_batch==0 and a.micro_batch>0 and 1<=a.epochs<=200
    torch.set_num_threads(2);seed(42);out=ROOT/(('smoke_' if a.smoke else '')+{'cross':'turboad','mf':'turbo_mfad','mf_plus':'turboad_plus','mf_plus_small':'turboad_plus_small'}[a.variant]);out.mkdir(exist_ok=True)
    if a.smoke:
        if a.variant=='mf_plus_small':raise ValueError('Use demo10_plus_check.py --small for real resized multiview smoke')
        assert a.epochs<=2
        rows=torch.load(BASE/('turboad/plus_samples.pt' if a.variant=='mf_plus' else 'turboad/samples.pt'),map_location='cpu',weights_only=False)['samples'];manifest_hash=hashlib.sha256((BASE/'turboad/manifest.json').read_bytes()).hexdigest()
    else:
        feature_info=json.loads((ROOT/({'mf_plus':'plus_features_complete.json','mf_plus_small':'plus_small_features_complete.json'}.get(a.variant,'features_complete.json'))).read_text());manifest_hash=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest();assert feature_info['manifest_sha256']==manifest_hash
        rows=[]
        for path in feature_info['files']:rows.extend(torch.load(path,map_location='cpu',weights_only=False))
        assert len(rows)==feature_info['samples']
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val']
    assert train and val and not ({rows[i]['log_name'] for i in train}&{rows[i]['log_name'] for i in val})
    manifest=json.loads((BASE/'turboad/manifest.json' if a.smoke else ROOT/'manifest.json').read_text())
    assert [(r['token'],r['log_name'],r['split']) for r in rows]==[(r['token'],r['log_name'],r['split']) for r in manifest]
    tokens=[r['token'] for r in rows];tensors={k:torch.stack([r[k] for r in rows]) for k in ('vision','lidar','status','target')};del rows
    def batch(ids):return [tensors[k][ids].cuda().float() for k in ('vision','lidar','status','target')]
    net=BimodalPlanner(a.variant,BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
    config=dict(variant=a.variant,smoke=a.smoke,seed=42,micro_batch=a.micro_batch,effective_batch=128,gradient_accumulation=128//a.micro_batch,world_size=1,epochs=a.epochs,lr=1e-4,min_lr=1e-6,warmup_epochs=3,scheduler_horizon=200,weight_decay=1e-4,betas=[.9,.95],precision='bf16',frozen_encoder=True,input=({'mf_plus':'F0/L0/R0 RGB 288x160','mf_plus_small':'F0 RGB 288x160 + L0/R0 RGB 144x80'}.get(a.variant,'front RGB 288x160')+' + LiDAR BEV 256x256; ego status'),manifest_sha256=manifest_hash)
    atomic_json(out/'config.json',config)
    def evaluate():
        net.eval();seed(123);ade=fde=loss_sum=0.;n=0
        with torch.no_grad():
            for i in range(0,len(val),a.micro_batch):
                ids=val[i:i+a.micro_batch];v,l,s,y=batch(ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(v,l,s)['trajectory']
                d=(pred.float()[:,:,:2]-y[:,:,:2]).norm(dim=-1);ade+=float(d.sum());fde+=float(d[:,-1].sum());n+=len(ids)
                net.train()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,l,s,target=y)['trajectory_loss']
                assert torch.isfinite(loss) and torch.isfinite(pred).all();loss_sum+=float(loss)*len(ids);net.eval()
        return dict(ade=ade/(n*8),fde=fde/n,val_loss=loss_sum/n)
    start=0;steps=0;best=float('inf');elapsed=0.;history=[]
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert state['config']==config and state['manifest_sha256']==manifest_hash
        net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);start=state['epoch'];steps=state['step'];best=state['best'];elapsed=state['elapsed'];history=json.loads((out/'history.json').read_text())[:start+1]
    else:
        checkpoint(out/'initial.pt',dict(model=net.state_dict(),variant=a.variant,config=config,manifest_sha256=manifest_hash,epoch=0,step=0))
        history=[dict(epoch=0,**evaluate())];atomic_json(out/'history.json',history)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();print('TRAIN_START',json.dumps(dict(train=len(train),val=len(val),config=config,start_epoch=start)),flush=True)
    for epoch in range(start+1,a.epochs+1):
        seed(42+epoch);order=[train[i] for i in torch.randperm(len(train)).tolist()];net.train();loss_sum=0.;norms=[];epoch_start=time.perf_counter();lr=lr_at(epoch-1)
        for group in opt.param_groups:group['lr']=lr
        for offset in range(0,len(order),128):
            effective=order[offset:offset+128];opt.zero_grad(set_to_none=True)
            for j in range(0,len(effective),a.micro_batch):
                ids=effective[j:j+a.micro_batch];v,l,s,y=batch(ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,l,s,target=y)['trajectory_loss']
                assert torch.isfinite(loss);(loss*(len(ids)/len(effective))).backward();loss_sum+=float(loss)*len(ids)
            norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)));opt.step();steps+=1
        metrics=evaluate();row=dict(epoch=epoch,step=steps,lr=lr,train_loss=loss_sum/len(train),grad_norm_mean=float(np.mean(norms)),seconds=time.perf_counter()-epoch_start,**metrics);history.append(row)
        improved=metrics['ade']<best;best=min(best,metrics['ade'])
        state=dict(model=net.state_dict(),optimizer=opt.state_dict(),variant=a.variant,config=config,manifest_sha256=manifest_hash,epoch=epoch,step=steps,best=best,elapsed=elapsed+time.perf_counter()-started)
        atomic_json(out/'history.json',history)
        if improved:checkpoint(out/'best_val_ade.pt',state)
        checkpoint(out/'latest.pt',state)
        atomic_json(out/'progress.json',dict(**row,best_val_ade=best,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=state['elapsed']))
        print(json.dumps(row),flush=True)
    summary=dict(config=config,train_samples=len(train),val_samples=len(val),epochs=a.epochs,steps=steps,seconds=elapsed+time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,initial=history[0],final=history[-1],best_val_ade=best,scope='legacy 129/338 smoke only' if a.smoke else 'first 20/200 shards, official navtrain; log-disjoint internal validation; fixed 200 navtest evaluated separately')
    atomic_json(out/'summary.json',summary);print('TRAIN_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
