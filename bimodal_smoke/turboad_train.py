"""TurboAD bounded IL pilot with log-disjoint validation; no test selection."""
import sys,json,time,random,hashlib,argparse
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import numpy as np,torch
from model import BimodalPlanner

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=32);p.add_argument('--run',default='pilot_seed42');a=p.parse_args()
    assert a.run in ('pilot_seed42','expanded_seed42')
    assert 1<=a.epochs<=40 and 1<=a.batch<=64
    torch.set_num_threads(2);seed(42);root=BASE/'turboad';out=root/a.run;out.mkdir(exist_ok=True)
    payload=torch.load(root/'samples.pt',map_location='cpu',weights_only=False);samples=payload['samples']
    train=[s for s in samples if s['split']=='train'];val=[s for s in samples if s['split']=='val']
    assert train and val and not ({s['log_name'] for s in train}&{s['log_name'] for s in val})
    keys=('vision','lidar','status','target')
    def batch(rows):return [torch.stack([s[k] for s in rows]).cuda().float() for k in keys]
    net=BimodalPlanner('cross',BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    def evaluate():
        net.eval();preds=[];targets=[];losses=[];seed(123)
        with torch.no_grad():
            for i in range(0,len(val),a.batch):
                v,l,s,y=batch(val[i:i+a.batch])
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(v,l,s)['trajectory']
                preds.append(pred.float().cpu());targets.append(y.cpu())
                net.train()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,l,s,target=y)['trajectory_loss']
                losses.append((float(loss),len(y)));net.eval()
        pred=torch.cat(preds);gt=torch.cat(targets);distance=(pred[:,:,:2]-gt[:,:,:2]).norm(dim=-1)
        stop=gt[:,:,:2].norm(dim=-1);moving=gt[:,-1,:2].norm(dim=-1)>2
        metrics=dict(val_loss=sum(x*n for x,n in losses)/len(val),ade=float(distance.mean()),fde=float(distance[:,-1].mean()),
            moving_ade=float(distance[moving].mean()) if moving.any() else None,moving_count=int(moving.sum()),
            stop_ade=float(stop.mean()),stop_fde=float(stop[:,-1].mean()))
        assert torch.isfinite(pred).all() and all(np.isfinite(v) for v in metrics.values() if v is not None)
        return metrics,pred
    initial,_=evaluate();history=[dict(epoch=0,**initial)];print(json.dumps(history[-1]),flush=True)
    best=float('inf');started=time.perf_counter();steps=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,a.epochs+1):
        seed(42+epoch);order=torch.randperm(len(train)).tolist();loss_sum=0;norms=[];net.train()
        for i in range(0,len(order),a.batch):
            rows=[train[j] for j in order[i:i+a.batch]];v,l,s,y=batch(rows);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,l,s,target=y)['trajectory_loss']
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            steps+=1;loss_sum+=float(loss)*len(rows);norms.append(float(norm))
        metrics,pred=evaluate();row=dict(epoch=epoch,step=steps,train_loss=loss_sum/len(train),grad_norm_mean=float(np.mean(norms)),**metrics)
        history.append(row);print(json.dumps(row),flush=True)
        checkpoint=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=steps,name='TurboAD',variant='cross',seed=42,
            config=dict(batch=a.batch,lr=1e-4,epochs=a.epochs),scope=payload['scope'],manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
        torch.save(checkpoint,out/'latest.pt')
        if metrics['ade']<best:
            best=metrics['ade'];torch.save(checkpoint,out/'best_val_ade.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'val_predictions.json').write_text(json.dumps([dict(token=s['token'],prediction=p.tolist(),target=s['target'].tolist()) for s,p in zip(val,pred)]))
    summary=dict(name='TurboAD',train_samples=len(train),val_samples=len(val),train_logs=len({s['log_name'] for s in train}),val_logs=len({s['log_name'] for s in val}),
        epochs=a.epochs,steps=steps,batch=a.batch,seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        initial=initial,final=history[-1],best_val_ade=best,scope='Internal log-disjoint navtrain validation, NOT navtest PDMS')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('TURBOAD_PILOT_OK',json.dumps(summary),flush=True)
if __name__=='__main__':main()
