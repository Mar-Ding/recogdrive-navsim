"""Bounded real-data training smoke, gradients, input dependence and strict reload."""
import sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from model import BimodalPlanner

ROOT=BASE/'bimodal_smoke'

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def batch(samples,ids):
    return [torch.stack([samples[i][key] for i in ids]).cuda() for key in ('vision','lidar','status','target')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['cross','mf'],required=True)
    parser.add_argument('--steps',type=int,default=50);args=parser.parse_args()
    assert 1<=args.steps<=500
    torch.set_num_threads(2);seed(42)
    data=torch.load(ROOT/'artifacts/debug_samples.pt',map_location='cpu',weights_only=False)
    samples=data['samples'];out=ROOT/'outputs'/args.variant;out.mkdir(parents=True,exist_ok=True)
    anchor=ROOT/'upstream/kmeans_navsim_traj_20.npy'
    net=BimodalPlanner(args.variant,anchor).cuda()
    common_initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('fusion.')}
    fusion_initial={k:v.detach().clone() for k,v in net.fusion.named_parameters()}
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    inputs=batch(samples,list(range(len(samples))))
    def fixed_loss():
        seed(123);net.train()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return float(net(*inputs[:3],target=inputs[3])['trajectory_loss'])
    before=fixed_loss();seed(42)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();rows=[];grad_report={}
    for step in range(args.steps):
        net.train();v,l,s,y=batch(samples,[step%len(samples)])
        # Input grads prove that both sensor streams reach the supervised trajectory objective.
        if step==0:v.requires_grad_();l.requires_grad_()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=net(v,l,s,target=y);loss=result['trajectory_loss']
        assert torch.isfinite(loss),f'Nonfinite loss at {step}'
        loss.backward()
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if step==0:
            grad_report={'rgb_input_grad_norm':float(v.grad.norm()),'lidar_input_grad_norm':float(l.grad.norm())}
            for name,module in [('fusion',net.fusion),('lidar_encoder',net.lidar_encoder),('decoder',net.trajectory_head)]:
                grad_report[name+'_grad_norm']=float(torch.stack([p.grad.float().norm().square() for p in module.parameters() if p.grad is not None]).sum().sqrt())
            assert all(x>0 and np.isfinite(x) for x in grad_report.values()),grad_report
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step()
        rows.append(dict(step=step+1,loss=float(loss),grad_norm=float(norm)))
        if step==0 or (step+1)%10==0:print(json.dumps(rows[-1]),flush=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-started
    after=fixed_loss()
    deltas={k:float((p.detach()-fusion_initial[k]).abs().max()) for k,p in net.fusion.named_parameters()}
    assert max(deltas.values())>0
    net.eval()
    def predict(v=inputs[0],l=inputs[1]):
        seed(999)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return net(v,l,inputs[2])['trajectory'].float()
    pred=predict();assert pred.shape==(len(samples),8,3) and torch.isfinite(pred).all()
    rgb_delta=float((pred-predict(v=torch.zeros_like(inputs[0]))).abs().max())
    lidar_delta=float((pred-predict(l=torch.zeros_like(inputs[1]))).abs().max())
    assert rgb_delta>0 and lidar_delta>0,{'rgb_delta':rgb_delta,'lidar_delta':lidar_delta}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')}
    payload=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=args.steps,variant=args.variant,
                 config={'width':256,'depth':6,'rgb_size':[160,288],'lidar_size':[256,256],'horizon':8,'trajectory_dim':3},
                 source_sha256=sources,encoder=data['encoder'],scope=data['scope'])
    torch.save(payload,out/'latest.pt')
    saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    reload_net=BimodalPlanner(args.variant,anchor).cuda().eval();reload_net.load_state_dict(saved['model'],strict=True)
    seed(999)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reloaded=reload_net(*inputs[:3])['trajectory'].float()
    reload_error=float((reloaded-pred).abs().max());assert reload_error==0,reload_error
    # Preserve common initial parameters for a direct A/B initialization audit.
    torch.save(common_initial,out/'common_initial.pt')
    report=dict(variant=args.variant,steps=args.steps,samples=len(samples),scope=data['scope'],
                device=torch.cuda.get_device_name(),physical_gpu=__import__('os').environ['CUDA_VISIBLE_DEVICES'],
                trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                fixed_loss_before=before,fixed_loss_after=after,loss_decreased=after<before,
                gradients=grad_report,fusion_max_parameter_change=max(deltas.values()),
                rgb_ablation_max_difference=rgb_delta,lidar_ablation_max_difference=lidar_delta,
                strict_reload_max_difference=reload_error,train_seconds=train_seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                checkpoint_sha256=hashlib.sha256((out/'latest.pt').read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'training.json').write_text(json.dumps(rows,indent=2))
    (out/'predictions.json').write_text(json.dumps([dict(token=s['token'],prediction=p.tolist(),ground_truth=s['target'].tolist()) for s,p in zip(samples,pred.cpu())],indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,s,p in zip(axs,samples,pred.cpu().numpy()):
        gt=s['target'].numpy();ax.plot(gt[:,0],gt[:,1],'o-',label='GT');ax.plot(p[:,0],p[:,1],'o-',label=args.variant)
        ax.set_aspect('equal');ax.set_title(s['token'][:8]);ax.legend();ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Training smoke on three debug samples; not a benchmark');fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True);print('BIMODAL_TRAIN_SMOKE_OK',args.variant,flush=True)

if __name__=='__main__':main()
