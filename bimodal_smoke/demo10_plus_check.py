"""Real encoded multiview forward/backward and strict checkpoint smoke."""
import argparse,json,sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner
p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');a=p.parse_args();label='plus_small' if a.small else 'plus';variant='mf_plus_small' if a.small else 'mf_plus'
torch.set_num_threads(2);torch.manual_seed(42);root=BASE/'demo10';path=sorted((root/(label+'_features')).glob('*.pt'))[0];rows=torch.load(path,map_location='cpu',weights_only=False)[:2]
v,l,s,y=[torch.stack([r[k] for r in rows]).cuda().float() for k in ('vision','lidar','status','target')];v.requires_grad_(True);l.requires_grad_(True)
net=BimodalPlanner(variant,BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda().train()
with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,l,s,target=y)['trajectory_loss']
assert torch.isfinite(loss);loss.backward();boundaries=[0,180,225,270] if a.small else [0,180,360,540]
grads=[float(v.grad[:,left:right].abs().sum()) for left,right in zip(boundaries[:-1],boundaries[1:])];grads.append(float(l.grad.abs().sum()))
assert all(x>0 for x in grads) and torch.isfinite(v.grad).all() and torch.isfinite(l.grad).all()
net.load_state_dict(net.state_dict(),strict=True);net.eval();torch.manual_seed(999)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=net(v.detach(),l.detach(),s)['trajectory']
assert prediction.shape==(2,8,3) and torch.isfinite(prediction).all()
report=dict(variant=variant,tokens=v.shape[1],loss=float(loss),front_left_right_lidar_gradient_sums=grads,prediction_shape=list(prediction.shape),finite=True)
(root/(label+'_model_check.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
