"""Real-cache flow forward/backward and fixed-noise overfit smoke."""
import json,time,sys
from pathlib import Path
sys.path.append('/data/ruiqiwang/recogdrive_baseline/env/lib/python3.10/site-packages')
import torch
from mf_flow import FlowPlanner,flow_weight
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
torch.set_num_threads(2);torch.manual_seed(42)
net=FlowPlanner('mf',BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda()
info=json.loads((BASE/'demo10/features_complete.json').read_text())
rows=torch.load(info['files'][0],map_location='cpu',weights_only=False)[:8]
v,l,s,y=[torch.stack([r[k] for r in rows]).cuda().float() for k in ('vision','lidar','status','target')]
opt=torch.optim.AdamW(net.parameters(),lr=1e-4)
noise={'rgb':torch.randn(8,180,768,device='cuda'),'lidar':torch.randn(8,256,256,device='cuda')}
t=torch.full((8,),.5,device='cuda')
def fixed():
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        return {m:float(net.flow_loss(v,l,m,noise[m],t)[0]) for m in noise}
initial=fixed();start=time.monotonic()
for step in range(20):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        aux,cos=net.flow_loss(v,l,('rgb','lidar')[step%2],noise[('rgb','lidar')[step%2]],t)
        traj=net(v,l,s,target=y)['trajectory_loss'];loss=traj+.1*aux
    assert torch.isfinite(loss)
    loss.backward()
    assert net.fusion.blocks[0].attention.parameters().__next__().grad is not None
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
final=fixed();assert all(final[k]<initial[k] for k in initial),(initial,final)
# At t=0 changing the target cannot affect the predicted velocity.
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    a=net.flow_loss(v,l,'rgb',noise['rgb'],torch.zeros_like(t),True)[2]
    b=net.flow_loss(v+3,l,'rgb',noise['rgb'],torch.zeros_like(t),True)[2]
    assert torch.equal(a,b),'RGB target leakage at t=0'
    a=net.flow_loss(v,l,'lidar',noise['lidar'],torch.zeros_like(t),True)[2]
    b=net.flow_loss(v,l+torch.randn_like(l),'lidar',noise['lidar'],torch.zeros_like(t),True)[2]
    assert torch.equal(a,b),'LiDAR target leakage at t=0'
assert flow_weight(1,100)==.1 and flow_weight(100,100)==0
result=dict(initial=initial,final=final,steps=20,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,finite=True,target_leakage_checks='passed')
(BASE/'demo10/flow_smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
