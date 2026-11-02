"""Before full optimization: 3090 main/aux backward and finite trajectory checks."""
import sys,json
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla')
sys.path.insert(0,str(BASE/'multiview'))
import torch
from train_turbo_cross_vla import CachedPlanner as Cross
from train_visual_mf_vla import CachedPlanner as MF
def main():
 torch.set_num_threads(2);torch.manual_seed(42);results={}
 for name,cls in [('cross',Cross),('mf',MF)]:
  net=cls(BASE.parent/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
  del net.vision;del net.language
  b=2;v=[torch.randn(b,n,768,device='cuda') for n in (180,45,45)];lang=torch.randn(b,12,768,device='cuda');mask=torch.ones(b,12,device='cuda',dtype=torch.long);ego=torch.randn(b,4,device='cuda')
  target=torch.zeros(b,8,3,device='cuda');target[:,:,0]=torch.arange(1,9,device='cuda')[None]
  net.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(v,lang,mask,ego,{'trajectory':target})['loss']
  assert torch.isfinite(loss);loss.backward();row=dict(loss=float(loss),flow={})
  if name=='mf':
   for modality in ('vision','language','action'):
    net.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):aux,metrics=net.flow_loss(v,lang,mask,ego,target,modality)
    assert torch.isfinite(aux);aux.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.fusion.parameters())
    row['flow'][modality]={k:float(x) for k,x in metrics.items()}
  net.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net(v,lang,mask,ego)['trajectory']
  assert pred.shape==(2,8,3) and torch.isfinite(pred).all();row['trajectory_shape']=list(pred.shape);results[name]=row
  del net;torch.cuda.empty_cache()
 (BASE/'outputs/migration_smoke.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
