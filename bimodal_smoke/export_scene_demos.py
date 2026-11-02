"""Run selected fixed-test scenes on lab GPU0; export actual predictions only."""
import sys,json,pickle,lzma
from pathlib import Path
import numpy as np
import torch
from pyquaternion import Quaternion
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla');LAB=BASE.parent
sys.path.insert(0,str(BASE/'multiview'))
from navsim.common.dataloader import MetricCacheLoader
from train_turbo_cross_vla import CachedPlanner as Cross
from train_visual_mf_vla import CachedPlanner as MF
from train_mfad_no_flow import CachedPlanner as NoFlow
torch.set_num_threads(2)
cache=torch.load(BASE/'outputs/shared/test_features.pt',map_location='cpu',weights_only=False)
manifest=json.loads((BASE/'outputs/shared/test_manifest.json').read_text())
# First occurrence of each navigation command, fixed manifest order; not score selected.
indices=[];seen=set()
for i,r in enumerate(manifest):
 if r['command'] not in seen:indices.append(i);seen.add(r['command'])
 if len(indices)==3:break
scenes=[]
for i in indices:
 r=manifest[i];frames=pickle.load((LAB/'data/openscene-v1.1/meta_datas/test'/f"{r['log']}.pkl").open('rb'))
 ix=next(j for j,f in enumerate(frames) if f['token']==r['token']);f=frames[ix]
 yaw=Quaternion(f['ego2global_rotation']).yaw_pitch_roll[0];pos=np.asarray(f['ego2global_translation'])[:2];c,s=np.cos(yaw),np.sin(yaw)
 gt=[]
 for a in frames[ix+1:ix+9]:
  d=np.asarray(a['ego2global_translation'])[:2]-pos;gt.append([float(c*d[0]+s*d[1]),float(-s*d[0]+c*d[1])])
 assert len(gt)==8
 scenes.append(dict(index=i,token=r['token'],log=r['log'],command=r['command'],images=r['images'],ground_truth_xy=gt,predictions={}))
for name,cls in [('turbo_cross_vla',Cross),('multiview_mf_vla',MF),('mfad_no_flow',NoFlow)]:
 out=BASE/'outputs'/name;ck=torch.load(out/'best_val_ade.pt',map_location='cpu',weights_only=False)
 net=cls(out/'train_kmeans_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
 del net.vision;del net.language
 net.load_state_dict(ck['model'],strict=True);net.eval()
 for scene in scenes:
  i=scene['index'];torch.manual_seed(999);torch.cuda.manual_seed_all(999)
  v=cache['vision'][i:i+1].cuda().float();cmd=cache['command'][i:i+1]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net([v[:,:180],v[:,180:225],v[:,225:]],cache['language'][cmd].cuda().float(),cache['mask'][cmd].cuda(),cache['ego'][i:i+1].cuda())['trajectory'][0].float().cpu().numpy()
  scene['predictions'][name]=dict(epoch=ck['epoch'],trajectory=pred.tolist(),ADE_m=float(np.linalg.norm(pred[:,:2]-np.asarray(scene['ground_truth_xy']),axis=1).mean()))
 del net,ck;torch.cuda.empty_cache()
dest=BASE/'outputs/scene_demos';dest.mkdir(exist_ok=True)
(dest/'predictions.json').write_text(json.dumps(dict(selection='first occurrence of each navigation command in fixed test manifest; no score filtering',scenes=scenes),indent=2))
print('EXPORTED',len(scenes),flush=True)
