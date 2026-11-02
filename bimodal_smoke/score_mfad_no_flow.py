"""Two migrated VLA methods, existing fixed-200 lab NAVSIM PDM protocol."""
import os,sys,json,lzma,pickle,time,hashlib,csv,argparse
from pathlib import Path
from dataclasses import asdict
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla');LAB=BASE.parent
sys.path.insert(0,str(BASE/'multiview'));sys.path.insert(1,str(LAB/'repo'))
os.environ['NUPLAN_MAPS_ROOT']=str(LAB/'data/maps');os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0'
import torch,numpy as np
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
def main(variant,smoke):
 from train_mfad_no_flow import CachedPlanner
 out=BASE/'outputs/mfad_no_flow'
 torch.set_num_threads(2)
 tensors=torch.load(BASE/'outputs/shared/test_features.pt',map_location='cpu',weights_only=False)
 tokens=json.loads((LAB/'turboad/navtest_subset_tokens.json').read_text());assert tokens==tensors['tokens']
 with initialize_config_dir(config_dir=str(LAB/'repo/navsim/planning/script/config/pdm_scoring'),version_base=None):cfg=compose(config_name='default_run_pdm_score',overrides=['train_test_split=navtest','worker=sequential'])
 sim=instantiate(cfg.simulator);scorer=instantiate(cfg.scorer);assert sim.proposal_sampling==scorer.proposal_sampling
 caches=MetricCacheLoader(LAB/'turboad/evaluation/metric_cache');assert set(tokens)<=set(caches.tokens)
 for label,filename in [('initial','initial.pt'),('trained','best_val_ade.pt')]:
  dest=out/('evaluation_smoke_'+label+'.json' if smoke else 'evaluation_'+label+'.json')
  if dest.exists():continue
  ck=torch.load(out/filename,map_location='cpu',weights_only=False)
  if not smoke and label=='trained':assert json.loads((out/'history.json').read_text())[-1]['epoch']==200
  net=CachedPlanner(out/'train_kmeans_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
  del net.vision;del net.language
  net.load_state_dict(ck['model'],strict=True);net.eval();rows=[];started=time.time()
  for i,token in enumerate(tokens[:1] if smoke else tokens):
   torch.manual_seed(999);torch.cuda.manual_seed_all(999)
   v=tensors['vision'][i:i+1].cuda().float();cmd=tensors['command'][i:i+1]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net([v[:,:180],v[:,180:225],v[:,225:]],tensors['language'][cmd].cuda().float(),tensors['mask'][cmd].cuda(),tensors['ego'][i:i+1].cuda())['trajectory'][0].float().cpu().numpy()
   assert pred.shape==(8,3) and np.isfinite(pred).all()
   with lzma.open(caches.metric_cache_paths[token],'rb') as f:cache=pickle.load(f)
   result=pdm_score(metric_cache=cache,model_trajectory=Trajectory(pred,TrajectorySampling(num_poses=8,interval_length=.5)),future_sampling=sim.proposal_sampling,simulator=sim,scorer=scorer)
   rows.append(dict(token=token,valid=True,**asdict(result)));assert np.isfinite(rows[-1]['score'])
   (out/'evaluation_progress.json').write_text(json.dumps(dict(label=label,done=len(rows),total=1 if smoke else 200,seconds=time.time()-started)))
  avg={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('token','valid')}
  result=dict(**avg,pdms_100=100*avg['score'],successful_scenarios=len(rows),failed_scenarios=0,checkpoint_epoch=ck['epoch'],seconds=time.time()-started,checkpoint_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest())
  dest.write_text(json.dumps(result,indent=2))
  with (out/('scores_smoke_'+label+'.csv' if smoke else 'scores_'+label+'.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows+[dict(token='average',valid=True,**avg)])
  print(label,json.dumps(result),flush=True);del net,ck;torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('variant',choices=['cross','mf']);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.variant,a.smoke)
