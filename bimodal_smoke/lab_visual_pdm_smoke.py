"""Real three-camera input through MF-VLA and original scorer; integration only."""
import os,sys,json,pickle,lzma
from pathlib import Path
from dataclasses import asdict
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla');LAB=BASE.parent
sys.path.insert(0,str(BASE/'multiview'));os.environ['NUPLAN_MAPS_ROOT']=str(LAB/'data/maps');os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0'
import torch,numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from visual_mf_vla import MFPlanner
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 r=json.loads((BASE/'outputs/shared/test_manifest.json').read_text())[0]
 net=MFPlanner(LAB/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda().eval()
 images=[]
 for path,size in zip(r['images'],[(160,288),(80,144),(80,144)]):
  tf=transforms.Compose([transforms.Resize(size),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
  with Image.open(path) as im:images.append(tf(im.convert('RGB'))[None].cuda())
 text=AutoTokenizer.from_pretrained(BASE/'weights/bert-base-uncased',local_files_only=True)(['navigation instruction: '+['turn left','go straight','turn right','unknown'][r['command']]],padding='max_length',max_length=12,truncation=True,return_tensors='pt').to('cuda')
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net(images,text['input_ids'],text['attention_mask'],torch.tensor([r['ego']],device='cuda'))['trajectory'][0].float().cpu().numpy()
 assert pred.shape==(8,3) and np.isfinite(pred).all()
 with initialize_config_dir(config_dir=str(LAB/'repo/navsim/planning/script/config/pdm_scoring'),version_base=None):cfg=compose(config_name='default_run_pdm_score',overrides=['train_test_split=navtest','worker=sequential'])
 sim=instantiate(cfg.simulator);scorer=instantiate(cfg.scorer);caches=MetricCacheLoader(LAB/'turboad/evaluation/metric_cache')
 with lzma.open(caches.metric_cache_paths[r['token']],'rb') as f:cache=pickle.load(f)
 result=pdm_score(metric_cache=cache,model_trajectory=Trajectory(pred,TrajectorySampling(num_poses=8,interval_length=.5)),future_sampling=sim.proposal_sampling,simulator=sim,scorer=scorer)
 result=dict(scope='integration only: random initialization, official anchors, real current three cameras + BERT command, native fixed-test cache/scorer; not a trained result',token=r['token'],trajectory_shape=list(pred.shape),**asdict(result));assert np.isfinite(result['score'])
 (BASE/'outputs/pdm_integration_smoke.json').write_text(json.dumps(result,indent=2));print('PDM_INTEGRATION_OK',json.dumps(result),flush=True)
if __name__=='__main__':main()
