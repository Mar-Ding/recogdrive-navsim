"""Frozen native VLM online + hybrid head, identical fixed NAVSIM PDM protocol."""
import os,sys,json,csv,lzma,pickle,time,argparse,hashlib
from pathlib import Path
from dataclasses import asdict
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.insert(0,str(BASE/'repo'));sys.path.insert(0,str(BASE/'bimodal_smoke'))
os.environ['NUPLAN_MAPS_ROOT']=str(BASE/'data/maps');os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0'
import torch,numpy as np
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader,MetricCacheLoader
from navsim.common.dataclasses import SensorConfig,Trajectory
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
from navsim.evaluate.pdm_score import pdm_score
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from recogdrive_dd import ReCogDriveDD,OUT,ROOT,save

def main(smoke=False):
    torch.set_num_threads(2)
    tokens=json.loads((BASE/'turboad/navtest_subset_tokens.json').read_text());assert len(tokens)==200 and len(set(tokens))==200
    with initialize_config_dir(config_dir=str(BASE/'repo/navsim/planning/script/config/pdm_scoring'),version_base=None):cfg=compose(config_name='default_run_pdm_score',overrides=['train_test_split=navtest','worker=sequential'])
    simulator=instantiate(cfg.simulator);scorer=instantiate(cfg.scorer);assert simulator.proposal_sampling==scorer.proposal_sampling
    filt=instantiate(cfg.train_test_split.scene_filter);filt.tokens=tokens[:1] if smoke else tokens
    loader=SceneLoader(BASE/'data/openscene-v1.1/meta_datas/test',BASE/'data/sensor_blobs/test',filt,SensorConfig(cam_f0=[3],cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False),load_image_path=True)
    caches=MetricCacheLoader(BASE/'turboad/evaluation/metric_cache');assert set(filt.tokens)<=set(caches.tokens) and set(loader.tokens)==set(filt.tokens)
    path=OUT/'best_val_ade.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    digest=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest();assert state['config']['manifest_sha256']==digest
    history=json.loads((OUT/'history.json').read_text());best=min((r for r in history if r['epoch']>0),key=lambda r:r['ade']);assert best['epoch']==state['epoch']
    if not smoke:assert history[-1]['epoch']==200,'Do not score an incomplete training run as the final result'
    torch.manual_seed(42);net=ReCogDriveDD().cuda().eval();net.load_state_dict(state['model'],strict=True)
    checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest();del state
    builder=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(BASE/'models/ReCogDrive-VLM-2B'),device='cuda:0')
    rows=[];start=time.perf_counter();prefix='evaluation_smoke' if smoke else 'evaluation'
    for token in filt.tokens:
        torch.manual_seed(999);torch.cuda.manual_seed_all(999)
        with torch.inference_mode():
            f=builder.compute_features(loader.get_agent_input_from_token(token));assert f['last_hidden_state'].shape==(2800,1536)
            assert torch.equal(f['last_hidden_state'],f['last_hidden_state'].bfloat16().float())
            f['last_hidden_state']=f['last_hidden_state'].bfloat16();f={k:v[None].cuda() for k,v in f.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(f)['trajectory'][0].float().cpu().numpy()
        assert pred.shape==(8,3) and np.isfinite(pred).all()
        trajectory=Trajectory(pred,TrajectorySampling(num_poses=8,interval_length=.5))
        with lzma.open(caches.metric_cache_paths[token],'rb') as stream:cache=pickle.load(stream)
        result=pdm_score(metric_cache=cache,model_trajectory=trajectory,future_sampling=simulator.proposal_sampling,simulator=simulator,scorer=scorer)
        rows.append(dict(token=token,valid=True,**asdict(result)))
        assert np.isfinite(rows[-1]['score'])
        save(prefix+'_progress.json',dict(done=len(rows),total=len(filt.tokens),seconds=time.perf_counter()-start,checkpoint_epoch=best['epoch']))
        save(prefix+'_rows.json',rows);print('SCORED',len(rows),token,rows[-1]['score'],flush=True)
    assert len(rows)==(1 if smoke else 200)
    average={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('token','valid')};average.update(token='average',valid=True)
    if not smoke:
        with (OUT/'scores.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows+[average])
    save(prefix+'.json',dict(**average,successful_scenarios=len(rows),failed_scenarios=0,seconds=time.perf_counter()-start,checkpoint_epoch=best['epoch'],checkpoint_sha256=checkpoint_hash,manifest_sha256=digest,token_list_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),scope='single-scene integration check' if smoke else 'fixed 200 navtest, native PDM scorer, best validation ADE',training_epochs=history[-1]['epoch']))
    print('EVAL_COMPLETE',average,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args.smoke)
    except Exception as exc:
        import traceback
        save('evaluation_smoke_error.json' if args.smoke else 'evaluation_error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
