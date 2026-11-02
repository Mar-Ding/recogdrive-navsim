"""Native ReCogDrive trajectory prediction with the identical upstream PDM scorer."""
import os,sys,json,csv,lzma,pickle,time
from pathlib import Path
from dataclasses import asdict
BASE=Path('/data/ruiqiwang/recogdrive_baseline');sys.path.insert(0,str(BASE/'repo'));sys.path.insert(0,str(BASE/'bimodal_smoke'))
os.environ['NUPLAN_MAPS_ROOT']=str(BASE/'data/maps');os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0'
import torch,numpy as np
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader,MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.evaluate.pdm_score import pdm_score
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from native_recogdrive import save_json
def main():
    torch.set_num_threads(2);out=BASE/'demo10/native_recogdrive';tokens=json.loads((BASE/'turboad/navtest_subset_tokens.json').read_text())
    with initialize_config_dir(config_dir=str(BASE/'repo/navsim/planning/script/config/pdm_scoring'),version_base=None):cfg=compose(config_name='default_run_pdm_score',overrides=['train_test_split=navtest','worker=sequential'])
    simulator=instantiate(cfg.simulator);scorer=instantiate(cfg.scorer);assert simulator.proposal_sampling==scorer.proposal_sampling
    filt=instantiate(cfg.train_test_split.scene_filter);filt.tokens=tokens
    loader=SceneLoader(BASE/'data/openscene-v1.1/meta_datas/test',BASE/'data/sensor_blobs/test',filt,SensorConfig(cam_f0=[3],cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False),load_image_path=True)
    caches=MetricCacheLoader(BASE/'turboad/evaluation/metric_cache');assert set(tokens)<=set(caches.tokens) and set(loader.tokens)==set(tokens)
    torch.manual_seed(42);agent=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=str(BASE/'models/ReCogDrive-VLM-2B'),checkpoint_path=str(out/'best_val_ade.ckpt'),cache_hidden_state=False)
    expected=torch.load(out/'best_val_ade.ckpt',map_location='cpu',weights_only=False)['state_dict'];agent.initialize();agent.eval()
    actual=agent.state_dict();assert all(torch.equal(actual[k.removeprefix('agent.')].cpu(),v) for k,v in expected.items())
    rows=[];start=time.perf_counter()
    for token in tokens:
        torch.manual_seed(999);torch.cuda.manual_seed_all(999)
        trajectory=agent.compute_trajectory(loader.get_agent_input_from_token(token));assert trajectory.poses.shape==(8,3) and np.isfinite(trajectory.poses).all()
        with lzma.open(caches.metric_cache_paths[token],'rb') as f:cache=pickle.load(f)
        result=pdm_score(metric_cache=cache,model_trajectory=trajectory,future_sampling=simulator.proposal_sampling,simulator=simulator,scorer=scorer)
        rows.append(dict(token=token,valid=True,**asdict(result)));save_json('evaluation_progress.json',dict(done=len(rows),total=200,seconds=time.perf_counter()-start));print('SCORED',len(rows),token,rows[-1]['score'],flush=True)
    assert len(rows)==200
    average={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('token','valid')};average.update(token='average',valid=True)
    with (out/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows+[average])
    save_json('evaluation.json',dict(**average,successful_scenarios=200,failed_scenarios=0,seconds=time.perf_counter()-start));print('EVAL_COMPLETE',average,flush=True)
if __name__=='__main__':main()
