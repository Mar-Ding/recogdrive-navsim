"""Internal log-disjoint validation PDMS; never official navtest."""
import os,sys,json,pickle,lzma,traceback,hashlib
from pathlib import Path
from dataclasses import asdict
BASE=Path('/root/autodl-tmp/mf-vla');OUT=BASE/'outputs/turbo_cross_vla';DEST=OUT/'pdms_val'
os.environ['NUPLAN_MAPS_ROOT']=str(BASE/'data/maps')
sys.path.insert(0,str(BASE/'multiview'))
import numpy as np
import torch
from train_turbo_cross_vla import CachedPlanner,seed
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(4);seed(123)
    rows=json.loads((OUT/'manifest.json').read_text());ids=[i for i,r in enumerate(rows) if r['split']=='val']
    predfile=DEST/'predictions.npz'
    if not predfile.exists():
        net=CachedPlanner(OUT/'train_kmeans_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
        del net.vision;del net.language
        ck=torch.load(OUT/'best_val_ade.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model'],strict=True);net.eval()
        tensors=torch.load(OUT/'features.pt',map_location='cpu',weights_only=False);pred=[]
        with torch.inference_mode():
            for start in range(0,len(ids),128):
                ix=ids[start:start+128];v=tensors['vision'][ix].cuda().float();cmd=tensors['command'][ix]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    p=net([v[:,:180],v[:,180:225],v[:,225:]],tensors['language'][cmd].cuda().float(),tensors['mask'][cmd].cuda(),tensors['ego'][ix].cuda())['trajectory']
                pred.append(p.float().cpu().numpy())
        np.savez(predfile,poses=np.concatenate(pred),tokens=np.array([rows[i]['token'] for i in ids]))
        del net,tensors,ck;torch.cuda.empty_cache()
    data=np.load(predfile);predictions=dict(zip(data['tokens'],data['poses']))
    sampling=TrajectorySampling(num_poses=40,interval_length=.1);sim=PDMSimulator(sampling);scorer=PDMScorer(sampling)
    processor=MetricCacheProcessor(str(BASE/'outputs/multiview_mf_vla/pdms_val/metric_cache'),False)
    resultsfile=DEST/'results.jsonl';done={json.loads(s)['token'] for s in resultsfile.read_text().splitlines()} if resultsfile.exists() else set()
    groups={}
    for i in ids:groups.setdefault(rows[i]['log'],[]).append(rows[i])
    for log,group in groups.items():
        frames=pickle.load(open(BASE/'data/navsim10/openscene-v1.1/meta_datas/trainval'/f'{log}.pkl','rb'));lookup={f['token']:i for i,f in enumerate(frames)}
        for r in group:
            token=r['token']
            if token in done:continue
            result=dict(token=token,valid=False)
            try:
                i=lookup[token];window=frames[i-3:i+11]
                if i<3 or len(window)!=14:raise ValueError('insufficient 4 history / 10 future frames')
                if any(abs((window[j]['timestamp']-window[3]['timestamp'])/1e6-(j-3)*.5)>.2 for j in range(14)):raise ValueError('temporal discontinuity')
                if not frames[i]['roadblock_ids']:raise ValueError('no route')
                scene=Scene.from_scene_dict_list(window,None,num_history_frames=4,num_future_frames=10,sensor_config=SensorConfig.build_no_sensors())
                scenario=NavSimScenario(scene,map_root=os.environ['NUPLAN_MAPS_ROOT'],map_version='nuplan-maps-v1.0')
                entry=processor.compute_metric_cache(scenario)
                with lzma.open(entry.file_name,'rb') as f:cache=pickle.load(f)
                score=pdm_score(cache,Trajectory(predictions[token].astype(np.float64)),sampling,sim,scorer)
                result.update(asdict(score));result['valid']=True
            except Exception as e:
                result['error']=repr(e);traceback.print_exc()
            with resultsfile.open('a') as f:f.write(json.dumps(result)+'\n')
            done.add(token);print('PROGRESS',len(done),len(ids),result,flush=True)
    results=[json.loads(s) for s in resultsfile.read_text().splitlines()];valid=[r for r in results if r['valid']]
    summary=dict(split='internal log-disjoint validation; NOT official navtest',requested=len(ids),successful=len(valid),failed=len(results)-len(valid),checkpoint='best_val_ade.pt',metrics={k:float(np.mean([r[k] for r in valid])) for k in valid[0] if k not in ('token','valid')} if valid else {})
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
