"""Native ReCogDrive feasibility probe on the exact demo10 manifest."""
import os,sys,json,time,hashlib,argparse,tarfile
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.insert(0,str(BASE/'repo'))
os.environ['HF_HOME']=str(BASE/'hf_cache');os.environ['TMPDIR']=str(BASE/'tmp')
import torch,numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
ROOT=BASE/'demo10';OUT=ROOT/'native_recogdrive';OUT.mkdir(exist_ok=True)
def save_json(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['probe']);a=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((ROOT/'manifest.json').read_text());selected=manifest[:8]
    meta=OUT/'metadata';meta.mkdir(exist_ok=True);wanted={r['log_name'] for r in selected}
    if not all((meta/(name+'.pkl')).exists() for name in wanted):
        with tarfile.open(BASE/'data/metadata_trainval.tgz','r|gz') as archive:
            for member in archive:
                name=Path(member.name).name
                if Path(name).stem in wanted and member.isfile():
                    (meta/name).write_bytes(archive.extractfile(member).read())
                    if all((meta/(x+'.pkl')).exists() for x in wanted):break
    loader=SceneLoader(meta,BASE/'data/sensor_blobs/trainval',SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=sorted(wanted),tokens=[r['token'] for r in selected]),SensorConfig(cam_f0=[3],cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False),load_image_path=True)
    builder=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(BASE/'models/ReCogDrive-VLM-2B'),device='cuda:0')
    cache=OUT/'probe_cache';cache.mkdir(exist_ok=True);rows=[];timings=[]
    for row in selected:
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():f=builder.compute_features(loader.get_agent_input_from_token(row['token']))
        torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        # Native builder casts BF16 hidden states to FP32; exact roundtrip is required.
        hidden=f['last_hidden_state'];assert torch.equal(hidden,hidden.bfloat16().float())
        f['last_hidden_state']=hidden.bfloat16()
        torch.save(dict(features=f,target=torch.tensor(row['target'],dtype=torch.float32)),cache/(row['token']+'.pt'))
        rows.append(dict(token=row['token'],shape=list(hidden.shape),bytes=(cache/(row['token']+'.pt')).stat().st_size))
        print('CACHE_PROBE',rows[-1],timings[-1],flush=True)
    del builder;import gc;gc.collect();torch.cuda.empty_cache()
    agent=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=str(BASE/'models/ReCogDrive-VLM-2B'),checkpoint_path=None,cache_hidden_state=True)
    # Native IL initialization: VLM pretrained, planner freshly initialized.
    opt=torch.optim.AdamW(agent.action_head.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95));bench=[]
    sample=torch.load(cache/(selected[0]['token']+'.pt'),weights_only=False)
    for bs in [1,2,4,8,16,32,64,128]:
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
            for step in range(3):
                features={k:v[None].expand(bs,*v.shape).cuda().float() for k,v in sample['features'].items()}
                target={'trajectory':sample['target'][None].expand(bs,8,3).cuda()};agent.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=agent(features,target);loss=agent.compute_loss(features,target,pred)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(agent.action_head.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            bench.append(dict(batch=bs,seconds=float(np.mean(times[1:])),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(loss)))
            print('TRAIN_PROBE',bench[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            bench.append(dict(batch=bs,oom=True));opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
    agent.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        predicted=agent({k:v[None].cuda().float() for k,v in sample['features'].items()})['pred_traj']
    assert predicted.shape==(1,8,3) and torch.isfinite(predicted).all()
    save_json('probe.json',dict(manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),cache_samples=rows,cache_seconds=timings,estimated_cache_gib=sum(r['bytes'] for r in rows)/len(rows)*len(manifest)/2**30,estimated_encoding_hours=float(np.mean(timings))*len(manifest)/3600,train_benchmark=bench,inference_shape=list(predicted.shape),native_code_unchanged=True,planner_initialization='random; native VLM pretrained'))
if __name__=='__main__':main()
