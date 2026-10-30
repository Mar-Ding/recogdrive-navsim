"""Bounded official-navtrain subset preparation for TurboAD."""
import sys,time,json,tarfile,pickle,hashlib,argparse
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import yaml,numpy as np,torch
from PIL import Image
from transformers import AutoModel
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.common.dataclasses import Scene
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
ROOT=BASE/'turboad';ROOT.mkdir(exist_ok=True)
SENSORS=BASE/'data/sensor_blobs/trainval'
CFG=BASE/'repo/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml'

def index(dense=False):
    cfg=yaml.safe_load(CFG.read_text());logs=set(cfg['log_names']);tokens=set(cfg['tokens']) if cfg.get('tokens') else None
    if dense:logs &= set(json.loads((ROOT/'selected_logs.json').read_text()))
    records=[];seen=set();archive=BASE/'data/metadata_trainval.tgz'
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if not m.isfile() or Path(m.name).stem not in logs or not m.name.endswith('.pkl'):continue
            frames=pickle.loads(tar.extractfile(m).read());name=Path(m.name).stem
            candidates=[]
            for start in range(0,len(frames)-13):
                i=start+3;f=frames[i]
                if not len(f['roadblock_ids']) or (tokens is not None and f['token'] not in tokens):continue
                poses=np.array([Scene._build_ego_status(x).ego_pose for x in frames[i:i+9]])
                target=convert_absolute_to_relative_se2_array(StateSE2(*poses[0]),poses[1:])
                candidates.append(dict(token=f['token'],log_name=name,image=f['cams']['CAM_F0']['data_path'],pcd=f['lidar_path'],
                    status=np.concatenate([f['driving_command'],f['ego_dynamic_state']]).tolist(),target=target.tolist(),map_location=f['map_location']))
            if not candidates:continue
            ids=np.arange(len(candidates)) if dense else np.linspace(0,len(candidates)-1,min(64,len(candidates)),dtype=int)
            records.extend(candidates[i] for i in ids);seen.add(name)
            if len(seen)%100==0:print('INDEX',len(seen),len(records),flush=True)
    (ROOT/'candidates.json').write_text(json.dumps(records));print('INDEX_DONE',len(seen),len(records),flush=True)

def dense_index():index(dense=True)

def sensors():
    candidates=json.loads((ROOT/'candidates.json').read_text());by_path={};by_log={}
    for r in candidates:
        by_log.setdefault(r['log_name'],[]).append(r)
        for k in ('image','pcd'):by_path[r[k]]=r['log_name']
    selected=json.loads((ROOT/'selected_logs.json').read_text()) if (ROOT/'selected_logs.json').exists() else []
    wanted={r[k] for log in selected for r in by_log[log] for k in ('image','pcd')}
    started=time.monotonic();saved=0
    for kind in ('camera','lidar'):
        archive=BASE/'data'/('trainval_'+kind+'_0.tgz')
        print('SENSOR_ARCHIVE',str(archive),flush=True)
        with tarfile.open(archive,mode='r|gz') as tar:
            for member_index,m in enumerate(tar):
                if time.monotonic()-started>2400:raise TimeoutError('Sensor scan time budget reached; saved files retained')
                if member_index<8 or member_index%2000==0:print('SCAN',member_index,m.name,flush=True)
                if not m.isfile():continue
                parts=Path(m.name).parts
                matches=[('/'.join(parts[i:]),by_path['/'.join(parts[i:])]) for i in range(len(parts)) if '/'.join(parts[i:]) in by_path]
                if not matches:continue
                rel,log=matches[0]
                if log not in selected and len(selected)<44:
                    selected.append(log);wanted.update(r[k] for r in by_log[log] for k in ('image','pcd'))
                    (ROOT/'selected_logs.json').write_text(json.dumps(selected));print('SELECT_LOG',len(selected),log,flush=True)
                if rel not in wanted:continue
                dest=(SENSORS/rel).resolve();assert dest.is_relative_to(SENSORS.resolve())
                if not dest.exists():
                    data=tar.extractfile(m).read();assert len(data)==m.size
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);saved+=1
                    if saved%100==0:print('SAVED',saved,flush=True)
                if len(selected)==44 and all((SENSORS/x).exists() for x in wanted):break
    complete=[r for log in selected for r in by_log[log] if all((SENSORS/r[k]).exists() for k in ('image','pcd'))]
    complete_logs=sorted({r['log_name'] for r in complete},key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    assert len(complete_logs)>=6,('Not enough paired logs',len(complete_logs))
    val_logs=set(complete_logs[:max(2,round(len(complete_logs)*.15))])
    for r in complete:r['split']='val' if r['log_name'] in val_logs else 'train'
    (ROOT/'manifest.json').write_text(json.dumps(complete,indent=2));print('PAIRED',len(complete),'logs',len(complete_logs),'train',sum(r['split']=='train' for r in complete),flush=True)

def encode():
    torch.set_num_threads(2);rows=json.loads((ROOT/'manifest.json').read_text())
    model_path='/home/yiding/TurboVLA-repro/assets/dinov3-vitb'
    model=AutoModel.from_pretrained(model_path,local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    samples=[]
    for offset in range(0,len(rows),32):
        chunk=rows[offset:offset+32]
        pixels=torch.stack([torch.from_numpy(np.array(Image.open(SENSORS/r['image']).convert('RGB').resize((288,160)),dtype=np.float32)/255).permute(2,0,1) for r in chunk]).cuda()
        pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):vision=model(pixel_values=pixels).last_hidden_state[:,5:].cpu()
        for r,v in zip(chunk,vision):
            points=LidarPointCloud.from_buffer((SENSORS/r['pcd']).read_bytes(),'pcd').points[:3].T
            points=points[(points[:,2]>.2)&(points[:,2]<100)]
            bev=np.histogramdd(points[:,:2],bins=(np.linspace(-32,32,257),)*2)[0]
            samples.append(dict(token=r['token'],log_name=r['log_name'],split=r['split'],vision=v,
                lidar=torch.tensor(np.minimum(bev,5)[None]/5,dtype=torch.float32),status=torch.tensor(r['status'],dtype=torch.float32),target=torch.tensor(r['target'],dtype=torch.float32)))
        if offset%256==0:print('ENCODE',offset,len(rows),flush=True)
    torch.save(dict(samples=samples,encoder=model_path,scope='Official navtrain subset with log-disjoint internal validation; not navtest'),ROOT/'samples.pt')
    print('TURBOAD_FEATURES_OK',len(samples),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['index','dense_index','sensors','encode']);a=p.parse_args();globals()[a.stage]()
