"""Reproducible first 20/200 OpenScene shards, official navtrain filtering."""
import concurrent.futures, hashlib, json, math, pickle, shutil, subprocess, sys, tarfile, time
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import numpy as np, requests, torch, yaml
from navsim.common.dataclasses import Scene
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
ROOT=BASE/'demo10'; ROOT.mkdir(exist_ok=True)
DATA=BASE/'data'; SENSORS=DATA/'sensor_blobs/trainval'
HOST='https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/'

def write(name,value):
    temp=ROOT/(name+'.tmp'); temp.write_text(json.dumps(value,indent=2));temp.replace(ROOT/name)

def budget():
    entries=[]
    for kind in ('camera','lidar'):
        response=requests.get(f'https://hf-mirror.com/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_{kind}?limit=1000',timeout=60);response.raise_for_status()
        files={Path(x['path']).name:x for x in response.json()}
        for shard in range(20):
            item=files[f'openscene_sensor_trainval_{kind}_{shard}.tgz'];path=DATA/f'trainval_{kind}_{shard}.tgz'
            entries.append(dict(kind=kind,shard=shard,bytes=item['size'],existing_bytes=path.stat().st_size if path.exists() else 0))
    value=dict(archives=entries,total_archive_bytes=sum(x['bytes'] for x in entries),remaining_download_bytes=sum(max(0,x['bytes']-x['existing_bytes']) for x in entries),disk_free_bytes=shutil.disk_usage(DATA).free,reserved_free_bytes=30*2**30)
    assert value['remaining_download_bytes']+70*2**30<value['disk_free_bytes'],'Insufficient storage for archives + estimated selected sensors/features + 30 GiB reserve'
    write('download_budget.json',value);print(json.dumps(value),flush=True)

def index():
    cfg=yaml.safe_load((BASE/'repo/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text())
    logs=set(cfg['log_names']); tokens=set(cfg['tokens']) if cfg.get('tokens') else None
    shards=json.loads((DATA/'trainval_shard_index.json').read_text())
    selected={log:i for i in range(20) for log in shards[f'openscene_sensor_trainval_{i}']}
    old=json.loads((BASE/'turboad/manifest.json').read_text()); fixed_val={r['log_name'] for r in old if r['split']=='val'}
    eligible=sorted(logs&set(selected),key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    val_logs=set(fixed_val); target_val=max(2,round(len(eligible)*.15))
    for log in eligible:
        if len(val_logs)>=target_val:break
        val_logs.add(log)
    rows=[]; counts={}; total=0; seen=set()
    with tarfile.open(DATA/'metadata_trainval.tgz','r|gz') as tar:
        for m in tar:
            log=Path(m.name).stem
            if not m.isfile() or not m.name.endswith('.pkl') or log not in logs:continue
            frames=pickle.loads(tar.extractfile(m).read()); count=0
            for start in range(0,len(frames)-cfg['num_history_frames']-cfg['num_future_frames']+1,cfg['frame_interval']):
                i=start+cfg['num_history_frames']-1; f=frames[i]
                if cfg['has_route'] and not f['roadblock_ids']:continue
                if tokens is not None and f['token'] not in tokens:continue
                count+=1
                if log not in selected:continue
                assert f['token'] not in seen;seen.add(f['token'])
                poses=np.array([Scene._build_ego_status(x).ego_pose for x in frames[i:i+9]])
                target=convert_absolute_to_relative_se2_array(StateSE2(*poses[0]),poses[1:])
                rows.append(dict(token=f['token'],log_name=log,shard=selected[log],split='val' if log in val_logs else 'train',
                    image=f['cams']['CAM_F0']['data_path'],views={v:f['cams'][v]['data_path'] for v in ('CAM_F0','CAM_L0','CAM_R0')},
                    pcd=f['lidar_path'],status=np.concatenate([f['driving_command'],f['ego_dynamic_state']]).tolist(),target=target.tolist()))
            counts[log]=count;total+=count
            if len(counts)%100==0:print('INDEX',len(counts),total,len(rows),flush=True)
    rows.sort(key=lambda r:(r['shard'],r['log_name'],r['token']))
    test=set(json.loads((BASE/'turboad/navtest_subset_tokens.json').read_text()))
    assert not (seen&test)
    assert not ({r['log_name'] for r in rows if r['split']=='train'} & val_logs)
    audit=dict(selection='first 20 of 200 sensor shards, then official navtrain scene filter; approximate 10%, not exact token 10%',
        shards=list(range(20)),official_navtrain_scenes=total,selected_scenes=len(rows),scene_fraction=len(rows)/total,
        official_navtrain_logs=len(counts),selected_logs=len({r['log_name'] for r in rows}),
        train_samples=sum(r['split']=='train' for r in rows),val_samples=sum(r['split']=='val' for r in rows),
        train_logs=len({r['log_name'] for r in rows if r['split']=='train'}),val_logs=len({r['log_name'] for r in rows if r['split']=='val'}),
        fixed_old_val_logs=sorted(fixed_val),navtest_token_overlap=0)
    write('manifest.json',rows);write('data_audit.json',audit);print('INDEX_DONE',json.dumps(audit),flush=True)

def download_extract(job):
    shard,kind=job;rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['shard']==shard]
    wanted={p for r in rows for p in (list(r['views'].values()) if kind=='camera' else [r['pcd']])}
    marker=ROOT/f'{kind}_{shard}.complete.json'
    if marker.exists() and all((SENSORS/p).is_file() for p in wanted):return
    if shutil.disk_usage(DATA).free<30*2**30:raise RuntimeError('Less than 30 GiB disk reserve')
    archive=DATA/f'trainval_{kind}_{shard}.tgz';rel=f'openscene_sensor_trainval_{kind}/openscene_sensor_trainval_{kind}_{shard}.tgz'
    # Obtain expected size and SHA256 from the hosting API; curl supports resumable downloads.
    api=f'https://hf-mirror.com/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_{kind}?limit=1000'
    response=requests.get(api,timeout=60);response.raise_for_status();entry=next(x for x in response.json() if x['path'].endswith('/'+Path(rel).name))
    expected=entry['size'];sha=entry.get('lfs',{}).get('oid')
    if not archive.exists() or archive.stat().st_size!=expected:
        print('DOWNLOAD',kind,shard,expected,flush=True)
        subprocess.run(['curl','-fL','--retry','8','--retry-delay','5','--connect-timeout','30','--speed-limit','1024','--speed-time','180','-C','-',HOST+rel+'?download=true','-o',str(archive)],check=True)
    assert archive.stat().st_size==expected,(archive,expected)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if sha:assert h.hexdigest()==sha,('Archive checksum mismatch',archive)
    remaining={p for p in wanted if not (SENSORS/p).exists()};saved=0
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if not remaining:break
            if not m.isfile():continue
            parts=Path(m.name).parts;relpath=next(('/'.join(parts[i:]) for i in range(len(parts)) if '/'.join(parts[i:]) in remaining),None)
            if relpath is None:continue
            dest=(SENSORS/relpath).resolve();assert dest.is_relative_to(SENSORS.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix(dest.suffix+'.partial')
            with tar.extractfile(m) as source,temp.open('wb') as out:shutil.copyfileobj(source,out)
            assert temp.stat().st_size==m.size;temp.replace(dest);remaining.remove(relpath);saved+=1
    assert not remaining,('Missing sensor files',kind,shard,len(remaining))
    write(marker.name,dict(kind=kind,shard=shard,files=len(wanted),saved=saved,archive_bytes=expected,sha256=h.hexdigest()))
    print('EXTRACT_DONE',kind,shard,len(wanted),flush=True)

def sensors():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download_extract,[(i,k) for i in range(20) for k in ('camera','lidar')]))
    rows=json.loads((ROOT/'manifest.json').read_text())
    assert all((SENSORS/p).is_file() for r in rows for p in [r['pcd'],*r['views'].values()])
    write('sensors_complete.json',dict(samples=len(rows),time=time.time()))

def encode():
    from PIL import Image
    from transformers import AutoModel
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    torch.set_num_threads(2);rows=json.loads((ROOT/'manifest.json').read_text());features=ROOT/'features';features.mkdir(exist_ok=True)
    path='/home/yiding/TurboVLA-repro/assets/dinov3-vitb'
    model=AutoModel.from_pretrained(path,local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    files=[]
    for offset in range(0,len(rows),256):
        dest=features/f'{offset:07d}.pt';files.append(str(dest))
        if dest.exists():continue
        samples=[]
        for j in range(offset,min(offset+256,len(rows)),32):
            chunk=rows[j:min(j+32,offset+256,len(rows))]
            x=torch.stack([torch.from_numpy(np.array(Image.open(SENSORS/r['image']).convert('RGB').resize((288,160)),dtype=np.float32)/255).permute(2,0,1) for r in chunk]).cuda()
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):vision=model(pixel_values=x).last_hidden_state[:,5:].cpu()
            for r,v in zip(chunk,vision):
                points=LidarPointCloud.from_buffer((SENSORS/r['pcd']).read_bytes(),'pcd').points[:3].T;points=points[(points[:,2]>.2)&(points[:,2]<100)]
                bev=np.histogramdd(points[:,:2],bins=(np.linspace(-32,32,257),)*2)[0]
                samples.append(dict(token=r['token'],log_name=r['log_name'],split=r['split'],vision=v.clone(),lidar=torch.tensor(np.minimum(bev,5)[None]/5,dtype=torch.float32),status=torch.tensor(r['status'],dtype=torch.float32),target=torch.tensor(r['target'],dtype=torch.float32)))
        temp=dest.with_suffix('.partial');torch.save(samples,temp);temp.replace(dest);print('ENCODE',offset,len(rows),flush=True)
    write('features_complete.json',dict(files=files,samples=len(rows),encoder=path,manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()))

if __name__=='__main__':globals()[sys.argv[1]]()
