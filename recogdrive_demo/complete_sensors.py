import json,pickle,tarfile,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
import numpy as np
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
B=Path('/data/ruiqiwang/recogdrive_baseline');root=B/'data/sensor_blobs/test'
rows=[]
for p in (B/'data/openscene-v1.1/meta_datas/test').glob('*.pkl'):
 if not (root/p.stem).exists():continue
 frames=pickle.load(p.open('rb'))
 indices={i for i,f in enumerate(frames) if (root/f['cams']['CAM_F0']['data_path']).exists()}
 demo={x['token'] for x in json.loads((B/'demo/results.json').read_text())}
 for i,f in enumerate(frames):
  if f['token'] in demo:indices.update(range(max(0,i-3),i+1))
 rows.extend(frames[i] for i in sorted(indices))
assert rows
wanted={c['data_path'] for f in rows for c in f['cams'].values()}|{f['lidar_path'] for f in rows}
print('TARGET',len(rows),'frames',len(wanted),'files',flush=True)
def download(kind):
 paths={p for p in wanted if (p.endswith('.pcd') if kind=='lidar' else p.endswith('.jpg'))}
 remaining={p for p in paths if not (root/p).exists()}
 if not remaining:return
 url=f'https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_{kind}/openscene_sensor_test_{kind}_0.tgz?download=true'
 for attempt in range(4):
  try:
   with requests.get(url,stream=True,timeout=(30,120)) as r:
    r.raise_for_status()
    with tarfile.open(fileobj=r.raw,mode='r|gz') as tar:
     for m in tar:
      if not m.isfile():continue
      parts=Path(m.name).parts
      if 'sensor_blobs' not in parts:continue
      rel=Path(*parts[parts.index('sensor_blobs')+1:])
      if rel.parts[0]=='test':rel=Path(*rel.parts[1:])
      name=rel.as_posix()
      if name not in remaining:continue
      dest=(root/rel).resolve();assert dest.is_relative_to(root.resolve())
      dest.parent.mkdir(parents=True,exist_ok=True)
      data=tar.extractfile(m).read();assert len(data)==m.size
      tmp=dest.with_suffix(dest.suffix+'.partial');tmp.write_bytes(data);tmp.replace(dest)
      remaining.remove(name);print(kind,'saved',name,'remaining',len(remaining),flush=True)
      if not remaining:return
   raise RuntimeError('Archive ended with missing paths: '+str(sorted(remaining)))
  except Exception as e:
   print(kind,'attempt',attempt+1,repr(e),flush=True)
   if attempt==3:raise
   time.sleep(3)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,['camera','lidar']))
report=[]
for f in rows:
 cameras={}
 for k,v in f['cams'].items():
  p=root/v['data_path']
  with Image.open(p) as image:image.verify()
  cameras[k]={'path':v['data_path'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'intrinsic':v['cam_intrinsic'].tolist(),'sensor2lidar_rotation':v['sensor2lidar_rotation'].tolist(),'sensor2lidar_translation':v['sensor2lidar_translation'].tolist()}
 p=root/f['lidar_path'];cloud=LidarPointCloud.from_buffer(p.read_bytes(), 'pcd').points
 assert cloud.shape[1]>0 and np.isfinite(cloud[:3]).all()
 report.append({'token':f['token'],'timestamp':int(f['timestamp']),'log_name':f['log_name'],'frame_idx':int(f['frame_idx']),'cameras':cameras,'lidar':{'path':f['lidar_path'],'shape':list(cloud.shape),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
(B/'demo/multisensor_manifest.json').write_text(json.dumps(report,indent=2))
print('MULTISENSOR_COMPLETE',len(report),'frames',len(wanted),'files',flush=True)
