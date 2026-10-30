import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
B=Path('/data/ruiqiwang/recogdrive_baseline');root=B/'data/sensor_blobs/test'
manifest=json.loads((B/'demo/multisensor_manifest.json').read_text())
results=json.loads((B/'demo/results.json').read_text())
loader=SceneLoader(B/'data/openscene-v1.1/meta_datas/test',root,SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=sorted({r['log_name'] for r in manifest})),SensorConfig.build_all_sensors(include=[0,1,2,3]),load_image_path=False)
checks=[]
for r in results:
 a=loader.get_agent_input_from_token(r['token'])
 assert len(a.cameras)==4 and len(a.lidars)==4
 for cams,lidar in zip(a.cameras,a.lidars):
  for name in ['cam_f0','cam_l0','cam_l1','cam_l2','cam_r0','cam_r1','cam_r2','cam_b0']:
   image=getattr(cams,name).image
   assert image.ndim==3 and image.shape[2]==3
  assert lidar.lidar_pc.shape[1]>0
 checks.append({'token':r['token'],'history_frames':4,'camera_files_loaded':32,'lidar_files_loaded':4})
row=next(x for x in manifest if x['token']==results[0]['token'])
fig,axs=plt.subplots(3,3,figsize=(15,10))
for ax,(name,cam) in zip(axs.flat,row['cameras'].items()):
 with Image.open(root/cam['path']) as im:ax.imshow(im.resize((480,280)))
 ax.set_title(name);ax.axis('off')
pc=LidarPointCloud.from_buffer((root/row['lidar']['path']).read_bytes(),'pcd').points
mask=(np.abs(pc[0])<50)&(np.abs(pc[1])<50)&(pc[2]>-3)&(pc[2]<5)
ax=axs.flat[8];ax.scatter(pc[0,mask],pc[1,mask],s=.1,c=pc[2,mask],cmap='viridis');ax.set_aspect('equal');ax.set_title('LiDAR BEV');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.suptitle('NAVSIM token '+row['token']);fig.tight_layout();fig.savefig(B/'demo/multisensor_preview.png',dpi=120)
report={'frames':len(manifest),'camera_files':sum(len(x['cameras']) for x in manifest),'lidar_files':len(manifest),'bytes':sum(sum(c['bytes'] for c in x['cameras'].values())+x['lidar']['bytes'] for x in manifest),'demo_loader_checks':checks,'scope':'Metadata-associated sensor files; individual camera exposure timestamps are not provided in this manifest.'}
(B/'demo/multisensor_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report));print('NAVSIM_ALL_SENSORS_LOAD_OK')
