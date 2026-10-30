import os,json,time
from pathlib import Path
import numpy as np,torch
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
base=Path('/data/ruiqiwang/recogdrive_baseline');os.chdir(base)
logs=list((base/'data/sensor_blobs/test').iterdir())
loader=SceneLoader(base/'data/openscene-v1.1/meta_datas/test',base/'data/sensor_blobs/test',SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=[p.name for p in logs]),SensorConfig(cam_f0=[3],cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False),load_image_path=True)
selected=[]
for token,frames in loader.scene_frames_dicts.items():
 image=base/'data/sensor_blobs/test'/frames[3]['cams']['CAM_F0']['data_path']
 if image.exists(): selected.append((token,image))
assert selected,'No matching real image and scene'
torch.manual_seed(42)
a=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),vlm_path=str(base/'models/ReCogDrive-VLM-2B'),checkpoint_path=str(base/'models/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt'),cache_hidden_state=False)
a.initialize();a.eval();out=base/'demo';out.mkdir(exist_ok=True)
results=[]
for token,image in selected[:3]:
 start=time.time();prediction=a.compute_trajectory(loader.get_agent_input_from_token(token)).poses
 assert prediction.shape==(8,3) and np.isfinite(prediction).all()
 frames=loader.scene_frames_dicts[token]
 # Ground truth is in the same relative coordinate system as model outputs.
 from pyquaternion import Quaternion
 from navsim.common.dataclasses import convert_absolute_to_relative_se2_array,StateSE2
 poses=np.array([[f['ego2global_translation'][0],f['ego2global_translation'][1],Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]] for f in frames[3:12]])
 gt=convert_absolute_to_relative_se2_array(StateSE2(*poses[0]),poses[1:])
 result=dict(token=token,image=str(image),prediction=prediction.tolist(),ground_truth=gt.tolist(),ade=float(np.linalg.norm(prediction[:,:2]-gt[:,:2],axis=1).mean()),seconds=time.time()-start)
 results.append(result);print(json.dumps(result),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
fig,axs=plt.subplots(1,2,figsize=(12,5));axs[0].imshow(Image.open(results[0]['image']));axs[0].axis('off');axs[0].set_title('Real NAVSIM front camera')
p=np.array(results[0]['prediction']);g=np.array(results[0]['ground_truth']);axs[1].plot(g[:,0],g[:,1],'o-',label='Ground truth');axs[1].plot(p[:,0],p[:,1],'o-',label='ReCogDrive 2B IL');axs[1].scatter([0],[0],c='black');axs[1].axis('equal');axs[1].legend();axs[1].set_xlabel('x (m)');axs[1].set_ylabel('y (m)');fig.tight_layout();fig.savefig(out/'demo.png',dpi=160)
print('REAL_NAVSIM_DEMO_OK',flush=True)
