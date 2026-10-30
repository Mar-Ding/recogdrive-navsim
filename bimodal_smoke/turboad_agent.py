"""TurboAD adapter for the unchanged ReCogDrive NAVSIM PDM evaluator."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig,Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from model import BimodalPlanner

class TurboADAgent(AbstractAgent):
    def __init__(self,checkpoint_path,encoder_path='/home/yiding/TurboVLA-repro/assets/dinov3-vitb'):
        super().__init__();self.checkpoint_path=checkpoint_path;self.encoder_path=encoder_path
    def name(self):return 'TurboAD'
    def get_sensor_config(self):
        config=SensorConfig.build_no_sensors();config.cam_f0=[3];config.lidar_pc=[3];return config
    def initialize(self):
        torch.set_num_threads(2)
        self.encoder=AutoModel.from_pretrained(self.encoder_path,local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        checkpoint=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False)
        variant=checkpoint.get('variant','cross')
        self.planner=BimodalPlanner(variant,Path('/data/ruiqiwang/recogdrive_baseline/bimodal_smoke/upstream/kmeans_navsim_traj_20.npy')).cuda().eval()
        self.planner.load_state_dict(checkpoint['model'],strict=True)
    def compute_trajectory(self,agent_input):
        image=Image.fromarray(agent_input.cameras[-1].cam_f0.image).convert('RGB').resize((288,160))
        pixels=torch.from_numpy(np.asarray(image,dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
        points=agent_input.lidars[-1].lidar_pc[:3].T
        points=points[(points[:,2]>.2)&(points[:,2]<100)]
        bev=np.histogramdd(points[:,:2],bins=(np.linspace(-32,32,257),)*2)[0]
        lidar=torch.tensor(np.minimum(bev,5)[None,None]/5,dtype=torch.float32,device='cuda')
        ego=agent_input.ego_statuses[-1]
        status=torch.tensor(np.concatenate([ego.driving_command,ego.ego_velocity,ego.ego_acceleration])[None],dtype=torch.float32,device='cuda')
        # Fix decoder noise for repeatable scores, independent of worker ordering.
        torch.manual_seed(999);torch.cuda.manual_seed_all(999)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            vision=self.encoder(pixel_values=pixels).last_hidden_state[:,5:]
            prediction=self.planner(vision,lidar,status)['trajectory'][0].float().cpu().numpy()
        assert prediction.shape==(8,3) and np.isfinite(prediction).all()
        return Trajectory(prediction,TrajectorySampling(num_poses=8,interval_length=.5))
