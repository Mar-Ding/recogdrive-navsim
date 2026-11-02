"""Frozen cached DINOv3 RGB adaptation of the DiffusionDrive action head.

Image-space conditioning grid (not a calibrated BEV); no LiDAR, BERT or MF.
"""
import torch
from torch import nn
import torch.nn.functional as F
from model import Planner, XYTrajectoryHead, TransfuserConfig

class DinoDiffusionDrive(Planner):
    def __init__(self, anchors, mode, bert_path, vision_path):
        nn.Module.__init__(self)
        self.mode='vl'
        # Frozen encoder features are loaded from the verified shared cache.
        self.vision=nn.Identity(); self.language=nn.Identity()
        self.vision_proj=nn.Linear(768,256)
        self.vision_pos=nn.Parameter(torch.randn(1,270,256)*.02)
        self.camera_embed=nn.Embedding(3,256)
        self.ego=nn.Sequential(nn.Linear(8,256),nn.ReLU(),nn.Linear(256,256))
        self.query=nn.Parameter(torch.randn(1,31,256)*.02)
        self.query_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,8,1024,0.,batch_first=True),2)
        self.head=XYTrajectoryHead(8,1024,256,str(anchors),TransfuserConfig())

    def context(self,image,ids,language_mask,ego):
        v=torch.cat([self.vision_proj(x)+self.camera_embed.weight[i] for i,x in enumerate(image)],1)+self.vision_pos
        c=torch.cat((v,self.ego(ego).unsqueeze(1)),1)
        return c,torch.zeros(c.shape[:2],dtype=torch.bool,device=c.device)

    def fuse(self,c,pad,noisy,steps):
        return c,None

    def decode(self,c,pad,noisy,steps):
        from model import gen_sineembed_for_position
        b=c.shape[0]
        grids=[c[:,:180].transpose(1,2).reshape(b,256,10,18)]
        for a,z in ((180,225),(225,270)):
            grids.append(F.interpolate(c[:,a:z].transpose(1,2).reshape(b,256,5,9),size=(10,9),mode='bilinear',align_corners=False))
        grid=F.interpolate(torch.cat((grids[1],grids[0],grids[2]),-1),size=(16,16),mode='bilinear',align_corners=False)
        q=self.query_decoder(self.query.expand(b,-1,-1),c,memory_key_padding_mask=pad)
        time=self.head.time_mlp(steps).view(-1,1,256)
        traj=self.head.plan_anchor_encoder(gen_sineembed_for_position(noisy,hidden_dim=64).flatten(-2))
        return self.head.diff_decoder(traj,noisy,grid,(16,16),q[:,1:],q[:,:1],time,c[:,-1:],None)
