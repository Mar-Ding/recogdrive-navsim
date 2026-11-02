"""ReCogDrive hidden-state adapter to the upstream DiffusionDrive trajectory head.
The resampled feature grid is a learned latent grid, not geometric BEV.
"""
import torch
from torch import nn
import torch.nn.functional as F
from model import Planner,XYTrajectoryHead
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

class RecogDriveDDHead(Planner):
    def __init__(self,anchors,input_dim=1536):
        nn.Module.__init__(self)
        self.mode='vl'
        self.projection=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,256))
        self.grid_queries=nn.Parameter(torch.randn(1,180,256)*.02)
        self.resampler=nn.MultiheadAttention(256,8,batch_first=True)
        self.ego=nn.Sequential(nn.Linear(20,256),nn.GELU(),nn.Linear(256,256))
        self.query=nn.Parameter(torch.randn(1,31,256)*.02)
        self.query_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,8,1024,0.,batch_first=True),2)
        self.head=XYTrajectoryHead(8,1024,256,str(anchors),TransfuserConfig())
    def context(self,image,ids,language_mask,ego):
        # image is the frozen VLM final hidden state, ids unused.
        hidden=self.projection(image[0])
        grid=self.resampler(self.grid_queries.expand(len(hidden),-1,-1),hidden,hidden,key_padding_mask=~language_mask.bool(),need_weights=False)[0]
        context=torch.cat((grid,self.ego(ego).unsqueeze(1)),1)
        return context,torch.zeros(context.shape[:2],device=context.device,dtype=torch.bool)
    def fuse(self,c,pad,noisy,steps):return c,None
