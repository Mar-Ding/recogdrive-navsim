"""TurboVLA bidirectional VL fusion + action cross-attention NAVSIM adaptation."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from model import Planner
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla')
path=BASE/'src/TurboVLA/turbovla/models/components/fusion.py'
spec=importlib.util.spec_from_file_location('turbo_fusion',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class CrossPlanner(Planner):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.fusion=nn.ModuleList([module.BiAttentionBlock(256,256,768,4,dropout=0.,drop_path=.1,residual_style='normalized',attention_backend='sdpa') for _ in range(6)])
  self.text_layers=nn.ModuleList([nn.TransformerEncoderLayer(256,4,768,0.,batch_first=True) for _ in range(6)])
  self.action_cross=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,4,768,0.,batch_first=True),2)
 def fuse(self,c,pad,noisy,steps):
  v,l,state=c[:,:270],c[:,270:-1],c[:,-1:];mask=pad[:,270:-1]
  for block,text in zip(self.fusion,self.text_layers):
   v,l=block(v,l,attention_mask_l=mask);l=text(l,src_key_padding_mask=mask);l=l.masked_fill(mask[:,:,None],0.)
  cond=torch.cat((v,l,state),1)
  action=self.action_codec(self.head.norm_odo(noisy).flatten(2))+self.types.weight[3]+self.noise_embed(steps.float().view(-1,1,1)/1000)
  action=self.action_cross(action,cond,memory_key_padding_mask=pad)
  return cond,action
