"""NAVSIM VLA adaptation: upstream MFBlock, SDPA and rotating latent flow.

The NAVSIM flow recipe is an explicit adaptation, not a claim of exact
equivalence to an unavailable TurboMFVLA training checkpoint/configuration.
"""
import sys
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla')
sys.path.insert(0,str(BASE/'src/Multimodal-Flow/src'))
from model import Planner
from mf.modeling.block import MFBlock
from mf.modeling.attention import SharedQKVOAttention
from mf.modeling.mrope import apply_rotary_pos_emb
from mf.modeling.layers import RMSNorm
if not hasattr(nn,"RMSNorm"): nn.RMSNorm=RMSNorm
from mf.modeling.packing import PackedRoutingLayout
from mf.modeling.embeddings import TimestepEmbedder


class RoutedSDPA(SharedQKVOAttention):
    """Use upstream QKVO, QK normalization and mRoPE; dense batched SDPA."""
    def _attend(self,qkv,packed_layout,**kwargs):
        p=packed_layout
        q,k,v=qkv.view(-1,3,self.num_heads,self.head_dim).unbind(1)
        q,k=self._normalize_packed_qk(q,k,p)
        cos,sin=self.rotary(p.positions,dtype=q.dtype)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        def dense(z):return z.view(p.batch_size,p.sequence_length,self.num_heads,self.head_dim).transpose(1,2)
        out=F.scaled_dot_product_attention(dense(q),dense(k),dense(v),attn_mask=p.flex_block_mask)
        return out.transpose(1,2).reshape(-1,self.hidden_size)


class VLAMFFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList()
        for _ in range(6):
            block=MFBlock(hidden_size=256,num_heads=4,head_dim=64,ffn_hidden_size=768,
                          modality_ids=(0,1,2,3),attention_mode='shared',ffn_mode='modality_specific')
            block.attention=RoutedSDPA(256,4,64)
            self.blocks.append(block)
        self.norm=RMSNorm(256)

    def forward(self,x,pad,nlanguage,naction,target='action'):
        b,n,d=x.shape;device=x.device
        assert n==270+nlanguage+1+naction
        coords=[]
        for camera,(h,w) in enumerate(((10,18),(5,9),(5,9))):
            a=torch.arange(h*w,device=device)
            coords.append(torch.stack((torch.full_like(a,camera),a//w,a%w)))
        def line(count,stream):
            a=torch.arange(count,device=device)
            return torch.stack((torch.full_like(a,stream),torch.zeros_like(a),a))
        pos=torch.cat((*coords,line(nlanguage,3),line(1,4),line(naction,5)),1)
        ids=torch.tensor([0]*270+[1]*nlanguage+[3]+[2]*naction,device=device)
        selected={'vision':ids==0,'language':ids==1,'action':ids==2}[target]
        # Clean conditions cannot see corrupted target tokens. Target sees all.
        allowed=~((~selected)[:,None]&selected[None,:])
        allowed=allowed[None,None]&(~pad[:,None,None,:])
        # Padding queries remain finite but their results are never used.
        layout=PackedRoutingLayout(positions=pos.repeat(1,b),modality_ids=ids.repeat(b),
            cu_seqlens=torch.arange(b+1,device=device,dtype=torch.int32)*n,
            active_indices=torch.arange(b*n,device=device),active_mask=torch.ones(b,n,device=device,dtype=torch.bool),
            dense_modality_ids=ids.expand(b,-1),batch_size=b,sequence_length=n,hidden_size=d,max_seqlen=n,
            flex_block_mask=allowed,flex_sequence_length=None,target_mask=selected.repeat(b))
        packed=layout.with_tokens(x)
        for block in self.blocks:packed=replace(packed,tokens=block.forward_packed(packed))
        return self.norm(packed.tokens).view(b,n,d)


class MFPlanner(Planner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fusion=VLAMFFusion()
        self.flow_time=TimestepEmbedder(256)
        self.flow_heads=nn.ModuleDict({'vision':nn.Linear(256,768),
                                     'language':nn.Linear(256,768),'action':nn.Linear(256,16)})

    def fuse(self,c,pad,noisy,steps):
        a=self.action_codec(self.head.norm_odo(noisy).flatten(2))+self.types.weight[3]
        a=a+self.noise_embed(steps.float().view(-1,1,1)/1000)
        tokens=torch.cat((c,a),1)
        padding=torch.cat((pad,torch.zeros(a.shape[:2],dtype=torch.bool,device=a.device)),1)
        fused=self.fusion(tokens,padding,c.shape[1]-271,a.shape[1],'action')
        return fused[:,:c.shape[1]],fused[:,c.shape[1]:]

    def flow_loss(self,views,language,mask,ego,trajectory,target,noise=None,t=None,details=False):
        assert target in ('vision','language','action')
        values={'vision':F.layer_norm(torch.cat(views,1).detach().float(),(768,)),
                'language':F.layer_norm(language.detach().float(),(768,)),
                'action':self.head.norm_odo(trajectory.detach().float()[...,:2]).flatten(1)[:,None]}
        clean=values[target]
        noise=torch.randn_like(clean) if noise is None else noise
        t=torch.rand(len(clean),device=clean.device) if t is None else t
        values[target]=(1-t[:,None,None])*noise+t[:,None,None]*clean
        v=self.vision_proj(values['vision'])+self.vision_pos+self.types.weight[0]
        v=v+torch.cat([self.camera_embed.weight[i].expand(n,-1) for i,n in enumerate((180,45,45))])[None]
        l=self.language_proj(values['language'])+self.types.weight[1]
        a=self.action_codec(values['action'])+self.types.weight[3]
        clocks={name:self.flow_time(t if name==target else torch.ones_like(t))[:,None] for name in values}
        v=v+clocks['vision'];l=l+clocks['language'];a=a+clocks['action']
        state=self.ego(ego)[:,None]+self.types.weight[2]
        pad=torch.cat((torch.zeros(len(v),270,device=v.device,dtype=torch.bool),~mask.bool(),
                       torch.zeros(len(v),2,device=v.device,dtype=torch.bool)),1)
        fused=self.fusion(torch.cat((v,l,state,a),1),pad,l.shape[1],1,target)
        selected={'vision':fused[:,:270],'language':fused[:,270:270+l.shape[1]],'action':fused[:,-1:]}[target]
        pred=self.flow_heads[target](selected).float();velocity=clean-noise
        active=mask.bool() if target=='language' else torch.ones(pred.shape[:2],device=pred.device,dtype=torch.bool)
        mse=(pred-velocity).square().mean(-1)[active].mean()
        cosine=F.cosine_similarity(pred,velocity,dim=-1)[active].mean()
        # NAVSIM adaptation: explicit cosine-direction term, separately logged.
        loss=mse+.1*(1-cosine)
        if details:return loss,dict(mse=mse,cosine=cosine,prediction=pred,velocity=velocity)
        return loss,dict(mse=mse.detach(),cosine=cosine.detach())


def flow_weight(step,total_steps):
    return .1*max(0.,min(1.,(total_steps-step)/(.4*total_steps)))
