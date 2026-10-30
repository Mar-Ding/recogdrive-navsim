"""Two sensor fusion choices, one unchanged upstream diffusion trajectory head."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from vendor.diffusion_decoder import TrajectoryHead

WIDTH=256

class CrossFusion(nn.Module):
    def __init__(self,depth=6):
        super().__init__()
        from turbovla.models.components.fusion import BiAttentionBlock
        self.cross=nn.ModuleList([BiAttentionBlock(256,256,256,4,dropout=0.,residual_style='pre_norm',attention_backend='sdpa') for _ in range(depth)])
        self.ffns=nn.ModuleList([nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.Linear(256,768),nn.GELU(),nn.Linear(768,256)) for _ in range(2)]) for _ in range(depth)])

    def forward(self,v,l):
        for cross,ffns in zip(self.cross,self.ffns):
            v,l=cross(v,l)
            v,l=v+ffns[0](v),l+ffns[1](l)
        return v,l

class MFFusion(nn.Module):
    """E005-style upstream MF blocks + SDPA, with sensor-specific spatial mRoPE.

    LiDAR uses local modality ID 3 (neither text=1 nor old robot action=2).
    Both sensors are clean conditioning streams; fully bidirectional visibility.
    This first smoke uses trajectory supervision, without auxiliary flow pretraining.
    """
    def __init__(self,depth=6):
        super().__init__()
        from mf.modeling.layers import RMSNorm
        if not hasattr(nn,'RMSNorm'):nn.RMSNorm=RMSNorm
        from mf.modeling.block import MFBlock
        from mf.modeling.attention import SharedQKVOAttention
        from mf.modeling.mrope import apply_rotary_pos_emb

        class SensorSDPA(SharedQKVOAttention):
            def _attend(self,qkv,packed_layout,**kwargs):
                p=packed_layout
                q,k,val=qkv.view(-1,3,self.num_heads,self.head_dim).unbind(1)
                q,k=self._normalize_packed_qk(q,k,p)
                cos,sin=self.rotary(p.positions,dtype=q.dtype)
                q,k=apply_rotary_pos_emb(q,k,cos,sin)
                def dense(z):return z.view(p.batch_size,p.sequence_length,self.num_heads,self.head_dim).transpose(1,2)
                y=F.scaled_dot_product_attention(dense(q),dense(k),dense(val),attn_mask=p.flex_block_mask)
                return y.transpose(1,2).reshape(-1,self.hidden_size)

        self.blocks=nn.ModuleList()
        for _ in range(depth):
            block=MFBlock(hidden_size=256,num_heads=4,head_dim=64,ffn_hidden_size=768,modality_ids=(0,3))
            block.attention=SensorSDPA(256,4,64)
            self.blocks.append(block)
        self.norm=RMSNorm(256)

    def forward(self,v,l):
        from mf.modeling.packing import PackedRoutingLayout
        b=v.shape[0];n=v.shape[1]+l.shape[1];device=v.device
        assert v.shape[1] in (180,270,540) and l.shape[1]==256
        nv=v.shape[1]
        rgb=torch.arange(nv,device=device);bev=torch.arange(256,device=device)
        pos=torch.cat([torch.stack([rgb//180,(rgb%180)//18,rgb%18]),
                       torch.stack([torch.full_like(bev,nv//180),bev//16,bev%16])],1)
        if nv==270:
            positions=[]
            for view,(height,width) in enumerate(((10,18),(5,9),(5,9))):
                index=torch.arange(height*width,device=device)
                positions.append(torch.stack([torch.full_like(index,view),index//width,index%width]))
            pos=torch.cat([*positions,torch.stack([torch.full_like(bev,3),bev//16,bev%16])],1)
        ids=torch.cat([torch.zeros_like(rgb),torch.full_like(bev,3)])
        layout=PackedRoutingLayout(positions=pos.repeat(1,b),modality_ids=ids.repeat(b),
            cu_seqlens=torch.arange(b+1,device=device,dtype=torch.int32)*n,
            active_indices=torch.arange(b*n,device=device),active_mask=torch.ones(b,n,device=device,dtype=torch.bool),
            dense_modality_ids=ids.expand(b,-1),batch_size=b,sequence_length=n,hidden_size=256,max_seqlen=n,
            flex_block_mask=None,flex_sequence_length=None,target_mask=None)
        packed=layout.with_tokens(torch.cat([v,l],1))
        for block in self.blocks:packed=replace(packed,tokens=block.forward_packed(packed))
        x=self.norm(packed.tokens).view(b,n,256)
        return x[:,:nv],x[:,nv:]

class BimodalPlanner(nn.Module):
    def __init__(self,variant,anchor_path):
        super().__init__()
        self.variant=variant
        # Construct common modules before the variant so same seed gives equal common initialization.
        self.vision_projection=nn.Sequential(nn.Linear(768,256),nn.LayerNorm(256))
        channels=[1,32,64,128,256]
        self.lidar_encoder=nn.Sequential(*[nn.Sequential(nn.Conv2d(channels[i],channels[i+1],3,2,1),nn.GroupNorm(8,channels[i+1]),nn.GELU()) for i in range(4)])
        self.rgb_position=nn.Parameter(torch.randn(1,{'mf_plus':540,'mf_plus_small':270}.get(variant,180),256)*.02)
        self.bev_position=nn.Parameter(torch.randn(1,256,256)*.02)
        self.sensor_embedding=nn.Parameter(torch.randn(2,256)*.02)
        self.status_encoder=nn.Linear(8,256)
        self.scene_queries=nn.Embedding(31,256)
        self.scene_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,8,1024,dropout=0.,batch_first=True),3)
        self.bev_adapter=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())
        config=SimpleNamespace(tf_d_model=256,tf_num_head=8,tf_d_ffn=1024,tf_dropout=0.,
                               lidar_max_x=32,lidar_max_y=32,trajectory_cls_weight=10.,trajectory_reg_weight=8.)
        self.trajectory_head=TrajectoryHead(8,1024,256,str(anchor_path),config)
        self.fusion=CrossFusion() if variant=='cross' else MFFusion()

    def forward(self,vision,lidar,status,target=None):
        v=self.vision_projection(vision)+self.rgb_position+self.sensor_embedding[0]
        l=self.lidar_encoder(lidar).flatten(2).transpose(1,2)+self.bev_position+self.sensor_embedding[1]
        _,l=self.fusion(v,l)
        bev=self.bev_adapter(l.transpose(1,2).reshape(-1,256,16,16))
        bev=F.interpolate(bev,size=(64,64),mode='bilinear',align_corners=False)
        state=self.status_encoder(status)[:,None]
        memory=torch.cat([l,state],dim=1)
        queries=self.scene_decoder(self.scene_queries.weight[None].expand(len(l),-1,-1),memory)
        return self.trajectory_head(queries[:,:1],queries[:,1:],bev,bev.shape[2:],state,
                                    targets=None if target is None else {'trajectory':target})
