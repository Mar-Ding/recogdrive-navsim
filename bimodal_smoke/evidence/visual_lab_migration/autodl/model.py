"""BERT-conditioned two- and three-modal NAVSIM research prototypes.

V+L: action anchors enter only DiffusionDrive. V+L+A: noisy action chunks also
enter the chunk-causal fusion stack at each diffusion step. This implementation
is MF-inspired and is not an official MF registered extension.
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
import navsim.agents
navsim.agents.__path__.append('/root/autodl-tmp/mf-vla/src/DiffusionDrive/navsim/agents')
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead
from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position

class XYTrajectoryHead(TrajectoryHead):
    def norm_odo(self,x):
        return torch.stack((2*(x[...,0]+1.2)/56.9-1,2*(x[...,1]+20)/46-1),-1)
    def denorm_odo(self,x):
        return torch.stack(((x[...,0]+1)/2*56.9-1.2,(x[...,1]+1)/2*46-20),-1)

class Planner(nn.Module):
    def __init__(self, anchors, mode, bert_path, vision_path):
        super().__init__()
        if mode not in ('vl','vla'): raise ValueError(mode)
        self.mode=mode
        self.vision=AutoModel.from_pretrained(vision_path,local_files_only=True).eval()
        self.language=AutoModel.from_pretrained(bert_path,local_files_only=True,add_pooling_layer=False).eval()
        for encoder in (self.vision,self.language):
            for p in encoder.parameters(): p.requires_grad_(False)
        self.vision_proj=nn.Linear(768,256)
        self.language_proj=nn.Linear(768,256)
        self.vision_pos=nn.Parameter(torch.randn(1,270,256)*.02)
        self.camera_embed=nn.Embedding(3,256)
        self.ego=nn.Linear(4,256)
        self.types=nn.Embedding(4,256)
        if mode=='vla':
            self.action_codec=nn.Sequential(nn.Linear(16,256),nn.GELU(),nn.Linear(256,256))
            self.noise_embed=nn.Sequential(nn.Linear(1,256),nn.SiLU(),nn.Linear(256,256))
        self.fusion=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,1024,0.,batch_first=True),6,enable_nested_tensor=False)
        self.query=nn.Parameter(torch.randn(1,31,256)*.02)
        self.query_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,8,1024,0.,batch_first=True),2)
        self.head=XYTrajectoryHead(8,1024,256,str(anchors),TransfuserConfig())
    def context(self,image,ids,language_mask,ego):
        self.vision.eval(); self.language.eval()
        with torch.no_grad():
            assert len(image)==3, "expected ordered [F0,L0,R0] images"
            patches=[self.vision(pixel_values=view).last_hidden_state[:,5:,:] for view in image]
            l=self.language(input_ids=ids,attention_mask=language_mask).last_hidden_state
        assert [x.shape[1] for x in patches]==[180,45,45], [x.shape for x in patches]
        visual=torch.cat([self.vision_proj(x)+self.camera_embed.weight[i] for i,x in enumerate(patches)],dim=1)
        visual=visual+self.vision_pos+self.types.weight[0]
        language=self.language_proj(l)+self.types.weight[1]
        state=self.ego(ego).unsqueeze(1)+self.types.weight[2]
        c=torch.cat((visual,language,state),dim=1)
        pad=torch.cat((torch.zeros(image[0].shape[0],270,device=image[0].device,dtype=torch.bool),~language_mask.bool(),torch.zeros(image[0].shape[0],1,device=image[0].device,dtype=torch.bool)),dim=1)
        return c,pad
    def fuse(self,c,pad,noisy,steps):
        n=c.shape[1]
        if self.mode=='vla':
            a=self.action_codec(self.head.norm_odo(noisy).flatten(2))+self.types.weight[3]
            a=a+self.noise_embed(steps.float().view(-1,1,1)/1000)
            tokens=torch.cat((c,a),1)
            padding=torch.cat((pad,torch.zeros(a.shape[:2],dtype=torch.bool,device=a.device)),1)
            mask=torch.zeros(tokens.shape[1],tokens.shape[1],dtype=torch.bool,device=tokens.device)
            mask[:n,n:]=True
            fused=self.fusion(tokens,mask=mask,src_key_padding_mask=padding)
            return fused[:,:n],fused[:,n:]
        return self.fusion(c,src_key_padding_mask=pad),None
    def decode(self,c,pad,noisy,steps):
        cond,action=self.fuse(c,pad,noisy,steps)
        v=cond[:,:180].transpose(1,2).reshape(-1,256,10,18)
        grid=F.interpolate(v,size=(16,16),mode='bilinear',align_corners=False)
        q=self.query_decoder(self.query.expand(noisy.shape[0],-1,-1),cond,memory_key_padding_mask=pad)
        time=self.head.time_mlp(steps).view(-1,1,256)
        pe=gen_sineembed_for_position(noisy,hidden_dim=64).flatten(-2)
        traj=self.head.plan_anchor_encoder(pe)
        if action is not None: traj=traj+action
        return self.head.diff_decoder(traj,noisy,grid,(16,16),q[:,1:],q[:,:1],time,cond[:,-1:],None)
    def forward(self,image,ids,language_mask,ego,targets=None):
        c,pad=self.context(image,ids,language_mask,ego)
        b=image[0].shape[0]; dev=image[0].device
        anchors=self.head.plan_anchor.unsqueeze(0).expand(b,-1,-1,-1)
        clean=self.head.norm_odo(anchors); noise=torch.randn_like(clean)
        if self.training:
            assert targets is not None
            steps=torch.randint(0,50,(b,),device=dev)
            noisy=self.head.denorm_odo(self.head.diffusion_scheduler.add_noise(clean,noise,steps).clamp(-1,1))
            regs,clss=self.decode(c,pad,noisy,steps)
            loss=sum(self.head.loss_computer(r,s,targets,anchors) for r,s in zip(regs,clss))
            index=clss[-1].argmax(-1)
            return {'loss':loss,'trajectory':regs[-1][torch.arange(b,device=dev),index]}
        self.head.diffusion_scheduler.set_timesteps(1000,dev)
        x=self.head.diffusion_scheduler.add_noise(clean,noise,torch.full((b,),8,device=dev,dtype=torch.long))
        for t in (10,0):
            noisy=self.head.denorm_odo(x.clamp(-1,1))
            steps=torch.full((b,),t,device=dev,dtype=torch.long)
            regs,clss=self.decode(c,pad,noisy,steps)
            x=self.head.diffusion_scheduler.step(self.head.norm_odo(regs[-1][...,:2]),t,x).prev_sample
        index=clss[-1].argmax(-1)
        return {'trajectory':regs[-1][torch.arange(b,device=dev),index]}
