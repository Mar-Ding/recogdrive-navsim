"""E005 backbone consuming registered tasks and native physical task batches."""
from dataclasses import replace
import torch
from torch import nn
from mf.registries import PHYSICAL_LAYOUT_REGISTRY, OUTPUT_HEAD_REGISTRY, OBJECTIVE_REGISTRY
from mf.codecs.registry import CODEC_REGISTRY
from mf.modeling.block import MFBlock
from mf.modeling.layers import RMSNorm
from mf.modeling.embeddings import TimestepEmbedder
from vla.model import SDPAChunkAttention
from vla.action_extension import ActionCodecConfig
from vla.extension_runtime import initialize, identity
from vla.extension_layout import bind_batch, pack_batch

MODALITIES = ('image','text','action')

class ExtensionMF(nn.Module):
    def __init__(self,vision_dim=384,text_dim=768,action_dim=7,width=256,depth=6,horizon=12):
        runtime = initialize()  # must freeze before nn.Module construction
        super().__init__()
        cfg = runtime['resolved_config']
        if (width,depth,horizon,action_dim) != (cfg['width'],cfg['depth'],cfg['horizon'],cfg['action_dim']):
            raise ValueError('Resolved model configuration mismatch')
        self.config = dict(vision_dim=vision_dim,text_dim=text_dim,action_dim=action_dim,
            width=width,depth=depth,horizon=horizon,attention_version='extension_v3')
        self.attention_version='extension_v3'
        self.horizon,self.width=horizon,width
        self.inputs=nn.ModuleDict({m:CODEC_REGISTRY.resolve(cfg['input_codecs'][m]).encoder_builder(
            ActionCodecConfig(width,d)) for m,d in zip(MODALITIES,(vision_dim,text_dim,action_dim))})
        head=OUTPUT_HEAD_REGISTRY.resolve(cfg['output_head'])
        self.heads=nn.ModuleDict({m:head(width,d) for m,d in zip(MODALITIES,(vision_dim,text_dim,action_dim))})
        self.modality=nn.Embedding(3,width)
        self.action_queries=nn.Parameter(torch.randn(1,horizon,width)*.02)
        self.time=TimestepEmbedder(width)
        self.blocks=nn.ModuleList()
        for _ in range(depth):
            block=MFBlock(hidden_size=width,num_heads=width//64,head_dim=64,ffn_hidden_size=width*3,modality_ids=(0,1,2))
            block.attention=SDPAChunkAttention(width,width//64,64)
            self.blocks.append(block)
        self.norm=RMSNorm(width)
        self._plans={};self._validated=set()

    def plan(self,lengths,task,device):
        key=(tuple(lengths[m] for m in MODALITIES),task,str(device))
        if key not in self._plans:
            self._plans[key]=PHYSICAL_LAYOUT_REGISTRY.resolve('vla_prefix')(lengths,task,device)
        return self._plans[key]

    def prepare(self,values,task,t,padding=None,sub=None,clean=None,noise=None):
        if set(values)!=set(MODALITIES):raise ValueError('All three modalities required')
        lengths={m:values[m].shape[1] for m in MODALITIES}
        if lengths['action']!=self.horizon:raise ValueError('Action horizon mismatch')
        b,device=len(values['image']),values['image'].device
        plan=self.plan(lengths,task,device)
        pad={m:(padding or {}).get(m,torch.zeros(b,lengths[m],device=device,dtype=torch.bool)) for m in MODALITIES}
        tokens=[]
        for c in plan.compiled.chunks:
            m=c.modality
            z=self.inputs[m](values[m])
            if m=='action':z=z+self.action_queries
            clock=t if c.target_mask else torch.ones_like(t)
            z=z+self.time(clock)[:,None]+self.modality.weight[c.modality_id]
            tokens.append(z.masked_fill(pad[m][...,None],0))
        batch=bind_batch(plan,torch.cat(tokens,1),pad,sub,clean,
            values[plan.target.modality] if clean is not None else None,t if clean is not None else None,noise)
        # Structural tensors/compiled contracts are immutable cached templates.
        # Validate every new shape/task/device; per-step finite checks are in training.
        key=(tuple(lengths.values()),task,b,str(device))
        if key not in self._validated:
            batch.validate()
            self._validated.add(key)
        return batch

    def forward_batch(self,batch):
        packed,valid=pack_batch(batch)
        for block in self.blocks:
            hidden=block.forward_packed(packed).masked_fill(~valid.reshape(-1,1),0)
            packed=replace(packed,tokens=hidden)
        hidden=self.norm(packed.tokens).view(*batch.physical_layout.token_embeddings.shape)
        return {c.modality:hidden[:,s.start:s.end].masked_fill(batch.padding[c.modality][...,None],0)
            for c,s in zip(batch.plan.compiled.chunks,batch.plan.compiled.physical_spans)}

    def forward(self,values,target,t,padding_masks=None,text_self_attention_masks=None):
        return self.forward_batch(self.prepare(values,f'vla_flow_{target}',t,padding_masks,text_self_attention_masks))

    def loss_batch(self,batch):
        hidden=self.forward_batch(batch)
        target=batch.plan.target
        # Head and objective selection originate in the task/compiled contract.
        from mf.contracts.task_registry import DEFAULT_TASK_REGISTRY
        task=DEFAULT_TASK_REGISTRY.resolve(batch.plan.task_name)
        return OBJECTIVE_REGISTRY.resolve(task.loss_components[0])(self.heads[target.modality](hidden[target.modality]),batch)

    def flow_loss(self,clean,target,padding_masks=None,noise=None,t=None,text_self_attention_masks=None):
        y=clean[target].detach()
        noise=torch.randn_like(y) if noise is None else noise
        t=torch.rand(len(y),device=y.device) if t is None else t
        values=dict(clean);values[target]=(1-t[:,None,None])*noise+t[:,None,None]*y
        return self.loss_batch(self.prepare(values,f'vla_flow_{target}',t,padding_masks,text_self_attention_masks,y,noise))

    def policy_tokens(self,vision,text,padding,text_self_attention_masks=None):
        values={'image':vision,'text':text,'action':vision.new_zeros(len(vision),self.horizon,7)}
        return self.forward_batch(self.prepare(values,'vla_act_policy',vision.new_zeros(len(vision)),
            {'text':padding},text_self_attention_masks))
