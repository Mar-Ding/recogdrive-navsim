"""MF action extension. Import registers definitions; constructs no model/data."""
__mf_extension_version__ = '1.0.0'
import torch
from torch import nn
from mf.contracts.task_registry import TaskDefinition, BranchRole, register_task
from mf.registries import (register_physical_layout, register_output_head,
    register_objective, register_sampler, register_decoder, register_generation)
from vla.action_extension import register_action
from vla.action_extension import ActionEncoder, ActionDecoder
from mf.codecs.registry import CODEC_REGISTRY, CodecDefinition

MODALITIES = ('image', 'text', 'action')
register_action()
for name, modality in (('vla_image_features','vision'),('vla_text_features','text')):
    CODEC_REGISTRY.register(CodecDefinition(name,ActionEncoder,ActionDecoder,
        frozenset({'continuous','flow_target','features'}),modality))
for i, target in enumerate(MODALITIES):
    register_task(TaskDefinition(101+i, f'vla_flow_{target}',
        {m: BranchRole.TARGET if m == target else BranchRole.CONDITION for m in MODALITIES},
        ('vla_flow',), sampler='vla_balanced', planner_order=i, packing_order=i, metric_order=i))
register_task(TaskDefinition(104, 'vla_act_policy',
    {'image': BranchRole.CONDITION, 'text': BranchRole.CONDITION, 'action': BranchRole.TARGET},
    ('vla_bc_teacher',), sampler='vla_balanced', planner_order=3, packing_order=3, metric_order=3))

def layout(*args, **kwargs):
    from vla.extension_layout import compile_layout
    return compile_layout(*args, **kwargs)

def velocity_head(hidden_size, latent_dim):
    return nn.Linear(hidden_size, latent_dim)

def flow_objective(prediction, batch):
    # The compiled physical target map, not caller chunk order, owns the target.
    physical = batch.physical_layout
    target = batch.target_modality
    target_id = batch.target_id
    clean = physical.target_latents[target_id]
    # x_t=(1-t)*noise+t*x_1 => velocity=(x_1-x_t)/(1-t).
    # Keep the original sampled noise for stable t close to one.
    velocity = clean - batch.noise.reshape_as(clean)
    valid = ~batch.padding[target].reshape(-1)
    error = (prediction.reshape_as(clean).float() - velocity.float()).square()
    return (error * valid[:, None]).sum() / (valid.sum() * clean.shape[-1]).clamp_min(1)

def bc_objective(prediction, target, teacher, valid, teacher_weight=.25):
    def masked(y):
        return ((prediction.float()-y.float()).abs()*valid[..., None]).sum()/(valid.sum()*y.shape[-1]).clamp_min(1)
    bc, distill = masked(target), masked(teacher)
    return bc + teacher_weight * distill, bc, distill

class BalancedSampler:
    """Exact consumed-index cursor, independent of DataLoader prefetch."""
    def __init__(self, size, batch_size, seed, shuffle=True):
        if size < batch_size or batch_size <= 0:
            raise ValueError('Invalid dataset/batch size')
        self.size, self.batch_size, self.seed, self.shuffle = size, batch_size, seed, shuffle
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.rng) if shuffle else torch.arange(size)
        self.cursor, self.step, self.epoch = 0, 0, 0
    def next(self):
        if self.cursor + self.batch_size > self.size:
            self.order = torch.randperm(self.size, generator=self.rng) if self.shuffle else torch.arange(self.size)
            self.cursor = 0
            self.epoch += 1
        ids = self.order[self.cursor:self.cursor+self.batch_size].clone()
        task = f'vla_flow_{MODALITIES[self.step % 3]}'
        self.cursor += self.batch_size
        self.step += 1
        return ids, task
    def state_dict(self):
        return dict(version=1, size=self.size, batch_size=self.batch_size, seed=self.seed,
            shuffle=self.shuffle, order=self.order.clone(), cursor=self.cursor, step=self.step,
            epoch=self.epoch, rng=self.rng.get_state())
    def load_state_dict(self, state):
        for key in ('size', 'batch_size', 'seed', 'shuffle'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Sampler resume mismatch: {key}')
        assert state['version'] == 1
        assert torch.equal(torch.sort(state['order']).values, torch.arange(self.size))
        assert 0 <= state['cursor'] <= self.size
        self.order, self.cursor, self.step, self.epoch = state['order'], state['cursor'], state['step'], state['epoch']
        self.rng.set_state(state['rng'])

def act_decoder(head, condition, state, padding, vision_length, horizon):
    from vla.attention_revision import decode_with_padding
    return decode_with_padding(head, condition, state, padding, vision_length, horizon)

def act_generation(core, head, vision, text, padding, sub, state):
    from mf.registries import DECODER_REGISTRY
    from vla.extension_runtime import initialize
    tokens = core.policy_tokens(vision, text, padding, sub)
    # Output ordering follows the compiled policy contract.
    plan = core.plan({m:tokens[m].shape[1] for m in MODALITIES}, 'vla_act_policy', vision.device)
    condition = torch.cat([tokens[c.modality] for c in plan.compiled.chunks], 1)
    return DECODER_REGISTRY.resolve(initialize()['resolved_config']['decoder'])(
        head, condition, state, padding, vision.shape[1], core.horizon)

def flow_generation(core, conditions, target='action', steps=20, padding=None, sub=None, generator=None):
    """Registered complete feature/action flow integration and codec decode path."""
    if steps <= 0 or target not in MODALITIES:
        raise ValueError('Invalid flow generation request')
    values = dict(conditions)
    template = values[target]
    values[target] = torch.randn(template.shape, device=template.device, dtype=template.dtype, generator=generator)
    for index in range(steps):
        t = template.new_full((len(template),), index/steps)
        batch = core.prepare(values, f'vla_flow_{target}', t, padding, sub)
        hidden = core.forward_batch(batch)[target]
        from mf.registries import DECODER_REGISTRY
        values[target] = values[target] + DECODER_REGISTRY.resolve('vla_continuous')(core.heads[target],hidden)/steps
    # Heads decode hidden states to the modality's continuous latent domain.
    return values[target]

register_physical_layout('vla_prefix', layout)
register_output_head('vla_velocity', velocity_head)
register_objective('vla_flow', flow_objective)
register_objective('vla_bc_teacher', bc_objective)
register_sampler('vla_balanced', BalancedSampler)
register_decoder('vla_act_decoder', act_decoder)
register_decoder('vla_continuous', lambda head, hidden: head(hidden))
register_generation('vla_act_forward', act_generation)
register_generation('vla_flow_euler', flow_generation)
