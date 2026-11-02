"""Sensor flow auxiliary objective, adapted from TurboMFVLA E010.

Fixed targets: frozen RGB features and nonlearned LiDAR BEV patches.
No target trajectory is supplied to auxiliary conditioning or inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import BimodalPlanner


class FlowPlanner(BimodalPlanner):
    def __init__(self, variant, anchor_path):
        assert variant == 'mf'
        super().__init__(variant, anchor_path)
        from mf.modeling.embeddings import TimestepEmbedder
        self.flow_time = TimestepEmbedder(256)
        self.flow_rgb_in = nn.Linear(768, 256)
        self.flow_lidar_in = nn.Linear(256, 256)
        self.flow_rgb_out = nn.Linear(256, 768)
        self.flow_lidar_out = nn.Linear(256, 256)

    def flow_loss(self, vision, lidar, target, noise=None, t=None, return_details=False):
        assert target in ('rgb', 'lidar')
        # Each nonoverlapping 16x16 BEV patch is one fixed 256-dimensional token.
        values = {'rgb': F.layer_norm(vision.detach().float(), (768,)),
                  'lidar': F.layer_norm(F.unfold(lidar.detach().float(), 16, stride=16).transpose(1, 2), (256,))}
        clean = values[target]
        noise = torch.randn_like(clean) if noise is None else noise
        t = torch.rand(len(clean), device=clean.device) if t is None else t
        values[target] = (1-t[:, None, None])*noise+t[:, None, None]*clean
        rgb_clock = t if target == 'rgb' else torch.ones_like(t)
        lidar_clock = t if target == 'lidar' else torch.ones_like(t)
        v = self.flow_rgb_in(values['rgb']) + self.rgb_position + self.sensor_embedding[0] + self.flow_time(rgb_clock)[:, None]
        l = self.flow_lidar_in(values['lidar']) + self.bev_position + self.sensor_embedding[1] + self.flow_time(lidar_clock)[:, None]
        v, l = self.fusion(v, l)
        prediction = self.flow_rgb_out(v) if target == 'rgb' else self.flow_lidar_out(l)
        velocity = clean-noise
        loss = F.mse_loss(prediction.float(), velocity)
        cosine = F.cosine_similarity(prediction.float(), velocity, dim=-1).mean().detach()
        if return_details:
            return loss, cosine, prediction, velocity
        return loss, cosine


def flow_weight(step, total_steps):
    return .1*max(0., min(1., (total_steps-step)/(.4*total_steps)))
