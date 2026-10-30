"""Extract upstream classes verbatim, without importing the unused TransFuser backbone."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'reference/DiffusionDrive'
OUT = ROOT / 'bimodal_smoke/vendor'
OUT.mkdir(exist_ok=True)

def classes(file, names):
    text = (SRC / file).read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    return '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names)

header = '''# Extracted from HUSTVL/DiffusionDrive (MIT); see reference/DiffusionDrive/LICENSE.
from __future__ import annotations
from typing import Dict
import math
import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from navsim.common.enums import StateSE2Index
'''
decoder = header + '\n' + classes('modules/conditional_unet1d.py', {'SinusoidalPosEmb'})
decoder += '\n\n' + classes('modules/blocks.py', {'linear_relu_ln','bias_init_with_prob','gen_sineembed_for_position','GridSampleCrossBEVAttention'})
decoder += '\n\n' + classes('modules/multimodal_loss.py', {'reduce_loss','weight_reduce_loss','py_sigmoid_focal_loss','LossComputer'})
decoder += '\n\n' + classes('transfuser_model_v2.py', {'DiffMotionPlanningRefinementModule','ModulationLayer','CustomTransformerDecoderLayer','_get_clones','CustomTransformerDecoder','TrajectoryHead'})
(OUT/'diffusion_decoder.py').write_text(decoder, encoding='utf-8')
(OUT/'__init__.py').write_text('', encoding='utf-8')
manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SRC.rglob('*') if p.is_file()}
(OUT/'source_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Upstream decoder extracted without editing class bodies')
