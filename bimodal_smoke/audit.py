"""Audit shared initialization, official extraction and experiment provenance on CPU."""
import sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import ast
import hashlib
import importlib.metadata
import json
import subprocess
import torch

ROOT=BASE/'bimodal_smoke'

def main():
    a=torch.load(ROOT/'outputs/cross/common_initial.pt',map_location='cpu',weights_only=False)
    b=torch.load(ROOT/'outputs/mf/common_initial.pt',map_location='cpu',weights_only=False)
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    def definitions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    original={}
    for path in [ROOT/'upstream/transfuser_model_v2.py',*list((ROOT/'upstream/modules').glob('*.py'))]:
        original.update(definitions(path))
    extracted=definitions(ROOT/'vendor/diffusion_decoder.py')
    assert all(original[k]==v for k,v in extracted.items())
    upstream={}
    for name,path in {'mf':'/home/yiding/Multimodal-Flow2VLA/upstream/Multimodal-Flow-main/src/mf',
                      'turbo':'/home/yiding/Multimodal-Flow2VLA/upstream/TurboVLA-native/turbovla'}.items():
        upstream[name]={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(path).rglob('*.py')}
    reports={v:json.loads((ROOT/'outputs'/v/'report.json').read_text()) for v in ('cross','mf')}
    assert all(r['loss_decreased'] and r['strict_reload_max_difference']==0 for r in reports.values())
    audit=dict(common_initial_tensors=len(a),common_initialization_equal=True,official_decoder_ast_equal=True,
               extracted_definitions=list(extracted),upstream_sha256=upstream,
               gpu_policy='Only CUDA_VISIBLE_DEVICES=0 or 2; GPU1 never selected.',
               train_scope='Three already-available test samples for debugging only; no benchmark.',
               mf_scope='Upstream MFBlock + E005 SDPA approach; sensor-specific IDs/positions. No auxiliary flow pretraining or full extension-registry lifecycle in this first smoke.')
    (ROOT/'artifacts/audit.json').write_text(json.dumps(audit,indent=2))
    packages={name:importlib.metadata.version(name) for name in ('torch','transformers','timm','numpy','diffusers','safetensors')}
    # diffusers is appended from the existing ReCogDrive overlay; report actual imported versions.
    import diffusers,transformers
    packages['diffusers']=diffusers.__version__;packages['transformers']=transformers.__version__
    (ROOT/'artifacts/environment.json').write_text(json.dumps(dict(python=sys.version,packages=packages),indent=2))
    print(json.dumps(dict(common_initial_tensors=len(a),official_decoder_ast_equal=True,packages=packages),indent=2))
    print('BIMODAL_AUDIT_OK')

if __name__=='__main__':main()
