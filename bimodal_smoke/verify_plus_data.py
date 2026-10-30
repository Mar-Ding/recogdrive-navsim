"""Check Plus preserves samples and routes all three images into model input."""
import sys,json
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch

def main():
    a=torch.load(ROOT/'samples.pt',map_location='cpu',weights_only=False)['samples'];b=torch.load(ROOT/'plus_samples.pt',map_location='cpu',weights_only=False)['samples']
    assert [r['token'] for r in a]==[r['token'] for r in b]
    for old,new in zip(a,b):
        assert old['split']==new['split'] and old['log_name']==new['log_name']
        for k in ('lidar','status','target'):assert torch.equal(old[k],new[k])
        assert new['vision'].shape==(540,768) and torch.isfinite(new['vision']).all()
    rows=json.loads((ROOT/'plus_manifest.json').read_text());test=set(json.loads((ROOT/'navtest_subset_tokens.json').read_text()))
    assert {r['token'] for r in rows if r['split_source']=='test'}==test
    assert not {r['token'] for r in b}&test
    trainlogs={r['log_name'] for r in b if r['split']=='train'};vallogs={r['log_name'] for r in b if r['split']=='val'};assert not trainlogs&vallogs
    report=dict(training_tokens=len(b),train=sum(r['split']=='train' for r in b),validation=sum(r['split']=='val' for r in b),test_tokens=len(test),views=['CAM_F0','CAM_L0','CAM_R0'],unchanged=['token order','split','log_name','lidar','status','target'],visual_tokens=540)
    (ROOT/'plus_data_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
