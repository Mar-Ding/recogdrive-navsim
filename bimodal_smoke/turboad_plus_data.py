"""Add aligned L0/R0 images to the exact previous training and test tokens."""
import sys,json,pickle,tarfile
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch,numpy as np
from PIL import Image
from transformers import AutoModel
VIEWS=['CAM_F0','CAM_L0','CAM_R0']

def main():
    train=json.loads((ROOT/'manifest.json').read_text());test=json.loads((ROOT/'navtest_subset_manifest.json').read_text())
    by_token={r['token']:dict(r,split_source=split) for split,rows in [('trainval',train),('test',test)] for r in rows}
    def consume(frames):
        for f in frames:
            if f['token'] in by_token:
                r=by_token[f['token']];r['views']={k:f['cams'][k]['data_path'] for k in VIEWS}
    logs={r['log_name'] for r in train}
    with tarfile.open(BASE/'data/metadata_trainval.tgz','r|gz') as tar:
        for m in tar:
            if m.isfile() and m.name.endswith('.pkl') and Path(m.name).stem in logs:consume(pickle.loads(tar.extractfile(m).read()))
    for name in {r['log_name'] for r in test}:consume(pickle.load((BASE/'data/openscene-v1.1/meta_datas/test'/f'{name}.pkl').open('rb')))
    assert all('views' in r for r in by_token.values())
    for split in ('trainval','test'):
        wanted={p for r in by_token.values() if r['split_source']==split for p in r['views'].values()}
        destroot=BASE/'data/sensor_blobs'/split
        with tarfile.open(BASE/'data'/f'{split}_camera_0.tgz','r|gz') as tar:
            for m in tar:
                if not m.isfile():continue
                parts=Path(m.name).parts;rel=next(('/'.join(parts[i:]) for i in range(len(parts)) if '/'.join(parts[i:]) in wanted),None)
                if rel is None:continue
                dest=(destroot/rel).resolve();assert dest.is_relative_to(destroot.resolve())
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(m).read())
        missing=[p for p in wanted if not (destroot/p).exists()];assert not missing,('Missing aligned cameras',split,len(missing))
        print('VIEWS_READY',split,len(wanted),flush=True)
    (ROOT/'plus_manifest.json').write_text(json.dumps(list(by_token.values()),indent=2))
    torch.set_num_threads(2)
    encoder=AutoModel.from_pretrained('/home/yiding/TurboVLA-repro/assets/dinov3-vitb',local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    payload=torch.load(ROOT/'samples.pt',weights_only=False);samples=payload['samples']
    for i in range(0,len(samples),8):
        chunk=samples[i:i+8];paths=[BASE/'data/sensor_blobs/trainval'/by_token[r['token']]['views'][v] for r in chunk for v in VIEWS]
        pixels=torch.stack([torch.from_numpy(np.array(Image.open(p).convert('RGB').resize((288,160)),dtype=np.float32)/255).permute(2,0,1) for p in paths]).cuda()
        pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(pixel_values=pixels).last_hidden_state[:,5:].reshape(len(chunk),540,768).cpu()
        for r,v in zip(chunk,features):r['vision']=v
        print('ENCODE_PLUS',i,len(samples),flush=True)
    torch.save(payload,ROOT/'plus_samples.pt');print('PLUS_DATA_OK',len(samples),flush=True)
if __name__=='__main__':main()
