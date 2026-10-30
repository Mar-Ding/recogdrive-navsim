"""Encode the three existing aligned NAVSIM debug samples with frozen DINOv3."""
import sys
from pathlib import Path
BASE = Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import json
import hashlib
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

def main():
    torch.set_num_threads(2)
    root=BASE/'bimodal_smoke'; out=root/'artifacts';out.mkdir(exist_ok=True)
    manifest=json.loads((BASE/'demo/multisensor_manifest.json').read_text())
    results=json.loads((BASE/'demo/results.json').read_text())
    model_path=Path('/home/yiding/TurboVLA-repro/assets/dinov3-vitb')
    encoder=AutoModel.from_pretrained(model_path,local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    samples=[]
    for r in results:
        row=next(x for x in manifest if x['token']==r['token'])
        sensor=BASE/'data/sensor_blobs/test'
        image=sensor/row['cameras']['CAM_F0']['path']
        pcd=sensor/row['lidar']['path']
        pixels=torch.from_numpy(np.array(Image.open(image).convert('RGB').resize((288,160)),dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            v=encoder(pixel_values=pixels).last_hidden_state[:,5:]
        assert v.shape==(1,180,768) and torch.isfinite(v).all()
        points=LidarPointCloud.from_buffer(pcd.read_bytes(),'pcd').points[:3].T
        points=points[(points[:,2]>.2)&(points[:,2]<100)]
        bins=np.linspace(-32,32,257)
        bev=np.histogramdd(points[:,:2],bins=(bins,bins))[0]
        bev=np.minimum(bev,5)/5
        # Match the official NAVSIM status order: command(4), velocity(2), acceleration(2).
        import pickle
        meta=BASE/'data/openscene-v1.1/meta_datas/test'/(row['log_name']+'.pkl')
        with meta.open('rb') as f: frames=pickle.load(f)
        current=next(f for f in frames if f['token']==r['token'])
        status=np.concatenate([current['driving_command'],current['ego_dynamic_state']])
        assert status.shape==(8,)
        samples.append(dict(token=r['token'],vision=v[0].float().cpu(),lidar=torch.tensor(bev[None],dtype=torch.float32),
                            status=torch.tensor(status,dtype=torch.float32),target=torch.tensor(r['ground_truth'],dtype=torch.float32),
                            image=str(image),pcd=str(pcd),camera_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            lidar_sha256=hashlib.sha256(pcd.read_bytes()).hexdigest()))
        print('ENCODED',r['token'],v.shape,len(points),flush=True)
    torch.save(dict(samples=samples,encoder=str(model_path),rgb_size=[160,288],rgb_grid=[10,18],
                    scope='Three NAVSIM test samples: debugging only, no validation/generalization claim.'),out/'debug_samples.pt')
    print('REAL_BIMODAL_CACHE_OK',flush=True)

if __name__=='__main__':main()
