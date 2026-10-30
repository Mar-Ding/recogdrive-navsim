"""Fresh-process checkpoint demo: raw RGB + PCD -> frozen encoder -> trained planner."""
import sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import argparse
import json
import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from model import BimodalPlanner

ROOT=BASE/'bimodal_smoke'

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['cross','mf'],required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(999)
    data=torch.load(ROOT/'artifacts/debug_samples.pt',map_location='cpu',weights_only=False)
    out=ROOT/'outputs'/args.variant
    checkpoint=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    model=BimodalPlanner(args.variant,ROOT/'upstream/kmeans_navsim_traj_20.npy').cuda().eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    encoder=AutoModel.from_pretrained(data['encoder'],local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    records=[];torch.cuda.reset_peak_memory_stats()
    for row in data['samples']:
        pixels=torch.tensor(np.array(Image.open(row['image']).convert('RGB').resize((288,160)),dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,:,None,None]
        cloud=LidarPointCloud.from_buffer(Path(row['pcd']).read_bytes(),'pcd').points[:3].T
        cloud=cloud[(cloud[:,2]>.2)&(cloud[:,2]<100)]
        bins=np.linspace(-32,32,257);bev=np.histogramdd(cloud[:,:2],bins=(bins,bins))[0]
        lidar=torch.tensor((np.minimum(bev,5)/5)[None,None],dtype=torch.float32,device='cuda')
        status=row['status'][None].cuda()
        torch.cuda.synchronize();start=time.perf_counter();torch.manual_seed(999)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            vision=encoder(pixel_values=pixels).last_hidden_state[:,5:]
            prediction=model(vision.float(),lidar,status)['trajectory'].float()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert prediction.shape==(1,8,3) and torch.isfinite(prediction).all()
        vision_error=float((vision.float().cpu()-row['vision'][None]).abs().max())
        assert vision_error==0,vision_error
        assert torch.equal(lidar.cpu()[0],row['lidar'])
        records.append(dict(token=row['token'],prediction=prediction[0].cpu().tolist(),ground_truth=row['target'].tolist(),
                            encode_and_plan_seconds=seconds,cached_encoder_max_difference=vision_error))
    report=dict(variant=args.variant,raw_sensor_demo_ok=True,samples=records,
                full_pipeline_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                full_pipeline_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                timing_scope='GPU encoder + planner, excludes disk I/O and CPU preprocessing; first sample is cold.')
    (out/'raw_demo.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True);print('RAW_RGB_LIDAR_CHECKPOINT_DEMO_OK',args.variant,flush=True)

if __name__=='__main__':main()
