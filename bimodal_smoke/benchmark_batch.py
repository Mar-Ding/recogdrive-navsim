"""Bounded cached-feature training memory/throughput probe; no checkpoints written."""
import argparse
import gc
import json
import time
import sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['cross','mf'],required=True);args=p.parse_args()
torch.set_num_threads(2)
root=BASE/'bimodal_smoke'
samples=torch.load(root/'artifacts/debug_samples.pt',map_location='cpu',weights_only=False)['samples']
rows=[]
for size in (1,8,16,32,64):
    torch.manual_seed(42)
    net=BimodalPlanner(args.variant,root/'upstream/kmeans_navsim_traj_20.npy').cuda().train()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    inputs=[torch.stack([samples[i%len(samples)][key] for i in range(size)]).cuda() for key in ('vision','lidar','status','target')]
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(8):
            if step==3:torch.cuda.synchronize();started=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=net(*inputs[:3],target=inputs[3])['trajectory_loss']
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        torch.cuda.synchronize();seconds=(time.perf_counter()-started)/5
        row=dict(batch=size,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds_per_step=seconds,samples_per_second=size/seconds)
    except torch.cuda.OutOfMemoryError:
        row=dict(batch=size,oom=True)
    rows.append(row);print(json.dumps(row),flush=True)
    del net,opt,inputs
    if 'loss' in globals():del loss
    gc.collect();torch.cuda.empty_cache()
    if row.get('oom'):break
out=root/'outputs'/args.variant/'batch_benchmark.json'
out.write_text(json.dumps(dict(variant=args.variant,scope='Repeated debug samples, cached frozen vision features, BF16 AdamW, 3 warmup + 5 measured steps',results=rows),indent=2))
