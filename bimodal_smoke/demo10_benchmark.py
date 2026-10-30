"""Single GPU cached-feature, real optimizer memory/throughput measurement."""
import argparse,gc,json,sys,time
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['cross','mf'],required=True);a=p.parse_args()
torch.set_num_threads(2);samples=torch.load(BASE/'turboad/samples.pt',map_location='cpu',weights_only=False)['samples'];results=[]
for size in (64,128):
    torch.manual_seed(42);net=BimodalPlanner(a.variant,BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda().train()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95));inputs=[torch.stack([samples[i%len(samples)][k] for i in range(size)]).cuda().float() for k in ('vision','lidar','status','target')]
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(5):
            if step==2:torch.cuda.synchronize();started=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(*inputs[:3],target=inputs[3])['trajectory_loss']
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();seconds=(time.perf_counter()-started)/3
        row=dict(batch=size,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds_per_step=seconds,samples_per_second=size/seconds)
    except torch.cuda.OutOfMemoryError:row=dict(batch=size,oom=True)
    results.append(row);print(json.dumps(row),flush=True);del net,opt,inputs
    if 'loss' in globals():del loss
    gc.collect();torch.cuda.empty_cache()
    if row.get('oom'):break
out=BASE/'demo10';out.mkdir(exist_ok=True);(out/f'benchmark_{a.variant}.json').write_text(json.dumps(dict(variant=a.variant,scope='cached frozen DINOv3; single 3090 BF16 AdamW, 2 warmup + 3 measured steps',results=results),indent=2))
