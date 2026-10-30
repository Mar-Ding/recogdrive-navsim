import sys,json
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner

def main():
    torch.set_num_threads(2)
    sample=torch.load(ROOT/'plus_samples.pt',weights_only=False)['samples'][0]
    net=BimodalPlanner('mf_plus',BASE/'bimodal_smoke/upstream/kmeans_navsim_traj_20.npy').cuda().eval()
    ckpt=torch.load(ROOT/'plus_seed42/best_val_ade.pt',map_location='cpu',weights_only=False);net.load_state_dict(ckpt['model'],strict=True)
    v,l,s=[sample[k][None].cuda().float() for k in ('vision','lidar','status')]
    predictions=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for view in (None,1,2):
            modified=v.clone()
            if view is not None:modified[:,view*180:(view+1)*180]=0
            torch.manual_seed(999);predictions.append(net(modified,l,s)['trajectory'].float())
    deltas=[float((p-predictions[0]).abs().max()) for p in predictions[1:]]
    assert all(x>0 for x in deltas)
    report=dict(checkpoint_epoch=ckpt['epoch'],trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),left_zero_max_abs_delta=deltas[0],right_zero_max_abs_delta=deltas[1],scope='Single-sample computational dependency check; not a performance ablation')
    (ROOT/'plus_model_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
