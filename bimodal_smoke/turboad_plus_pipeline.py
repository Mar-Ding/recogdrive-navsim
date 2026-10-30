"""Bounded Plus training and identical official subset evaluation."""
import os,time,subprocess,json,sys
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad';CODE=BASE/'bimodal_smoke'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner

def run(stage,args,gpu,timeout):
    (ROOT/'plus_status.json').write_text(json.dumps({'stage':stage}));print(stage,flush=True)
    with (ROOT/'logs'/f'{stage}.log').open('w') as log:
        subprocess.run(['bash','run.sh',*args],cwd=CODE,env={**os.environ,'GPU':str(gpu)},stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def main():
    start=time.monotonic()
    while not (ROOT/'plus_samples.pt').exists():
        if time.monotonic()-start>1200:raise TimeoutError('Plus feature preparation timeout')
        time.sleep(5)
    run('plus_train',['turboad_train.py','--variant','mf_plus','--run','plus_seed42','--epochs','20','--batch','32'],0,1200)
    torch.manual_seed(42);net=BimodalPlanner('mf_plus',CODE/'upstream/kmeans_navsim_traj_20.npy')
    torch.save(dict(model=net.state_dict(),variant='mf_plus',name='TurboAD-Plus',step=0),ROOT/'plus_initial.pt');del net
    for label,path in [('plus_initial',ROOT/'plus_initial.pt'),('plus',ROOT/'plus_seed42/best_val_ade.pt')]:
        run('score_'+label,['turboad_eval.py','score','--tokens',str(ROOT/'navtest_subset_tokens.json'),'--checkpoint',str(path),'--label',label],2,1200)
    (ROOT/'plus_status.json').write_text(json.dumps({'stage':'complete'}))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'plus_status.json').write_text(json.dumps({'stage':'failed','error':repr(e)}));raise
