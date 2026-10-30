"""Bounded MF initial/final evaluation on the identical TurboAD subset."""
import sys,os,time,subprocess,json
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad';CODE=BASE/'bimodal_smoke'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner

def main():
    torch.manual_seed(42)
    net=BimodalPlanner('mf',CODE/'upstream/kmeans_navsim_traj_20.npy')
    torch.save(dict(model=net.state_dict(),name='Turbo-MFAD',variant='mf',seed=42,step=0),ROOT/'mf_initial_seed42.pt');del net
    for label,checkpoint in [('mf_initial',ROOT/'mf_initial_seed42.pt'),('mf_expanded',ROOT/'mf_expanded_seed42/best_val_ade.pt')]:
        if label=='mf_expanded':
            start=time.monotonic()
            while not (ROOT/'mf_expanded_seed42/summary.json').exists():
                if time.monotonic()-start>600:raise TimeoutError('MF training did not finish')
                time.sleep(5)
        with (ROOT/'logs'/('score_'+label+'.log')).open('w') as log:
            subprocess.run(['bash','run.sh','turboad_eval.py','score','--tokens',str(ROOT/'navtest_subset_tokens.json'),'--checkpoint',str(checkpoint),'--label',label],cwd=CODE,env={**os.environ,'GPU':'2'},stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
        print('SCORE_COMPLETE',label,flush=True)
    (ROOT/'mf_evaluation_status.json').write_text(json.dumps({'stage':'complete'}))
if __name__=='__main__':main()
