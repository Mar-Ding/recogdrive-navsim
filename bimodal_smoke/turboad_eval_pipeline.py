"""Evaluate identical initial/IL models with the upstream scorer on fixed tokens."""
import sys,time,json,subprocess,os
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad';CODE=BASE/'bimodal_smoke'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import torch
from model import BimodalPlanner
status=ROOT/'evaluation_status.json'
def report(stage,**extra):
    row=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**extra);status.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def run(stage,args,timeout):
    report(stage)
    with (ROOT/'logs'/(stage+'.log')).open('w') as log:
        subprocess.run(['bash','run.sh',*args],cwd=CODE,env={**os.environ,'GPU':'2'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def main():
    report('waiting_navtest_subset');tokens=ROOT/'navtest_subset_tokens.json';started=time.monotonic()
    while not tokens.exists():
        if time.monotonic()-started>1200:raise TimeoutError('navtest data wait exceeded')
        time.sleep(10)
    torch.manual_seed(42);net=BimodalPlanner('cross',CODE/'upstream/kmeans_navsim_traj_20.npy')
    torch.save(dict(model=net.state_dict(),name='TurboAD',seed=42,step=0),ROOT/'initial_seed42.pt');del net
    run('metric_cache',['turboad_eval.py','cache','--tokens',str(tokens)],1800)
    run('score_initial',['turboad_eval.py','score','--tokens',str(tokens),'--checkpoint',str(ROOT/'initial_seed42.pt'),'--label','initial'],1800)
    run('score_pilot',['turboad_eval.py','score','--tokens',str(tokens),'--label','pilot'],1800)
    report('pilot_evaluation_complete',tokens=len(json.loads(tokens.read_text())))
try:main()
except Exception as error:report('failed',error=repr(error));raise
