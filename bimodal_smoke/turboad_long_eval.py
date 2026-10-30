"""Evaluate each long run once after fixed training budget completes."""
import os,time,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad';CODE=BASE/'bimodal_smoke'
def evaluate(label,gpu):
    start=time.monotonic();summary=ROOT/(label+'_seed42')/'summary.json'
    while not summary.exists():
        if time.monotonic()-start>1800:raise TimeoutError(label+' training timeout')
        time.sleep(5)
    assert json.loads(summary.read_text())['epochs']==100
    with (ROOT/'logs'/('score_'+label+'.log')).open('w') as log:
        subprocess.run(['bash','run.sh','turboad_eval.py','score','--tokens',str(ROOT/'navtest_subset_tokens.json'),'--checkpoint',str(summary.parent/'best_val_ade.pt'),'--label',label],cwd=CODE,env={**os.environ,'GPU':str(gpu)},stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
    print('COMPLETE',label,flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(evaluate,label,gpu) for label,gpu in [('mf_long',0),('plus_long',2)]]
        for job in jobs:job.result()
    (ROOT/'long_evaluation_status.json').write_text(json.dumps({'stage':'complete'}))
