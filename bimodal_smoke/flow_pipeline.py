"""Isolated flow run, then unchanged NAVSIM scoring with inference-only weights."""
import csv,json,os,subprocess,time,traceback,sys
from pathlib import Path
import torch
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'demo10';OUT=ROOT/'turbo_mfad_flow';CODE=BASE/'bimodal_smoke'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
LOGS=ROOT/'logs'

def status(stage,**kw):
    p=ROOT/'flow_pipeline_status.tmp';p.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));p.replace(ROOT/'flow_pipeline_status.json')

def run(name,args,timeout):
    with (LOGS/(name+'.log')).open('a') as f:
        subprocess.run(['bash','run.sh',*args],cwd=CODE,env={**os.environ,'GPU':'2'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def main():
    status('training',gpu=2,epochs=200,effective_batch=128)
    deadline=time.monotonic()+6*3600
    while not (OUT/'summary.json').exists():
        active=subprocess.run(['pgrep','-f','^/home/yiding/TurboVLA-repro/env/bin/python -u demo10_flow_train.py --variant mf --epochs 200 --micro-batch 128$'],capture_output=True).returncode==0
        if not active:
            run('train_mf_flow',['demo10_flow_train.py','--variant','mf','--epochs','200','--micro-batch','128'],6*3600)
            break
        if time.monotonic()>deadline:raise TimeoutError('Flow training timed out')
        time.sleep(10)
    status('evaluating',gpu=2)
    from model import BimodalPlanner
    policy=BimodalPlanner('mf',CODE/'upstream/kmeans_navsim_traj_20.npy')
    state=torch.load(OUT/'best_val_ade.pt',map_location='cpu',weights_only=False)
    keys=set(policy.state_dict());inference={k:v for k,v in state['model'].items() if k in keys}
    assert all(k.startswith('flow_') for k in set(state['model'])-keys)
    policy.load_state_dict(inference,strict=True)
    checkpoint=OUT/'best_val_ade_inference.pt'
    torch.save(dict(model=inference,variant='mf',epoch=state['epoch'],config=state['config'],manifest_sha256=state['manifest_sha256']),checkpoint)
    tokens=BASE/'turboad/navtest_subset_tokens.json'
    run('score_demo10_mf_flow',['flow_eval.py','score','--tokens',str(tokens),'--checkpoint',str(checkpoint),'--label','demo10_mf_flow'],3600)
    files=list((BASE/'turboad/evaluation/scores_demo10_mf_flow').glob('*.csv'));assert len(files)==1
    rows=list(csv.DictReader(files[0].open()));scenes=[r for r in rows if r['token']!='average']
    assert len(scenes)==200 and {r['token'] for r in scenes}==set(json.loads(tokens.read_text())) and all(r['valid']=='True' for r in scenes)
    (ROOT/'score_mf_flow.csv').write_bytes(files[0].read_bytes())
    result=next(r for r in rows if r['token']=='average');(OUT/'evaluation.json').write_text(json.dumps(result,indent=2));status('complete',result=result)

if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
