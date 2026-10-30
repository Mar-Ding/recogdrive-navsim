"""Durable download -> encode -> paired single GPU training -> official evaluation."""
import concurrent.futures,csv,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'demo10';CODE=BASE/'bimodal_smoke';LOGS=ROOT/'logs';LOGS.mkdir(exist_ok=True)

def status(stage,**kwargs):
    value=dict(stage=stage,time=time.time(),**kwargs);temp=ROOT/'pipeline_status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(ROOT/'pipeline_status.json');print(json.dumps(value),flush=True)
def run(name,args,gpu,timeout):
    with (LOGS/(name+'.log')).open('a') as log:
        subprocess.run(['bash','run.sh',*args],cwd=CODE,env={**os.environ,'GPU':str(gpu)},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def worker(variant,gpu):
    out=ROOT/('turboad' if variant=='cross' else 'turbo_mfad')
    bench=json.loads((ROOT/f'benchmark_{variant}.json').read_text());safe=[r['batch'] for r in bench['results'] if not r.get('oom') and r['reserved_gib']<20]
    batch=max(safe);assert batch in (64,128)
    if not (out/'summary.json').exists():run('train_'+variant,['demo10_train.py','--variant',variant,'--micro-batch',str(batch),'--epochs','200'],gpu,6*3600)
    assert json.loads((out/'summary.json').read_text())['epochs']==200
    tokenpath=BASE/'turboad/navtest_subset_tokens.json';tokens=set(json.loads(tokenpath.read_text()))
    metrics={}
    for suffix,ckpt in [('_initial','initial.pt'),('','best_val_ade.pt')]:
        label='demo10_'+variant+suffix;dest=BASE/'turboad/evaluation'/('scores_'+label)
        if not list(dest.glob('*.csv')):run('score_'+label,['turboad_eval.py','score','--tokens',str(tokenpath),'--checkpoint',str(out/ckpt),'--label',label],gpu,2400)
        files=list(dest.glob('*.csv'));assert len(files)==1
        rows=list(csv.DictReader(files[0].open()));scenes=[r for r in rows if r['token']!='average']
        assert len(scenes)==len(tokens)==200 and {r['token'] for r in scenes}==tokens and all(r['valid']=='True' for r in scenes)
        shutil.copy2(files[0],ROOT/(label+'.csv'));metrics[label]=next(r for r in rows if r['token']=='average')
    (out/'evaluation.json').write_text(json.dumps(metrics,indent=2));return metrics

def main():
    status('downloading',expected_shards=20,models=['TurboAD','Turbo-MFAD'],epochs=200,effective_batch=128)
    deadline=time.monotonic()+6*3600
    # Adopt the already launched download; retry resumably if its process exits.
    while not (ROOT/'sensors_complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Download exceeded six hours; partial downloads retained')
        active=subprocess.run(['pgrep','-f','^/home/yiding/TurboVLA-repro/env/bin/python -u navsim_demo10_data.py sensors$'],capture_output=True).returncode==0
        if active:time.sleep(15);continue
        for attempt in range(4):
            try:run('download',['navsim_demo10_data.py','sensors'],0,6*3600);break
            except subprocess.CalledProcessError:
                if attempt==3:raise
                time.sleep(15)
    status('encoding')
    if not (ROOT/'features_complete.json').exists():run('encode',['navsim_demo10_data.py','encode'],0,3600)
    status('training_and_evaluation',gpu0='TurboAD',gpu2='Turbo-MFAD',epochs=200,effective_batch=128)
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(worker,variant,gpu):variant for variant,gpu in [('cross',0),('mf',2)]}
        for future in concurrent.futures.as_completed(jobs):results[jobs[future]]=future.result()
    (ROOT/'comparison.json').write_text(json.dumps(results,indent=2));status('complete',comparison=results)

if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=str(error),traceback=traceback.format_exc());raise
