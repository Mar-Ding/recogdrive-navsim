import os,sys,json,time,subprocess,traceback
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla')
OUT=BASE/'outputs/dino_diffusiondrive'
def status(stage,**kw):
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),indent=2))
def run(script,*args):
    with (OUT/(script+'.log')).open('a') as f:
        subprocess.run([sys.executable,'-u',str(BASE/'multiview'/script),*args],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    status('training')
    run('train_dino_diffusiondrive.py')
    status('evaluating')
    run('score_dino_diffusiondrive.py','cross','--smoke')
    run('score_dino_diffusiondrive.py','cross')
    result=json.loads((OUT/'evaluation_trained.json').read_text());assert result['successful_scenarios']==200
    status('complete',result=result)
    with (BASE/'RUN_LOG.md').open('a') as f:f.write('\nFrozen DINOv3 DiffusionDrive finished: '+json.dumps(result)+'\n')
except Exception as e:
    status('failed',error=repr(e),traceback=traceback.format_exc());raise
