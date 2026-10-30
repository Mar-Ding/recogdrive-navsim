"""Run bounded data preparation and one TurboAD pilot, with persistent status."""
import json,time,subprocess,os,shutil
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'turboad';CODE=BASE/'bimodal_smoke'
status=ROOT/'expanded_status.json'
def report(stage,**extra):
    row=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
    status.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def main():
    expected={'trainval_camera_0.tgz':4704972041,'trainval_lidar_0.tgz':3456836376}
    report('waiting_sensor_archives');started=time.monotonic()
    while not all((BASE/'data'/p).exists() and (BASE/'data'/p).stat().st_size==size for p,size in expected.items()):
        if time.monotonic()-started>1200:raise TimeoutError('Archive download wait exceeded 20 minutes')
        time.sleep(10)
    assert shutil.disk_usage(BASE).free>30*2**30,'Less than 30 GiB free'
    for stage,cmd,timeout in [('pair_sensors',['turboad_data.py','sensors'],1800),('encode',['turboad_data.py','encode'],900),('train',['turboad_train.py','--epochs','20','--batch','32','--run','expanded_seed42'],3600)]:
        report(stage)
        with (ROOT/'logs'/('expanded_'+stage+'.log')).open('w') as log:
            subprocess.run(['bash','run.sh',*cmd],cwd=CODE,env={**os.environ,'GPU':'0'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    report('training_complete',summary=str(ROOT/'expanded_seed42/summary.json'))
try:main()
except Exception as error:report('failed',error=repr(error));raise

