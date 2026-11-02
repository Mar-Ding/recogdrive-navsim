"""Durable dual-GPU migrated VLA training, then automatic fixed-200 PDMS."""
import os,sys,json,time,subprocess,concurrent.futures,traceback
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla');OUT=BASE/'outputs';LOG=OUT/'logs';LOG.mkdir(parents=True,exist_ok=True)
def status(stage,**kw):
 p=OUT/'pipeline_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),indent=2));t.replace(p)
def run(name,script,gpu,*args):
 assert gpu in (0,2)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
 with (LOG/(name+'.log')).open('a') as f:subprocess.run([sys.executable,'-u',str(BASE/'multiview'/script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def worker(variant,gpu):
 name='turbo_cross_vla' if variant=='cross' else 'multiview_mf_vla';out=OUT/name
 run('train_'+variant,'train_turbo_cross_vla.py' if variant=='cross' else 'train_visual_mf_vla.py',gpu)
 assert json.loads((out/'history.json').read_text())[-1]['epoch']==200
 run('score_smoke_'+variant,'lab_visual_score.py',gpu,variant,'--smoke')
 run('score_'+variant,'lab_visual_score.py',gpu,variant)
 evaluation=json.loads((out/'evaluation_trained.json').read_text());assert evaluation['successful_scenarios']==200
 with (BASE/'RUN_LOG.md').open('a') as f:f.write('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+' '+name+' complete: PDMS '+str(evaluation['pdms_100'])+' / 100\n')
 return evaluation
def main():
 status('preparing_shared_cache',gpu0='Turbo cross-attention VLA',gpu2='MF-VLA rotating flow',gpu1='untouched')
 run('prepare','lab_visual_prepare.py',0)
 status('training_then_automatic_pdms',epochs=200,effective_batch=128,train=8688,val=1930,test=200)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(worker,v,g):v for v,g in [('cross',0),('mf',2)]};results={}
  for future in concurrent.futures.as_completed(jobs):results[jobs[future]]=future.result()
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2));status('complete',results=results)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
