"""Full and reduced auxiliary cameras, identical demo10 tokens and IL budget."""
import argparse,concurrent.futures,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline');ROOT=BASE/'demo10';CODE=BASE/'bimodal_smoke'
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import numpy as np,torch
from PIL import Image
from transformers import AutoModel

def encode(small):
    torch.set_num_threads(2);label='plus_small' if small else 'plus';folder=ROOT/(label+'_features');folder.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text());source=json.loads((ROOT/'features_complete.json').read_text());sha=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest();assert source['manifest_sha256']==sha
    encoder=AutoModel.from_pretrained('/home/yiding/TurboVLA-repro/assets/dinov3-vitb',local_files_only=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    files=[];offset=0
    for path in source['files']:
        rows=torch.load(path,map_location='cpu',weights_only=False);metadata=manifest[offset:offset+len(rows)];offset+=len(rows);dest=folder/Path(path).name;files.append(str(dest))
        assert [r['token'] for r in rows]==[r['token'] for r in metadata]
        if not dest.exists():
            for i in range(0,len(rows),16):
                chunk=metadata[i:i+16];pixels=[]
                for r in chunk:
                    for view in ('CAM_L0','CAM_R0'):
                        p=BASE/'data/sensor_blobs/trainval'/r['views'][view];size=(144,80) if small else (288,160)
                        pixels.append(torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize(size),dtype=np.float32)/255).permute(2,0,1))
                x=torch.stack(pixels).cuda();x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):v=encoder(pixel_values=x).last_hidden_state[:,5:].reshape(len(chunk),-1,768).cpu()
                for row,side in zip(rows[i:i+16],v):row['vision']=torch.cat([row['vision'],side],dim=0)
            temp=dest.with_suffix('.partial');torch.save(rows,temp);temp.replace(dest)
        # Verify every cached chunk, including resumes: front/lidar/status/targets/splits unchanged.
        original=torch.load(path,map_location='cpu',weights_only=False);result=torch.load(dest,map_location='cpu',weights_only=False)
        assert len(original)==len(result)
        for old,new in zip(original,result):
            assert old['token']==new['token'] and old['split']==new['split'] and old['log_name']==new['log_name']
            assert new['vision'].shape==(270 if small else 540,768) and torch.equal(old['vision'],new['vision'][:180]) and torch.isfinite(new['vision']).all()
            assert all(torch.equal(old[k],new[k]) for k in ('lidar','status','target'))
        print('ENCODE_VERIFIED',label,offset,len(manifest),flush=True)
    assert offset==len(manifest)
    info=dict(files=files,samples=offset,manifest_sha256=sha,views=['CAM_F0','CAM_L0','CAM_R0'],sizes=[[288,160],[144,80] if small else [288,160],[144,80] if small else [288,160]],front_and_other_inputs_verified_equal=True)
    (ROOT/(label+'_features_complete.json')).write_text(json.dumps(info,indent=2))

def run(label,args,gpu,timeout):
    with (ROOT/'logs'/(label+'.log')).open('a') as log:subprocess.run(['bash','run.sh',*args],cwd=CODE,env={**os.environ,'GPU':str(gpu)},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def worker(small,gpu):
    label='plus_small' if small else 'plus';variant='mf_plus_small' if small else 'mf_plus';out=ROOT/('turboad_'+label);state=ROOT/(label+'_status.json')
    def status(stage,**extra):state.write_text(json.dumps(dict(stage=stage,time=time.time(),gpu=gpu,**extra),indent=2))
    try:
        status('encoding')
        if not (ROOT/(label+'_features_complete.json')).exists():run(label+'_encode',['demo10_plus.py','encode','--small'] if small else ['demo10_plus.py','encode'],gpu,3600)
        run(label+'_check',['demo10_plus_check.py','--small'] if small else ['demo10_plus_check.py'],gpu,600)
        status('benchmark')
        run(label+'_benchmark',['demo10_benchmark.py','--variant',variant],gpu,600)
        bench=json.loads((ROOT/f'benchmark_{variant}.json').read_text());batch=max(r['batch'] for r in bench['results'] if not r.get('oom') and r['reserved_gib']<21)
        status('training',micro_batch=batch,effective_batch=128,epochs=200)
        if not (out/'summary.json').exists():run(label+'_train',['demo10_train.py','--variant',variant,'--micro-batch',str(batch),'--epochs','200'],gpu,8*3600)
        assert json.loads((out/'summary.json').read_text())['steps']==13600
        status('evaluation');metrics={};tokens=set(json.loads((BASE/'turboad/navtest_subset_tokens.json').read_text()))
        for suffix,checkpoint in [('_initial','initial.pt'),('','best_val_ade.pt')]:
            name='demo10_'+label+suffix;folder=BASE/'turboad/evaluation'/('scores_'+name)
            if not list(folder.glob('*.csv')):run(name+'_score',['turboad_eval.py','score','--tokens',str(BASE/'turboad/navtest_subset_tokens.json'),'--checkpoint',str(out/checkpoint),'--label',name],gpu,3600)
            files=list(folder.glob('*.csv'));assert len(files)==1;rows=list(csv.DictReader(files[0].open()));scenes=[r for r in rows if r['token']!='average']
            assert len(scenes)==200 and {r['token'] for r in scenes}==tokens and all(r['valid']=='True' for r in scenes)
            shutil.copy2(files[0],ROOT/(name+'.csv'));metrics[name]=next(r for r in rows if r['token']=='average')
        (out/'evaluation.json').write_text(json.dumps(metrics,indent=2));status('complete',metrics=metrics);return metrics
    except Exception as error:status('failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['encode','pipeline']);p.add_argument('--small',action='store_true');a=p.parse_args()
    if a.mode=='encode':encode(a.small)
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(worker,small,gpu) for small,gpu in [(False,0),(True,2)]]
            results=[job.result() for job in jobs]
        (ROOT/'plus_comparison.json').write_text(json.dumps(results,indent=2))
