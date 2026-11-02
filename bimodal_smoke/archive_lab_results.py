"""Download completed lab artifacts without data/encoder caches or credentials."""
import hashlib,json,stat,time,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/lab_archive_20261102'
LAB='/data/ruiqiwang/recogdrive_baseline'; BASE=LAB+'/visual_vla'
cfg=dict(x.split(':',1) for x in (ROOT/'login.md').read_text().splitlines() if ':' in x)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=30)
records=[]
with c.open_sftp() as s:
 def get(remote,relative):
  p=DEST/relative;p.parent.mkdir(parents=True,exist_ok=True);size=s.stat(remote).st_size
  if not p.exists() or p.stat().st_size!=size:
   tmp=p.with_name(p.name+'.partial');s.get(remote,str(tmp));tmp.replace(p)
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  records.append(dict(path=str(p.relative_to(DEST)).replace('\\','/'),bytes=size,sha256=digest))
  print(str(relative),size,flush=True)
 def tree(remote,relative):
  for a in s.listdir_attr(remote):
   if a.filename in ('__pycache__','.git'):continue
   r=remote+'/'+a.filename;local=Path(relative)/a.filename
   if stat.S_ISDIR(a.st_mode):tree(r,local)
   elif stat.S_ISREG(a.st_mode):get(r,local)
 for variant in ('turbo_cross_vla','multiview_mf_vla','mfad_no_flow','dino_diffusiondrive'):
  remote=BASE+'/outputs/'+variant
  for a in s.listdir_attr(remote):
   if stat.S_ISREG(a.st_mode) and (a.filename.endswith(('.json','.csv','.log','.npy')) or a.filename in ('best_val_ade.pt','initial.pt')):
    get(remote+'/'+a.filename,Path('outputs')/variant/a.filename)
 for n in ('manifest.json','test_manifest.json','protocol.json'):get(BASE+'/outputs/shared/'+n,Path('outputs/shared')/n)
 tree(BASE+'/multiview','multiview')
 for n in ('RUN_LOG.md','run.sh','migration_sources.json'):get(BASE+'/'+n,n)
 for repo in ('DiffusionDrive','Multimodal-Flow','TurboVLA'):
  tree(BASE+'/src/'+repo,Path('src')/repo)
 tree(BASE+'/outputs/logs','outputs/logs')
 get(LAB+'/demo10/recogdrive_dd/best_val_ade.pt','outputs/recogdrive_td_front/best_val_ade.pt')
 for a in s.listdir_attr(LAB+'/demo10/recogdrive_dd'):
  if stat.S_ISREG(a.st_mode) and a.filename.endswith(('.json','.csv','.npy')):get(LAB+'/demo10/recogdrive_dd/'+a.filename,Path('outputs/recogdrive_td_front')/a.filename)
 _,o,e=c.exec_command('/home/yiding/TurboVLA-repro/env/bin/python -m pip freeze',timeout=60)
 (DEST/'requirements.lab-freeze.txt').write_text(o.read().decode(),encoding='utf-8')
c.close()
(DEST/'SHA256SUMS.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('COMPLETE',len(records),'files',sum(x['bytes'] for x in records),flush=True)
