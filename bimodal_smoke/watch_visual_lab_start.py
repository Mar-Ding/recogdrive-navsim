"""Bounded startup check; retain compact local evidence, stop after first epochs."""
import time,json,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
ROOT=Path(__file__).resolve().parent/'evidence/visual_lab_migration';ROOT.mkdir(exist_ok=True)
BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla/outputs/'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25);c.get_transport().set_keepalive(20)
s=c.open_sftp()
for _ in range(30):
 result={}
 for key,path in [('pipeline','pipeline_status.json'),('cache','shared/preparation_status.json'),('cross','turbo_cross_vla/status.json'),('mf','multiview_mf_vla/status.json')]:
  try:result[key]=json.loads(s.open(BASE+path).read())
  except FileNotFoundError:pass
 (ROOT/'live_status.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
 if any(v.get('stage')=='failed' for v in result.values()):raise RuntimeError('Remote pipeline failed; see live_status.json')
 if all(result.get(k,{}).get('epoch',0)>=2 for k in ('cross','mf')):break
 time.sleep(40)
s.close();c.close()
