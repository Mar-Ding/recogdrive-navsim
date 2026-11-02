import json,sys,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
HERE=Path(__file__).resolve().parent;BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25)
if '--smoke' in sys.argv:
 with c.open_sftp() as s:s.put(str(HERE/'lab_visual_smoke.py'),BASE+'/multiview/lab_visual_smoke.py')
 _,o,e=c.exec_command('cd '+BASE+'; CUDA_VISIBLE_DEVICES=2 bash run.sh multiview/lab_visual_smoke.py',timeout=120)
 print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));print('SMOKE_EXIT',o.channel.recv_exit_status())
if '--pdm-smoke' in sys.argv:
 with c.open_sftp() as s:s.put(str(HERE/'lab_visual_pdm_smoke.py'),BASE+'/multiview/lab_visual_pdm_smoke.py')
 _,o,e=c.exec_command('cd '+BASE+'; CUDA_VISIBLE_DEVICES=2 bash run.sh multiview/lab_visual_pdm_smoke.py',timeout=120)
 print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));print('PDM_SMOKE_EXIT',o.channel.recv_exit_status())
with c.open_sftp() as s:
 for name in ('pipeline_status.json','shared/preparation_status.json','turbo_cross_vla/status.json','multiview_mf_vla/status.json','migration_smoke.json'):
  try:print(name,s.open(BASE+'/outputs/'+name).read().decode()[:3500])
  except FileNotFoundError:pass
 for variant in ('turbo_cross_vla','multiview_mf_vla'):
  try:
   history=json.loads(s.open(BASE+'/outputs/'+variant+'/history.json').read());print(variant,'HISTORY_LAST',json.dumps(history[-3:]))
  except FileNotFoundError:pass
_,o,e=c.exec_command('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; tail -5 '+BASE+'/outputs/logs/train_cross.log; tail -5 '+BASE+'/outputs/logs/train_mf.log',timeout=30)
print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));c.close()
