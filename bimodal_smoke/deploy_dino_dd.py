"""Deploy an isolated frozen-DINO DiffusionDrive run on authorized GPU0."""
import json,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
HERE=Path(__file__).resolve().parent
BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25)
with c.open_sftp() as s:
 train=s.open(BASE+'/multiview/train_turbo_cross_vla.py').read().decode()
 train=train.replace('from turbo_cross_vla import CrossPlanner as Planner','from dino_diffusiondrive import DinoDiffusionDrive as Planner').replace("outputs/turbo_cross_vla","outputs/dino_diffusiondrive")
 start=train.index('class CachedPlanner(Planner):');end=train.index('\ndef main():',start)
 train=train[:start]+'class CachedPlanner(Planner):\n    pass\n\n'+train[end:]
 train=train.replace("tensors['ego'][ids].cuda()","torch.cat((torch.nn.functional.one_hot(cmd,num_classes=4).float(),tensors['ego'][ids]),-1).cuda()")
 train=train.replace("hashlib.sha256((BASE/'multiview/model.py').read_bytes())","hashlib.sha256((BASE/'multiview/dino_diffusiondrive.py').read_bytes())")
 train=train.replace("fusion='TurboVLA upstream BiAttentionBlock x6 + text enhancement; action cross decoder x2; no MF auxiliary loss'","fusion='Frozen DINOv3 cached RGB; image-space conditioning grid; scene queries x2; command+ego numeric; DiffusionDrive head; no language/MF/flow'")
 score=(HERE/'lab_visual_score.py').read_text()
 start=score.index(" if variant=='cross':");end=score.index(' torch.set_num_threads',start)
 score=score[:start]+" from train_dino_diffusiondrive import CachedPlanner\n out=BASE/'outputs/dino_diffusiondrive'\n"+score[end:]
 score=score.replace("tensors['ego'][i:i+1].cuda()","torch.cat((torch.nn.functional.one_hot(cmd,num_classes=4).float(),tensors['ego'][i:i+1]),-1).cuda()")
 files={'dino_diffusiondrive.py':(HERE/'dino_diffusiondrive.py').read_text(),'train_dino_diffusiondrive.py':train,'score_dino_diffusiondrive.py':score,'pipeline_dino_diffusiondrive.py':(HERE/'pipeline_dino_diffusiondrive.py').read_text()}
 for name,data in files.items():
  (HERE/name).write_text(data,encoding='utf-8')
  with s.open(BASE+'/multiview/'+name,'w') as f:f.write(data)
 _,o,e=c.exec_command('mkdir -p '+BASE+'/outputs/dino_diffusiondrive; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader',timeout=30)
 print(o.read().decode());print(e.read().decode());assert o.channel.recv_exit_status()==0
 for name in ('manifest.json','features.pt'):
  dest=BASE+'/outputs/dino_diffusiondrive/'+name
  try:s.lstat(dest)
  except FileNotFoundError:s.symlink(BASE+'/outputs/shared/'+name,dest)
 with s.open(BASE+'/RUN_LOG.md','a') as f:f.write('\n2026-11-02: Frozen DINOv3 DiffusionDrive RGB adaptation baseline: same 8688/1930, fixed 200 test, 200 epochs, effective batch128, seed42. Three cameras F288x160 / L,R144x80, command+ego numeric. No BERT/MF/flow/LiDAR. Image-space grid is not calibrated BEV. Trainable projection, scene decoder, upstream DD trajectory head. GPU0 only; automatic initial/trained PDMS. Outputs/dino_diffusiondrive.\n')
c.close()
