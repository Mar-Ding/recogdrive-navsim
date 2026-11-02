"""Isolated MFAD no-auxiliary ablation, matching the existing MF+flow run."""
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
HERE=Path(__file__).resolve().parent
BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25)
with c.open_sftp() as s:
 train=s.open(BASE+'/multiview/train_turbo_cross_vla.py').read().decode()
 train=train.replace('from turbo_cross_vla import CrossPlanner as Planner','from visual_mf_vla import MFPlanner as Planner').replace('outputs/turbo_cross_vla','outputs/mfad_no_flow')
 train=train.replace("fusion='TurboVLA upstream BiAttentionBlock x6 + text enhancement; action cross decoder x2; no MF auxiliary loss'","fusion='MFBlock x6 width256 heads4 sharedQKVO modalityFFN mRoPE; no auxiliary flow',flow_enabled=False,flow_weight=0.,source_sha256={n:hashlib.sha256((BASE/'multiview'/n).read_bytes()).hexdigest() for n in ('visual_mf_vla.py','train_mfad_no_flow.py')}")
 # Fail loudly if any future edit accidentally invokes the auxiliary loss.
 train=train.replace('class CachedPlanner(Planner):','class CachedPlanner(Planner):\n    def flow_loss(self,*args,**kwargs):\n        raise AssertionError("Flow auxiliary must never execute in this ablation")\n')
 needle="torch.save(dict(model=net.state_dict(),epoch=0,config=config),OUT/'initial.pt')"
 assert needle in train
 train=train.replace(needle,needle+"\n        reference=torch.load(BASE/'outputs/multiview_mf_vla/initial.pt',map_location='cpu',weights_only=False)['model']\n        assert reference.keys()==net.state_dict().keys()\n        assert all(torch.equal(reference[k],v.cpu()) for k,v in net.state_dict().items()), 'MF initial weights differ'\n        save_json('initialization_match.json',dict(exact=True,reference='multiview_mf_vla/initial.pt',auxiliary_executed=False))\n        del reference")
 assert '.flow_loss(' not in train
 score=(HERE/'lab_visual_score.py').read_text()
 start=score.index(" if variant=='cross':");end=score.index(' torch.set_num_threads',start)
 score=score[:start]+" from train_mfad_no_flow import CachedPlanner\n out=BASE/'outputs/mfad_no_flow'\n"+score[end:]
 pipe=(HERE/'pipeline_dino_diffusiondrive.py').read_text().replace('dino_diffusiondrive','mfad_no_flow').replace('Frozen DINOv3 DiffusionDrive finished','MFAD no-flow auxiliary finished')
 files={'train_mfad_no_flow.py':train,'score_mfad_no_flow.py':score,'pipeline_mfad_no_flow.py':pipe}
 for name,data in files.items():
  compile(data,name,'exec')
  (HERE/name).write_text(data,encoding='utf-8')
  with s.open(BASE+'/multiview/'+name,'w') as f:f.write(data)
 _,o,e=c.exec_command('mkdir -p '+BASE+'/outputs/mfad_no_flow; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader',timeout=30)
 print(o.read().decode());print(e.read().decode());assert o.channel.recv_exit_status()==0
 for name in ('manifest.json','features.pt'):
  dest=BASE+'/outputs/mfad_no_flow/'+name
  try:s.lstat(dest)
  except FileNotFoundError:s.symlink(BASE+'/outputs/shared/'+name,dest)
 with s.open(BASE+'/RUN_LOG.md','a') as f:f.write('\n2026-11-02 MFAD no-flow ablation: identical MFPlanner architecture and initial tensors checked against existing MF+flow initial.pt. No flow_loss call (guard raises). Same F288x160 L/R144x80, frozen DINO/BERT, language+ego+noisy action, 8688/1930 split, seed42, 200 epochs, effective batch128, train-only20 anchors, best validation ADE, fixed200 initial/trained PDMS automatic. GPU0 only; outputs/mfad_no_flow.\n')
c.close()
