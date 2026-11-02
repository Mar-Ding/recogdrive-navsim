"""Write a durable server-side run report after the training process exits."""
import json,time,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta

out=Path('/root/autodl-tmp/mf-vla/outputs/multiview_mf_vla')
pid=int(sys.argv[1])
def read(name,default):
    try:return json.loads((out/name).read_text())
    except (OSError,json.JSONDecodeError):return default
def alive():
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except OSError:return False
while alive():
    status=read('status.json',{})
    if status.get('stage') in ('training_complete','failed'):break
    time.sleep(15)
history=read('history.json',[]);status=read('status.json',{});config=read('config.json',{})
epochs=[r for r in history if r.get('epoch',0)>0]
report=dict(recorded_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    state=status.get('stage','unknown'),process_pid=pid,
    completed_epochs=epochs[-1]['epoch'] if epochs else 0,
    training_validation_seconds=sum(r['seconds'] for r in epochs),
    initial=history[0] if history else None,final=history[-1] if history else None,
    best_validation=min(history,key=lambda r:r['ade']) if history else None,
    config=config,last_status=status,
    pdms='Not evaluated: test data, maps and metric cache are not prepared on this server.')
if report['state'] not in ('training_complete','failed'):report['state']='process_exited_before_completion'
(out/'completion_summary.json').write_text(json.dumps(report,indent=2))
text=['# TurboMFAD VLA training completion log',
      '',f"Recorded (Asia/Shanghai): {report['recorded_at']}",f"State: {report['state']}",
      f"Completed epochs: {report['completed_epochs']}/200",
      f"Training + validation: {report['training_validation_seconds']/3600:.3f} hours",
      '', 'Model: frozen DINOv3 + BERT; trainable upstream MFBlock (6 x 256), DiffusionDrive trajectory head.',
      'Inputs: front RGB 288x160 + left/right RGB 144x80, navigation instruction, ego state, noisy action.',
      'Flow: optimizer-step rotation V -> L -> A; MSE + 0.1*(1-cosine).',
      'Auxiliary weight: 0.1 for first 60% of updates, linearly decays to 0.',
      'This is the NAVSIM adaptation, not an exact reproduction of the old TurboMFVLA recipe.',
      f"Data: train={config.get('train')}, validation={config.get('val')}; log-disjoint; effective batch=128.",
      '', 'Initial / best / final validation:',json.dumps({k:report[k] for k in ('initial','best_validation','final')},indent=2),
      '',report['pdms'],'', 'Artifacts: pipeline.log, history.json, config.json, smoke.json, latest.pt, best_val_ade.pt.',
      'No claim of PDMS improvement is made from ADE alone.']
(out/'COMPLETION_LOG.md').write_text('\n'.join(text))
with (out/'RUN_LOG.md').open('a') as f:f.write('\n\n## Completion\n\n'+'\n'.join(text[2:7])+'\nSee COMPLETION_LOG.md and completion_summary.json.\n')
print(json.dumps(report),flush=True)
