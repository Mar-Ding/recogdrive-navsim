"""Stop only the audited native experiment tree and preserve timing evidence."""
import os,signal,subprocess,json,time
from pathlib import Path
root=Path('/data/ruiqiwang/recogdrive_baseline/demo10/native_recogdrive')
lines=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]
processes={int(a[0]):(int(a[1]),a[2]) for line in lines if len(a:=line.strip().split(None,2))==3}
roots=[p for p,(_,cmd) in processes.items() if cmd.endswith('bimodal_smoke/native_recogdrive_run.py pipeline') and processes[p][0]==1]
assert len(roots)<=1,roots
selected=set(roots)
while True:
    more={p for p,(parent,_) in processes.items() if parent in selected}
    if more<=selected:break
    selected|=more
for pid in roots:
    os.kill(pid,signal.SIGSTOP)
for pid in sorted(selected-set(roots),reverse=True):
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
for pid in roots:
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
history=json.loads((root/'history.json').read_text())
epochs=[r for r in history if r['epoch']>0]
mean=sum(r['seconds'] for r in epochs)/len(epochs)
record=dict(status='stopped_by_user',completed_epoch=epochs[-1]['epoch'],measured_epoch_seconds=[r['seconds'] for r in epochs],mean_epoch_seconds=mean,projected_200_epoch_hours=200*mean/3600,scope='Measured training plus validation per epoch, extrapolated to 200; excludes encoding and final PDM evaluation; not completed runtime.',reason='User requests replacing native DiT planner with existing DiffusionDrive action generation',stopped_pids=sorted(selected),timestamp=time.time())
(root/'stopped_timing.json').write_text(json.dumps(record,indent=2))
(root/'pipeline_status.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
