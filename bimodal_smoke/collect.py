"""Fetch lightweight evidence only; large checkpoints stay on the server."""
from pathlib import Path
import paramiko
cfg=dict(line.split(':',1) for line in Path('login.md').read_text().splitlines() if ':' in line)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=30)
try:
    with c.open_sftp() as s:
        for variant in ('cross','mf'):
            out=Path('bimodal_smoke/evidence')/variant;out.mkdir(parents=True,exist_ok=True)
            for name in ('report.json','training.json','predictions.json','trajectories.png','raw_demo.json'):
                s.get('/data/ruiqiwang/recogdrive_baseline/bimodal_smoke/outputs/'+variant+'/'+name,str(out/name))
        for name in ('audit.json','environment.json'):
            s.get('/data/ruiqiwang/recogdrive_baseline/bimodal_smoke/artifacts/'+name,str(Path('bimodal_smoke/evidence')/name))
finally:c.close()
