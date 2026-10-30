"""Upload only this experiment and the explicitly vendored reference files."""
from pathlib import Path
import paramiko
cfg=dict(line.split(':',1) for line in Path('login.md').read_text().splitlines() if ':' in line)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=30)
root='/data/ruiqiwang/recogdrive_baseline/bimodal_smoke'
try:
    with c.open_sftp() as s:
        def mkdir(path):
            try:s.stat(path)
            except FileNotFoundError:
                mkdir(path.rsplit('/',1)[0]);s.mkdir(path)
        mkdir(root)
        for p in Path('bimodal_smoke').rglob('*'):
            if not p.is_file() or p.parts[1] in {'outputs','artifacts','__pycache__'}:continue
            dest=root+'/'+p.relative_to('bimodal_smoke').as_posix();mkdir(dest.rsplit('/',1)[0]);s.put(str(p),dest)
        for p in Path('reference/DiffusionDrive').rglob('*'):
            if not p.is_file():continue
            dest=root+'/upstream/'+p.relative_to('reference/DiffusionDrive').as_posix();mkdir(dest.rsplit('/',1)[0]);s.put(str(p),dest)
    print('Uploaded to',root)
finally:c.close()
