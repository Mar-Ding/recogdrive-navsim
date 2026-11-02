import json,warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
R=Path(__file__).resolve().parents[1];cfg=dict(x.split(':',1) for x in (R/'login.md').read_text().splitlines() if ':' in x)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=30)
data=json.loads((R/'output/charts/scene_demo_predictions.json').read_text());dest=R/'output/charts/demo_raw';dest.mkdir(exist_ok=True)
with c.open_sftp() as s:
 for i,row in enumerate(data['scenes']):
  for j,path in enumerate(row['images']):s.get(path,str(dest/f'scene_{i}_{j}.jpg'))
c.close();print('DOWNLOADED 9 camera images')
