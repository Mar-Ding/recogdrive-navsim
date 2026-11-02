import json, os, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
ROOT=Path(__file__).resolve().parent/'evidence'/'visual_lab_migration'
ROOT.mkdir(parents=True,exist_ok=True)
def connect(host,port,user,password):
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host,port=port,username=user,password=password,timeout=20)
    return c
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
lab=connect(cfg['ip'],int(cfg['port']),cfg['user'],cfg['passwd'])
auto=connect('connect.nmb1.seetacloud.com',12513,'root',os.environ['AUTODL_PASSWORD'])
with auto.open_sftp() as s:
    dst=ROOT/'autodl';dst.mkdir(exist_ok=True)
    for name in s.listdir('/root/autodl-tmp/mf-vla/multiview'):
        if name.endswith('.py'):s.get('/root/autodl-tmp/mf-vla/multiview/'+name,str(dst/name))
    for name in ('turbo_cross_vla','multiview_mf_vla','multiview_vl'):
        print('AUTO',name)
        for fn in ('config.json','status.json','evaluation.json','pdms_summary.json'):
            try:
                b=s.open('/root/autodl-tmp/mf-vla/outputs/'+name+'/'+fn).read();(dst/(name+'_'+fn)).write_bytes(b)
                print(fn,b.decode()[:3000])
            except FileNotFoundError:pass
with lab.open_sftp() as s:
    s.get('/data/ruiqiwang/recogdrive_baseline/demo10/manifest.json',str(ROOT/'lab_manifest.json'))
    print('LAB manifest',json.loads((ROOT/'lab_manifest.json').read_text()).__class__.__name__)
for c,cmd in [(lab,"/data/ruiqiwang/recogdrive_baseline/env/bin/python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)'; ls /data/ruiqiwang/recogdrive_baseline/models; ls /data/ruiqiwang/recogdrive_baseline/bimodal_smoke/upstream; df -h /data"),(auto,"du -sh /root/autodl-tmp/mf-vla/weights/*; head -55 /root/autodl-tmp/mf-vla/multiview/model.py; head -35 /root/autodl-tmp/mf-vla/multiview/visual_mf_vla.py")]:
    _,o,e=c.exec_command(cmd,timeout=60);print(o.read().decode());print(e.read().decode())
lab.close();auto.close()
