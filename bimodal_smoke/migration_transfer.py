import os,json,warnings,stat
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
def connect(host,port,user,password):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(host,port=port,username=user,password=password,timeout=25);return c
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
lab=connect(cfg['ip'],int(cfg['port']),cfg['user'],cfg['passwd']);auto=connect('connect.nmb1.seetacloud.com',12513,'root',os.environ['AUTODL_PASSWORD'])
BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla';SRC='/root/autodl-tmp/mf-vla'
def run(c,cmd):
 _,o,e=c.exec_command(cmd,timeout=120);print(o.read().decode(errors='replace'));print(e.read().decode(errors='replace'));assert o.channel.recv_exit_status()==0
run(lab,'mkdir -p '+BASE+'/multiview '+BASE+'/weights '+BASE+'/src '+BASE+'/outputs; /home/yiding/TurboVLA-repro/env/bin/python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"; ls /home/yiding/TurboVLA-repro/assets; ls /data/ruiqiwang/recogdrive_baseline/data/sensor_blobs/test | head -2')
a=auto.open_sftp();b=lab.open_sftp()
def mkdir(p):
 try:b.mkdir(p)
 except IOError:pass
def copytree(src,dst):
 mkdir(dst)
 for item in a.listdir_attr(src):
  if item.filename in ('__pycache__','.git') or item.filename.endswith('.pyc'):continue
  sp=src+'/'+item.filename;dp=dst+'/'+item.filename
  if stat.S_ISDIR(item.st_mode):copytree(sp,dp)
  else:
   try:
    if b.stat(dp).st_size==item.st_size:continue
   except IOError:pass
   print('TRANSFER',sp,item.st_size,flush=True)
   with a.open(sp,'rb') as x,b.open(dp+'.partial','wb') as y:
    if item.st_size>1024*1024:x.prefetch(file_size=item.st_size)
    y.set_pipelined(True)
    while True:
     z=x.read(1024*1024)
     if not z:break
     y.write(z)
   try:b.remove(dp)
   except IOError:pass
   b.rename(dp+'.partial',dp)
def digest(c,path):
 _,o,e=c.exec_command('sha256sum '+path,timeout=60);v=o.read().decode();assert o.channel.recv_exit_status()==0;return v.split()[0]
for name,asset in [('bert-base-uncased','bert-base-uncased'),('dinov3-vitb16','dinov3-vitb')]:
 if digest(auto,SRC+'/weights/'+name+'/model.safetensors')!=digest(lab,'/home/yiding/TurboVLA-repro/assets/'+asset+'/model.safetensors'):
  copytree(SRC+'/weights/'+name,BASE+'/weights/'+name)
  assert digest(auto,SRC+'/weights/'+name+'/model.safetensors')==digest(lab,BASE+'/weights/'+name+'/model.safetensors')
  continue
 mkdir(BASE+'/weights/'+name)
 for item in b.listdir_attr('/home/yiding/TurboVLA-repro/assets/'+asset):
  if not stat.S_ISREG(item.st_mode):continue
  dp=BASE+'/weights/'+name+'/'+item.filename
  try:b.remove(dp)
  except IOError:pass
  b.symlink('/home/yiding/TurboVLA-repro/assets/'+asset+'/'+item.filename,dp)
 print('REUSE_VERIFIED_WEIGHTS',name,flush=True)
for source in ('Multimodal-Flow/src','DiffusionDrive/navsim/agents/diffusiondrive','TurboVLA/turbovla/models/components'):
 parts=source.split('/');p=BASE+'/src'
 for part in parts[:-1]:p+='/'+part;mkdir(p)
 copytree(SRC+'/src/'+source,BASE+'/src/'+source)
for name in ('model.py','visual_mf_vla.py','turbo_cross_vla.py','train_turbo_cross_vla.py','train_visual_mf_vla.py'):
 data=a.open(SRC+'/multiview/'+name).read().decode().replace(SRC,BASE)
 with b.open(BASE+'/multiview/'+name,'w') as f:f.write(data)
print('TRANSFER_COMPLETE',flush=True)
a.close();b.close();auto.close();lab.close()
