import os,tarfile,requests
from pathlib import Path
base=Path('/data/ruiqiwang/recogdrive_baseline')
url='https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_0.tgz?download=true'
r=requests.get(url,stream=True,timeout=(30,120));r.raise_for_status()
count=0
with tarfile.open(fileobj=r.raw,mode='r|gz') as t:
 for m in t:
  if m.isfile() and '/CAM_F0/' in m.name:
   parts=Path(m.name).parts
   dest=base/'data'/'sensor_blobs'/Path(*parts[parts.index('sensor_blobs')+1:])
   dest.parent.mkdir(parents=True,exist_ok=True)
   dest.write_bytes(t.extractfile(m).read());count+=1
   print(str(dest),flush=True)
   if count>=16: break
print('CAMERA_SUBSET_DONE',count,flush=True)
