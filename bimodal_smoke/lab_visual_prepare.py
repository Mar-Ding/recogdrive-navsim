"""Exact lab demo10 manifest -> shared visual/language cache; no resampling."""
import os,sys,json,pickle,hashlib,time,tarfile
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline/visual_vla');LAB=BASE.parent
sys.path.insert(0,str(BASE/'multiview'))
import torch,numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModel,AutoTokenizer
ROOT=BASE/'outputs/shared';ROOT.mkdir(parents=True,exist_ok=True)
def save(name,obj):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
 torch.set_num_threads(2)
 original=LAB/'demo10/manifest.json';rows=json.loads(original.read_text());assert len(rows)==10618
 converted=[dict(token=r['token'],log=r['log_name'],split=r['split'],images=[str(LAB/'data/sensor_blobs/trainval'/r['views'][c]) for c in ('CAM_F0','CAM_L0','CAM_R0')],ego=r['status'][4:],command=int(np.argmax(r['status'][:4])),target=r['target']) for r in rows]
 assert sum(r['split']=='train' for r in converted)==8688 and sum(r['split']=='val' for r in converted)==1930
 tokens=json.loads((LAB/'turboad/navtest_subset_tokens.json').read_text());assert len(tokens)==200 and not set(tokens)&{r['token'] for r in rows}
 test=[];wanted=set(tokens)
 for p in sorted((LAB/'data/openscene-v1.1/meta_datas/test').glob('*.pkl')):
  frames=pickle.load(p.open('rb'))
  for f in frames:
   if f['token'] in wanted:
    test.append(dict(token=f['token'],log=p.stem,images=[str(LAB/'data/sensor_blobs/test'/f['cams'][c]['data_path']) for c in ('CAM_F0','CAM_L0','CAM_R0')],ego=np.asarray(f['ego_dynamic_state']).tolist(),command=int(np.argmax(f['driving_command']))))
 lookup={r['token']:r for r in test};assert len(lookup)==200;test=[lookup[t] for t in tokens]
 missing={str(Path(p).relative_to(LAB/'data/sensor_blobs/test')) for r in test for p in r['images'] if not Path(p).is_file()}
 if missing:
  with tarfile.open(LAB/'data/test_camera_0.tgz','r|gz') as tar:
   for m in tar:
    if not missing:break
    if not m.isfile():continue
    parts=Path(m.name).parts;rel=next(('/'.join(parts[i:]) for i in range(len(parts)) if '/'.join(parts[i:]) in missing),None)
    if rel:
     dest=LAB/'data/sensor_blobs/test'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(m).read());missing.remove(rel)
 assert not missing
 assert all(Path(p).is_file() for r in converted+test for p in r['images'])
 save('manifest.json',converted);save('test_manifest.json',test)
 save('protocol.json',dict(original_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),train=8688,val=1930,test=200,epochs=200,effective_batch=128,seed=42,split='exact existing lab demo10; log disjoint',cameras=['F0 288x160','L0 144x80','R0 144x80'],modalities=['vision','language instruction','action'],ego='condition',metric='original ReCogDrive PDM scorer, fixed existing 200 navtest tokens',command_templates=['turn left','go straight','turn right','unknown']))
 vision=AutoModel.from_pretrained(BASE/'weights/dinov3-vitb16',local_files_only=True).cuda().eval().requires_grad_(False)
 language=AutoModel.from_pretrained(BASE/'weights/bert-base-uncased',local_files_only=True,add_pooling_layer=False).cuda().eval().requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(BASE/'weights/bert-base-uncased',local_files_only=True)
 text=tok(['navigation instruction: '+x for x in ['turn left','go straight','turn right','unknown']],padding='max_length',max_length=12,truncation=True,return_tensors='pt').to('cuda')
 with torch.no_grad():lang=language(**text).last_hidden_state.cpu().half()
 tf=[transforms.Compose([transforms.Resize(size),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]) for size in [(160,288),(80,144),(80,144)]]
 for name,group in [('features.pt',converted),('test_features.pt',test)]:
  if (ROOT/name).exists():continue
  feats=[];started=time.time()
  for offset in range(0,len(group),32):
   chunk=group[offset:offset+32];views=[]
   for cam in range(3):
    pixels=[]
    for r in chunk:
     with Image.open(r['images'][cam]) as im:pixels.append(tf[cam](im.convert('RGB')))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f=vision(pixel_values=torch.stack(pixels).cuda()).last_hidden_state[:,5:]
    assert f.shape[1]==[180,45,45][cam];views.append(f.cpu().half())
   feats.append(torch.cat(views,1));elapsed=time.time()-started
   save('preparation_status.json',dict(stage='encoding',file=name,done=offset+len(chunk),total=len(group),seconds=elapsed,remaining_seconds=elapsed/(offset+len(chunk))*(len(group)-offset-len(chunk))))
   if offset%512==0:print(name,offset,len(group),elapsed,flush=True)
  tensors=dict(vision=torch.cat(feats),language=lang,mask=text['attention_mask'].cpu(),command=torch.tensor([r['command'] for r in group]),ego=torch.tensor([r['ego'] for r in group]),tokens=[r['token'] for r in group])
  if name=='features.pt':tensors['target']=torch.tensor([r['target'] for r in group])
  torch.save(tensors,ROOT/(name+'.partial'));(ROOT/(name+'.partial')).replace(ROOT/name)
 for variant in ('turbo_cross_vla','multiview_mf_vla'):
  out=BASE/'outputs'/variant;out.mkdir(exist_ok=True)
  for name in ('manifest.json','features.pt'):
   dest=out/name
   if not dest.exists():dest.symlink_to(ROOT/name)
 save('preparation_status.json',dict(stage='complete',samples=len(converted),test=len(test)))
if __name__=='__main__':main()
