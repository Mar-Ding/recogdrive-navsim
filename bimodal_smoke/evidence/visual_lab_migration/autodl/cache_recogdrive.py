import os,sys,json,pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from transformers import AutoModel,AutoTokenizer
from torchvision import transforms
BASE=Path('/root/autodl-tmp/mf-vla');OUT=BASE/'outputs/recogdrive_diffusiondrive';SOURCE=BASE/'outputs/multiview_mf_vla';ROOT=BASE/'weights/recogdrive-vlm-2b'
torch.set_num_threads(4)
rows=json.loads((SOURCE/'manifest.json').read_text());cache=OUT/'features';cache.mkdir(exist_ok=True);states=[];logs={}
for r in rows:
 if r['log'] not in logs:
  frames=pickle.load(open(BASE/'data/navsim10/openscene-v1.1/meta_datas/trainval'/f"{r['log']}.pkl",'rb'));logs[r['log']]=(frames,{f['token']:i for i,f in enumerate(frames)})
 frames,lookup=logs[r['log']];i=lookup[r['token']];f=frames[i];yaw=Quaternion(f['ego2global_rotation']).yaw_pitch_roll[0];c,s=np.cos(yaw),np.sin(yaw);pos=np.array(f['ego2global_translation'])[:2];history=[]
 for h in frames[i-3:i+1]:
  d=np.array(h['ego2global_translation'])[:2]-pos;a=Quaternion(h['ego2global_rotation']).yaw_pitch_roll[0]-yaw;history.append([c*d[0]+s*d[1],-s*d[0]+c*d[1],np.arctan2(np.sin(a),np.cos(a))])
 states.append(np.r_[f['driving_command'],f['ego_dynamic_state'],np.array(history).flatten()])
states=torch.tensor(np.array(states),dtype=torch.float32);torch.save(states,OUT/'states.pt');del logs
from navsim.agents.recogdrive.recogdrive_backbone import system_message
model=AutoModel.from_pretrained(str(ROOT),torch_dtype=torch.bfloat16,trust_remote_code=True,use_flash_attn=False,low_cpu_mem_usage=True).cuda().eval()
for p in model.parameters():p.requires_grad=False
tokenizer=AutoTokenizer.from_pretrained(str(ROOT),trust_remote_code=True,use_fast=False);model.img_context_token_id=tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
sys.path.insert(0,str(ROOT));from conversation import get_conv_template
tf=transforms.Compose([transforms.Resize((448,896),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
for i,r in enumerate(rows):
 if i>=int(os.environ.get('RECOG_LIMIT',str(len(rows)))):break
 target=cache/(r['token']+'.pt')
 if target.exists():continue
 with Image.open(r['images'][0]) as im:
  full=tf(im.convert('RGB'));thumb=transforms.functional.resize(full,[448,448]);pixels=torch.stack([full[:,:,:448],full[:,:,448:],thumb]).cuda().bfloat16()
 h=states[i,8:].reshape(4,3);history=' '.join(f'   - t-{3-j}: ({h[j,0]:.2f}, {h[j,1]:.2f}, {h[j,2]:.2f})' for j in range(4));command=['turn left','go straight','turn right','unknown'][r['command']]
 prompt='<image>\nAs an autonomous driving system, predict the vehicle trajectory based on front camera view, historical motion context: '+history+' and navigation command: '+command+'. Predict 8 future (x,y,heading) trajectory points for the next 4 seconds.'
 conv=get_conv_template('internvl2_5');conv.system_message=system_message;conv.append_message(conv.roles[0],prompt);conv.append_message(conv.roles[1],None);text=conv.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*768+'</img>',1);inputs=tokenizer(text,return_tensors='pt').to('cuda')
 with torch.inference_mode():
  output=model(pixel_values=pixels,**inputs,image_flags=torch.ones(3,device='cuda',dtype=torch.long),output_hidden_states=True,return_dict=True);feature=output.hidden_states[-1][0].cpu().half()
 torch.save(feature,target);del output
 (OUT/'status.json').write_text(json.dumps(dict(stage='encoding',done=i+1,total=len(rows))))
 print('ENCODING',i+1,len(rows),feature.shape,flush=True)
print('ENCODING_COMPLETE',flush=True)
