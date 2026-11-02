import argparse, sys, pickle
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from model import Planner
sys.path.insert(0,'/root/autodl-tmp/mf-vla/smoke')
from smoke_dinov3 import find_sample

BERT='/root/autodl-tmp/mf-vla/weights/bert-base-uncased'
VISION='/root/autodl-tmp/mf-vla/weights/dinov3-vitb16'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('vl','vla'),required=True); args=p.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    image_path,trajectory,state,token=find_sample(Path('/root/autodl-tmp/mf-vla/data/navmini'))
    t=np.arange(1,9,dtype=np.float32)*.5
    anchors=np.stack([np.stack((t*5,t*t*k),-1) for k in np.linspace(-.2,.2,20,dtype=np.float32)])
    anchor_path=Path('/root/autodl-tmp/mf-vla/multiview/synthetic_anchors.npy'); np.save(anchor_path,anchors)
    tf=transforms.Compose([transforms.Resize((160,288)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    log_name=image_path.parent.parent.name
    root=Path('/root/autodl-tmp/mf-vla/data/navmini/openscene-v1.1')
    frames=pickle.load(open(root/'meta_datas'/'mini'/(log_name+'.pkl'),'rb'))
    frame=next(f for f in frames if f['token']==token)
    auxiliary=transforms.Compose([transforms.Resize((80,144)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    side_paths=[root/'sensor_blobs'/'mini'/frame['cams'][cam]['data_path'] for cam in ('CAM_L0','CAM_R0')]
    assert all(p.is_file() for p in side_paths),side_paths
    image=[tf(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()]+[auxiliary(Image.open(p).convert('RGB')).unsqueeze(0).cuda() for p in side_paths]
    command=['turn left','go straight','turn right','unknown'][int(state[:4].argmax())]
    tokenizer=AutoTokenizer.from_pretrained(BERT,local_files_only=True)
    text=tokenizer('navigation instruction: '+command,return_tensors='pt',padding='max_length',truncation=True,max_length=12).to('cuda')
    alternative=tokenizer('navigation instruction: turn left',return_tensors='pt',padding='max_length',truncation=True,max_length=12).to('cuda')
    ego=torch.from_numpy(state[4:]).unsqueeze(0).cuda()
    target={'trajectory':torch.from_numpy(trajectory).unsqueeze(0).cuda()}
    model=Planner(anchor_path,args.mode,BERT,VISION).cuda()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    model.train(); optimizer.zero_grad(set_to_none=True)
    out=model(image,text['input_ids'],text['attention_mask'],ego,target)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    weights=[('vision projection',model.vision_proj.weight),('language projection',model.language_proj.weight)]
    if args.mode=='vla': weights.append(('action encoder',model.action_codec[0].weight))
    for name,p in weights:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(p.grad is None for p in model.language.parameters())
    optimizer.step(); model.eval()
    with torch.no_grad():
        c,pad=model.context(image,text['input_ids'],text['attention_mask'],ego)
        alt,altpad=model.context(image,alternative['input_ids'],alternative['attention_mask'],ego)
        noisy=torch.from_numpy(anchors).unsqueeze(0).cuda(); step=torch.tensor([10],device='cuda')
        f,a=model.fuse(c,pad,noisy,step)
        altf,alta=model.fuse(alt,altpad,noisy,step)
        assert not torch.allclose(f[:,:180],altf[:,:180]),'language failed to affect fusion'
        if args.mode=='vla':
            f2,a2=model.fuse(c,pad,noisy+2,step)
            assert torch.allclose(f,f2,atol=1e-6),'action leaked into condition'
            assert not torch.allclose(a,a2),'action chunk ignored'
            assert not torch.allclose(a,alta),'action chunk ignores language'
        # Test each camera affects the fused representation, including side views.
        for index in (1,2):
            changed=list(image); changed[index]=torch.zeros_like(image[index])
            cx,px=model.context(changed,text['input_ids'],text['attention_mask'],ego)
            fx,_=model.fuse(cx,px,noisy,step)
            assert not torch.allclose(f[:,:180],fx[:,:180]),f'side view {index} ignored'
        prediction=model(image,text['input_ids'],text['attention_mask'],ego)['trajectory']
        assert prediction.shape==(1,8,3) and torch.isfinite(prediction).all()
    print('mode:',args.mode,'NAVSIM token:',token,'instruction:',command)
    print('image encoder: frozen DINOv3 ViT-B/16; language encoder: frozen BERT-base-uncased')
    print('train loss:',round(float(out['loss']),4),'language sensitivity: pass','action mask/sensitivity:', 'pass' if args.mode=='vla' else 'n/a')
    print('views: F0 288x160; L0/R0 144x80; side sensitivity: pass'); print('prediction:',tuple(prediction.shape),'peak GiB:',round(torch.cuda.max_memory_allocated()/1024**3,3))
    print('SMOKE PASS')
if __name__=='__main__': main()
