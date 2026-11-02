"""Real-data VLA training for the separate 4090 server; preserves model.py."""
import os,sys,json,time,pickle,hashlib,math,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torchvision import transforms
from transformers import AutoTokenizer
import yaml
BASE=Path('/root/autodl-tmp/mf-vla')
sys.path.insert(0,str(BASE/'multiview'))
from visual_mf_vla import MFPlanner as Planner, flow_weight
OUT=BASE/'outputs/multiview_mf_vla';OUT.mkdir(parents=True,exist_ok=True)
DATA=BASE/'data/navsim10/openscene-v1.1'
torch.set_num_threads(4)
def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def save_json(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def status(stage,**kw):save_json('status.json',dict(stage=stage,updated=time.time(),**kw))
def manifest():
    cfg=yaml.safe_load((BASE/'src/recogdrive/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text())
    rows=[]
    for log in sorted(cfg['log_names']):
        path=DATA/'meta_datas/trainval'/f'{log}.pkl'
        if not path.exists() or not (DATA/'sensor_blobs/trainval'/log).exists():continue
        frames=pickle.load(path.open('rb'))
        for start in range(0,len(frames)-13,cfg['frame_interval']):
            i=start+3;f=frames[i]
            if not f['roadblock_ids']:continue
            if not all(abs((a['timestamp']-f['timestamp'])/1e6-j*.5)<.2 for j,a in enumerate(frames[i+1:i+9],1)):continue
            images=[str(DATA/'sensor_blobs/trainval'/f['cams'][cam]['data_path']) for cam in ('CAM_F0','CAM_L0','CAM_R0')]
            if not all(Path(p).is_file() for p in images):raise RuntimeError(f'Missing camera for {log} {i}')
            yaw=Quaternion(f['ego2global_rotation']).yaw_pitch_roll[0];pos=np.asarray(f['ego2global_translation'])[:2];c,s=np.cos(yaw),np.sin(yaw)
            target=[]
            for j,a in enumerate(frames[i+1:i+9],1):
                assert abs((a['timestamp']-f['timestamp'])/1e6-j*.5)<.2
                d=np.asarray(a['ego2global_translation'])[:2]-pos;ang=Quaternion(a['ego2global_rotation']).yaw_pitch_roll[0]-yaw
                target.append([c*d[0]+s*d[1],-s*d[0]+c*d[1],np.arctan2(np.sin(ang),np.cos(ang))])
            state=np.r_[f['driving_command'],f['ego_dynamic_state']].astype(np.float32);assert state.shape==(8,)
            rows.append(dict(token=f['token'],log=log,images=images,target=target,ego=state[4:].tolist(),command=int(state[:4].argmax())))
    rows=sorted(rows,key=lambda r:hashlib.sha256(r['token'].encode()).hexdigest())[:10618]
    logs=sorted({r['log'] for r in rows},key=lambda x:hashlib.sha256(x.encode()).hexdigest());val=set(logs[:max(1,round(len(logs)*.15))])
    for r in rows:r['split']='val' if r['log'] in val else 'train'
    assert len(rows)>100 and len({r['token'] for r in rows})==len(rows)
    save_json('manifest.json',rows);return rows

class CachedPlanner(Planner):
    def context(self,image,ids,language_mask,ego):
        visual=torch.cat([self.vision_proj(x)+self.camera_embed.weight[i] for i,x in enumerate(image)],1)
        visual=visual+self.vision_pos+self.types.weight[0]
        language=self.language_proj(ids)+self.types.weight[1]
        state=self.ego(ego).unsqueeze(1)+self.types.weight[2]
        c=torch.cat((visual,language,state),1)
        pad=torch.cat((torch.zeros(ego.shape[0],270,device=ego.device,dtype=torch.bool),~language_mask.bool(),torch.zeros(ego.shape[0],1,device=ego.device,dtype=torch.bool)),1)
        return c,pad

def main():
    seed(42);status('preparing_manifest')
    rows=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else manifest()
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val']
    anchors=OUT/'train_kmeans_20.npy'
    if not anchors.exists():
        y=np.asarray([rows[i]['target'] for i in train],dtype=np.float32)[:,:,:2]
        points=torch.from_numpy(y.reshape(len(y),-1));centers=points[torch.randperm(len(points))[:20]].clone()
        for _ in range(100):
            labels=torch.cdist(points,centers).argmin(1)
            updated=torch.stack([points[labels==k].mean(0) if (labels==k).any() else centers[k] for k in range(20)])
            converged=torch.allclose(centers,updated,atol=1e-4);centers=updated
            if converged:break
        np.save(anchors,centers.numpy().reshape(20,8,2))
    net=CachedPlanner(anchors,'vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
    cache=OUT/'features.pt';started=time.time()
    if cache.exists():tensors=torch.load(cache,map_location='cpu',weights_only=False)
    else:
        status('encoding',samples=len(rows),train=len(train),val=len(val))
        tokenizer=AutoTokenizer.from_pretrained(BASE/'weights/bert-base-uncased',local_files_only=True)
        text=tokenizer(['navigation instruction: '+x for x in ['turn left','go straight','turn right','unknown']],padding='max_length',max_length=12,truncation=True,return_tensors='pt').to('cuda')
        net.eval()
        with torch.no_grad(): language=net.language(**text).last_hidden_state.cpu().half()
        tf=[transforms.Compose([transforms.Resize(size),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]) for size in [(160,288),(80,144),(80,144)]]
        features=[]
        for offset in range(0,len(rows),32):
            chunk=rows[offset:offset+32];views=[]
            for camera in range(3):
                batch=[]
                for r in chunk:
                    with Image.open(r['images'][camera]) as im:batch.append(tf[camera](im.convert('RGB')))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f=net.vision(pixel_values=torch.stack(batch).cuda()).last_hidden_state[:,5:]
                assert f.shape[1]==[180,45,45][camera];views.append(f.cpu().half())
            features.append(torch.cat(views,1))
            elapsed=time.time()-started
            status('encoding',done=min(offset+32,len(rows)),samples=len(rows),seconds=elapsed,remaining_seconds=elapsed/(offset+len(chunk))*(len(rows)-offset-len(chunk)))
        tensors=dict(vision=torch.cat(features),language=language,mask=text['attention_mask'].cpu(),command=torch.tensor([r['command'] for r in rows]),ego=torch.tensor([r['ego'] for r in rows]),target=torch.tensor([r['target'] for r in rows]))
        torch.save(tensors,cache);save_json('encoding.json',dict(seconds=time.time()-started,samples=len(rows)))
    # Frozen encoders are unnecessary during cached-feature optimization.
    del net.vision;del net.language;torch.cuda.empty_cache()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
    def batch(ids):
        v=tensors['vision'][ids].cuda().float();cmd=tensors['command'][ids]
        return [v[:,:180],v[:,180:225],v[:,225:]],tensors['language'][cmd].cuda().float(),tensors['mask'][cmd].cuda(),tensors['ego'][ids].cuda(),{'trajectory':tensors['target'][ids].cuda()}
    # Probe realistic microbatch sizes without updating weights.
    micro=8
    for candidate in (8,16,32,64,128):
        try:
            net.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            args=batch(train[:candidate])
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(*args)['loss']
            assert torch.isfinite(loss);loss.backward();del loss
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=net.flow_loss(*args[:-1],args[-1]['trajectory'],'vision')
            assert torch.isfinite(loss);loss.backward();peak=torch.cuda.max_memory_allocated()/2**30
            del loss,args;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();micro=candidate
            print('BATCH_PROBE',candidate,peak,flush=True)
            if peak>18:break
        except torch.cuda.OutOfMemoryError:
            if 'loss' in locals():del loss
            if 'args' in locals():del args
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
    def evaluate():
        net.eval();seed(123);ade=fde=0.;n=0
        with torch.no_grad():
            for start in range(0,len(val),micro):
                args=batch(val[start:start+micro])
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(*args[:-1])['trajectory']
                dist=(pred.float()[...,:2]-args[-1]['trajectory'][...,:2]).norm(dim=-1)
                assert torch.isfinite(dist).all();ade+=dist.sum().item();fde+=dist[:,-1].sum().item();n+=len(dist)
        return dict(ade=ade/n/8,fde=fde/n)
    config=dict(mode='vla',epochs=200,effective_batch=128,micro_batch=micro,seed=42,train=len(train),val=len(val),model_sha256=hashlib.sha256((BASE/'multiview/model.py').read_bytes()).hexdigest(),anchors='train-only kmeans20',fusion='upstream MFBlock 6x256, shared QKVO, modality FFN, mRoPE, SDPA',flow_targets=['vision','language','action'],flow_mse_weight=1.,flow_cosine_weight=.1,flow_weight='0.1 first60pct then linear to0',flow_recipe='NAVSIM adaptation; not exact old TurboMFVLA reproduction',source_sha256={name:hashlib.sha256((BASE/'multiview'/name).read_bytes()).hexdigest() for name in ('visual_mf_vla.py','train_visual_mf_vla.py')},manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
    save_json('config.json',config);history=[];best=float('inf');first=1
    if (OUT/'latest.pt').exists():
        ck=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);assert ck['config']==config
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];best=ck['best'];first=ck['epoch']+1
    else:history=[dict(epoch=0,**evaluate())];save_json('history.json',history)
    for epoch in range(first,201):
        tick=time.time();seed(42+epoch);net.train();aux_sums={k:[0.,0.,0] for k in ('vision','language','action')};order=[train[i] for i in torch.randperm(len(train)).tolist()];total=0.
        lr=1e-4*epoch/3 if epoch<=3 else 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*(epoch-4)/197))
        for group in opt.param_groups:group['lr']=lr
        for offset in range(0,len(order),128):
            effective=order[offset:offset+128];opt.zero_grad(set_to_none=True)
            for start in range(0,len(effective),micro):
                ids=effective[start:start+micro]
                args=batch(ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(*args)['loss']
                assert torch.isfinite(loss);(loss*len(ids)/len(effective)).backward();total+=loss.item()*len(ids)
                del loss
                step=(epoch-1)*math.ceil(len(train)/128)+offset//128
                target=('vision','language','action')[step%3]
                weight=flow_weight(step+1,200*math.ceil(len(train)/128))
                with torch.autocast('cuda',dtype=torch.bfloat16):aux,metrics=net.flow_loss(*args[:-1],args[-1]['trajectory'],target)
                assert torch.isfinite(aux);(weight*aux*len(ids)/len(effective)).backward()
                aux_sums[target][0]+=float(metrics['mse'])*len(ids);aux_sums[target][1]+=float(metrics['cosine'])*len(ids);aux_sums[target][2]+=len(ids)
                del aux,args
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            if offset%1024==0:status('training',epoch=epoch,done=offset+len(effective),samples=len(train),micro_batch=micro,flow_target=target,flow_weight=weight)
        row=dict(epoch=epoch,loss=total/len(train),flow_weight=weight,flow={k:dict(mse=v[0]/max(1,v[2]),cosine=v[1]/max(1,v[2]),samples=v[2]) for k,v in aux_sums.items()},lr=lr,**evaluate(),seconds=time.time()-tick);history.append(row)
        improved=row['ade']<best;best=min(best,row['ade'])
        state=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,config=config,history=history)
        torch.save(state,OUT/'latest.tmp');(OUT/'latest.tmp').replace(OUT/'latest.pt')
        if improved:torch.save(state,OUT/'best_val_ade.pt')
        save_json('history.json',history)
        avg=np.mean([r['seconds'] for r in history[-3:] if 'seconds' in r]);status('training',**row,best_ade=best,projected_200_hours=float(avg*200/3600),remaining_hours=float(avg*(200-epoch)/3600),peak_gib=torch.cuda.max_memory_allocated()/2**30)
        print(json.dumps(row),flush=True)
    status('training_complete',epochs=200,best_ade=best,pdms='not run; test data/maps/cache not yet prepared')
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));raise
