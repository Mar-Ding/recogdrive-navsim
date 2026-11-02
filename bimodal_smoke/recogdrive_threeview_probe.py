"""GPU2-only real RGB -> frozen ReCogDrive VLM -> DD head feasibility test."""
import os,sys,json,time,traceback,gc,hashlib
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
OUT=BASE/'demo10/recogdrive_dd_threeview';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'repo'));sys.path.insert(0,str(BASE/'bimodal_smoke'))
os.environ['HF_HOME']=str(BASE/'hf_cache');os.environ['TMPDIR']=str(BASE/'tmp')
os.environ['MPLCONFIGDIR']=str(BASE/'tmp/matplotlib')
import torch,numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder,format_number
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
from navsim.agents.recogdrive.utils.conversation import get_conv_template
from navsim.agents.recogdrive.recogdrive_backbone import system_message
from recogdrive_dd import ReCogDriveDD

def save(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((BASE/'demo10/manifest.json').read_text());selected=manifest[:2]
    save('status.json',dict(stage='loading_vlm',gpu=2,samples=2))
    sensors=SensorConfig(cam_f0=[3],cam_l0=[3],cam_r0=[3],cam_l1=False,cam_l2=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False)
    loader=SceneLoader(BASE/'demo10/native_recogdrive/metadata',BASE/'data/sensor_blobs/trainval',SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=sorted({r['log_name'] for r in selected}),tokens=[r['token'] for r in selected]),sensors,load_image_path=True)
    builder=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(BASE/'models/ReCogDrive-VLM-2B'),device='cuda:0')
    backbone=builder.backbone;items=[];stats=[]
    for row in selected:
        agent=loader.get_agent_input_from_token(row['token']);es=agent.ego_statuses
        hist=torch.tensor([[float(e.ego_pose[j]) for j in range(3)] for e in es[:4]])
        cmd=torch.tensor(es[-1].driving_command,dtype=torch.float32)
        state=torch.cat((cmd,torch.tensor(es[-1].ego_velocity),torch.tensor(es[-1].ego_acceleration))).float()
        cam=agent.cameras[-1]
        # Native-resolution front dynamic tiling unchanged; each side only 1 tile.
        paths=[str(cam.cam_f0.image),str(cam.cam_l0.image),str(cam.cam_r0.image)]
        pixels=[load_image(p,max_num=m) for p,m in zip(paths,(12,1,1))]
        counts=[len(x) for x in pixels]
        command=next((['turn left','go straight','turn right'][i] for i,v in enumerate(cmd[:3]) if v==1),'unknown')
        hs=' '.join(f' - t-{3-i}: ({format_number(h[0])}, {format_number(h[1])}, {format_number(h[2])})' for i,h in enumerate(hist.tolist()))
        question=f'Front camera: <image>\nLeft camera (simplified): <image>\nRight camera (simplified): <image>\nAs an autonomous driving system, predict the vehicle trajectory based on the three camera views, historical motion context (last 4 timesteps):{hs}\nActive navigation command: [{command.upper()}]\nOutput 8 future trajectory points as (x:float, y:float, heading:float), encapsulated in [PT, ...], precision 2 decimal places.'
        template=get_conv_template('internvl2_5');template.system_message=system_message;template.append_message(template.roles[0],question);template.append_message(template.roles[1],None);query=template.get_prompt()
        for n in counts:query=query.replace('<image>','<img>'+'<IMG_CONTEXT>'*(256*n)+'</img>',1)
        backbone.tokenizer.padding_side='left'
        text=backbone.tokenizer([query],return_tensors='pt',padding='max_length',max_length=2800,truncation=False)
        expected=256*sum(counts);assert int((text['input_ids']==backbone.img_context_token_id).sum())==expected
        ids=text['input_ids'].cuda();mask=text['attention_mask'].cuda();pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,1)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():
            outputs=backbone.model(pixel_values=torch.cat(pixels).cuda().bfloat16(),input_ids=ids,attention_mask=mask,position_ids=pos,image_flags=torch.ones(sum(counts),device='cuda',dtype=torch.long),output_hidden_states=True,return_dict=True)
            hidden=outputs.hidden_states[-1][0].detach().cpu().bfloat16()
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        assert hidden.shape[1]==1536 and torch.isfinite(hidden).all()
        # Store only valid tokens; training must pad/mask if lengths vary.
        valid=hidden[mask[0].cpu().bool()].contiguous()
        item=dict(features=dict(last_hidden_state=hidden,history_trajectory=hist,status_feature=state,high_command_one_hot=cmd),target=torch.tensor(row['target'],dtype=torch.float32))
        torch.save(item,OUT/(row['token']+'.pt'));items.append(item)
        stats.append(dict(token=row['token'],tiles=counts,sequence_length=len(hidden),valid_tokens=len(valid),encode_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,full_feature_gib=hidden.numel()*2/2**30,valid_feature_gib=valid.numel()*2/2**30))
        save('status.json',dict(stage='encoding_probe',done=len(items),details=stats));del outputs,ids,mask,pos;gc.collect();torch.cuda.empty_cache()
    del builder,backbone;gc.collect();torch.cuda.empty_cache()
    net=ReCogDriveDD().cuda();opt=torch.optim.AdamW(net.parameters(),lr=1e-4);bench=[]
    for bs in (8,32,64,128):
        features={k:v[None].repeat(bs,*([1]*v.ndim)).cuda() for k,v in items[0]['features'].items()};target=items[0]['target'][None].repeat(bs,1,1).cuda()
        net.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(features,target)['trajectory_loss']
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        secs=time.perf_counter()-tick
        net.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net(features)['trajectory']
        assert pred.shape==(bs,8,3) and torch.isfinite(pred).all()
        bench.append(dict(batch=bs,train_step_seconds=secs,loss=float(loss),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,inference_shape=list(pred.shape)))
        del features,target,loss,pred;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();save('head_benchmark.json',bench)
    report=dict(stage='probe_passed',gpu=2,front='native448 dynamic max12 unchanged',sides='448 each, max1 tile; not 144x80 DINO preprocessing',encoding=stats,head_benchmark=bench,full_training_started=False,estimated_full_cache_gib=sum(x['full_feature_gib'] for x in stats)/len(stats)*10618,estimated_encoding_hours=stats[-1]['encode_seconds']*10618/3600)
    save('probe.json',report);save('status.json',report)
    with (BASE/'visual_vla/RUN_LOG.md').open('a') as f:f.write('\nReCogDrive-TD three-view GPU2 probe passed: '+json.dumps(report)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:save('status.json',dict(stage='failed',error=repr(e),traceback=traceback.format_exc()));raise
