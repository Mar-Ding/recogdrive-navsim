"""Real-cache gradients, rotation, conditioning visibility and timing checks."""
import json,time
import torch
from train_visual_mf_vla import BASE,OUT,CachedPlanner,seed

seed(42);torch.set_num_threads(4)
data=torch.load(OUT/'features.pt',map_location='cpu',weights_only=False)
net=CachedPlanner(OUT/'train_kmeans_20.npy','vla',str(BASE/'weights/bert-base-uncased'),str(BASE/'weights/dinov3-vitb16')).cuda()
del net.vision;del net.language;torch.cuda.empty_cache()
def batch(n):
    v=data['vision'][:n].cuda().float();cmd=data['command'][:n]
    return [v[:,:180],v[:,180:225],v[:,225:]],data['language'][cmd].cuda().float(),data['mask'][cmd].cuda(),data['ego'][:n].cuda(),data['target'][:n].cuda()

result={'source':'real cached NAVSIM sample','flow':{},'inference':{},'timing':[]}
args=batch(2)
for target in ('vision','language','action'):
    net.train();net.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,detail=net.flow_loss(*args,target,details=True)
    assert torch.isfinite(loss);loss.backward()
    groups={'fusion':net.fusion,'target_head':net.flow_heads[target]}
    grads={name:sum(float(p.grad.float().square().sum()) for p in mod.parameters() if p.grad is not None)**.5 for name,mod in groups.items()}
    assert all(v>0 for v in grads.values())
    result['flow'][target]={'loss':float(loss),'mse':float(detail['mse']),'cosine':float(detail['cosine']),'grad_norm':grads}

net.eval();net.zero_grad(set_to_none=True)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    c,pad=net.context(*args[:-1]);a=net.head.plan_anchor[None].expand(2,-1,-1,-1);t=torch.zeros(2,device='cuda',dtype=torch.long)
    cond,act=net.fuse(c,pad,a,t);other,other_act=net.fuse(c,pad,a+5,t)
    condition_delta=float((cond-other).abs().max());action_delta=float((act-other_act).abs().max())
    assert condition_delta<1e-5 and action_delta>1e-4
    pred=net(*args[:-1])['trajectory'];assert pred.shape==(2,8,3) and torch.isfinite(pred).all()
result['inference']={'condition_change_when_noisy_action_changes':condition_delta,'action_change':action_delta,'trajectory_shape':list(pred.shape)}
result['target_isolation']={}
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for target,shape in (('vision',(2,270,768)),('language',(2,12,768)),('action',(2,1,16))):
        fixed_noise=torch.randn(shape,device='cuda');zero=torch.zeros(2,device='cuda')
        _,original=net.flow_loss(*args,target,noise=fixed_noise,t=zero,details=True)
        changed=list(args)
        if target=='vision':changed[0]=[torch.randn_like(x) for x in args[0]]
        elif target=='language':changed[1]=torch.randn_like(args[1])
        else:changed[4]=args[4]+torch.randn_like(args[4])
        _,perturbed=net.flow_loss(*changed,target,noise=fixed_noise,t=zero,details=True)
        delta=float((original['prediction']-perturbed['prediction']).abs().max())
        assert delta<1e-5
        result['target_isolation'][target]=delta
result['fixed_noise_overfit']={}
optimizer=torch.optim.AdamW(net.parameters(),lr=5e-4)
for target,shape in (('vision',(2,270,768)),('language',(2,12,768)),('action',(2,1,16))):
    fixed_noise=torch.randn(shape,device='cuda');t=torch.full((2,),.5,device='cuda');net.train()
    def objective():return net.flow_loss(*args,target,noise=fixed_noise,t=t)[0]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before=float(objective())
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=float(objective())
    assert after<before,(target,before,after)
    result['fixed_noise_overfit'][target]={'before':before,'after':after}
del optimizer
del loss,detail,cond,act,other,other_act,pred,args,c,pad,a
for n in (8,16,32,64):
    args=batch(n);net.train();net.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(*args[:-1],{'trajectory':args[-1]})['loss']
    assert torch.isfinite(loss);loss.backward();del loss
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=net.flow_loss(*args,'vision')
    assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
    result['timing'].append(dict(batch=n,forward_backward_seconds=time.perf_counter()-tick,peak_gib=torch.cuda.max_memory_allocated()/2**30))
    del loss,args;net.zero_grad(set_to_none=True);torch.cuda.empty_cache()
(OUT/'smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
