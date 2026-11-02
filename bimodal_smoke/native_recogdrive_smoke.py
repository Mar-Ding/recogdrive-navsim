"""Check native cached inference and one real full batch before the long run."""
from native_recogdrive_run import *
torch.set_num_threads(2);seed(42)
net=ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),cache_hidden_state=True)
samples=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((OUT/'probe_cache').glob('*.pt'))]
features={k:torch.stack([samples[i%len(samples)]['features'][k] for i in range(128)]) for k in samples[0]['features']}
targets={'trajectory':torch.stack([samples[i%len(samples)]['target'] for i in range(128)]).cuda()}
optimizer=torch.optim.AdamW(net.action_head.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.95))
net.train();optimizer.zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(features,targets);loss=net.compute_loss(features,targets,pred)
assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.action_head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
net.eval();start=time.perf_counter()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=net(features)['pred_traj']
torch.cuda.synchronize();assert pred.shape==(128,8,3) and torch.isfinite(pred).all()
save_json('smoke.json',dict(loss=loss.item(),grad_norm=float(norm),prediction_shape=list(pred.shape),inference_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,native_modules_unchanged=True))
print('NATIVE_FULL_BATCH_SMOKE_OK',flush=True)
