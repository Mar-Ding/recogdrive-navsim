"""E009/E010: shared anonymous RAM features, dense trajectory windows."""
import os,json,mmap,hashlib
import numpy as np
import torch
from vla.common import ROOT
from vla.extension_runtime import identity as extension_identity,verify_identity as verify_extension
from vla.turbo_experiment import bank as sparse_bank

SCHEDULE=os.environ.get('DENSE_FLOW','fixed')
assert SCHEDULE in ('fixed','decay')
EXPERIMENT='E009' if SCHEDULE=='fixed' else 'E010'
OUTPUT=ROOT/f'outputs/turbo_mf_dense_{SCHEDULE}_seed42'
COMPONENT=ROOT/'outputs/mf_extension_seed7'
CACHE=ROOT/'outputs/dense_shared'
_BANK=None

def weight(step):
    return .1 if SCHEDULE=='fixed' else .1*max(0.,min(1.,(10000-step)/4000))

def identity():
    return {**extension_identity(),'dense_experiment':{'experiment':EXPERIMENT,
        'schedule':SCHEDULE,'updates':10000,'flow_weight':'0.1 constant' if SCHEDULE=='fixed' else '0.1 through6000 then linear to0 at10000',
        'dataset':{k:v for k,v in json.loads((CACHE/'manifest.json').read_text()).items() if k!='feature_seconds'},
        'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}}

def verify_identity(saved):
    for key,value in identity().items():
        if saved.get(key)!=value:raise ValueError('Dense identity mismatch '+key)

class Gather:
    def __init__(self,array,indices):self.array,self.indices=array,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,ids):
        if torch.is_tensor(ids):ids=ids.numpy()
        return torch.from_numpy(np.array(self.array[self.indices[ids]],copy=True))

def bank():
    global _BANK
    if _BANK is not None:return _BANK
    manifest=json.loads((CACHE/'manifest.json').read_text())
    fd=int(os.environ['DENSE_FEATURE_FD']);region=mmap.mmap(fd,manifest['bytes'],access=mmap.ACCESS_READ)
    features=np.ndarray(tuple(manifest['shape']),dtype=np.float16,buffer=region)
    metadata=torch.load(CACHE/'metadata.pt',map_location='cpu',weights_only=False)
    # Keep the original 1600 validation examples/512-point selection subset.
    sparse,holdout,_=sparse_bank();del sparse
    ids=metadata['train_ids'].numpy();task=metadata['task'].numpy()
    data=[Gather(features,ids),Gather(metadata['text'].numpy(),task[ids]),
        Gather(metadata['pad'].numpy(),task[ids]),Gather(metadata['sub'].numpy(),task[ids])]
    data += [Gather(metadata[k].numpy(),ids) for k in ('state','target','mask','teacher')]
    data += [metadata['task'][ids],metadata['episode'][ids]]
    # mmap remains alive through features.base; retain explicit reference too.
    _BANK=(data,holdout,[{'manifest':manifest}]);return _BANK

class TaskSampler:
    """Uniform task, then uniform dense frame; exact per-task cursor restore."""
    def __init__(self,data,batch,seed):
        self.batch=batch;self.rng=torch.Generator().manual_seed(seed);self.step=0
        self.groups=[torch.where(data[8]==i)[0] for i in range(40)]
        self.orders=[g[torch.randperm(len(g),generator=self.rng)] for g in self.groups]
        self.cursors=[0]*40
    def next(self):
        selected=[]
        for task in torch.randint(40,(self.batch,),generator=self.rng).tolist():
            cursor=self.cursors[task];order=self.orders[task]
            if cursor==len(order):
                order=self.groups[task][torch.randperm(len(order),generator=self.rng)];self.orders[task]=order;cursor=0
            selected.append(order[cursor]);self.cursors[task]=cursor+1
        ids=torch.stack(selected)
        flow=('image','text','action')[self.step%3];self.step+=1
        return ids,'vla_flow_'+flow
    def state_dict(self):return dict(orders=self.orders,cursors=self.cursors,step=self.step,rng=self.rng.get_state())
    def load_state_dict(self,state):
        self.orders=state['orders'];self.cursors=state['cursors'];self.step=state['step'];self.rng.set_state(state['rng'])
