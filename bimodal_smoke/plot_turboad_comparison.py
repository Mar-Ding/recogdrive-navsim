"""Plot archived training evidence without writing outside the workspace."""
import os
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
r=root/'evidence/turboad'
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for folder,label in [('expanded','TurboAD'),('mf_expanded','Turbo-MFAD'),('plus','TurboAD-Plus')]:
    h=json.loads((r/folder/'history.json').read_text())
    for ax,key in zip(axes,['val_loss','ade','fde']):
        ax.plot([x['epoch'] for x in h],[x[key] for x in h],label=label)
        ax.set_xlabel('Epoch');ax.set_ylabel(key);ax.grid(alpha=.25)
axes[0].legend();fig.suptitle('Same 129 train / 338 validation samples; seed 42')
fig.tight_layout();fig.savefig(r/'training_comparison.png',dpi=160)
