"""Publication-style vector diagram, traced against the archived lab code."""
import os
from pathlib import Path
R=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(R/'output/charts/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
D=R/'bimodal_smoke/assets/navsim'
fig,ax=plt.subplots(figsize=(19,9.3));ax.set(xlim=(0,19),ylim=(0,9.3));ax.axis('off')
ink='#253348';muted='#687789';blue='#3978AD';green='#368B78';purple='#8865AD';orange='#C98C42'
def txt(x,y,s,size=10,c=ink,ha='center',weight='normal'):
    ax.text(x,y,s,fontsize=size,color=c,ha=ha,va='center',weight=weight)
def box(x,y,w,h,title,sub='',color=blue,fill='#EFF5FA',fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.09',fc=fill,ec=color,lw=1.1))
    txt(x+w/2,y+h*(.64 if sub else .5),title,fs,weight='bold')
    if sub:txt(x+w/2,y+h*.27,sub,8.5,muted)
def line(points,c=muted,dash=False,arrow=True):
    xs,ys=zip(*points);ax.plot(xs,ys,color=c,lw=1.2,ls='--' if dash else '-',zorder=1)
    if arrow:ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='-|>',color=c,lw=1.2,mutation_scale=11))
def tokens(x,y,n,c):
    for i in range(n):ax.add_patch(Rectangle((x+i*.105,y),.075,.32,fc=c,ec='white',lw=.3))
def panel(x,y,w,h,label):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.12',fc='#FAFBFD',ec='#DCE2E9',lw=.8,zorder=-2))
    txt(x+.18,y+h-.29,label,11,ha='left',weight='bold')
txt(.2,8.95,'TurboMFAD',23,ha='left',weight='bold')
txt(3.2,8.94,'Three-modal planning with shared multimodal attention',14,ha='left')
txt(18.7,8.94,'NAVSIM  /  implementation diagram',9,muted,ha='right')
panel(.15,2.65,5.25,5.65,'(a) Modality encoding')
panel(5.65,2.65,5.25,5.65,'(b) Multimodal fusion')
panel(11.15,2.65,7.55,5.65,'(c) Diffusion trajectory generation')
# Actual sensor thumbnails, not generated illustrations.
for j,(x,y,w,h) in enumerate([(.4,6.52,1.65,.91),(.4,5.83,.79,.45),(1.26,5.83,.79,.45)]):
    im=Image.open(R/f'output/charts/demo_raw/scene_1_{j}.jpg')
    ax.imshow(im,extent=(x,x+w,y,y+h),aspect='auto',zorder=3)
txt(1.22,7.65,'Three-view RGB',10,weight='bold')
txt(1.22,5.55,'F: 288×160  |  L/R: 144×80',8,muted)
box(2.65,6.15,2.45,1.18,'DINOv3 ViT-B/16','Frozen encoder · 768-d',blue)
line([(2.08,6.73),(2.65,6.73)],blue)
box(.4,4.42,1.66,.72,'Navigation text','Template instruction',green,'#EFF7F3',9)
box(2.65,4.42,2.45,.72,'BERT-base','Frozen encoder · 12 tokens',green,'#EFF7F3')
line([(2.08,4.78),(2.65,4.78)],green)
box(.4,3.11,1.66,.72,'Noisy anchors','20 × 8 XY + time',purple,'#F4F0F8',9)
box(2.65,3.11,2.45,.72,'Action / time MLP','16 → 256 per anchor',purple,'#F4F0F8')
line([(2.08,3.47),(2.65,3.47)],purple)
# Fusion input tokens and a deliberately restrained block inset.
for y,c,label,n in [(6.74,blue,'270 visual tokens',9),(4.78,green,'12 language tokens',6),(3.47,purple,'20 action tokens',6)]:
    line([(5.12,y),(5.98,y)],c)
    tokens(6.02,y-.16,n,c)
    txt(6.65,y+.35,label,8.5,c)
    line([(7.05,y),(7.5,y)],c)
box(7.5,3.1,2.96,4.53,'',color='#76899E',fill='white')
txt(8.98,7.24,'MFBlock × 6',15,weight='bold')
box(7.75,6.12,2.46,.65,'Shared Q/K/V/O','mRoPE + routed mask',blue,fs=10)
line([(8.98,6.09),(8.98,5.77)])
box(7.75,4.98,2.46,.76,'Modality-specific FFN','Visual · language · action · ego',green,'#EFF7F3',10)
txt(8.98,4.51,'RMSNorm + residuals',9,muted)
txt(8.98,4.08,'256 channels · 4 heads',9,muted)
txt(8.98,3.58,'Conditions cannot attend\nto the noisy target',8.5,purple)
box(6.02,2.89,1.0,.37,'Ego MLP',color=orange,fill='#FCF5EA',fs=8)
line([(7.04,3.06),(7.28,3.06),(7.28,3.86),(7.5,3.86)],orange)
txt(6.51,2.72,'4-d state → 1 token',7.5,muted)
# The fused context branches: only the first 180 front-view tokens form the grid.
line([(10.48,6.74),(11.35,6.74)])
box(11.35,6.28,2.6,.92,'Visual grid adapter','Front 180 tokens → 16×16',blue,fs=10)
line([(10.48,5.45),(11.35,5.45)])
box(11.35,4.99,2.6,.92,'31 scene queries','2-layer query decoder',green,'#EFF7F3',10)
line([(10.48,3.75),(11.35,3.75)])
box(11.35,3.29,2.6,.92,'Fused action tokens','Add to anchor embeddings',purple,'#F4F0F8',10)
for y,c in [(6.74,blue),(5.45,green),(3.75,purple)]:
    line([(13.98,y),(14.43,y)],c)
box(14.43,3.29,2.15,3.91,'',color=orange,fill='#FCF5EA')
txt(15.505,6.68,'DiffusionDrive',12,weight='bold')
txt(15.505,6.28,'Trajectory head',11)
txt(15.505,5.57,'20 anchors',10)
txt(15.505,5.13,'2 decoder layers',10)
txt(15.505,4.69,'2 inference steps',10)
txt(15.505,3.84,'+ ego / time',9,muted)
line([(16.6,5.25),(17.0,5.25)])
# Tiny vector trajectory glyph, followed by concrete output specification.
for shift,c in [(-.16,'#D4DEE6'),(.16,'#D4DEE6'),(0,green)]:
    yy=np.linspace(4.5,6.1,8);xx=17.62+shift+.26*((yy-4.5)/1.6)**2
    ax.plot(xx,yy,'o-',c=c,lw=1.8,ms=3,zorder=4)
txt(17.75,6.54,'Trajectory',10,weight='bold')
txt(17.75,3.99,'8 poses / 4 s',9,muted)
txt(14.95,7.72,'Repeat fusion + decoding at each diffusion step',9,muted)
# Auxiliary objective is a separate pass sharing the same fusion parameters.
panel(.15,.2,18.55,2.13,'(d) Rotating flow supervision  ·  training only')
box(.44,.62,3.45,.91,'Rotate target: V → L → A','One target modality per optimizer step',purple,'#F4F0F8',10)
line([(3.92,1.08),(4.32,1.08)],purple,True)
box(4.32,.62,3.55,.91,'Interpolate target latent','z(t) = (1 − t) ε + t z',purple,'#F4F0F8',10)
line([(7.9,1.08),(8.3,1.08)],purple,True)
box(8.3,.62,3.55,.91,'Shared MFBlock + flow head','Other modalities remain clean conditions',purple,'#F4F0F8',10)
line([(11.88,1.08),(12.28,1.08)],purple,True)
box(12.28,.62,5.99,.91,'Velocity loss: MSE + 0.1 (1 − cosine)','Target: z − ε   |   auxiliary weight: 0.1 → 0',purple,'#F4F0F8',10)
txt(18.4,1.98,'No-flow ablation: identical planner, auxiliary branch disabled',8.5,muted,ha='right')
for ext in ('png','svg','pdf'):
    fig.savefig(D/f'mfad_architecture_detailed.{ext}',dpi=220,bbox_inches='tight',pad_inches=.12,facecolor='white')
plt.close(fig)
print('Paper-style diagram exported: PNG / SVG / PDF')
