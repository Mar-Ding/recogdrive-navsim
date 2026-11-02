"""Build GitHub figures from measured experiment evidence (no synthetic curves)."""
import os,json,csv,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]; D=R/'bimodal_smoke/assets/navsim';D.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(R/'output/charts/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
rows=json.loads((R/'output/charts/NAVSIM_detailed_metrics.json').read_text(encoding='utf-8'))
colors=['#277DA8','#5276B8','#087F79','#8E6AB2','#BA8437'];names=['DINOv3-DD','TurboAD\nCross','TurboMFAD\n+ flow','TurboMFAD\nno-flow','ReCogDrive-TD\nfront']
def save(fig,name):
 for ext in ('png','svg'):fig.savefig(D/(name+'.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def style(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig,axs=plt.subplots(1,3,figsize=(16,5.2));fig.suptitle('NAVSIM 约10%数据 Demo：训练与评测结果',fontsize=20,weight='bold',color='#20334D')
for ax,key,title in zip(axs,['PDMS','val_ADE_m','train_val_minutes'],['固定200场景 PDMS ↑（百分制）','内部验证 ADE ↓（m）','训练 + 验证耗时 ↓（分钟）']):
 vals=[r[key] for r in rows];bars=ax.bar(names,vals,color=colors);style(ax);ax.set_title(title);ax.tick_params(axis='x',labelsize=9)
 ax.set_ylim(0,max(vals)*1.19 if key!='PDMS' else 100)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.025,f'{v:.4f}' if key=='val_ADE_m' else f'{v:.2f}',ha='center',fontsize=10)
fig.text(.04,.01,'8,688训练 / 1,930验证 · 200 epochs · seed42 · 有效batch128；前视VLM与三视角DINO输入不同。耗时不含编码和测试评分。',fontsize=10,color='#52657A');fig.tight_layout(rect=[0,.07,1,.9]);save(fig,'results_overview')

fig,axs=plt.subplots(1,2,figsize=(13,5));keys=['dino_diffusiondrive','turbo_cross_vla','multiview_mf_vla','mfad_no_flow'];labels=['DINOv3-DD','TurboAD Cross','TurboMFAD + flow','TurboMFAD no-flow']
curve=[]
for key,label,col in zip(keys,labels,colors):
 h=json.loads((R/'output/charts/evidence'/f'{key}_history.json').read_text())
 for ax in axs:ax.plot([x['epoch'] for x in h],[x['ade'] for x in h],label=label,color=col,lw=1.7)
 curve.extend(dict(model=label,epoch=x['epoch'],validation_ADE_m=x['ade']) for x in h)
axs[0].set_yscale('log');axs[0].set_title('全部训练过程（对数纵轴）');axs[1].set_xlim(40,200);axs[1].set_ylim(.65,1.1);axs[1].set_title('后期收敛细节（epoch 40–200）')
for ax in axs:style(ax);ax.set_xlabel('Epoch');ax.set_ylabel('验证 ADE（m，越低越好）')
axs[0].legend(fontsize=9);fig.suptitle('实际训练日志：验证误差收敛',fontsize=18,weight='bold');fig.tight_layout(rect=[0,0,1,.93]);save(fig,'training_convergence')
with (D/'training_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)

fig,ax=plt.subplots(figsize=(16,8));ax.set_xlim(0,16);ax.set_ylim(0,8);ax.axis('off')
def box(x,y,w,h,text,color='#EDF3F8',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',fc=color,ec='#C7D7E3'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#20334D')
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#52657A',lw=1.6))
ax.text(.2,7.55,'TurboAD / TurboMFAD：纯视觉三模态适配结构',fontsize=22,weight='bold',color='#20334D')
box(.3,5.6,3,1,'RGB：前视 288×160\n左 / 右各 144×80');box(3.8,5.6,3,1,'冻结 DINOv3-B/16\n270视觉 tokens → 256维');arrow(3.3,6.1,3.7,6.1)
box(.3,3.9,3,1,'导航指令模板\n最长12个语言 tokens');box(3.8,3.9,3,1,'冻结 BERT-base\n语言特征 → 256维');arrow(3.3,4.4,3.7,4.4)
box(.3,2.2,3,1,'带噪轨迹 / diffusion时间\n+ ego 4维条件');box(3.8,2.2,3,1,'动作编码 + 时间嵌入\nego投影');arrow(3.3,2.7,3.7,2.7)
box(7.5,2.2,3.7,4.4,'融合模块（选择其一）\n\nTurboAD：双向 VL attention\n+ action cross-attention\n\nTurboMFAD：MFBlock ×6\n256维 / 4 heads\n共享QKVO、模态FFN、mRoPE','#E2F1ED')
for y in (6.1,4.4,2.7):arrow(6.8,y,7.4,y)
box(12,4.9,3.5,1.5,'场景查询（31 queries）\n+ DiffusionDrive轨迹头\n20 anchors / 推理2步');arrow(11.2,4.9,11.9,5.6)
box(12,2.8,3.5,1.1,'未来4秒 / 8个轨迹点\n→ NAVSIM 原生PDM评分');arrow(13.75,4.9,13.75,4)
box(.3,.35,15.2,1.05,'仅 MF + flow 训练：按 step 轮换视觉 → 语言 → 动作；MSE + 0.1×(1−cosine)，外层权重0.1后衰减至0。\nno-flow 使用同结构、同初始化，关闭辅助损失；所有版本推理均不运行辅助 flow。','#FFF3DF',11)
save(fig,'architecture')
for ext in ('png','svg'):shutil.copy2(R/'output/charts'/('NAVSIM_detailed_metrics.'+ext),D/('pdm_components.'+ext))
shutil.copy2(R/'output/charts/NAVSIM_detailed_metrics.csv',D/'metrics.csv')
print('FIGURES_READY',D)
