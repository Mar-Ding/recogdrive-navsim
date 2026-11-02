"""Evidence-backed report chart; all artifacts and matplotlib cache stay in F workspace."""
import os,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/charts';OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':11})
E=OUT/'evidence'
def load(v,n):return json.loads((E/(v+'_'+n+'.json')).read_text())
keys=['dino_diffusiondrive','turbo_cross_vla','multiview_mf_vla']
scores=[load(v,'evaluation_trained')['pdms_100'] for v in keys]
old=json.loads((ROOT/'bimodal_smoke/evidence/recogdrive_dd_final/evaluation.json').read_text())
scores.append(old.get('pdms_100',old['score']*100))
names=['DINOv3\nDiffusionDrive','TurboAD\nCross-attention','TurboMFAD\nMF + flow','ReCogDrive-TD\n前视 VLM']
colors=['#277DA8','#5276B8','#087F79','#BA8437']
fig=plt.figure(figsize=(20,15),facecolor='white')
gs=GridSpec(3,2,figure=fig,height_ratios=[4.5,4.9,1.7],hspace=.5,wspace=.22)
fig.text(.055,.963,'NAVSIM Demo：已完成模型的训练与评测对比',fontsize=25,weight='bold',color='#20334D')
fig.text(.055,.933,'实验室 RTX 3090 · 相同数据划分与固定 200 场景 · PDMS 使用百分制（越高越好）',fontsize=13,color='#52657A')
for k,(lo,hi,title) in enumerate([(0,100,'完整刻度：各模型训练后 PDMS'),(75,90,'局部放大：75–90 分（截断纵轴）')]):
 ax=fig.add_subplot(gs[0,k]);bars=ax.bar(range(4),scores,color=colors,width=.57,zorder=3)
 ax.set_ylim(lo,hi);ax.set_xticks(range(4),names);ax.set_ylabel('PDMS（0–100）');ax.set_title(title,loc='left',fontsize=15,pad=17)
 ax.grid(axis='y',alpha=.16,zorder=0);ax.spines[['top','right']].set_visible(False)
 for bar,s in zip(bars,scores):ax.text(bar.get_x()+bar.get_width()/2,s+(1.2 if k==0 else .25),f'{s:.2f}',ha='center',fontsize=17,weight='bold',color='#20334D')
 if k==1:
  ax.text(.02,.96,'MF − DINO 基线：+2.51 分\nMF − Cross-attention：+4.52 分',transform=ax.transAxes,va='top',fontsize=11,color='#087F79',bbox=dict(facecolor='white',edgecolor='#D7E6E4',pad=7))
ax=fig.add_subplot(gs[1,:]);ax.axis('off');ax.set_title('具体输入与结构：冻结编码器，训练投影／适配器与轨迹生成模块',loc='left',fontsize=16,pad=15)
rows=[
['DINOv3\nDiffusionDrive','F0 288×160\nL0 / R0 各144×80\nRGB + 数值指令 + ego','冻结 DINOv3-B/16\n270×768 视觉特征\n不使用 BERT','768→256 投影 + 相机/位置编码\n三视角图像网格 → 16×16条件网格\n31个场景查询，2层 decoder\n无额外 V-L / MF 融合','DiffusionDrive 轨迹头\n20个训练集聚类 anchors\n2层扩散 decoder；推理2步',f"{scores[0]:.2f}\n最佳验证 ADE 0.7543 m"],
['TurboAD\nCross-attention','同上三相机 RGB\n导航文本（BERT）\n+ ego + 噪声动作 tokens','冻结 DINOv3-B/16\n视觉270 tokens\n冻结 BERT，文本12 tokens','TurboVLA BiAttentionBlock ×6\n视觉与语言双向 cross-attention\n文本增强；动作 cross decoder ×2\n融合宽度256；不加 flow 辅助','DiffusionDrive 轨迹头\n20个训练集聚类 anchors\n2层扩散 decoder；推理2步',f'{scores[1]:.2f}\n最佳验证 ADE 0.7561 m'],
['TurboMFAD\nMF + flow','同上三相机 RGB\n导航文本（BERT）\n+ ego + 噪声动作 tokens','冻结 DINOv3-B/16\n视觉270 tokens\n冻结 BERT，文本12 tokens','MFBlock ×6；256维 / 4头\n共享 QKVO + 各模态独立 FFN\n空间 mRoPE；视觉/语言/动作轮换 flow\nMSE + 0.1(1−cos)，辅助权重0.1→0','DiffusionDrive 轨迹头\n20个训练集聚类 anchors\n2层扩散 decoder；推理2步',f'{scores[2]:.2f}\n最佳验证 ADE 0.7136 m'],
['ReCogDrive-TD\n前视 VLM','仅 F0 前视 RGB\n448×448 动态切图（max12）\n导航指令 + ego + 4帧历史','冻结 ReCogDrive\nInternVL 2B VLM\n完整2800×1536 hidden states','VLM 内部视觉/语言融合\n1536→256 投影\n256个 latent queries → 16×16网格\n31个场景查询，3层 decoder','DiffusionDrive 轨迹头\n已有官方20 anchors\nlatent网格 + CNN适配器\n替换原生 ReCogDrive DiT',f'{scores[3]:.2f}\n最佳验证 ADE 0.7107 m']]
table=ax.table(cellText=rows,colLabels=['模型','输入','冻结编码器 / 特征','融合与条件构造','轨迹生成','测试结果'],colWidths=[.115,.18,.155,.29,.175,.12],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
table.auto_set_font_size(False);table.set_fontsize(10.8)
for (r,c),cell in table.get_celld().items():
 cell.set_edgecolor('#DBE4EB');cell.PAD=.07
 if r==0:cell.set_facecolor('#20334D');cell.set_text_props(color='white',weight='bold');cell.set_height(.105)
 else:cell.set_facecolor('#F1F6F8' if r%2 else 'white');cell.set_height(.224)
ax=fig.add_subplot(gs[2,:]);ax.axis('off')
ax.text(0,1,'统一训练与评测协议',fontsize=15,weight='bold',color='#087F79',va='top')
ax.text(0,.73,'约10% NAVSIM数据池：10,618场景 = 8,688训练 + 1,930验证；按日志隔离。200 epoch，seed42，有效batch128。\nAdamW：lr 1e−4 / wd 1e−4 / betas(0.9,0.95)，warmup3 + cosine，BF16，梯度裁剪1。\n按最低验证 ADE 选权重；原 ReCogDrive / NAVSIM PDM simulator + scorer，固定200个 navtest 场景，各版200/200评分成功。',fontsize=12,linespacing=1.7,va='top',color='#344B62')
fig.text(.055,.06,'读取方式：这是同一内部测试集上的 demo 对比；ReCogDrive-TD 的输入、编码器、历史状态与 anchors 不同，不能把差距全部归因于 fusion。',fontsize=11,color='#53657A')
fig.text(.055,.035,'待做：ReCogDrive-TD 三视角＋侧视简化，尚未训练，因此不画成绩。以上模型均不使用 LiDAR；16×16网格不是传感器测量的 BEV。',fontsize=11,color='#53657A')
fig.subplots_adjust(left=.055,right=.97,top=.89,bottom=.125)
for ext in ('png','pdf','svg'):fig.savefig(OUT/('NAVSIM_model_comparison.'+ext),dpi=180,facecolor='white')
with (OUT/'NAVSIM_model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','PDMS_100']);w.writerows(zip([n.replace('\n',' ') for n in names],scores))
print(OUT/'NAVSIM_model_comparison.png')

