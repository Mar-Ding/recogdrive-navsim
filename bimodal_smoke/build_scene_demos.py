"""Render real sensor/prediction evidence plus editable architecture diagrams."""
import os,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'bimodal_smoke/assets/navsim';os.environ['MPLCONFIGDIR']=str(R/'output/charts/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
data=json.loads((R/'output/charts/scene_demo_predictions.json').read_text())
labels={'turbo_cross_vla':'TurboAD Cross','multiview_mf_vla':'TurboMFAD + flow','mfad_no_flow':'TurboMFAD no-flow'};colors=['#5276B8','#087F79','#8E6AB2']
for i,row in enumerate(data['scenes']):
 fig=plt.figure(figsize=(16,8),facecolor='white');g=fig.add_gridspec(3,3,width_ratios=[1,1,1.1],height_ratios=[1,1,.32],wspace=.2,hspace=.15)
 for j,(pos,title) in enumerate([((0,slice(0,2)),'F0 前视（网络输入288×160）'),((1,0),'L0 左侧（网络输入144×80）'),((1,1),'R0 右侧（网络输入144×80）')]):
  ax=fig.add_subplot(g[pos]);ax.imshow(Image.open(R/f'output/charts/demo_raw/scene_{i}_{j}.jpg'));ax.axis('off');ax.set_title(title,fontsize=11,color='#20334D')
 ax=fig.add_subplot(g[:2,2]);gt=np.vstack(([0,0],row['ground_truth_xy']));ax.plot(-gt[:,1],gt[:,0],'o-',color='#263238',label='真实未来轨迹',lw=2.5,ms=4)
 for (key,label),col in zip(labels.items(),colors):
  p=np.vstack(([0,0],np.asarray(row['predictions'][key]['trajectory'])[:,:2]));ax.plot(-p[:,1],p[:,0],'.--',color=col,label=f"{label}\n场景ADE {row['predictions'][key]['ADE_m']:.2f} m",lw=1.6,ms=5)
 ax.scatter(0,0,s=100,c='#D89027',marker='^',zorder=6);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.set_xlabel('向右位移（m）');ax.set_ylabel('向前位移（m）');ax.set_title('真实推理：未来4秒 / 8点');ax.legend(fontsize=8,loc='best')
 text=fig.add_subplot(g[2,:]);text.axis('off');text.text(0,.8,f"场景 {i+1} · 导航：{['左转','直行','右转','未知'][row['command']]} · token: {row['token']}\n图片为原始传感器画面预览；右侧为自车坐标系轨迹示意，未投影到RGB。单场景ADE不是PDMS。",fontsize=11,color='#52657A',va='top')
 fig.suptitle('NAVSIM 真实场景演示：三视角输入与预测轨迹',fontsize=20,weight='bold',color='#20334D');fig.subplots_adjust(top=.89,bottom=.06)
 fig.savefig(D/f'scene_demo_{i+1}.png',dpi=150,bbox_inches='tight');plt.close(fig)
shutil.copy2(R/'output/charts/scene_demo_predictions.json',D/'scene_demo_predictions.json')

fig,ax=plt.subplots(figsize=(16,9));ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off')
def b(x,y,w,h,t,c='#EDF3F8',fs=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.1',fc=c,ec='#C9D7E3'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color='#20334D')
def a(x,y,xx,yy,col='#52657A',rad=0):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.6,color=col,connectionstyle=f'arc3,rad={rad}'))
ax.text(.2,8.45,'TurboMFAD：视觉 / 语言 / 动作融合与轮换 Flow 辅助',fontsize=21,weight='bold',color='#20334D')
for y,title,encoder,col in [(6.5,'三视角 RGB\nF0:288×160 · L/R:144×80','冻结 DINOv3-B/16\n270 tokens → 256维','#E3F1F9'),(4.7,'导航语言\n指令模板，最长12 tokens','冻结 BERT-base\n语言 tokens → 256维','#E7F3EC'),(2.9,'带噪动作 + diffusion时间\n另输入 ego 条件','动作/时间编码\nego 独立投影','#F0E9F7')]:
 b(.3,y,3,1.15,title,col);b(3.8,y,3,1.15,encoder,col);a(3.3,y+.57,3.7,y+.57);a(6.8,y+.57,7.5,y+.57)
 b(7.6,y,3.6,1.15,{'6.5':'','4.7':'','2.9':''}.get(str(y),'') or ('视觉 tokens' if y==6.5 else '语言 tokens' if y==4.7 else '动作 tokens + ego'),col)
ax.add_patch(FancyBboxPatch((7.3,2.0),4.2,6.1,boxstyle='round,pad=.05',fc='none',ec='#087F79',lw=2));ax.text(9.4,7.9,'MFBlock ×6',ha='center',fontsize=14,weight='bold',color='#087F79');a(9.4,6.5,9.4,5.9);a(9.4,4.7,9.4,4.1)
b(12.1,5.8,3.5,1.6,'融合场景表征\n31个场景 queries\nDiffusionDrive 轨迹头\n20 anchors / 推理2步','#E3F1EF');a(11.6,6.5,12,6.5)
b(12.1,3.3,3.5,1.25,'未来4秒 / 8点轨迹\nNAVSIM PDM评测','#FFF3DF');a(13.85,5.8,13.85,4.7)
ax.text(9.4,2.3,'共享 QKVO · 模态专属 FFN\n空间 mRoPE · 256维 / 4 heads',ha='center',fontsize=11,color='#087F79')
b(.3,.3,15.3,1.25,'训练辅助分支（推理关闭）：按 optimizer step 轮换视觉 → 语言 → 动作\nz_t = (1−t)·noise + t·clean；速度目标 clean−noise；Laux = MSE + 0.1·(1−cosine)\n外层权重前60%更新为0.1，随后线性归零；no-flow版保留相同结构与初始化，完全关闭辅助loss。','#FFF3DF',11)
ax.text(.4,1.9,'框内为联合融合示意；条件 tokens 不能读取带噪目标 tokens。',fontsize=10,color='#52657A')
for ext in ('png','svg'):fig.savefig(D/f'mfad_architecture_detailed.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig);print('DEMOS_AND_ARCHITECTURE_READY')
