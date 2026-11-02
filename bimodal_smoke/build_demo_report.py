"""Build a data-backed Chinese PDF report; all outputs stay in the workspace."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import fitz

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'bimodal_smoke/evidence/demo10'
OUT=ROOT/'output/pdf';TMP=ROOT/'tmp/pdfs'
OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/msyhbd.ttc'))
models=[('TurboAD','turboad','cross'),('Turbo-MFAD','turbo_mfad','mf'),('TurboAD-Plus','turboad_plus','plus'),('Plus-small','turboad_plus_small','plus_small')]
data=[]
for name,folder,label in models:
    summary=json.loads((E/folder/'summary.json').read_text())
    history=json.loads((E/folder/'history.json').read_text())
    scores=[]
    for suffix in ('_initial',''):
        rows=list(csv.DictReader((E/f'score_{label}{suffix}.csv').open()))
        scenes=[r for r in rows if r['token']!='average']
        assert len(scenes)==200 and all(r['valid']=='True' for r in scenes)
        scores.append(float(next(r['score'] for r in rows if r['token']=='average')))
    data.append(dict(name=name,summary=summary,history=history,scores=scores))

colors=['#748194','#2563eb','#0f766e','#d18b22']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#465469','xtick.color':'#465469','ytick.color':'#465469','figure.facecolor':'white','axes.grid':True,'grid.alpha':.15})
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for d,color in zip(data,colors):
    h=d['history'];axes[0].plot([r['epoch'] for r in h],[r['ade'] for r in h],color=color,label=d['name'],lw=1.8)
    h=[r for r in h if r['epoch']>=1];axes[1].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],color=color,lw=1.8)
axes[0].set(yscale='log',xlabel='Epoch',ylabel='Validation ADE (m, log scale)')
axes[1].set(xlabel='Epoch',ylabel='Training loss');axes[0].legend(fontsize=9)
fig.savefig(TMP/'curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained')
import numpy as np
x=np.arange(4)
ax.bar(x-.18,[d['scores'][0] for d in data],.32,label='Initial',color='#cbd5e1')
bars=ax.bar(x+.18,[d['scores'][1] for d in data],.32,label='Trained / best validation ADE',color=colors)
ax.bar_label(bars,fmt='%.5f',padding=4,fontsize=11)
ax.axhline(.726,color='#ac4f43',ls='--',lw=1.6,label='DiffusionDrive 10% (LFG Table 4): 0.726 [external]')
ax.set(xticks=x,xticklabels=[d['name'] for d in data],ylabel='PDMS (0-1)',ylim=(0,1.0));ax.legend(loc='upper left',fontsize=9)
fig.savefig(TMP/'scores.png',dpi=180);plt.close(fig)

W,H=960,540
c=canvas.Canvas(str(OUT/'TurboAD_NAVSIM_demo_report.pdf'),pagesize=(W,H))
c.setTitle('TurboAD / Turbo-MFAD - NAVSIM 跑通实验报告')
c.setAuthor('AD Research')
def text(x,y,s,size=15,color='#24344b',bold=False):
    c.setFillColor(color);c.setFont('CNB' if bold else 'CN',size);c.drawString(x,y,s)
def box(x,y,w,h,fill='#eef3f8'):
    c.setFillColor(fill);c.roundRect(x,y,w,h,10,fill=1,stroke=0)
def page(n,title,sub):
    c.setFillColor('#ffffff');c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor('#0f766e');c.rect(42,H-48,38,5,fill=1,stroke=0)
    text(42,450,title,27,bold=True);text(42,422,sub,11,'#66758a')
    text(42,22,'NAVSIM DEMO  /  2026-10-31  /  单种子、固定 200 场景内部评测',9,'#66758a')
    text(882,22,f'{n:02d} / 04',9,'#66758a')
def finish():c.showPage()
def table(headers,rows,x=42,y=370,widths=None,rowh=39,size=12):
    widths=widths or [876/len(headers)]*len(headers)
    c.setFillColor('#edf2f7');c.rect(x,y-rowh,sum(widths),rowh,fill=1,stroke=0)
    for ri,row in enumerate([headers]+rows):
        yy=y-ri*rowh
        if ri>0:
            c.setStrokeColor('#e1e7ee');c.line(x,yy-rowh,x+sum(widths),yy-rowh)
        xx=x
        for val,w in zip(row,widths):text(xx+10,yy-rowh+13,str(val),size,bold=ri==0);xx+=w

page(1,'TurboAD 系列：10% 数据 Demo 已跑通','4 个版本均完成训练与原生 NAVSIM 评分  |  本轮最高 PDMS 0.75558（TurboAD-Plus）')
box(42,333,876,71)
text(57,382,'数据集：约 10% NAVSIM navtrain 数据池',14,'#0f766e',True)
text(57,361,'前 20 / 200 个官方传感器数据包 → 10,618 场景 / 126 日志，占完整 103,288 场景的 10.28%。',11)
text(57,343,'训练 8,688（完整集 8.41%）+ 验证 1,930；按日志隔离。',11)
box(42,218,430,105);box(488,218,430,105)
text(57,302,'训练协议：四版统一预算',14,'#0f766e',True)
text(57,283,'单卡 RTX 3090；200 epoch / 13,600 updates',10.5)
text(57,265,'micro batch = 有效 batch = 128；seed42；BF16',10.5)
text(57,247,'AdamW：lr1e-4 / wd1e-4 / betas(0.9,0.95)',10.5)
text(57,229,'warmup3 + cosine；min lr1e-6；梯度裁剪1',10.5)
text(503,302,'评测协议：固定测试集、统一评分',14,'#0f766e',True)
text(503,283,'冻结 DINOv3；训练融合、LiDAR 分支和轨迹头',10.5)
text(503,265,'按验证 ADE 最低选 checkpoint',10.5)
text(503,247,'200 个 navtest 场景；与训练 / 验证无重叠',10.5)
text(503,229,'ReCogDrive 原生 PDM scorer；四版均 200/200 有效',10.5)
text(44,198,'模型结构：视觉与点云融合 → BEV / 场景查询 → 扩散轨迹解码',13,'#0f766e',True)
c.drawImage(str(ROOT/'output/imagegen/turboad_architecture.png'),42,39,width=876,height=151,mask='auto',preserveAspectRatio=True,anchor='c')
finish()

page(2,'四个版本：输入、融合方法与共用轨迹解码','双感知模态：RGB + LiDAR；附加 8维 ego 状态；F0 / L0 / R0 为前视 / 左前 / 右前相机')
text(44,396,'共用编码：冻结 DINOv3（768 → 256维）；LiDAR 256×256 BEV → 4层 CNN → 256 tokens。',12)
table(['版本','RGB 输入 / 视觉 token','方法结构'],[
['TurboAD','F0 288×160 / 180','6层双向 Cross-attention + 双分支 FFN'],
['Turbo-MFAD','F0 288×160 / 180','6层 MFBlock：共享 QKVO + 空间 mRoPE'],
['TurboAD-Plus','F0/L0/R0 均 288×160 / 540','沿用 MF 融合，扩展三视角 token'],
['Plus-small','F0 288×160；L0/R0 144×80 / 270','沿用 MF 融合，仅缩小左右辅助图像'],
],y=380,widths=[159,329,388],rowh=36,size=11)
box(42,103,876,84)
text(57,164,'融合方法的区别',14,'#0f766e',True)
text(57,143,'Cross：TurboVLA BiAttentionBlock，视觉 ↔ LiDAR 双向交互；每层包含各自残差 FFN。',12)
text(57,122,'MF：RGB / LiDAR token 联合打包、双向可见；模态路由 + mRoPE；本轮未做额外 flow 预训练。',12)
box(42,42,876,51)
text(57,72,'共用解码：融合后的 LiDAR token → BEV adapter（64×64）+ 31 个场景查询',13,bold=True)
text(57,52,'场景查询由 3层 Transformer Decoder 更新；结合 ego → DiffusionDrive 轨迹头 → 8 航点 / 4 秒。',11)
finish()

page(3,'评测结果：训练前后对比与 DiffusionDrive 参考','采用 ReCogDrive 仓库原生 PDM scorer；按内部验证 ADE 选权重，测试分数不参与权重选择')
c.drawImage(str(TMP/'scores.png'),42,211,width=876,height=200,mask='auto')
table(['模型','初始 PDMS','训练后 PDMS','最佳验证 ADE'],[
[d['name'],f"{d['scores'][0]:.5f}",f"{d['scores'][1]:.5f}",f"{d['summary']['best_val_ade']:.4f} m"] for d in data]
+ [['DiffusionDrive（LFG Table 4）','外部 10% 数据','0.726','非本次复现']]
,y=202,widths=[250,200,200,226],rowh=22,size=10)
text(44,57,'外部参考：3Cam + LiDAR、端到端 BEV ResNet；本次为自建划分 + 200 场景，评测口径不同。',10,'#966115')
text(44,41,'来源：LFG，arXiv:2602.22091v1，Table 4；72.6 换算为 0.726，不能据此直接判优。',9,'#66758a')
c.linkURL('https://arxiv.org/html/2602.22091v1#S4.SS2.SSS4',(42,37,850,53),relative=0)

finish()

page(4,'阶段结论：跑通成立，性能优势仍需验证','辅助视角简化方案已验证可用；后续以统一协议和完整测试提升结论可靠性')
table(['模型','峰值 allocated 显存','训练 + 验证耗时'],[
[d['name'],f"{d['summary']['peak_allocated_gib']:.2f} GiB",f"{d['summary']['seconds']/60:.1f} 分钟"] for d in data
],y=393,widths=[270,305,301],rowh=31,size=12)
small=data[3];full=data[2]
mem=100*(1-small['summary']['peak_allocated_gib']/full['summary']['peak_allocated_gib'])
tm=100*(1-small['summary']['seconds']/full['summary']['seconds'])
text(44,207,f'Plus-small 相比 Plus：显存减少 {mem:.1f}%，训练耗时减少 {tm:.1f}%，PDMS 低 0.00338。',14,'#0f766e',True)
text(44,174,'当前观察：MF 高于 cross 基线；新增视角的增益较小，尚不能确认稳定收益。',13)
text(44,145,'下一步：按官方日志划分、seed0、验证 loss 选权重，补完整 navtest；必要时多种子复验。',13)
text(44,116,'当前限制：单种子 / 200 测试场景 / 自建数据划分；仅借鉴 Stage 2 IL 的主要优化预算。',12,'#966115')
text(44,87,'证据：bimodal_smoke/evidence/demo10 中的 summary、history、evaluation 和逐场景 CSV。',10,'#66758a')
text(44,67,'代码快照：4d496c2（部署版本）；模型架构保留自定义融合与 DiffusionDrive 解码，非原版 ReCogDrive。',10,'#66758a')
finish();c.save()
doc=fitz.open(OUT/'TurboAD_NAVSIM_demo_report.pdf')
for i,p in enumerate(doc):p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(str(TMP/f'page_{i+1}.png'))
assert len(doc)==4
assert all('�' not in p.get_text() for p in doc)
print(OUT/'TurboAD_NAVSIM_demo_report.pdf')
