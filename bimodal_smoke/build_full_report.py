"""Evidence-backed full report; all generated artifacts stay in this workspace."""
import csv,json,html
from pathlib import Path
import fitz
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

R=Path(__file__).resolve().parents[1]; E=R/'bimodal_smoke/evidence/demo10'; O=R/'output/pdf'
O.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/msyhbd.ttc'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCN',fontName='CN',fontSize=10,leading=17,spaceAfter=8,textColor=colors.HexColor('#24344b'),wordWrap='CJK'))
styles.add(ParagraphStyle(name='TitleCN',fontName='CNB',fontSize=23,leading=32,spaceAfter=18,textColor=colors.HexColor('#0f766e')))
styles.add(ParagraphStyle(name='HeadCN',fontName='CNB',fontSize=13,leading=20,spaceBefore=10,spaceAfter=8,textColor=colors.HexColor('#0f766e')))
styles.add(ParagraphStyle(name='CellCN',fontName='CN',fontSize=8.5,leading=13,wordWrap='CJK'))
story=[];md=['# TurboAD 系列 NAVSIM 完整实验报告\n\n日期：2026-10-31\n']
def p(s):
    story.append(Paragraph(html.escape(s),styles['BodyCN']));md.append(s+'\n')
def h(s):
    story.append(Paragraph(html.escape(s),styles['HeadCN']));md.append('## '+s+'\n')
def page(s):
    if story:story.append(PageBreak())
    story.append(Paragraph(html.escape(s),styles['TitleCN']));md.append('\n# '+s+'\n')
def table(headers,rows,widths):
    cells=[[Paragraph(html.escape(str(v)),styles['CellCN']) for v in row] for row in [headers]+rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f2f1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d9e2eb'))]))
    story.extend([t,Spacer(1,10)])
    md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+[''])
models=[('TurboAD','turboad','cross'),('Turbo-MFAD','turbo_mfad','mf'),('TurboAD-Plus','turboad_plus','plus'),('TurboAD-Plus-small','turboad_plus_small','plus_small')]
data=[]
for name,folder,label in models:
    s=json.loads((E/folder/'summary.json').read_text());hist=json.loads((E/folder/'history.json').read_text())
    vals=[];means=[]
    for suffix in ['_initial','']:
        rows=list(csv.DictReader((E/f'score_{label}{suffix}.csv').open()));scenes=[x for x in rows if x['token']!='average']
        assert len(scenes)==200 and all(x['valid']=='True' for x in scenes)
        means.append(next(x for x in rows if x['token']=='average'));vals.append(float(means[-1]['score']))
    data.append(dict(name=name,s=s,hist=hist,vals=vals,mean=means[-1]))

page('01  实验目标与总体结论')
p('本阶段目标是完成低成本自动驾驶规划 Demo：在去除 VLM/LLM 的视觉编码方案下，打通 RGB 与 LiDAR 跨模态融合、轨迹训练和 NAVSIM 原生评分，并验证增加视角与缩小辅助图像的可行性。')
p('四个主版本均完成同一约 10% 数据池上的 200 epoch 训练，并在固定 200 个 navtest 场景上全部有效。训练后 PDMS 均明显高于初始化，支持“流程已跑通、模型能够正常优化”的阶段判断。')
table(['版本','主要变化','训练后 PDMS'],[[d['name'],v,f"{d['vals'][1]:.5f}"] for d,v in zip(data,['双向 Cross-attention 基线','替换为 MF attention 融合','MF 融合扩展至三相机','保留前视、缩小左右辅助图像'])],[140,280,91])
h('本阶段可以给出的判断')
p('当前固定测试集上，TurboAD-Plus 得分最高；Turbo-MFAD 高于 Cross-attention 基线；辅助视角缩小后仍保持接近的分数，并降低资源开销。这些是单种子、内部固定子集的观察，尚未证明稳定性能优势。')
h('与老师要求的对应关系')
p('先以前视 RGB 跑通双模态基线，再补齐左右前视图；Plus-small 仅把辅助视角从 288×160 缩小到 144×80，前视保持 288×160。LiDAR 与 ego 状态继续保留。')
p('本报告主实验采用轨迹监督训练，MF 指融合模块的迁移；不将其表述为完整 Multimodal Flow 训练配方的复现。')

page('02  数据集与输入设计')
h('“10% 数据”的准确含义')
p('选取 OpenScene 官方传感器分片 0–19 / 200，按固定 ReCogDrive 提交 5831ee83 的 navtrain 配置过滤。得到 126 段日志、10,618 个场景，占完整 103,288 个场景的 10.28%。这是前 10% 分片对应的数据池，不是随机抽取的精确 10% 训练样本。')
table(['划分','场景数','日志数','用途'],[['训练','8,688','107','梯度更新；占完整集 8.41%'],['内部验证','1,930','19','每轮验证、选择 checkpoint'],['测试','200','固定 navtest token','统一 PDM 评分；不参与选权重']],[75,75,75,286])
p('训练与验证按日志隔离；验证日志通过固定 SHA256 排序规则确定并保留旧验证日志。训练、验证与测试 token 检查无重叠。四版共享同一 manifest：e0a72e1eb409f01f577c3ed7c14510a6a7eb1463f62d00ba5d6a53289ad73f81。')
h('输入模态及多视角区别')
p('感知模态为 RGB 图像与 LiDAR 点云；ego 状态作为额外条件输入。F0、L0、R0 分别为前视、左前、右前相机，它们是同一视觉模态的不同视角。本轮使用当前帧图像，未扩展为多帧视觉时序输入。')
table(['版本','相机输入分辨率','视觉 token'],[['TurboAD / Turbo-MFAD','F0：288×160','180'],['TurboAD-Plus','F0 / L0 / R0：均 288×160','540'],['TurboAD-Plus-small','F0：288×160；L0 / R0：144×80','270']],[150,286,75])
p('四版 LiDAR 均转为 256×256 单通道 BEV，编码为 256 tokens；ego 状态为 8维。辅助图像先真实缩小再经视觉编码器提取特征，没有以裁切旧特征替代图像缩放。')
p('多视角数据对齐逐条核验了前视、LiDAR、状态、目标、token、日志与 split；数据下载校验上游文件大小与 SHA256。')

page('03  模型结构与方法实现')
p('整体结构：RGB → 冻结 DINOv3 → 视觉 token；LiDAR BEV → CNN → 点云 token；两路融合后，以 LiDAR token 构造 BEV 与场景查询，结合 ego 状态，通过 DiffusionDrive 轨迹头预测未来 4秒的 8个航点。')
asset=R/'output/imagegen/turboad_architecture.png'
if asset.exists():
    im=fitz.Pixmap(str(asset));story.extend([Image(str(asset),width=511,height=511*im.height/im.width),Spacer(1,6)])
h('视觉与 LiDAR 编码')
p('DINOv3 输出 768维 patch 特征，冻结并离线缓存；训练时线性投影至 256维并 LayerNorm。LiDAR CNN 包含 4层 stride2 卷积，通道为 1→32→64→128→256，每层使用 GroupNorm 与 GELU，输出 16×16 tokens。两路加入位置与传感器嵌入。')
h('TurboAD：双向 Cross-attention')
p('使用 TurboVLA 的 BiAttentionBlock，深度 6层、隐藏维 256、4头，pre-norm、SDPA。视觉与 LiDAR 双向交互，每层配各自的残差 FFN（256→768→256），作为普通 attention fusion 基线。')
h('Turbo-MFAD / Plus：MF attention')
p('使用 6层 MFBlock，隐藏维 256、4头、head_dim64、FFN768；RGB 与 LiDAR 联合打包并双向可见，使用共享 QKVO、模态路由和空间 mRoPE，采用 SDPA 后端。Plus 与 Plus-small 沿用这一融合方法，仅扩展视觉 token 与位置布局。')
h('共同的轨迹解码')
p('融合后的 LiDAR token 重排为 16×16 BEV，经卷积 adapter 和双线性上采样得到 64×64 BEV。31个查询经 3层 Transformer Decoder 更新，memory 为 LiDAR tokens 与编码后的 ego 状态。随后调用迁移的 DiffusionDrive TrajectoryHead；分类、回归权重分别为 10与8。')

page('04  训练与评测协议')
table(['项目','本次统一设置'],[['训练硬件','单模型单卡 RTX 3090；使用 GPU0 / GPU2'],['数据与初始化','train8688 / val1930；seed42；各版从初始化训练'],['预算','200 epoch；每轮68次更新；共13,600次更新'],['batch','micro batch128 × 累积1 × world size1 = 有效 batch128'],['优化器','AdamW；lr1e-4；wd1e-4；betas(0.9,0.95)'],['调度与精度','warmup3 epoch + cosine；最低lr1e-6；BF16；梯度裁剪1'],['可训练部分','投影、LiDAR CNN、融合、场景查询与轨迹头；DINOv3冻结'],['选权重与随机性','内部验证 ADE 最低；验证噪声seed123；测试噪声seed999'],['评测','原生 ReCogDrive PDM scorer；固定200个 navtest token']],[120,391])
p('每轮末批为112个样本，按实际样本数处理损失。所有版本保持相同数据清单和总预算。Plus 的位置参数形状随 token 数变化，因此不是严格相同初始化的单因素消融。')
h('与 ReCogDrive 官方协议的对齐程度')
p('固定版本常规 IL 脚本为8卡、每卡batch16、累积1，有效batch128，200 epoch。本次单卡batch128保持该有效batch及主要优化器/调度预算。官方另有8卡×batch32、100 epoch的 Accelerate 路径，本轮未采用。')
p('对齐的是常规 IL 的主要优化预算和原生评分实现。模型改为冻结视觉编码器与自定义传感器融合，数据为约10%分片池加内部验证，seed与选权重规则也未完全按官方流程执行。因此本次不能称为原版 ReCogDrive 复现；也没有执行其 VLM 预训练或 GRPO/RL。')
h('指标解释')
p('PDMS 全部按原生 0–1 数值报告；ADE/FDE 单位为米。ADE 是内部验证轨迹误差，PDMS 来自测试场景的原生仿真评分，两者含义不同，验证误差更低不保证测试 PDMS 更高。')

page('05  ReCogDrive 训练评测协议核对结果')
p('本页汇总实际完成的协议核对。这里的“结果”指四个自定义模型在所迁移训练预算与原生评分流程下的结果，不代表原版 ReCogDrive 完整训练后的成绩。')
table(['协议项','ReCogDrive / 原生流程','本次执行结果'],[
['IL 总预算','常规脚本覆盖为200 epoch','四版均完成200 epoch / 13,600 updates'],
['有效 batch','8卡 × 每卡16 × 累积1 = 128','单卡128 × 累积1 = 128；已对齐'],
['优化参数','AdamW；lr/wd均1e-4；betas(0.9,0.95)','已采用相同参数；裁剪范数1'],
['学习率调度','3 epoch warmup + cosine，最低1e-6','已采用200 epoch调度'],
['随机种子','默认训练配置seed0','本次seed42；未对齐'],
['数值精度','默认trainer为16-mixed；另列DeepSpeed BF16配置','本次BF16；不能称精度配置完全一致'],
['数据划分','官方 train/val/test 配置','本次近似10%池、内部按日志划分；未完全对齐'],
['权重选择','官方训练框架的checkpoint流程','本次最低内部验证ADE；未宣称已对齐'],
['评分代码','run_metric_caching / run_pdm_score','复用原生入口、simulator与scorer'],
['评测范围','官方navtest配置','固定200 token子集；不是完整navtest'],
],[90,211,210])
h('原生评分的具体执行设置')
p('由原始 default_metric_caching / default_run_pdm_score 配置组合，train_test_split=navtest，worker=sequential。对固定 token 构建和复用 metric cache，评分前检查所有请求 token 均存在缓存。地图为 nuplan-maps-v1.0。')
p('采用原生 PDMSimulator / PDMScorer，proposal sampling 为40个0.1秒采样点；模型提供未来4秒轨迹。没有重新定义 PDMS 公式或替换评分实现。模型输入与推理由自定义 TurboADAgent 接入。')
h('训练评测完成情况')
p('四版均完成初始化与最佳验证权重两次评测，共8次 × 200场景；每次200/200有效、失败0。最佳权重仅按内部验证ADE选择，测试PDMS不参与选择。最终PDMS分别为 TurboAD 0.70645、Turbo-MFAD 0.75241、Plus 0.75558、Plus-small 0.75220。')
p('结论：主要IL优化预算及原生评分链路已迁移并验证可运行；训练数据、seed、精度、选权重与测试覆盖范围仍有差异，尚未完成严格同协议的原版 ReCogDrive 复现。')

page('06  实验结果与资源成本')
table(['版本','初始PDMS','训练后PDMS','最佳验证ADE'],[[d['name'],f"{d['vals'][0]:.5f}",f"{d['vals'][1]:.5f}",f"{d['s']['best_val_ade']:.4f} m"] for d in data],[155,110,110,136])
p('四版的初始和训练后评测均为200/200有效。Turbo-MFAD 比 TurboAD 高0.04596；Plus 比单前视 MF 高0.00318；Plus-small 比 Plus 低0.00338。差异以 PDMS 原始绝对值表示，未乘100。')
table(['版本','无责任碰撞','可行驶区域','TTC','进度','舒适度'],[[d['name']]+[f"{float(d['mean'][k]):.4f}" for k in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']] for d in data],[151,76,76,65,70,73])
p('各分项直接来自原生 CSV；PDMS 应使用逐场景评分后平均的结果，不能由表中各分项均值重新组合得到。')
table(['版本','峰值 allocated','训练 + 验证'],[[d['name'],f"{d['s']['peak_allocated_gib']:.2f} GiB",f"{d['s']['seconds']/60:.1f} 分钟"] for d in data],[200,155,156])
full,small=data[2]['s'],data[3]['s'];mem=100*(1-small['peak_allocated_gib']/full['peak_allocated_gib']);tm=100*(1-small['seconds']/full['seconds'])
p(f'Plus-small 相比 Plus：峰值 allocated 显存降低{mem:.1f}%，训练与验证耗时减少{tm:.1f}%。四版合计训练与验证约{sum(d["s"]["seconds"] for d in data)/3600:.2f} GPU小时。')
p('上述资源数字来自 summary 实测，DINOv3 冻结并预先缓存；不包含下载、解压、一次性特征编码和测试评分。不同 GPU 时段的负载可能影响耗时，不应当作纯算子速度对比。')

page('07  外部参考、结论与复现索引')
h('LFG 中 DiffusionDrive 的 10% 结果')
p('LFG 论文 v1 的 Table 4 报告 DiffusionDrive 在10%标签数据下 PDMS为72.6，输入为3Cam+LiDAR，采用端到端 BEV ResNet。换算至本报告尺度为0.726；这是论文的外部结果，不是本次重新运行的 DiffusionDrive 基线。[1]')
p('本次 Plus / Plus-small 同样使用三相机与LiDAR，但视觉骨干、训练方式、输入尺寸、数据子集和测试场景未与论文严格对齐；本次仅固定200场景，也未验证为论文的10%清单。因此只能作数值背景，不能由0.75558高于0.726宣称优于 DiffusionDrive。')
h('阶段结论')
p('已完成传感器数据对齐、冻结视觉特征缓存、双模态融合训练、DiffusionDrive 轨迹解码与原生 NAVSIM 评分的端到端工程闭环。新增视角和仅缩小辅助图像的两个版本也已完成同协议训练评测，达成“先跑通”的目标。')
p('当前结果支持继续研究 MF 融合与多视角输入；尚不能确认稳定增益。主要限制为单种子、自建近似10%数据池、200场景测试、非严格同初始化，以及未完整对齐外部协议。若后续需要性能结论，应优先统一数据与完整测试协议，再做多种子重复。')
h('复现证据与实现位置')
p('本地证据：bimodal_smoke/evidence/demo10/。四个模型目录中的 summary.json / history.json 记录预算、收敛与资源；score_cross、score_mf、score_plus、score_plus_small 及各自 initial CSV 保存逐场景评分。official_budget/ 保存所核对的官方代码副本。')
p('实现：model.py（融合与规划模型）、vendor/diffusion_decoder.py（轨迹头）、demo10_train.py（训练）、demo10_plus.py（多视角流程）、turboad_agent.py / turboad_eval.py（评测接入）。服务器实验根目录为 /data/ruiqiwang/recogdrive_baseline/demo10。')
h('参考资料')
p('[1] Learning to Drive is a Free Gift，arXiv:2602.22091v1，Table 4 与 Appendix B。https://arxiv.org/html/2602.22091v1')
p('[2] ReCogDrive：https://github.com/xiaomi-research/recogdrive；预算核查以本地固定版本副本为准。')
p('[3] DiffusionDrive：https://github.com/hustvl/DiffusionDrive；本次轨迹头迁移来源另见 vendor/source_hashes.json。')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#d9e2eb'));c.line(42,37,553,37)
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#66758a'));c.drawString(42,24,'TurboAD / NAVSIM  |  2026-10-31  |  阶段实验报告');c.drawRightString(553,24,str(doc.page))
pdf=O/'TurboAD_NAVSIM_full_report.pdf'
SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=48,title='TurboAD 系列 NAVSIM 完整实验报告',author='AD Research').build(story,onFirstPage=footer,onLaterPages=footer)
(O/'TurboAD_NAVSIM_full_report.md').write_text('\n'.join(md),encoding='utf-8')
tmp=R/'tmp/pdfs/full_report';tmp.mkdir(parents=True,exist_ok=True)
doc=fitz.open(pdf)
for i,pg in enumerate(doc):pg.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(str(tmp/f'page_{i+1}.png'))
assert len(doc)==7,len(doc)
assert all('�' not in pg.get_text() for pg in doc)
print(pdf, '\nPages:',len(doc))
