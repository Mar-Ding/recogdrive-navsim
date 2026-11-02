"""Concise teacher report from locally verified completed experiments only."""
from pathlib import Path

source = Path(__file__).with_name('build_full_report.py').read_text(encoding='utf-8')
exec(compile(source.split("page('01  实验目标与总体结论')")[0], __file__, 'exec'))
from reportlab.graphics.shapes import Drawing, Rect, String, Line

md=['# TurboAD 系列：NAVSIM 阶段实验汇报\n']
page('01  TurboAD 系列：训练与评测已跑通')
p('选择了 10% 的 NAVSIM 数据集进行测试。以 RGB 与 LiDAR 双模态融合为起点，采用 DiffusionDrive 轨迹解码，再扩展辅助相机并缩小辅助图像。四个版本已完成训练与原生 NAVSIM 评分闭环。')
table(['完成版本','每版训练预算','每版有效测试'],[['4 个','200 epoch / 13,600 次更新','200 / 200 场景']],[130,205,176])
h('数据：约 10% 分片池，训练与评测隔离')
p('取官方前 20 / 200 个传感器分片，经 navtrain 过滤得到 10,618 个场景（完整 103,288 场景的 10.28%）。按日志分为训练 8,688、验证 1,930；实际训练量占完整集 8.41%。固定 200 个 navtest 场景独立测试，三者 token 无重叠。')
h('训练与评测协议')
p('四版同数据、seed42、有效 batch128、200 epoch；单卡执行。AdamW，lr / wd 均 1e-4，betas(0.9, 0.95)，warmup3 + cosine，最低 lr1e-6，BF16，梯度裁剪1。按内部验证 ADE 最低选择权重。')
p('训练预算参照 ReCogDrive 常规 IL 的有效 batch128 与 200 epoch；测试复用其原生 NAVSIM PDM simulator / scorer，固定同一 200 场景。PDMS 统一按 0–100 分展示。这里完成的是自定义模型的训练评测，不是原版 ReCogDrive 的完整复现。')
h('实验路线：从原生模型到轻量跨模态规划')
table(['阶段','方法与实验目的','当前记录'],[
['1. ReCogDrive','保留原生 VLM + DiT 规划头，先测同规模训练成本','试训'],
['2. ReCogDrive-TD','保留 VLM，改用 DiffusionDrive 截断扩散轨迹生成','试训'],
['3. TurboAD','去掉 VLM，以视觉编码器 + Cross-attention 融合 RGB / LiDAR','200 轮训练与评分完成'],
['4. TurboMFAD','使用 Multimodal Flow attention 进行跨模态融合','200 轮训练与评分完成'],
['5. TurboMFAD-plus','拓展左右侧视；另做辅助视角缩小版本','两版训练与评分完成'],
],[112,257,142])
p('后文 Turbo-MFAD 即 TurboMFAD；TurboAD-Plus 即 TurboMFAD-plus。本地 10% 的数据集训练原生 ReCogDrive 预计需要约 43 小时，所以这部分都仅作测试。')

page('02  输入模态与四版方法结构')
p('两类感知模态：RGB 图像 + LiDAR 点云；另有 8 维 ego 状态条件。前视、左前、右前相机属于同一 RGB 模态的不同视角。')
table(['版本','输入 / 视觉 token','融合方法'],[
['TurboAD','前视 F0：288×160；180 tokens','6 层双向 Cross-attention'],
['Turbo-MFAD','前视 F0：288×160；180 tokens','6 层 MF attention'],
['TurboAD-Plus','F0 / L0 / R0：均 288×160；540 tokens','6 层 MF attention'],
['Plus-small','F0：288×160；L0 / R0：144×80；270 tokens','6 层 MF attention'],
],[112,244,155])
h('共同的编码与轨迹头')
p('RGB → 冻结 DINOv3 ViT-B（768 维）→ 256 维投影。LiDAR → 256×256 BEV → 四层 CNN → 256 个 256 维 token。加入位置与传感器嵌入后进行融合。')
p('融合后的 LiDAR tokens 构造 BEV，并通过 31 个场景查询结合 ego 状态；迁移 HUSTVL DiffusionDrive TrajectoryHead 生成轨迹（20 个 anchors、2 层 decoder、评测 2 步采样）。四版共用该解码路径。')
h('TurboAD 系列共同架构')
asset=R/'output/imagegen/turboad_architecture.png'
if asset.exists():
    im=fitz.Pixmap(str(asset));story.append(Image(str(asset),width=511,height=511*im.height/im.width))

page('03  实测结果：训练后规划评分明显改善')
p('同一固定 200 个 navtest 场景、同一原生 PDM scorer。初始化与训练后各版均为 200 / 200 有效；训练后权重只按内部验证 ADE 选择。')
drawing=Drawing(511,235)
palette=['#758499','#2563eb','#0f766e','#cf8b22']
for tick in range(7):
    y=30+tick*30
    drawing.add(Line(30,y,505,y,strokeColor=colors.HexColor('#e3e8ed')))
    drawing.add(String(7,y-3,f'{70+tick:.0f}',fontSize=8,fillColor=colors.HexColor('#66758a')))
for i,d in enumerate(data):
    x=57+i*119
    height=3000*(d['vals'][1]-.70)
    drawing.add(Rect(x,30,48,height,fillColor=colors.HexColor(palette[i]),strokeColor=None))
    drawing.add(String(x+24,38+height,f"{100*d['vals'][1]:.2f}",textAnchor='middle',fontSize=9))
    drawing.add(String(x+24,15,['TurboAD','Turbo-MFAD','Plus','Plus-small'][i],textAnchor='middle',fontSize=9))
drawing.add(String(30,226,'训练后 PDMS 局部放大（纵轴 70–76 分，非零起点）',fontName='CN',fontSize=9))
reference_y=30+3000*(0.726-.70)
drawing.add(Line(30,reference_y,505,reference_y,strokeColor=colors.HexColor('#b45309'),strokeWidth=1,strokeDashArray=[4,3]))
drawing.add(String(35,reference_y+17,'LFG / DiffusionDrive',fontSize=7,fillColor=colors.HexColor('#b45309')))
drawing.add(String(35,reference_y+5,'虚线：10% 数据，72.6 分',fontName='CN',fontSize=7,fillColor=colors.HexColor('#b45309')))
story.append(drawing)
table(['版本','初始 PDMS','训练后 PDMS','最佳验证 ADE'],[[d['name'].replace('TurboAD-Plus-small','Plus-small'),f"{100*d['vals'][0]:.2f}",f"{100*d['vals'][1]:.2f}",f"{d['s']['best_val_ade']:.4f} m"] for d in data],[155,104,119,133])
p('外部参考：LFG 论文 Table 4 的 DiffusionDrive（3Cam + LiDAR，10% 标签）PDMS 为 72.6 分。')
story.append(Paragraph('<link href="https://arxiv.org/html/2602.22091v1#S4.T4" color="#0f766e">来源：Learning to Drive is a Free Gift，Table 4（arXiv:2602.22091v1）</link>',styles['CellCN']))
h('结果')
p('四版从初始化到训练后均明显改善，证明训练信号能够转化为测试场景中的规划评分提升：不仅代码可运行，模型也能够正常学习。')
p(f"本轮 MF 版相对 Cross-attention 的 PDMS 高 {100*(data[1]['vals'][1]-data[0]['vals'][1]):.2f} 分；Plus 为 {100*data[2]['vals'][1]:.2f} 分，Plus-small 为 {100*data[3]['vals'][1]:.2f} 分。")

page('04  资源实测与训练预算预测')
table(['版本','峰值 allocated 显存','训练 + 验证耗时'],[[d['name'].replace('TurboAD-Plus-small','Plus-small'),f"{d['s']['peak_allocated_gib']:.2f} GiB",f"{d['s']['seconds']/60:.1f} 分钟"] for d in data],[183,164,164])
full,small=data[2]['s'],data[3]['s']
p(f"Plus-small 相比 Plus，峰值 allocated 显存降低 {100*(1-small['peak_allocated_gib']/full['peak_allocated_gib']):.1f}%，训练与验证耗时减少 {100*(1-small['seconds']/full['seconds']):.1f}%；PDMS 差值为 {100*(data[3]['vals'][1]-data[2]['vals'][1]):.2f} 分。辅助视角简化已完成可运行验证。")
p('耗时来自单卡 RTX 3090 实测；DINOv3 特征预先缓存。以上不包含下载、解压、一次性编码及测试评分；不同运行时段的 GPU 负载也可能影响耗时。')
h('ReCogDrive 两版：试训预算与运行状态')
table(['模型 / batch128','规划训练峰值显存','整轮实测均值','200 轮外推'],[
['ReCogDrive（原生）','约 16.52 GiB','784.07 秒 / 轮','43.56 小时'],
['ReCogDrive-TD','约 4.40 GiB','59.20 秒 / 轮','3.29 小时'],
],[159,127,116,109])
p('ReCogDrive-TD 已跑完；1 号服务器重启后尚未拉取结果，实际耗时与最终 PDMS 待同步。表中 3.29 小时仍保留为此前试训外推预算。')
p('原生版基于前三轮训练与验证实测外推，后按要求停止；表中两项均为预算预测，未计入最终 PDM 评分或 VLM 特征编码。显存为冻结 VLM 特征后的规划训练 / 测速记录，不代表在线加载 VLM 的总显存。')
p('TD 另需一次缓存预载 11.62 分钟，计入后约 3.48 小时；VLM 特征缓存约 85 GiB，预载需相应 CPU 内存及额外余量。原生版反复读盘，TD 改为内存预载，因此耗时变化不能全部归因于替换轨迹头。')
h('已经完成的实验闭环')
p('数据下载与传感器对齐 → 训练 / 验证 / 测试隔离 → 冻结视觉特征缓存 → Cross-attention / MF 融合训练 → DiffusionDrive 轨迹预测 → 原生 NAVSIM PDM 评分 → 保存逐场景结果。')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#d9e2eb'));c.line(42,37,553,37)
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#66758a'))
    c.drawString(42,24,'TurboAD / NAVSIM  |  阶段实验汇报  |  已完成实验记录')
    c.drawRightString(553,24,f'{doc.page} / 4')
pdf=O/'TurboAD_NAVSIM_teacher_report.pdf'
SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=48,title='TurboAD 系列 NAVSIM 阶段实验汇报',author='AD Research').build(story,onFirstPage=footer,onLaterPages=footer)
(O/'TurboAD_NAVSIM_teacher_report.md').write_text('\n'.join(md),encoding='utf-8')
tmp=R/'tmp/pdfs/teacher_report';tmp.mkdir(parents=True,exist_ok=True)
doc=fitz.open(pdf)
assert len(doc)==4,len(doc)
from PIL import Image as PILImage
sheet=PILImage.new('RGB',(1190,1684),'#d9e2eb')
for i,pg in enumerate(doc):
    assert '�' not in pg.get_text()
    target=tmp/f'page_{i+1}.png';pg.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(str(target))
    thumbnail=PILImage.open(target);thumbnail.thumbnail((595,842));sheet.paste(thumbnail,((i%2)*595,(i//2)*842))
sheet.save(tmp/'contact_sheet.png')
print(pdf,'Pages:',len(doc))
