import os,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/charts';E=OUT/'evidence'
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
metrics={'PDMS':'score','NC 无责碰撞':'no_at_fault_collisions','DAC 可行驶区域':'drivable_area_compliance','EP 自车进度':'ego_progress','TTC 碰撞时间':'time_to_collision_within_bound','C 舒适性':'comfort','DDC 行驶方向':'driving_direction_compliance'}
variants=[('dino_diffusiondrive','DINOv3-DD'),('turbo_cross_vla','TurboAD Cross'),('multiview_mf_vla','TurboMFAD + flow')]
if (E/'mfad_no_flow_evaluation_trained.json').exists():variants.append(('mfad_no_flow','MFAD no-flow'))
rows=[]
for key,label in variants:
 ev=json.loads((E/(key+'_evaluation_trained.json')).read_text());h=json.loads((E/(key+'_history.json')).read_text());best=next(x for x in h if x['epoch']==ev['checkpoint_epoch'])
 rows.append(dict(model=label,**{name:ev[k]*100 for name,k in metrics.items()},val_ADE_m=best['ade'],val_FDE_m=best['fde'],checkpoint_epoch=ev['checkpoint_epoch'],training_epochs=h[-1]['epoch'],train_val_minutes=sum(x.get('seconds',0) for x in h)/60,eval_seconds=ev['seconds'],successful_scenarios=ev['successful_scenarios'],failed_scenarios=ev['failed_scenarios']))
oldroot=ROOT/'bimodal_smoke/evidence/recogdrive_dd_final';ev=json.loads((oldroot/'evaluation.json').read_text());s=json.loads((oldroot/'summary.json').read_text());h=s['history'];best=next(x for x in h if x['epoch']==ev['checkpoint_epoch'])
rows.append(dict(model='ReCogDrive-TD front',**{name:ev[k]*100 for name,k in metrics.items()},val_ADE_m=best['ade'],val_FDE_m=best['fde'],checkpoint_epoch=ev['checkpoint_epoch'],training_epochs=200,train_val_minutes=sum(x.get('seconds',0) for x in h)/60,eval_seconds=ev['seconds'],successful_scenarios=ev['successful_scenarios'],failed_scenarios=ev['failed_scenarios']))
with (OUT/'NAVSIM_detailed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'NAVSIM_detailed_metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
note='PDMS及六个PDM分项均乘100展示，越高越好；验证ADE/FDE单位m，越低越好。ADE/FDE来自所选权重的内部验证集，PDM来自固定200个navtest场景。训练耗时不含编码、下载、缓存加载和测试评分。DDC按当前原生scorer配置记录，不一定参与最终PDMS加权；PDMS是逐场景计算后取均值，不能用这些分项均值直接还原。'
headers=list(rows[0]);md='# NAVSIM 完整指标对比\n\n'+note+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
for r in rows:md+='| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r.values())+' |\n'
md+='\n模型输入与融合细节见 NAVSIM_model_comparison.png。ReCogDrive-TD为前视VLM，与三视角DINO各组输入不同。无flow MFAD若无最终评分则未纳入。\n'
(OUT/'NAVSIM_detailed_metrics.md').write_text(md,encoding='utf-8')
fig,axes=plt.subplots(2,4,figsize=(20,10),facecolor='white');colors=['#277DA8','#5276B8','#087F79','#8E6AB2','#BA8437'] if len(rows)==5 else ['#277DA8','#5276B8','#087F79','#BA8437']
for ax,name in zip(axes.flat,metrics):
 vals=[r[name] for r in rows];bars=ax.bar(range(len(rows)),vals,color=colors,width=.6,zorder=3)
 ax.set_title(name,loc='left',fontsize=14,pad=14);ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_xticks(range(len(rows)),[r['model'].replace(' ','\n',1) for r in rows],fontsize=8);ax.grid(axis='y',alpha=.18,zorder=0);ax.spines[['top','right']].set_visible(False)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v:.2f}',ha='center',fontsize=11,weight='bold')
ax=axes.flat[-1];ax.axis('off');ax.text(0,1,'训练 / 评测说明',va='top',fontsize=15,weight='bold',color='#087F79');ax.text(0,.86,'同一 8,688 训练 / 1,930 验证\n200 epoch · seed42 · 有效batch128\n最低验证ADE选择权重\n固定200场景，各版200/200有效\n\n分项均按百分制展示（↑）\nADE / FDE及耗时见配套CSV\n\n无flow MFAD已完成训练与评测\n三视角ReCogDrive-TD已取消',va='top',fontsize=12,linespacing=1.7)
fig.suptitle('NAVSIM Demo：PDMS 与六个评分分项',fontsize=24,x=.06,ha='left',color='#20334D',weight='bold')
fig.text(.06,.025,'ReCogDrive-TD使用前视VLM，其余使用三视角DINOv3。分项为逐场景均值，不能直接按均值复算PDMS。',fontsize=11,color='#52657A')
fig.subplots_adjust(left=.055,right=.98,bottom=.12,top=.88,wspace=.26,hspace=.55)
for ext in ['png','svg']:fig.savefig(OUT/('NAVSIM_detailed_metrics.'+ext),dpi=180)
print(json.dumps(rows,ensure_ascii=False,indent=2))
