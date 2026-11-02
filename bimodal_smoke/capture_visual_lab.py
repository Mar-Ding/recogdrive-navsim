import json,warnings,time
from pathlib import Path
warnings.filterwarnings('ignore')
import paramiko
HERE=Path(__file__).resolve().parent;DEST=HERE/'evidence/visual_lab_migration';BASE='/data/ruiqiwang/recogdrive_baseline/visual_vla'
cfg=dict(l.split(':',1) for l in Path('login.md').read_text().splitlines() if ':' in l)
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=25);s=c.open_sftp();summary={}
for name in ('migration_sources.json','outputs/shared/protocol.json','outputs/migration_smoke.json','outputs/pdm_integration_smoke.json','outputs/pipeline_status.json'):
 s.get(BASE+'/'+name,str(DEST/name.replace('/','_')))
for variant in ('turbo_cross_vla','multiview_mf_vla'):
 for name in ('config.json','history.json','status.json'):
  s.get(BASE+'/outputs/'+variant+'/'+name,str(DEST/(variant+'_'+name)))
 hist=json.loads((DEST/(variant+'_history.json')).read_text());rates=[r['seconds'] for r in hist[-5:] if 'seconds' in r];last=hist[-1]
 summary[variant]=dict(completed_epochs=last['epoch'],initial_ade=hist[0]['ade'],current_ade=last['ade'],seconds_per_epoch=sum(rates)/len(rates),projected_total_training_minutes=sum(rates)/len(rates)*200/60,remaining_training_minutes=sum(rates)/len(rates)*(200-last['epoch'])/60)
(DEST/'startup_summary.json').write_text(json.dumps(summary,indent=2))
with s.open(BASE+'/RUN_LOG.md','a') as f:f.write('\n## 迁移启动实测\n\n模型前向/反向、三路 flow 梯度检查通过；真实三相机 + BERT + MF-VLA 到原生 PDM scorer 接口检查通过。共享缓存已完成。两个模型均已正式训练，batch128，有效 batch128。以下时间是前几轮速率外推，评分另计：\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
s.put(str(HERE/'evidence/recogdrive_dd_final/RESULTS.md'),'/data/ruiqiwang/recogdrive_baseline/demo10/recogdrive_dd/RESULTS_RECOVERED.md')
print(json.dumps(summary,indent=2));s.close();c.close()
