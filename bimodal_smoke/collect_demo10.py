"""Collect lightweight evidence without checkpoints or credentials in output."""
import json
from pathlib import Path
import paramiko
cfg=dict(line.split(':',1) for line in Path('login.md').read_text().splitlines() if ':' in line)
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect(cfg['ip'],port=int(cfg['port']),username=cfg['user'],password=cfg['passwd'],timeout=30)
BASE='/data/ruiqiwang/recogdrive_baseline';evidence=Path('bimodal_smoke/evidence')
try:
    with client.open_sftp() as sftp:
        def get(remote,local):
            try:sftp.stat(remote)
            except FileNotFoundError:return
            local.parent.mkdir(parents=True,exist_ok=True);sftp.get(remote,str(local))
        for label in ('mf_long','plus_long'):
            for name in ('summary.json','history.json'):get(f'{BASE}/turboad/{label}_seed42/{name}',evidence/'turboad'/label/name)
            folder=f'{BASE}/turboad/evaluation/scores_{label}'
            for name in sftp.listdir(folder):
                if name.endswith('.csv'):get(folder+'/'+name,evidence/'turboad'/f'score_{label}.csv')
                if name.endswith('_config.yaml'):get(folder+'/'+name,evidence/'turboad'/label/'score_config.yaml')
        for name in ('data_audit.json','pipeline_status.json','download_budget.json','benchmark_cross.json','benchmark_mf.json','features_complete.json','sensors_complete.json','comparison.json'):
            get(f'{BASE}/demo10/{name}',evidence/'demo10'/name)
        for label in ('smoke_turboad','smoke_turbo_mfad','turboad','turbo_mfad'):
            for name in ('summary.json','history.json','config.json','progress.json','evaluation.json'):get(f'{BASE}/demo10/{label}/{name}',evidence/'demo10'/label/name)
        for label in ('cross_initial','cross','mf_initial','mf'):get(f'{BASE}/demo10/demo10_{label}.csv',evidence/'demo10'/f'score_{label}.csv')
        for name,path in {
            'official_il_2b.sh':'scripts/training/run_recogdrive_train_multi_node_2b.sh',
            'official_accelerate_il_2b.sh':'scripts/training_accelerate/run_recogdrive_train_multi_node_2b.sh',
            'official_default_training.yaml':'navsim/planning/script/config/training/default_training.yaml',
            'official_optimizer_agent.py':'navsim/agents/recogdrive/recogdrive_agent.py',
            'official_lr_scheduler.py':'navsim/agents/recogdrive/utils/lr_scheduler.py',
        }.items():get(f'{BASE}/repo/{path}',evidence/'demo10'/'official_budget'/name)
    print('Collected available completed results and current demo10 status; no checkpoints downloaded.')
finally:client.close()
