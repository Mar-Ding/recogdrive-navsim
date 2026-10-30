"""Compose upstream evaluation config and invoke its unmodified scoring entrypoint."""
import sys,os,argparse,json
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
os.environ['NUPLAN_MAPS_ROOT']=str(BASE/'data/maps')
os.environ['NUPLAN_MAP_VERSION']='nuplan-maps-v1.0'
os.environ['NAVSIM_EXP_ROOT']=str(BASE/'turboad/evaluation')
os.environ['OPENSCENE_DATA_ROOT']=str(BASE/'data')
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','score']);p.add_argument('--tokens',required=True);p.add_argument('--checkpoint',default=str(BASE/'turboad/pilot_seed42/best_val_ade.pt'));p.add_argument('--label',choices=['initial','pilot','expanded','mf_initial','mf_expanded','plus_initial','plus'],default='pilot');a=p.parse_args()
    tokens=json.loads(Path(a.tokens).read_text());assert tokens and len(tokens)==len(set(tokens))
    kind='metric_caching' if a.mode=='cache' else 'pdm_scoring'
    name='default_metric_caching' if a.mode=='cache' else 'default_run_pdm_score'
    config_dir=BASE/'repo/navsim/planning/script/config'/kind
    with initialize_config_dir(config_dir=str(config_dir),version_base=None):
        cfg=compose(config_name=name,overrides=['train_test_split=navtest','worker=sequential'])
    OmegaConf.set_struct(cfg,False)
    cfg.navsim_log_path=str(BASE/'data/openscene-v1.1/meta_datas/test')
    cfg.train_test_split.scene_filter.tokens=tokens
    cfg.output_dir=str(BASE/'turboad/evaluation'/('cache_metadata' if a.mode=='cache' else 'scores_'+a.label))
    cfg.experiment_name='turboad_navtest_subset'
    Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    if a.mode=='cache':
        cfg.cache.cache_path=str(BASE/'turboad/evaluation/metric_cache')
        from navsim.planning.script.run_metric_caching import main as entry
    else:
        cfg.sensor_blobs_path=str(BASE/'data/sensor_blobs/test')
        cfg.metric_cache_path=str(BASE/'turboad/evaluation/metric_cache')
        cfg.agent=OmegaConf.create(dict(_target_='turboad_agent.TurboADAgent',checkpoint_path=a.checkpoint))
        from navsim.common.dataloader import MetricCacheLoader
        cache=MetricCacheLoader(Path(cfg.metric_cache_path))
        missing=set(tokens)-set(cache.tokens)
        assert not missing,f'Missing metric cache for {len(missing)} requested tokens'
        from navsim.planning.script.run_pdm_score import main as entry
    OmegaConf.save(cfg,Path(cfg.output_dir)/'turboad_resolved_config.yaml',resolve=True)
    entry.__wrapped__(cfg)
if __name__=='__main__':main()
