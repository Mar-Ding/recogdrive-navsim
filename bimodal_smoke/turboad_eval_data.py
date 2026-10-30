"""Prepare official-navtest tokens with complete current sensors; never training data."""
import sys,json,pickle,tarfile,zipfile,argparse,subprocess
from pathlib import Path
BASE=Path('/data/ruiqiwang/recogdrive_baseline')
sys.path.append(str(BASE/'env/lib/python3.10/site-packages'))
import yaml
ROOT=BASE/'turboad';SENSORS=BASE/'data/sensor_blobs/test'

def maps():
    dest=BASE/'data/maps';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(BASE/'data/maps.zip') as z:
        for info in z.infolist():
            parts=Path(info.filename).parts
            if not parts:continue
            if parts[0]=='nuplan-maps-v1.0':parts=parts[1:]
            if not parts:continue
            target=(dest/Path(*parts)).resolve();assert target.is_relative_to(dest.resolve())
            if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.stat().st_size!=info.file_size:target.write_bytes(z.read(info))
    assert (dest/'nuplan-maps-v1.0.json').exists()
    print('MAPS_OK',str(dest),flush=True)

def prepare():
    cfg=yaml.safe_load((BASE/'repo/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml').read_text())
    allowed=set(cfg.get('tokens') or []);logs=set(cfg['log_names']);rows=[]
    for file in sorted((BASE/'data/openscene-v1.1/meta_datas/test').glob('*.pkl')):
        if file.stem not in logs:continue
        frames=pickle.load(file.open('rb'))
        for start in range(len(frames)-13):
            f=frames[start+3]
            if not len(f['roadblock_ids']) or (allowed and f['token'] not in allowed):continue
            rows.append(dict(token=f['token'],log_name=file.stem,image=f['cams']['CAM_F0']['data_path'],pcd=f['lidar_path']))
    (ROOT/'navtest_candidates.json').write_text(json.dumps(rows,indent=2));print('NAVTEST_CANDIDATES',len(rows),flush=True)
    by_path={r[k]:r for r in rows for k in ('image','pcd')}
    for kind in ('camera','lidar'):
        archive=BASE/'data'/('test_'+kind+'_0.tgz')
        url=f'https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_{kind}/openscene_sensor_test_{kind}_0.tgz?download=true'
        if not archive.exists():
            subprocess.run(['curl','-fL','--connect-timeout','20','--max-time','900',url,'-o',str(archive)+'.partial'],check=True,timeout=920)
            Path(str(archive)+'.partial').rename(archive)
        saved=0
        with tarfile.open(archive,'r|gz') as tar:
            for m in tar:
                if not m.isfile():continue
                parts=Path(m.name).parts
                paths=['/'.join(parts[i:]) for i in range(len(parts))]
                rel=next((x for x in paths if x in by_path),None)
                if rel is None:continue
                dest=(SENSORS/rel).resolve();assert dest.is_relative_to(SENSORS.resolve())
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(m).read());saved+=1
                if saved%100==0:print(kind,'saved',saved,flush=True)
    paired=[r for r in rows if all((SENSORS/r[k]).exists() for k in ('image','pcd'))]
    # Fixed deterministic subset: this is NOT the complete navtest benchmark.
    paired=sorted(paired,key=lambda r:r['token'])[:200]
    assert paired,'No official navtest paired tokens in shard 0'
    (ROOT/'navtest_subset_manifest.json').write_text(json.dumps(paired,indent=2))
    (ROOT/'navtest_subset_tokens.json').write_text(json.dumps([r['token'] for r in paired]))
    print('NAVTEST_SUBSET_READY',len(paired),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['maps','prepare']);a=p.parse_args();globals()[a.stage]()
