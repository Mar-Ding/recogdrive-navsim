import csv,json
from pathlib import Path
root=Path('bimodal_smoke/evidence/turboad')
tokens=set(json.loads((root/'navtest_subset_tokens.json').read_text()))
report={}
for label in ('initial','pilot','expanded','mf_initial','mf_expanded','plus_initial','plus','mf_long','plus_long'):
    rows=list(csv.DictReader((root/f'score_{label}.csv').open()))
    scenes=[r for r in rows if r['token']!='average']
    assert len(scenes)==200 and {r['token'] for r in scenes}==tokens
    assert all(r['valid']=='True' for r in scenes)
    avg=next(r for r in rows if r['token']=='average')
    report[label]={'scenarios':len(scenes),'failed':0,'average':avg}
for label in ('pilot','expanded'):
    manifest=json.loads((root/label/'manifest.json').read_text())
    assert not tokens & {r['token'] for r in manifest}
    train={r['log_name'] for r in manifest if r['split']=='train'}
    val={r['log_name'] for r in manifest if r['split']=='val'}
    assert not train & val
(root/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
