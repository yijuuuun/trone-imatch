"""Verify archived Kaggle text artifacts without ML dependencies or a GPU."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

BASE=Path(__file__).resolve().parents[1]/'results/kaggle_2026-11-14'
RUN=BASE/'run_20261114T034717155134Z'

def rows(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def verify():
    inventory=json.loads((BASE/'archive_manifest.json').read_text(encoding='utf-8'))
    for name,expected in inventory['preserved_files'].items():
        actual=hashlib.sha256((RUN/name).read_bytes()).hexdigest()
        assert actual==expected,f'Changed archive artifact: {name}'
    manifest=rows(RUN/'manifest.csv')
    ids={r['sample_id'] for r in manifest}
    assert len(ids)==len(manifest)==1738
    gallery={r['gallery_id'] for r in rows(RUN/'gallery.csv')}
    comparison={r['model']:r for r in rows(RUN/'comparison.csv')}
    for model in ('Bearing-UAV','MEAN','Sample4Geo'):
        data=rows(RUN/model/'full/predictions.csv')
        summary=json.loads((RUN/model/'full/summary.json').read_text(encoding='utf-8'))
        assert {(r['sample_id'],r['repeat']) for r in data}=={(i,str(j)) for i in ids for j in range(3)}
        assert len(data)==len(ids)*3
        times=sorted(float(r['seconds']) for r in data)
        assert all(math.isfinite(t) and t>0 for t in times)
        def percentile(q):
            k=(len(times)-1)*q; i=int(k); j=min(i+1,len(times)-1)
            return (times[i]+(times[j]-times[i])*(k-i))*1000
        metrics=dict(mean_ms=statistics.mean(times)*1000,p50_ms=percentile(.5),p95_ms=percentile(.95),fps=1/statistics.mean(times))
        if model=='Bearing-UAV':
            for threshold in (10,20,32):
                metrics[f'accuracy_{threshold}m_pct']=sum(float(r['error_m'])<threshold for r in data if r['repeat']=='0')/len(ids)*100
        else:
            assert all(r['gallery_id'] in gallery for r in data)
            assert all(summary[f'accuracy_{t}m_pct'] is None for t in (10,20,32))
        for key,value in metrics.items():
            assert math.isclose(value,float(summary[key]),rel_tol=1e-9,abs_tol=1e-8),(model,key,'summary')
            assert math.isclose(value,float(comparison[model][key]),rel_tol=1e-9,abs_tol=1e-8),(model,key,'comparison')
        print(model,': 1738 IDs, 5214 rows, metrics verified')
    assert comparison['iMatch']['status']=='not_completed'
    print('All preserved archive hashes verified; iMatch remains not_completed.')

if __name__=='__main__': verify()
