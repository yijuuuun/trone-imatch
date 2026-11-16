"""Recompute Bearing-UAV sequence results from archived CSVs (no model inference)."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE/'run'
OUT = HERE/'analysis'
REF = HERE.parent/'2026-11-14_visloc_nav'

def scored(frame):
    return frame[frame.k != frame.groupby('set').k.transform('min')].copy()

def stats(s, g):
    return dict(segments=len(g), steps=len(s), w10=round(float((s.err_m<=10).mean()*100),1),
        w20=round(float((s.err_m<=20).mean()*100),1), w32=round(float((s.err_m<=32).mean()*100),1),
        med_err_m=round(float(s.err_m.median()),1), final_med_m=round(float(g.final_err_m.median()),1),
        final_le40=round(float((g.final_err_m<=40).mean()*100),1),
        unknown=round(float(s.unknown.mean()*100),1),outside_block=round(float(s.outside_block.mean()*100),1))

def groups(s, key):
    rows=[]
    for name,g in s.groupby(key,observed=True):
        rows.append(dict(group=str(name),steps=len(g),w10=100*(g.err_m<=10).mean(),w20=100*(g.err_m<=20).mean(),
            w32=100*(g.err_m<=32).mean(),median_m=g.err_m.median(),mean_m=g.err_m.mean(),
            outside_pct=100*g.outside_block.mean()))
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True)
    for rel,expected in json.loads((RUN/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((RUN/'source'/rel).read_bytes()).hexdigest()==expected, rel
    full=pd.read_csv(RUN/'Bearing-UAV/full/steps.csv')
    seg=pd.read_csv(RUN/'Bearing-UAV/full/segments.csv')
    saved=json.loads((RUN/'Bearing-UAV/full/summary.json').read_text())
    audit=json.loads((RUN/'data_audit.json').read_text())
    assert not full.duplicated(['set','k']).any()
    assert len(full)==audit['frames'] and len(seg)==audit['segments']
    anchors=full[full.k==full.groupby('set').k.transform('min')]
    assert (anchors.err_m==0).all() and (anchors.sec==0).all()
    for axis in ['x','y']:
        assert np.allclose(anchors['est_'+axis],anchors['real_'+axis])
        expected=np.clip(np.round((full['guess_'+axis]-256)/256),0,14)
        assert np.array_equal(expected,full['block_'+axis]), axis
    assert np.allclose(full.err_m,np.hypot(full.est_x-full.real_x,full.est_y-full.real_y)/4)
    assert np.isfinite(full.select_dtypes('number')).all().all()
    s=scored(full); metrics=stats(s,seg)
    for key,value in metrics.items(): assert saved[key]==value,(key,saved[key],value)
    for st,g in s.groupby('set'):
        r=seg.set_index('set').loc[st]
        assert len(g)==r.n_steps
        assert np.isclose(g.iloc[-1].err_m,r.final_err_m)
        for threshold in [10,20,32]: assert np.isclose((g.err_m<=threshold).mean(),r[f'w{threshold}'])
    assert len(s)==1020 and len(seg)==92
    s['flight']=s['set'].str.extract(r'visloc_(\d+)_')[0]
    s['distance']=pd.cut(s.dist_m,[0,300,600,np.inf],labels=['0-300m','300-600m','600m+'],include_lowest=True)
    groups(s,'flight').to_csv(OUT/'per_flight.csv',index=False)
    groups(s,'distance').to_csv(OUT/'per_distance.csv',index=False)
    groups(s,'outside_block').to_csv(OUT/'by_block_coverage.csv',index=False)
    flight_end=seg.assign(flight=seg['set'].str.extract(r'visloc_(\d+)_')[0]).groupby('flight').agg(
        segments=('set','size'),final_median_m=('final_err_m','median'),final_le40=('final_err_m',lambda e:100*(e<=40).mean()))
    flight_end.to_csv(OUT/'per_flight_endpoints.csv')
    references=[]
    endpoint=seg[['set','final_err_m']].rename(columns={'final_err_m':'bearing_final_m'})
    for variant in ['dr','r0','r1','r4g0n5w20']:
        t=pd.read_csv(REF/f'visloc_loma-{variant}_steps.csv');g=pd.read_csv(REF/f'visloc_loma-{variant}_segs.csv')
        paired=full.merge(t,on=['set','k'],validate='one_to_one',suffixes=('_bearing','_ref'))
        assert len(paired)==len(full)==len(t)
        for c in ['real_x','real_y','dist_m']: assert np.allclose(paired[c+'_bearing'],paired[c+'_ref'])
        if variant=='dr':
            # Check the actual odometry increments, not just matching seed labels.
            for axis in ['x','y']:
                bearing_motion=full['guess_'+axis]-full.groupby('set')['est_'+axis].shift()
                reference_motion=t['guess_'+axis]-t.groupby('set')['est_'+axis].shift()
                moves=full[['set','k']].assign(bearing_motion=bearing_motion).merge(
                    t[['set','k']].assign(reference_motion=reference_motion),on=['set','k'])
                moves=moves.dropna()
                assert np.allclose(moves.bearing_motion,moves.reference_motion,atol=1e-8), axis
        a=t[t.k==t.groupby('set').k.transform('min')]
        references.append(dict(method=variant,**stats(scored(t),g),anchor_median_m=float(a.err_m.median()),
            anchor_max_m=float(a.err_m.max()),initialization='known start' if variant=='dr' else 'relocalized first frame'))
        endpoint=endpoint.merge(g[['set','final_err_m']].rename(columns={'final_err_m':variant+'_final_m'}),on='set',validate='one_to_one')
    pd.DataFrame([dict(method='bearing_r0',**metrics,anchor_median_m=0,anchor_max_m=0,initialization='known start')]+references).to_csv(OUT/'reference_comparison.csv',index=False)
    endpoint.to_csv(OUT/'paired_endpoints.csv',index=False)
    dr=endpoint.dr_final_m<=40; b=endpoint.bearing_final_m<=40
    won=int((b&~dr).sum());lost=int((~b&dr).sum());n=won+lost
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(won,lost)+1))/2**n) if n else 1.
    diagnostics=dict(**metrics,total_frames=len(full),start_frames=len(anchors),final_successes=int((seg.final_err_m<=40).sum()),
        mean_error_m=float(s.err_m.mean()),p90_error_m=float(s.err_m.quantile(.9)),max_error_m=float(s.err_m.max()),
        mean_inference_ms=float(s.sec.mean()*1000),p50_inference_ms=float(s.sec.median()*1000),p95_inference_ms=float(s.sec.quantile(.95)*1000),
        inference_sum_s=float(s.sec.sum()),wall_s=saved['wall_s'],
        correction_improves_guess_pct=float((s.err_m<s.guess_err_m).mean()*100),
        bearing_only_final_success=won,dr_only_final_success=lost,endpoint_mcnemar_exact_p=p,
        caveat='LoMa reference relocalizes its first frame; reference comparisons do not isolate model effects.')
    (OUT/'metrics.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf8')
    print(json.dumps(diagnostics,indent=2))
    for file in ['per_flight.csv','per_distance.csv','reference_comparison.csv']:
        print(file);print(pd.read_csv(OUT/file).to_string(index=False))
    print('PASS: source hashes, coverage, coordinates, predicted block selection, known starts, summaries, reference coordinate alignment')

if __name__=='__main__': main()
