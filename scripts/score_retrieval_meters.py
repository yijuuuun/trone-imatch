"""Post-hoc retrieval-center errors. Requires geographiclib==2.1; no GPU/inference.

Coordinates are recovered from original export_visloc samples, not inferred from
retrieval predictions. Grid scale fit is checked against 64 m exporter spacing.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from geographiclib.geodesic import Geodesic

BASE=Path(__file__).resolve().parents[1]/'results/kaggle_2026-11-14'
RUN=BASE/'run_20261114T034717155134Z'
OUT=BASE/'meter_accuracy_2026-11-15'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def scales(lat):
    p=math.radians(lat)
    return 111412.84*math.cos(p)-93.5*math.cos(3*p),111132.92-559.82*math.cos(2*p)+1.175*math.cos(4*p)

def fit_axis(rows, coord, offset, sign, estimate):
    ds=[float(r[coord])-float(rows[0][coord]) for r in rows]
    os=[sign*32*(float(r[offset])-float(rows[0][offset])) for r in rows]
    steps=[round((d*estimate-o)/64) for d,o in zip(ds,os)]
    ts=[64*k+o for k,o in zip(steps,os)]
    denominator=sum(d*d for d in ds)
    if denominator<=0:raise ValueError('Insufficient variation to recover map scale')
    fitted=sum(d*t for d,t in zip(ds,ts))/denominator
    residual=max(abs(d*fitted-t) for d,t in zip(ds,ts))
    assert residual<.001,(coord,residual)
    assert abs(fitted/estimate-1)<.001,(coord,fitted,estimate)
    assert steps==[round((d*fitted-o)/64) for d,o in zip(ds,os)]
    return fitted,residual

def main():
    OUT.mkdir(exist_ok=True)
    original=read(OUT/'samples.csv');manifest=read(RUN/'manifest.csv')
    by_id={f"{r['set']}:{int(r['k'])}":r for r in original}
    assert len(by_id)==len(original)==len(manifest)==1738
    assert set(by_id)=={r['sample_id'] for r in manifest}
    for r in manifest:
        s=by_id[r['sample_id']]
        assert float(s['x_norm'])==float(r['gt_x']) and float(s['y_norm'])==float(r['gt_y'])
        assert (s['uav_w'],s['uav_h'],s['block_w'],s['block_h'])==('256','256','512','512')
        # This run is uncorrected; reject instead of silently ignoring corrections.
        assert float(s['along_m'])==float(s['right_m'])==0
        assert float(s['x_norm'])==float(s['x_norm_raw'])
        assert float(s['y_norm'])==float(s['y_norm_raw'])
        assert -90<=float(s['lat'])<=90 and -180<=float(s['lon'])<=180
    groups=defaultdict(list)
    for s in original:groups[s['flight']].append(s)
    fitted={};audits=[]
    for flight,rs in sorted(groups.items()):
        mx0,my0=scales(statistics.mean(float(r['lat']) for r in rs))
        mx,rx=fit_axis(rs,'lon','x_norm_raw',1,mx0)
        my,ry=fit_axis(rs,'lat','y_norm_raw',-1,my0)
        # Independently recover the common latitude reference from each axis.
        low,high=0.,60.
        for _ in range(60):
            mid=(low+high)/2
            if scales(mid)[0]>mx:low=mid
            else:high=mid
        lat_ref=(low+high)/2
        assert abs(scales(lat_ref)[1]-my)<.02
        fitted[flight]=(mx,my)
        audits.append(dict(flight=flight,n=len(rs),meters_per_degree_east=mx,meters_per_degree_north=my,
                           inferred_lat_ref=lat_ref,max_grid_residual_m=max(rx,ry)))
    queries={};centers={};coordinate_rows=[]
    for r in manifest:
        s=by_id[r['sample_id']];mx,my=fitted[s['flight']]
        lat,lon=float(s['lat']),float(s['lon'])
        # north-up map: positive normalized y points south.
        center_lat=lat+32*float(s['y_norm_raw'])/my
        center_lon=lon-32*float(s['x_norm_raw'])/mx
        entry=dict(sample_id=r['sample_id'],flight=s['flight'],query_lat=lat,query_lon=lon,
                   gallery_id=r['block_sha256'],gallery_lat=center_lat,gallery_lon=center_lon)
        queries[r['sample_id']]=entry
        if r['block_sha256'] in centers:
            prev=centers[r['block_sha256']]
            assert Geodesic.WGS84.Inverse(prev['gallery_lat'],prev['gallery_lon'],center_lat,center_lon)['s12']<.001
        centers[r['block_sha256']]=entry;coordinate_rows.append(entry)
    gallery=read(RUN/'gallery.csv')
    assert set(centers)=={r['gallery_id'] for r in gallery}
    assert all(r['gallery_id']==r['image_sha256'] for r in gallery)
    write(OUT/'coordinates.csv',coordinate_rows)
    write(OUT/'flight_coordinate_audit.csv',audits)
    summaries=[];checks={}
    for model in ('MEAN','Sample4Geo'):
        predictions=read(RUN/model/'full/predictions.csv')
        assert len(predictions)==5214
        assert {(r['sample_id'],int(r['repeat'])) for r in predictions}=={(i,j) for i in queries for j in range(3)}
        scored=[]
        for r in predictions:
            q=queries[r['sample_id']];g=centers[r['gallery_id']]
            error=Geodesic.WGS84.Inverse(q['query_lat'],q['query_lon'],g['gallery_lat'],g['gallery_lon'])['s12']
            same=q['flight']==g['flight']
            planar=None
            if same:
                mx,my=fitted[q['flight']]
                planar=math.hypot((g['gallery_lon']-q['query_lon'])*mx,(g['gallery_lat']-q['query_lat'])*my)
            scored.append(dict(**{k:v for k,v in r.items() if k!='error_m'},error_m=error,
                               same_flight=same,exporter_plane_error_m=planar,
                               paired_block_exact_match=r['gallery_id']==q['gallery_id']))
        write(OUT/(model+'_predictions.csv'),scored)
        first=[r for r in scored if r['repeat']=='0'];errors=[r['error_m'] for r in first]
        summary=dict(model=model,n=len(first),prediction='retrieved gallery block center',distance='WGS84 ellipsoidal geodesic',position_ranking_valid=False)
        for t in (10,20,32):
            summary[f'success_under_{t}m']=sum(e<t for e in errors)
            summary[f'accuracy_{t}m_pct']=100*sum(e<t for e in errors)/len(errors)
        summary.update(mean_error_m=statistics.mean(errors),median_error_m=statistics.median(errors),
                       same_flight_retrieval_pct=100*sum(r['same_flight'] for r in first)/len(first),
                       paired_block_exact_match_pct=100*sum(r['paired_block_exact_match'] for r in first)/len(first))
        summaries.append(summary)
        checks[model]={'planar_vs_geodesic_threshold_disagreements_same_flight':{str(t):sum((r['error_m']<t)!=(r['exporter_plane_error_m']<t) for r in first if r['same_flight']) for t in (10,20,32)},
                       'min_distance_to_threshold_m':{str(t):min(abs(e-t) for e in errors) for t in (10,20,32)},
                       'prediction_sha256':digest(RUN/model/'full/predictions.csv')}
    write(OUT/'retrieval_meter_summary.csv',summaries)
    metadata=dict(samples_sha256=digest(OUT/'samples.csv'),manifest_sha256=digest(RUN/'manifest.csv'),
                  samples_matched=1738,max_grid_residual_m=max(r['max_grid_residual_m'] for r in audits),
                  method='Grid-consistent flight scales inferred from GPS and raw normalized offsets; independent of retrieved IDs',
                  units='128 px/unit * 0.25 m/px = 32 m/unit; 256 px grid stride = 64 m',
                  corrections='all along_m/right_m are zero; raw and corrected normalized GT are equal',
                  checks=checks,summaries=summaries,position_ranking_valid=False)
    (OUT/'verification.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
