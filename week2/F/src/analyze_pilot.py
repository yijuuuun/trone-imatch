"""Summarize immutable real GPU records. Independent metadata metrics, not team scores."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path
import cv2
import numpy as np
from loma_backend import ROOT
from pilot_matrix import CONDITIONS

def csv_new(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def summary(name,records):
    rows=[r['metrics'] for r in records];n=len(rows);lat=[r['e2e_ms'] for r in records]
    errors=[r['error_m'] for r in rows if r['error_m'] is not None]
    large=sum(r['large_error'] is True for r in rows);detected=sum(r['large_error_detected'] is True for r in rows)
    c=records[0]['condition']
    return dict(condition=name,precision=c['precision'],headings=c['headings'],input_scale=c['scale'],
        lowlight=c['light'],enhancement=c['enhance'],n=n,cache=False,
        mean_latency_ms=float(np.mean(lat)),median_latency_ms=float(np.median(lat)),p95_latency_ms=float(np.percentile(lat,95)),
        fps=1000/float(np.mean(lat)),peak_allocated_mib=max(r['peak_allocated_mib'] for r in records),
        peak_reserved_mib=max(r['peak_reserved_mib'] for r in records),
        mean_matches=mean([r['matches'] for r in rows]),median_matches=float(np.median([r['matches'] for r in rows])),
        mean_inliers=mean([r['inliers'] for r in rows]),median_inliers=float(np.median([r['inliers'] for r in rows])),
        homography_success_pct=100*sum(r['valid_homography'] for r in rows)/n,
        homography_failure_pct=100*sum(not r['valid_homography'] for r in rows)/n,
        rejection_pct=100*sum(r['rejected'] for r in rows)/n,
        success_10m_pct=100*sum(r['success_10m'] for r in rows)/n,
        raw_success_10m_pct=100*sum(r['raw_success_10m'] for r in rows)/n,
        success_count=sum(r['success_10m'] for r in rows),
        mean_error_m=mean(errors),median_error_m=float(np.median(errors)) if errors else None,
        error_defined_n=len(errors),no_pose_n=n-len(errors),
        large_error_n=large,detected_large_error_n=detected,
        large_error_recall_pct=100*detected/large if large else None,
        mean_haversine_error_m=mean([r['haversine_error_m'] for r in rows]),
        metric_scope='independent Oracle T1 metadata scoring; not team official')

def full_h(record):
    c=record['selected'];H=c['homography']
    if H is None or c['center'] is None:return None
    A=np.vstack([cv2.getRotationMatrix2D((128.,128.),c['angle'],1.),[0,0,1]])
    return np.asarray(H)@A

def original_points(record,folder):
    angle=record['selected']['angle']
    with np.load(folder/record['correspondences']) as z:
        p=z[f'a{angle}_uav'];q=z[f'a{angle}_satellite']
    inv=cv2.invertAffineTransform(cv2.getRotationMatrix2D((128.,128.),angle,1.))
    return p@inv[:,:2].T+inv[:,2],q

def correspondence_overlap(a,b,fa,fb):
    p,q=original_points(a,fa);u,v=original_points(b,fb)
    if len(p)==0 or len(u)==0:return 0.,0.,0.
    # Mutual nearest joint correspondence; <=2px in BOTH original images.
    dp=((p[:,None]-u[None])**2).sum(2)
    dq=((q[:,None]-v[None])**2).sum(2)
    cost=dp+dq;nearest=cost.argmin(1);reverse=cost.argmin(0)
    i=np.arange(len(p));good=(reverse[nearest]==i)&(dp[i,nearest]<=4)&(dq[i,nearest]<=4)
    count=int(good.sum());union=len(p)+len(u)-count
    return count/len(p),count/len(u),count/union if union else 0.

def compare(name,reference,changed,run):
    paired=[]
    for a,b in zip(reference,changed):
        am,bm=a['metrics'],b['metrics'];assert am['k']==bm['k'];k=am['k']
        Ha,Hb=full_h(a),full_h(b);rms=None;delta=None
        if Ha is not None and Hb is not None:
            probes=np.array([[[0,0],[255,0],[255,255],[0,255],[128,128]]],dtype=np.float64)
            pa=cv2.perspectiveTransform(probes,Ha);pb=cv2.perspectiveTransform(probes,Hb)
            if np.isfinite(pa).all() and np.isfinite(pb).all():rms=float(np.sqrt(((pa-pb)**2).sum(2).mean()))
            delta=float(np.linalg.norm(np.asarray(a['selected']['center'])-b['selected']['center'])*.25)
        overlap=correspondence_overlap(a,b,run/a['condition']['id'],run/b['condition']['id'])
        paired.append(dict(comparison=name,k=k,scene=am['scene'],reference=a['condition']['id'],changed=b['condition']['id'],
             same_angle=am['selected_angle']==bm['selected_angle'],
             reference_angle=am['selected_angle'],changed_angle=bm['selected_angle'],
             homography_validity_changed=am['valid_homography']!=bm['valid_homography'],
             rejection_changed=am['rejected']!=bm['rejected'],
             matches_delta=bm['matches']-am['matches'],inliers_delta=bm['inliers']-am['inliers'],
             center_shift_m=delta,homography_probe_rms_px=rms,
             correspondence_reference_retained_pct=100*overlap[0],correspondence_changed_retained_pct=100*overlap[1],
             correspondence_jaccard_2px_pct=100*overlap[2],
             success_delta=int(bm['success_10m'])-int(am['success_10m']),
             reference_error_m=am['error_m'],changed_error_m=bm['error_m'],
             latency_ratio_reference_over_changed=a['e2e_ms']/b['e2e_ms']))
    # Scene-cluster paired bootstrap reflects repeated satellite scenes; exploratory Pilot only.
    scenes=sorted({p['scene'] for p in paired});clusters=[[p['success_delta'] for p in paired if p['scene']==scene] for scene in scenes]
    rng=np.random.default_rng(20261110);boot=[]
    for _ in range(5000):
        values=[v for i in rng.integers(0,len(clusters),len(clusters)) for v in clusters[i]]
        boot.append(np.mean(values)*100)
    out=dict(comparison=name,n=len(paired),scene_n=len(scenes),
        same_angle_pct=100*sum(p['same_angle'] for p in paired)/len(paired),
        homography_validity_changed_n=sum(p['homography_validity_changed'] for p in paired),
        rejection_changed_n=sum(p['rejection_changed'] for p in paired),
        mean_matches_delta=mean([p['matches_delta'] for p in paired]),mean_inliers_delta=mean([p['inliers_delta'] for p in paired]),
        mean_center_shift_m=mean([p['center_shift_m'] for p in paired]),
        max_center_shift_m=max((p['center_shift_m'] for p in paired if p['center_shift_m'] is not None),default=None),
        mean_homography_probe_rms_px=mean([p['homography_probe_rms_px'] for p in paired]),
        mean_correspondence_jaccard_2px_pct=mean([p['correspondence_jaccard_2px_pct'] for p in paired]),
        success_delta_pp=100*mean([p['success_delta'] for p in paired]),
        success_delta_scene_bootstrap_ci95_low_pp=float(np.percentile(boot,2.5)),
        success_delta_scene_bootstrap_ci95_high_pp=float(np.percentile(boot,97.5)),
        successful_to_failed_n=sum(p['success_delta']==-1 for p in paired),
        failed_to_successful_n=sum(p['success_delta']==1 for p in paired),
        speedup=float(np.mean([r['e2e_ms'] for r in reference])/np.mean([r['e2e_ms'] for r in changed])))
    return paired,out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=20)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    global CONDITIONS
    CONDITIONS=[c for c in CONDITIONS if c['suite'].startswith('F')]
    data={};summaries=[];sample_rows=[];profiles=[]
    for c in CONDITIONS:
        folder=args.run/c['id'];paths=[folder/f'k{k:03d}.json' for k in range(args.limit)]
        if not all(path.exists() for path in paths):raise RuntimeError('Incomplete condition: '+c['id'])
        records=[json.loads(path.read_text()) for path in paths];data[c['id']]=records
        summaries.append(summary(c['id'],records))
        for r in records:
            sample_rows.append(dict(condition=c['id'],precision=c['precision'],headings=c['headings'],input_scale=c['scale'],
                lowlight=c['light'],enhancement=c['enhance'],**r['metrics'],
                e2e_ms=r['e2e_ms'],preprocess_ms=r['preprocess_ms'],feature_gpu_ms=r['feature_gpu_ms'],
                matching_gpu_ms=r['matching_gpu_ms'],geometry_ms=r['geometry_ms'],
                peak_allocated_mib=r['peak_allocated_mib'],peak_reserved_mib=r['peak_reserved_mib']))
        for k in range(5):
            q=folder/f'profile{k:03d}.json'
            if not q.exists():raise RuntimeError('Missing separate profile: '+str(q))
            profiles.append(dict(condition=c['id'],k=k,**json.loads(q.read_text())))
    comparisons=[('fp16_vs_fp32_8','F0','F1'),('fp16_vs_fp32_4','F2','F3'),
        ('4_vs_8_fp32','F0','F2'),('4_vs_8_fp16','F1','F3'),
        ('scale075_vs_scale1','F1','F_scale075'),('scale050_vs_scale1','F1','F_scale050'),
        ('mild_vs_day','F1','E_mild'),('severe_vs_day','F1','E_severe'),
        ('gamma_vs_mild','E_mild','E_mild_gamma'),('clahe_vs_mild','E_mild','E_mild_clahe'),
        ('gamma_vs_severe','E_severe','E_severe_gamma'),('clahe_vs_severe','E_severe','E_severe_clahe')]
    pairs=[];paired_summaries=[]
    for name,ref,changed in comparisons:
        if ref not in data or changed not in data:continue
        records,s=compare(name,data[ref],data[changed],args.run);pairs.extend(records);paired_summaries.append(s)
    csv_new(args.out/'samples.csv',sample_rows);csv_new(args.out/'summary.csv',summaries)
    csv_new(args.out/'paired_samples.csv',pairs);csv_new(args.out/'paired_summary.csv',paired_summaries)
    csv_new(args.out/'latency_breakdown.csv',profiles)
    dump={'conditions':summaries,'paired':paired_summaries,'profile_summary':[
        dict(condition=c['id'],n=5,**{key:mean([r[key] for r in profiles if r['condition']==c['id']]) for key in ['feature_wall_ms','transformer_wall_ms','magsac_wall_ms','profile_e2e_ms']}) for c in CONDITIONS]}
    with (args.out/'summary.json').open('x') as f:json.dump(dump,f,indent=2,allow_nan=False)
    print(json.dumps(dump,indent=2))

if __name__=='__main__':main()
