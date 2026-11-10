"""Summarize measured real-night frame matching and create five local examples."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from loma_backend import ROOT,digest
from lowlight import enhance

OUT=ROOT/'experiments/2026-11-13_lowlight/nat2021_frame_matching_v1'

def csv_new(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(method,records):
    n=len(records);lat=np.array([r['mean_latency_ms'] for r in records]);ratios=[r['inlier_ratio'] for r in records if r['inlier_ratio'] is not None]
    overlaps=[r['overlap_H_proxy'] for r in records if r['overlap_H_proxy'] is not None]
    return dict(method=method,n_pairs=n,n_sequences=len({r['sequence'] for r in records}),
        mean_matches=float(np.mean([r['matches'] for r in records])),median_matches=float(np.median([r['matches'] for r in records])),
        mean_inliers=float(np.mean([r['inliers'] for r in records])),median_inliers=float(np.median([r['inliers'] for r in records])),
        mean_pair_inlier_ratio_pct=100*float(np.mean(ratios)) if ratios else None,
        pooled_inlier_ratio_pct=100*sum(r['inliers'] for r in records)/sum(r['matches'] for r in records) if sum(r['matches'] for r in records) else None,
        valid_homography_pct=100*sum(r['valid_homography'] for r in records)/n,
        geometry_pass_pct=100*sum(r['geometry_pass'] for r in records)/n,
        low_inlier_or_invalid_H_n=sum(not r['geometry_pass'] for r in records),
        mean_latency_ms=float(lat.mean()),median_latency_ms=float(np.median(lat)),p95_latency_ms=float(np.percentile(lat,95)),fps=1000/float(lat.mean()),
        peak_allocated_mib=max(r['peak_allocated_mib'] for r in records),
        mean_overlap_H_proxy=float(np.mean(overlaps)) if overlaps else None,min_overlap_H_proxy=min(overlaps) if overlaps else None,
        overlap_proxy_ge080_n=sum(v>=.8 for v in overlaps),overlap_proxy_defined_n=len(overlaps),
        mean_native_pair_Y255=float(np.mean([r['original_pair_mean_Y255'] for r in records])),
        mean_processed_pair_Y255=float(np.mean([r['enhanced_pair_mean_Y255'] for r in records])),
        preprocess_ms=float(np.mean([r['preprocess_ms'] for r in records])),feature_ms=float(np.mean([r['feature_ms'] for r in records])),
        matching_ms=float(np.mean([r['matching_ms'] for r in records])),magsac_ms=float(np.mean([r['magsac_ms'] for r in records])),
        Success_at_10m='',scope='NAT2021 native temporal consistency; no correspondence/position GT')

def example(pair,records,examples):
    a=cv2.cvtColor(cv2.imread(str(ROOT/pair['a_path'])),cv2.COLOR_BGR2RGB)
    b=cv2.cvtColor(cv2.imread(str(ROOT/pair['b_path'])),cv2.COLOR_BGR2RGB)
    panels=[]
    for method in ['none','gamma','clahe']:
        r=records[method][pair['pair_id']];x=enhance(a,method);y=enhance(b,method)
        panel=np.hstack([cv2.cvtColor(x,cv2.COLOR_RGB2BGR),cv2.cvtColor(y,cv2.COLOR_RGB2BGR)])
        with np.load(OUT/method/r['correspondence_file']) as z:p,q,mask=z['p'],z['q'],z['mask'].reshape(-1).astype(bool)
        for good,color,limit in [(True,(0,230,0),60),(False,(0,0,255),20)]:
            ids=np.flatnonzero(mask==good)
            if len(ids)>limit:ids=ids[np.linspace(0,len(ids)-1,limit,dtype=int)]
            for i in ids:
                left=tuple(np.rint(p[i]).astype(int));right=tuple(np.rint(q[i]+[a.shape[1],0]).astype(int))
                cv2.line(panel,left,right,color,1,cv2.LINE_AA);cv2.circle(panel,left,3,color,1);cv2.circle(panel,right,3,color,1)
        panel=cv2.resize(panel,(1280,360),interpolation=cv2.INTER_AREA)
        header=np.full((32,1280,3),245,np.uint8)
        label=f"{pair['pair_id']} {method}: matches={r['matches']}, inliers={r['inliers']}, ratio={100*r['inlier_ratio']:.1f}%, H={r['valid_homography']}"
        cv2.putText(header,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1,cv2.LINE_AA)
        panels.append(np.vstack([header,panel]))
    path=examples/(pair['pair_id']+'_comparison.png')
    if path.exists():raise FileExistsError(path)
    assert cv2.imwrite(str(path),np.vstack(panels))
    raw=np.hstack([cv2.cvtColor(a,cv2.COLOR_RGB2BGR),cv2.cvtColor(b,cv2.COLOR_RGB2BGR)])
    assert cv2.imwrite(str(examples/(pair['pair_id']+'_original_pair.png')),raw)
    return path.name

if __name__=='__main__':
    manifest=json.loads((OUT/'PAIR_MANIFEST.json').read_text());lookup={p['pair_id']:p for p in manifest['pairs']}
    methods=['none','gamma','clahe'];data={};summaries=[];samples=[]
    for method in methods:
        records=[json.loads((OUT/method/(p['pair_id']+'.json')).read_text()) for p in manifest['pairs']]
        data[method]={r['pair_id']:r for r in records};summaries.append(aggregate(method,records))
        for r in records:samples.append({k:v for k,v in r.items() if k!='homography'})
    csv_new(OUT/'samples.csv',samples);csv_new(OUT/'matching_summary.csv',summaries)
    native=[data['none'][p['pair_id']]['original_pair_mean_Y255'] for p in manifest['pairs']]
    q1,q2=np.quantile(native,[1/3,2/3])
    bins={p['pair_id']:('dark' if data['none'][p['pair_id']]['original_pair_mean_Y255']<=q1 else 'middle' if data['none'][p['pair_id']]['original_pair_mean_Y255']<=q2 else 'bright') for p in manifest['pairs']}
    stratified=[]
    for method in methods:
        for group in ['dark','middle','bright']:
            rows=[r for key,r in data[method].items() if bins[key]==group]
            stratified.append(dict(brightness_bin=group,bin_thresholds_original_Y255=f'{q1:.8f};{q2:.8f}',**aggregate(method,rows)))
    csv_new(OUT/'brightness_groups.csv',stratified)
    paired=[];paired_summary=[]
    for method in ['gamma','clahe']:
        rows=[]
        for key,a in data['none'].items():
            b=data[method][key]
            row=dict(method=method,pair_id=key,sequence=a['sequence'],brightness_bin=bins[key],
                matches_delta=b['matches']-a['matches'],inliers_delta=b['inliers']-a['inliers'],
                inlier_ratio_delta_pp=100*(b['inlier_ratio']-a['inlier_ratio']) if a['inlier_ratio'] is not None and b['inlier_ratio'] is not None else None,
                geometry_pass_before=a['geometry_pass'],geometry_pass_after=b['geometry_pass'],
                valid_H_before=a['valid_homography'],valid_H_after=b['valid_homography'],
                latency_delta_ms=b['mean_latency_ms']-a['mean_latency_ms'])
            paired.append(row);rows.append(row)
        seq=sorted({r['sequence'] for r in rows});groups=[[r['inliers_delta'] for r in rows if r['sequence']==s] for s in seq]
        rng=np.random.default_rng(20261110);boot=[]
        for _ in range(5000):boot.append(np.mean([v for i in rng.integers(0,len(groups),len(groups)) for v in groups[i]]))
        paired_summary.append(dict(method=method,n_pairs=40,n_sequences=20,
            mean_matches_delta=float(np.mean([r['matches_delta'] for r in rows])),
            mean_inliers_delta=float(np.mean([r['inliers_delta'] for r in rows])),
            inliers_delta_sequence_bootstrap_ci95_low=float(np.percentile(boot,2.5)),inliers_delta_sequence_bootstrap_ci95_high=float(np.percentile(boot,97.5)),
            inliers_improved_n=sum(r['inliers_delta']>0 for r in rows),inliers_worsened_n=sum(r['inliers_delta']<0 for r in rows),
            pass_to_fail_n=sum(r['geometry_pass_before'] and not r['geometry_pass_after'] for r in rows),
            fail_to_pass_n=sum(not r['geometry_pass_before'] and r['geometry_pass_after'] for r in rows)))
    csv_new(OUT/'paired_samples.csv',paired);csv_new(OUT/'paired_summary.csv',paired_summary)
    # Sequence-level rank association avoids treating the two frames as independent sequences.
    sequence=[]
    for seq in manifest['pairs'][::2]:
        rows=[r for r in data['none'].values() if r['sequence']==seq['sequence']]
        sequence.append(dict(sequence=seq['sequence'],mean_Y255=float(np.mean([r['original_pair_mean_Y255'] for r in rows])),
            mean_inliers=float(np.mean([r['inliers'] for r in rows])),mean_inlier_ratio=float(np.mean([r['inlier_ratio'] for r in rows]))))
    def ranks(x):
        x=np.array(x);return np.array([(np.sum(x<v)+.5*(np.sum(x==v)-1)) for v in x])
    rank_r=float(np.corrcoef(ranks([r['mean_Y255'] for r in sequence]),ranks([r['mean_inliers'] for r in sequence]))[0,1])
    csv_new(OUT/'sequence_summary.csv',sequence)
    # Five different sequences chosen by original-brightness quintile representatives.
    sorted_seq=sorted(sequence,key=lambda r:r['mean_Y255'])
    chosen=[sorted_seq[round(i*(len(sorted_seq)-1)/4)]['sequence'] for i in range(5)]
    examples=OUT/'visual_examples';examples.mkdir(exist_ok=False)
    examples_rows=[]
    for name in chosen:
        pair=next(p for p in manifest['pairs'] if p['sequence']==name and p['anchor']=='first')
        image=example(pair,data,examples)
        r=data['none'][pair['pair_id']]
        examples_rows.append(dict(pair_id=pair['pair_id'],sequence=name,original_Y255=r['original_pair_mean_Y255'],
            matches=r['matches'],inliers=r['inliers'],inlier_ratio=r['inlier_ratio'],geometry_pass=r['geometry_pass'],image=image))
    csv_new(OUT/'visual_examples_manifest.csv',examples_rows)
    failures=[r for r in samples if not r['geometry_pass']]
    weakest=min(data['none'].values(),key=lambda r:(r['inliers'],r['inlier_ratio']))
    strongest=max(data['none'].values(),key=lambda r:(r['inliers'],r['inlier_ratio']))
    for r in [weakest,strongest]:
        target=examples/(r['pair_id']+'_comparison.png')
        if not target.exists():example(lookup[r['pair_id']],data,examples)
    csv_new(OUT/'cases.csv',[dict(case_type='weakest_observed' if r is weakest else 'strongest_observed',
        pair_id=r['pair_id'],inliers=r['inliers'],matches=r['matches'],inlier_ratio=r['inlier_ratio'],
        geometry_pass=r['geometry_pass'],valid_H=r['valid_homography'],overlap_H_proxy=r['overlap_H_proxy']) for r in [weakest,strongest]])
    if failures:csv_new(OUT/'geometry_failure_cases.csv',[{k:v for k,v in r.items() if k!='homography'} for r in failures])
    else:
        with (OUT/'geometry_failure_cases.csv').open('x',newline='') as f:csv.writer(f).writerow(['method','pair_id','reason'])
    bearing=list(csv.DictReader((ROOT/'experiments/2026-11-13_lowlight/enhancement_comparison.csv').open()))
    csv_new(OUT/'bearing_uav_existing_reference.csv',bearing)
    table='| 영상 보정 | 쌍 | Mean matches | Mean inliers | Mean pair inlier ratio | 유효 H | 기하통과≥15 | Mean / Median / P95 ms |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for s in summaries:table+=f"| {s['method']} | {s['n_pairs']} | {s['mean_matches']:.1f} | {s['mean_inliers']:.1f} | {s['mean_pair_inlier_ratio_pct']:.1f}% | {s['valid_homography_pct']:.1f}% | {s['geometry_pass_pct']:.1f}% | {s['mean_latency_ms']:.1f} / {s['median_latency_ms']:.1f} / {s['p95_latency_ms']:.1f} |\n"
    bt='| 원본 밝기 층 | n | Mean Y′ /255 | Mean inliers | Inlier ratio | 기하통과 | H 겹침 proxy 평균 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for s in stratified:
        if s['method']=='none':bt+=f"| {s['brightness_bin']} | {s['n_pairs']} | {s['mean_native_pair_Y255']:.2f} | {s['mean_inliers']:.1f} | {s['mean_pair_inlier_ratio_pct']:.1f}% | {s['geometry_pass_pct']:.1f}% | {s['mean_overlap_H_proxy']:.3f} |\n"
    bearing_table='| Bearing-UAV 조건 | 보정 | n | Mean matches | Mean inliers | 유효 H | 독립 위치 Success@10m |\n|---|---|---:|---:|---:|---:|---:|\n'
    for s in bearing:bearing_table+=f"| {s['lowlight']} | {s['enhancement']} | {s['n']} | {float(s['mean_matches']):.1f} | {float(s['mean_inliers']):.1f} | {float(s['homography_success_pct']):.0f}% | {float(s['success_10m_pct']):.0f}% |\n"
    original=summaries[0]
    report='''# 실제 NAT2021 야간 영상의 Full LoMa-B 매칭 — 결론 보고서

**실제 NAT2021 원본40쌍과 Gamma·CLAHE를 각각 적용한40쌍, 총120건의 GPU 추론을 완료했습니다.** 같은 시퀀스의 두 프레임 사이 매칭이며 위성 지도 검색·위치 추정 실험이 아닙니다. 재학습이나 새 모델은 없습니다.

## 실험 조건

- 기존 공식 Full LoMa-B pretrained strict loading, FP32/TF32 off, RTX A6000 GPU0. keypoints2048,threshold0.1,USAC_MAGSAC3px,maxIters10000,confidence0.999.
- 기존에 고정한20시퀀스에서 첫·중간 anchor 각1쌍. 모두+5프레임(공식30fps 기준 약0.167초),전체1280×720화면,단일방향,caching 없음. 입력을14배수1274×714로 맞추고 원래 pixel좌표로 복원해기하계산.
- 원본 이미지40개는 기존 표본 재사용,짝의+5프레임40개만 공식공개배포에서추가확보. ZIP CRC/SHA확인,PAIR_MANIFEST.json에모든경로·hash저장. 출력으로표본을선별하거나실패를제외하지않음.
- Gamma=x^0.6,CLAHE=LAB-L clip2/grid8. 실제야간입력에합성을더하지않으며 **두프레임모두 같은보정** 적용. 원본파일은변경하지않음.
- 조건별10pair warm-up제외,CUDA동기화. 시간은decoded RGB→보정/resize→양쪽특징→matching/filter→MAGSAC. decode/모델로드/저장/겹침및진단채점은제외. 고정조건순서1회측정이며독립반복아님.

## 실제 야간 매칭 성능

'''+table+'''
Inlier ratio는MAGSAC자체일관성입니다. 정답대응점precision이나GT정확도를뜻하지않습니다. 유효H는finite/rank3/inlier≥4,기하통과는추가로inlier≥15인진단기준입니다. 두지표모두지리적성공률이아닙니다. **위성지도·위치GT가없으므로Success@10m는미측정/공란**입니다.

## 낮은 밝기와 매칭

'''+bt+f'''
밝기층은원본40쌍의평균Y′를3등분한고정그룹입니다(경계{q1:.3f},{q2:.3f}/255). Gamma/CLAHE에도같은그룹을적용해보정후그룹변경을막았습니다.20시퀀스평균밝기와원본inlier수의Spearman상관은{rank_r:.3f}입니다. 이는장면·텍스처·움직임과밝기가혼재한관찰결과이며,조도만의인과효과나주간대비손실로해석하지않습니다.

## 겹침 통제의 범위

시간간격·화면범위·처리법은고정했지만실제물리적겹침은GT없이동일하게강제할수없습니다. 원본의H기반교집합면적proxy는평균{original['mean_overlap_H_proxy']:.3f},최소{original['min_overlap_H_proxy']:.3f},0.8이상{original['overlap_proxy_ge080_n']}/{original['overlap_proxy_defined_n']}쌍입니다. 이proxy는모델의추정에서나온값이므로독립overlap GT가아닙니다. 조건간H가바뀌면proxy도바뀔수있습니다. 이값으로쌍을제외하지않았습니다. 전체표는40쌍모두포함합니다.

## 보정의 짝지은 변화

'''
    for s in paired_summary:
        report+=f"- {s['method']} 대 원본: 평균match차{s['mean_matches_delta']:+.1f},inlier차{s['mean_inliers_delta']:+.1f};20시퀀스단위paired bootstrap95%구간[{s['inliers_delta_sequence_bootstrap_ci95_low']:.1f},{s['inliers_delta_sequence_bootstrap_ci95_high']:.1f}]. inlier증가{s['inliers_improved_n']}쌍/감소{s['inliers_worsened_n']}쌍,기하통과→실패{s['pass_to_fail_n']}쌍/반대{s['fail_to_pass_n']}쌍.\n"
    report+='''
방향성은위의실측값으로판단하되inlier증가를정답매칭정확도향상이라고부르지않습니다. 같은시퀀스2쌍은독립장면이아니므로bootstrap은시퀀스단위로수행했습니다. 표본40쌍이실제야간전체성능을대표한다고주장하지않습니다.

## 성공·실패 사례와 예시5개

'''+f"원본최강관측사례: {strongest['pair_id']} (matches{strongest['matches']},inliers{strongest['inliers']}). 원본최약관측사례: {weakest['pair_id']} (matches{weakest['matches']},inliers{weakest['inliers']}).\n\n"
    if not failures:report+='기하통과기준에서실패한샘플은120건중0건입니다. 실패를만들어내지않고최약사례를별도표시했습니다. 이기준의통과가정답대응점이나항법정확도를보장하지는않습니다.\n\n'
    else:report+=f'기하통과기준에서실패한샘플은120건중{len(failures)}건입니다. geometry_failure_cases.csv에원인확인용지표를보존했습니다.\n\n'
    report+='''서로다른원본밝기의5시퀀스를고정순위로골라visual_examples_manifest.csv와visual_examples/에원본쌍및세방법의매칭그림을저장했습니다. 녹색선은MAGSACinlier,빨간선은outlier의고정개수시각표본이며전체대응점은NPZ에있습니다. 표시용resize는모델입력과별개입니다. 원본영상의이용조건은이전감사의범위를유지하며로컬예시를외부재배포하지않았습니다.

## 기존 Bearing-UAV 합성과 별도 비교

아래는새실험이아닌기존결과복사입니다. **Bearing-UAV는UAV–Satellite Oracle T1/FP16/8방향/256및512px**,위NAT는**같은야간영상의시간간격5프레임/FP32/1방향/native720p**입니다. 태스크·도메인·입력크기·정밀도·방향수가달라matching수나latency를직접적인성능우열/동일정확도로해석하지않습니다. 기존합성은UAV만보정했고이번NAT는두프레임모두보정했습니다.

'''+bearing_table+'''
## 재현·보존

protocol.json,environment.json,model.json,dtype_audit.json,PAIR_MANIFEST.json과원시조건별JSON/NPZ를보존했습니다. samples.csv120행,matching_summary.csv3행,brightness_groups.csv9행,paired/sequence통계,예시5개와cases.csv가산출물입니다. 단계별시간과allocated/reservedVRAM도기록했습니다.

```bash
cd ${WEEK2_WORKSPACE}
# 이미완료된결과는건너뛰며프로토콜변경시재개를거부합니다.
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -u standalone/run_nat_matching.py
# 보고서는기존파일을덮어쓰지않으므로새경로에서만재생성합니다.
.venv-gpu/bin/python standalone/report_nat_matching.py
```

기존260건GPU측정·T4R감사·원본이미지·가중치·코드는변경하지않았습니다. T4R복원·추가카메라모델·학습·GitCommit/Push/PR은수행하지않았습니다. 팀위치시험k100..999·잠긴지역·항법궤적은사용하지않았습니다. NAT2021은공식공개외부데이터이며이실험을팀공식위치평가라고부르지않습니다.
'''
    with (OUT/'REPORT.md').open('x') as f:f.write(report)
    with (OUT/'summary.json').open('x') as f:json.dump(dict(summary=summaries,paired=paired_summary,
        brightness_bins=[float(q1),float(q2)],sequence_brightness_inlier_spearman=rank_r,
        geometry_failures=len(failures),location_accuracy_measured=False),f,indent=2)
    print(json.dumps(dict(summary=summaries,paired=paired_summary,geometry_failures=len(failures)),indent=2))
