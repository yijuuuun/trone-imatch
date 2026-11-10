"""CPU-only descriptive statistics. Original data and synthesis code stay unchanged."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

from runtime_paths import AUDIT,ROOT as PROJECT
sys.path.insert(0,str(PROJECT/'standalone'))
from lowlight import synthesize,SYNTHESIS

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def linear(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)

def stats(rgb):
    code=rgb.astype(np.float64)/255.;lin=linear(code)
    weights=np.array([.2126,.7152,.0722]);yc=code@weights;yl=lin@weights
    result={'width':rgb.shape[1],'height':rgb.shape[0]}
    for name,y in [('srgb_code',yc),('assumed_linear',yl)]:
        result[name+'_mean']=float(y.mean());result[name+'_median']=float(np.median(y))
        for p in [1,5,10,25,75,90,95,99]:result[f'{name}_p{p:02d}']=float(np.percentile(y,p))
        result[name+'_std']=float(y.std())
    result['srgb_code_mean_255']=result['srgb_code_mean']*255
    result['dark_code_lt005_ratio']=float(np.mean(yc<.05))
    result['global_code_lt20of255_ratio_NOT_LAI']=float(np.mean(yc<20/255))
    result['dark_linear_lt001_ratio']=float(np.mean(yl<.01))
    result['black_all_channels_zero_ratio']=float(np.mean((rgb==0).all(2)))
    result['any_channel_zero_ratio']=float(np.mean((rgb==0).any(2)))
    result['any_channel_255_ratio']=float(np.mean((rgb==255).any(2)))
    saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float64)/255
    result['hsv_saturation_mean']=float(saturation.mean())
    result['hsv_saturation_p95']=float(np.percentile(saturation,95))
    result['hsv_s_gt095_and_v_gt005_ratio']=float(np.mean((saturation>.95)&(code.max(2)>.05)))
    result['contrast_p95_minus_p05']=result['srgb_code_p95']-result['srgb_code_p05']
    h,w=yc.shape;tiles=yc[:h//16*16,:w//16*16].reshape(h//16,16,w//16,16).transpose(0,2,1,3)
    std=tiles.std(axis=(2,3)).ravel()
    result['tile16_std_median']=float(np.median(std));result['tile16_std_p95']=float(np.percentile(std,95))
    hh=(yc[:-1,:-1]-yc[:-1,1:]-yc[1:,:-1]+yc[1:,1:])/2
    local=(yc[:-1,:-1]+yc[:-1,1:]+yc[1:,:-1]+yc[1:,1:])/4
    dark=hh[local<.1]
    result['dark_hh_samples']=int(len(dark))
    result['dark_hh_mad_proxy_NOT_sensor_sigma']=float(np.median(np.abs(dark-np.median(dark)))/.6744897501960817) if len(dark) else None
    for c,name in enumerate(['r','g','b']):
        v=code[:,:,c];result[name+'_mean']=float(v.mean());result[name+'_median']=float(np.median(v));result[name+'_std']=float(v.std())
        result[name+'_p05']=float(np.percentile(v,5));result[name+'_p95']=float(np.percentile(v,95))
    hist={name:np.histogram(v,bins=64,range=(0,1))[0].tolist() for name,v in [('srgb_code',yc),('assumed_linear',yl),('r',code[:,:,0]),('g',code[:,:,1]),('b',code[:,:,2])]}
    return result,hist

def trace_synthesis(rgb,condition,seed):
    exposure,photons,sigma=SYNTHESIS[condition];l=linear(rgb.astype(np.float64)/255)
    rng=np.random.default_rng(seed);noisy=rng.poisson(l*exposure*photons)/photons+rng.normal(0,sigma,l.shape)
    clipped=np.clip(noisy,0,1)
    encoded=np.where(clipped<=.0031308,12.92*clipped,1.055*clipped**(1/2.4)-.055)
    result=np.rint(np.clip(encoded,0,1)*255).astype(np.uint8)
    assert np.array_equal(result,synthesize(rgb,condition,seed))
    return dict(preclip_channel_below_zero_ratio=float(np.mean(noisy<0)),preclip_channel_above_one_ratio=float(np.mean(noisy>1)),
        preclip_noise_residual_mean=float(np.mean(noisy-exposure*l)),preclip_noise_residual_variance=float(np.var(noisy-exposure*l)),
        mean_theoretical_variance=float(np.mean(exposure*l/photons+sigma*sigma)),
        clipped_linear_bias_mean=float(np.mean(clipped-exposure*l)))

if __name__=='__main__':
    # Protect all pre-existing experimental results, code, and current report.
    paths=[]
    for folder in [PROJECT/'runs',PROJECT/'experiments',PROJECT/'meeting_2026-11-11_v1']:
        paths += [p for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(AUDIT)]
    paths += list((PROJECT/'standalone').glob('*.py'))+[PROJECT/'WEEK2_REPORT.md']
    protected={str(p.relative_to(PROJECT)):digest(p) for p in paths if p.is_file()}
    with (AUDIT/'PRESERVED_RESULTS_SHA256.json').open('x') as f:json.dump(protected,f,indent=2)
    manifest=json.loads((AUDIT/'NAT2021_SAMPLE_MANIFEST.json').read_text())
    real=[];synthetic=[];histograms={};trace=[]
    for i,record in enumerate(manifest['records']):
        path=AUDIT/record['relative_path'];assert digest(path)==record['sha256']
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);measured,hist=stats(rgb)
        real.append(dict(dataset='NAT2021',split='test_descriptive_only',sequence=path.parent.name,frame=path.name,
            scene_type='unlabelled; official per-sequence scene taxonomy unavailable',official_LAI='',source_sha256=record['sha256'],
            color_assumption='decoded RGB treated as sRGB; no RAW/EXIF/ICC calibration',**measured))
        histograms[record['member']]=hist
        print(f'NAT {i+1}/60 {path.parent.name}/{path.name}',flush=True)
    for k in range(20):
        path=PROJECT/'data/singapore_dev100/3d_ho37bc'/f'k{k:03d}_uav.jpg'
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        for condition in ['daylight','mild','severe']:
            im=synthesize(rgb,condition,20261110+k);measured,hist=stats(im)
            synthetic.append(dict(dataset='Bearing-UAV development',k=k,condition=condition,seed=20261110+k,
                evidence_grade='C Experimental',source_sha256=digest(path),**measured))
            histograms[f'dev_k{k:03d}_{condition}']=hist
            if condition!='daylight':trace.append(dict(k=k,condition=condition,seed=20261110+k,**trace_synthesis(rgb,condition,20261110+k)))
    write_csv(AUDIT/'NAT2021_STATISTICS.csv',real)
    write_csv(AUDIT/'SYNTHESIS_STATISTICS.csv',synthetic)
    write_csv(AUDIT/'SYNTHESIS_CLIPPING_AUDIT.csv',trace)
    with (AUDIT/'PIXEL_HISTOGRAMS_64BINS.json').open('x') as f:json.dump(histograms,f)
    metrics=[k for k,v in real[0].items() if isinstance(v,(float,int))]
    summaries=[]
    for label,records in [('NAT2021_sample60',real)]+[(f'dev_{c}20',[r for r in synthetic if r['condition']==c]) for c in ['daylight','mild','severe']]:
        summary={'group':label,'frames':len(records)}
        for key in metrics:
            values=[r[key] for r in records if r.get(key) is not None]
            for suffix,fn in [('mean_of_frames',np.mean),('median_of_frames',np.median),('min_frame',np.min),('max_frame',np.max)]:
                summary[key+'_'+suffix]=float(fn(values)) if values else None
        summaries.append(summary)
    with (AUDIT/'STATISTICS_SUMMARY.json').open('x') as f:json.dump(summaries,f,indent=2)
    seq=[]
    for name in manifest['selected_sequences']:
        values=[r for r in real if r['sequence']==name]
        seq.append(dict(sequence=name,n_frames=len(values),scene_type='unlabelled',
            mean_srgb_code_255=np.mean([r['srgb_code_mean_255'] for r in values]),
            min_frame_mean_255=min(r['srgb_code_mean_255'] for r in values),max_frame_mean_255=max(r['srgb_code_mean_255'] for r in values),
            mean_linear_luminance=np.mean([r['assumed_linear_mean'] for r in values]),
            mean_dark_code_lt005_ratio=np.mean([r['dark_code_lt005_ratio'] for r in values])))
    write_csv(AUDIT/'NAT2021_SEQUENCE_STATISTICS.csv',seq)
    black=np.zeros((256,256,3),np.uint8)
    controls={c:dict(mean_code=float(synthesize(black,c,20261110).mean()),
        any_channel_zero_ratio=float(np.mean((synthesize(black,c,20261110)==0).any(2))),
        **trace_synthesis(black,c,20261110)) for c in ['mild','severe']}
    with (AUDIT/'BLACK_INPUT_DIAGNOSTIC.json').open('x') as f:json.dump(controls,f,indent=2)
    # Native source examples are only resized for this local audit visualization.
    visuals=AUDIT/'visual_examples';visuals.mkdir()
    first=[next(r for r in real if r['sequence']==seq) for seq in manifest['selected_sequences']]
    montage=np.full((5*170,4*320,3),240,np.uint8)
    for i,row in enumerate(first):
        path=AUDIT/'nat2021_sample'/row['sequence']/row['frame']
        im=cv2.imread(str(path));preview=cv2.resize(im,(320,144),interpolation=cv2.INTER_AREA)
        y,x=(i//4)*170,(i%4)*320;montage[y+26:y+170,x:x+320]=preview
        cv2.putText(montage,f"{row['sequence']} mean={row['srgb_code_mean_255']:.1f}",(x+3,y+18),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),1,cv2.LINE_AA)
    cv2.imwrite(str(visuals/'NAT2021_first_frames_local_only.png'),montage)
    panel=np.full((3*282,3*256,3),240,np.uint8)
    for i,k in enumerate([0,5,19]):
        rgb=cv2.cvtColor(cv2.imread(str(PROJECT/'data/singapore_dev100/3d_ho37bc'/f'k{k:03d}_uav.jpg')),cv2.COLOR_BGR2RGB)
        for j,c in enumerate(['daylight','mild','severe']):
            im=cv2.cvtColor(synthesize(rgb,c,20261110+k),cv2.COLOR_RGB2BGR);y,x=i*282,j*256
            panel[y+26:y+282,x:x+256]=im
            cv2.putText(panel,f'k{k:03d} {c}',(x+3,y+18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imwrite(str(visuals/'existing_synthesis_k000_k005_k019.png'),panel)
    assert all(digest(PROJECT/path)==want for path,want in protected.items())
    with (AUDIT/'CPU_AUDIT_VERIFICATION.json').open('x') as f:json.dump(dict(nat_frames=60,nat_sequences=20,
        synthetic_frames=60,clipping_traces=40,protected_files=len(protected),all_existing_hashes_unchanged=True,
        same_seed_trace_matches_existing_function=True,new_gpu_inference=False,new_parameters=False),f,indent=2)
    print(json.dumps([{k:r[k] for k in ['group','srgb_code_mean_255_mean_of_frames','dark_code_lt005_ratio_mean_of_frames','assumed_linear_mean_mean_of_frames']} for r in summaries],indent=2))
