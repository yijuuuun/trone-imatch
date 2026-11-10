"""Resumable development-only LoMa pilot. No team accuracy equivalence claimed."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from loma_backend import Backend,ROOT,digest
from smoke_gpu import infer
from lowlight import SYNTHESIS

CONDITIONS=[
 dict(id='F1',suite='F',precision='fp16',headings=8,scale=1.,light='daylight',enhance='none'),
 dict(id='F3',suite='F',precision='fp16',headings=4,scale=1.,light='daylight',enhance='none'),
 dict(id='F0',suite='F',precision='fp32',headings=8,scale=1.,light='daylight',enhance='none'),
 dict(id='F2',suite='F',precision='fp32',headings=4,scale=1.,light='daylight',enhance='none'),
 *[dict(id=name,suite='E',precision='fp16',headings=8,scale=1.,light=light,enhance=method)
   for name,light,method in [('E_mild','mild','none'),('E_severe','severe','none'),
       ('E_mild_gamma','mild','gamma'),('E_mild_clahe','mild','clahe'),
       ('E_severe_gamma','severe','gamma'),('E_severe_clahe','severe','clahe')]],
 dict(id='F_scale075',suite='F_resolution',precision='fp16',headings=8,scale=.75,light='daylight',enhance='none'),
 dict(id='F_scale050',suite='F_resolution',precision='fp16',headings=8,scale=.5,light='daylight',enhance='none'),
]

def dump_new(path,obj):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
    if path.exists():raise FileExistsError(path)
    temporary.rename(path)

class Scorer:
    """Source metadata scoring is outside infer; GT cannot choose a heading."""
    def __init__(self):
        data=ROOT/'data/singapore_dev100'
        self.rows={int(s['k']):s for s in csv.DictReader((data/'samples.csv').open()) if int(s['k'])<100}
        self.mapping={int(s['k']):s for s in csv.DictReader((ROOT/'archive_reference/datasets/singapore/source_mapping.csv').open()) if int(s['k'])<100}
        self.uav=json.loads((data/'uav_metadata.json').read_text())
        self.rsi=next(v for v in json.loads((data/'rsi_metadata.json').read_text()).values() if abs(v.get('lng',0)-103.84197619336068)<1e-6)
        # Validate direct metadata coordinate mapping for all development samples.
        discrepancies=[]
        for k,row in self.rows.items():
            bx,by=self.block(k);m=self.rsi;u=self.uav[str(k)]
            x=(u['lng']-m['lng'])/m['lng_per_pixel']+m['width_pixel']/2
            y=(m['lat']-u['lat'])/m['lat_per_pixel']+m['height_pixel']/2
            discrepancies.extend([x-(bx+1)*256-float(row['x_norm'])*128,
                                  y-(by+1)*256-float(row['y_norm'])*128])
        self.max_discrepancy=max(abs(v) for v in discrepancies)
        if self.max_discrepancy>1e-5:raise ValueError('GT/georeference audit failed')

    def block(self,k):
        return tuple(map(int,re.search(r'block_(\d+)_(\d+)_',self.mapping[k]['p1_path']).groups()))

    def score(self,k,result):
        s=result['selected'];center=s['center'];rejected=center is None or s['inliers']<15
        base={'k':k,'scene':self.mapping[k]['p1_path'].rsplit('_base_',1)[0],
              'selected_angle':s['angle'],'matches':s['matches'],'inliers':s['inliers'],
              'valid_homography':center is not None,'rejected':rejected,
              'error_m':None,'haversine_error_m':None,'pred_lng':None,'pred_lat':None,
              'raw_success_10m':False,'success_10m':False,'large_error':None,'large_error_detected':None}
        if center is None:return base
        m=self.rsi;bx,by=self.block(k);row=self.rows[k]
        x,y=center;gtx=256+128*float(row['x_norm']);gty=256+128*float(row['y_norm'])
        err=math.hypot((x-gtx)*m['lngm_per_pixel'],(y-gty)*m['latm_per_pixel'])
        gx=bx*256+x;gy=by*256+y
        lng=m['lng']+(gx-m['width_pixel']/2)*m['lng_per_pixel']
        lat=m['lat']-(gy-m['height_pixel']/2)*m['lat_per_pixel']
        gt=self.uav[str(k)];dlon=math.radians(lng-gt['lng']);dlat=math.radians(lat-gt['lat'])
        a=math.sin(dlat/2)**2+math.cos(math.radians(lat))*math.cos(math.radians(gt['lat']))*math.sin(dlon/2)**2
        haversine=6371000*2*math.asin(math.sqrt(min(1.,max(0.,a))))
        base.update(error_m=err,haversine_error_m=haversine,pred_lng=lng,pred_lat=lat,
                    raw_success_10m=err<10,success_10m=err<10 and not rejected,
                    large_error=err>32,large_error_detected=err>32 and rejected)
        return base

def profile(backend,uav,sat,condition,seed):
    """Separate synchronized wall profiling; Transformer excludes match filtering."""
    timings={'feature_wall_ms':0.,'transformer_wall_ms':0.,'magsac_wall_ms':0.}
    original_features=backend.features;original_forward=backend.model.forward;original_h=cv2.findHomography
    def wrap(fn,key,cuda):
        def call(*args,**kwargs):
            if cuda:torch.cuda.synchronize()
            start=time.perf_counter();result=fn(*args,**kwargs)
            if cuda:torch.cuda.synchronize()
            timings[key]+=(time.perf_counter()-start)*1000
            return result
        return call
    backend.features=wrap(original_features,'feature_wall_ms',True)
    backend.model.forward=wrap(original_forward,'transformer_wall_ms',True)
    cv2.findHomography=wrap(original_h,'magsac_wall_ms',False)
    try:
        angles=list(range(0,360,45 if condition['headings']==8 else 90))
        r=infer(backend,uav,sat,angles,condition['light'],condition['enhance'],seed,input_scale=condition['scale'])
        timings['profile_e2e_ms']=r['e2e_ms']
    finally:
        backend.features=original_features;backend.model.forward=original_forward;cv2.findHomography=original_h
    return timings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--only',nargs='+',choices=[c['id'] for c in CONDITIONS])
    args=parser.parse_args()
    if not 5<=args.limit<=100:parser.error('Development only: 5..100')
    conditions=[c for c in CONDITIONS if not args.only or c['id'] in args.only]
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('GPU occupied; refusing interference: '+busy)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    scorer=Scorer();args.out.mkdir(parents=True,exist_ok=True)
    protocol={'conditions':CONDITIONS,'cohort':'development k0..limit-1; max99','seed_base':20261110,
       'map':'Oracle T1','cache':False,'keypoints':2048,'matching_threshold':.1,'magsac_px':3.,
       'magsac_maxiters':10000,'magsac_confidence':.999,'selection':'valid-H max inlier; ties max matches then first angle',
       'rotation':'same original canvas, black border, INTER_LINEAR',
       'resize':'nearest14 antialias bilinear; scale both UAV and satellite tensors; restore original coords before MAGSAC',
       'synthesis':SYNTHESIS,'synthesis_provenance':'temporary development model; T4R/NAT2021 settings unavailable',
       'gamma':.6,'clahe':{'clip':2.,'grid':[8,8]},'warmup':'10 single-heading pairs per condition; excluded',
       'success':'independent planar metadata error<10m AND not rejected; denominator all samples; raw success separately',
       'rejection':'invalid H or selected inliers<15','large_error':'raw valid-pose error>32m; no-pose error undefined, separately counted',
       'error_scale':'official source metadata 0.25m/px, normalized GT128px; no clipping or fallback',
       'gt_used_in_infer':False,'team_pipeline_verified':False,
       'coordinate_metadata_max_discrepancy_px':scorer.max_discrepancy,
       'source_hashes':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('loma_backend.py'),Path(__file__).with_name('smoke_gpu.py'),Path(__file__).with_name('lowlight.py')]}}
    config=args.out/'protocol.json'
    if config.exists():
        if json.loads(config.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Resume protocol changed')
    else:dump_new(config,protocol)
    if not (args.out/'environment.json').exists():
        dump_new(args.out/'environment.json',{'python':sys.executable,'version':sys.version,'torch':torch.__version__,
            'torch_cuda':torch.version.cuda,'opencv':cv2.__version__,'numpy':np.__version__,
            'gpu':subprocess.check_output(['nvidia-smi'],text=True),'command':sys.argv,'tf32':False,
            'hostname':subprocess.check_output(['hostname'],text=True).strip()})
    backend=Backend()
    if not (args.out/'model.json').exists():dump_new(args.out/'model.json',backend.meta)
    def images(k):
        result=[]
        for kind in ['uav','block']:
            path=ROOT/'data/singapore_dev100/3d_ho37bc'/f'k{k:03d}_{kind}.jpg'
            bgr=cv2.imread(str(path))
            if bgr is None:raise ValueError(path)
            result.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        return result
    for c in conditions:
        case=args.out/c['id'];case.mkdir(exist_ok=True)
        if all((case/f'k{k:03d}.json').exists() for k in range(args.limit)) and all((case/f'profile{k:03d}.json').exists() for k in range(5)):
            print('SKIP COMPLETE '+c['id'],flush=True);continue
        backend.set_precision(c['precision']=='fp16')
        print('START '+json.dumps(c),flush=True)
        for i in range(10):infer(backend,*images(i%5),[0],c['light'],c['enhance'],20261110+i%5,input_scale=c['scale'])
        angles=list(range(0,360,45 if c['headings']==8 else 90))
        for k in range(args.limit):
            target=case/f'k{k:03d}.json'
            if target.exists():continue
            try:
                result=infer(backend,*images(k),angles,c['light'],c['enhance'],20261110+k,True,c['scale'])
                pairs=result.pop('_correspondences')
                # A unique file name preserves any orphaned artifact from an interrupted run.
                correspondence=case/f'k{k:03d}_matches_{time.time_ns()}.npz'
                with correspondence.open('xb') as f:np.savez_compressed(f,**pairs)
                result.update(condition=c,metrics=scorer.score(k,result),correspondences=correspondence.name)
                dump_new(target,result)
                m=result['metrics']
                print(json.dumps({'condition':c['id'],'k':k,'latency_ms':round(result['e2e_ms'],2),
                    'inliers':m['inliers'],'error_m':m['error_m'],'rejected':m['rejected']}),flush=True)
            except Exception as exc:
                fail=case/f'failure_k{k:03d}_{time.time_ns()}.json'
                dump_new(fail,{'k':k,'error':repr(exc),'condition':c})
                raise
        for k in range(5):
            target=case/f'profile{k:03d}.json'
            if not target.exists():dump_new(target,profile(backend,*images(k),c,20261110+k))
        print('COMPLETE '+c['id'],flush=True)
    print('MATRIX COMPLETE',flush=True)

if __name__=='__main__':main()
