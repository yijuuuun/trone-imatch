"""Actual native-resolution NAT2021 temporal matching; no location GT or training."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from loma_backend import Backend,ROOT,digest
from lowlight import enhance

OUT=ROOT/'experiments/2026-11-13_lowlight/nat2021_frame_matching_v1'

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def brightness(rgb):
    return float(np.mean(rgb.astype(np.float64)@np.array([.2126,.7152,.0722])))

def geometry_quality(H,mask,p,q,shape):
    n=int(mask.sum()) if mask is not None else 0
    valid=H is not None and np.isfinite(H).all() and np.linalg.matrix_rank(H)==3 and n>=4
    overlap=None;coverage=None;reprojection=None
    if valid:
        inside=mask.reshape(-1).astype(bool)
        projected=cv2.perspectiveTransform(p[None].astype(np.float64),H)[0]
        errors=np.linalg.norm(projected-q,axis=1)
        if np.isfinite(errors[inside]).all():reprojection=float(np.median(errors[inside]))
        hull=cv2.convexHull(p[inside].astype(np.float32))
        coverage=float(cv2.contourArea(hull)/(shape[0]*shape[1]))
        h,w=shape
        corners=np.array([[[0.,0.],[w-1.,0.],[w-1.,h-1.],[0.,h-1.]]],np.float64)
        warped=cv2.perspectiveTransform(corners,H)[0].astype(np.float32)
        canvas=corners[0].astype(np.float32)
        if np.isfinite(warped).all() and cv2.isContourConvex(warped):
            area,_=cv2.intersectConvexConvex(warped,canvas)
            overlap=float(np.clip(area/((w-1)*(h-1)),0.,1.))
    return dict(matches=len(p),inliers=n,inlier_ratio=n/len(p) if len(p) else None,
        valid_homography=bool(valid),geometry_pass=bool(valid and n>=15),
        overlap_H_proxy=overlap,inlier_source_hull_fraction=coverage,median_inlier_reprojection_px=reprojection)

def infer(backend,a,b,method,seed):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    x,y=enhance(a,method),enhance(b,method)
    ta,tb=backend.preprocess(x),backend.preprocess(y)
    torch.cuda.synchronize();pre=time.perf_counter()
    fa,fb=backend.features(ta),backend.features(tb)
    backend.check_features(fa);backend.check_features(fb)
    torch.cuda.synchronize();feat=time.perf_counter()
    p,q=backend.match(fa,fb,a.shape[:2],b.shape[:2])
    torch.cuda.synchronize();match=time.perf_counter()
    cv2.setRNGSeed(seed);H,mask=None,None
    if len(p)>=4:H,mask=cv2.findHomography(p.astype(np.float32),q.astype(np.float32),cv2.USAC_MAGSAC,3.,maxIters=10000,confidence=.999)
    torch.cuda.synchronize();end=time.perf_counter()
    quality=geometry_quality(H,mask,p,q,a.shape[:2])
    result=dict(**quality,mean_latency_ms=(end-start)*1000,preprocess_ms=(pre-start)*1000,
        feature_ms=(feat-pre)*1000,matching_ms=(match-feat)*1000,magsac_ms=(end-match)*1000,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        enhanced_pair_mean_Y255=(brightness(x)+brightness(y))/2,
        homography=H.tolist() if H is not None and np.isfinite(H).all() else None)
    return result,p,q,mask

if __name__=='__main__':
    manifest=json.loads((OUT/'PAIR_MANIFEST.json').read_text())
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('GPU occupied: '+busy)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    protocol=dict(task='same-sequence native RGB frame matching',pairs=40,sequences=20,gap_frames=5,
        nominal_frame_gap_seconds=5/30,precision='FP32; TF32 disabled',headings=[0],cache=False,
        keypoints=2048,matching_threshold=.1,magsac_px=3.,maxiters=10000,confidence=.999,
        resize='native1280x720→1274x714 nearest14 antialias bilinear; restore native coordinates before MAGSAC',
        methods=['none','gamma','clahe'],gamma=.6,clahe={'clipLimit':2.,'tileGridSize':[8,8]},
        enhancement_scope='both temporal frames; no synthesis',warmup='10 pairs per method; excluded',
        valid_H='finite rank3 H and >=4 inliers',geometry_pass='valid H and >=15 inliers; consistency proxy, NOT GT accuracy',
        overlap='H-warped full-frame intersection fraction, post-hoc proxy; no filtering',
        latency='decoded RGB through enhancement,preprocessing,two feature extractions,matching,MAGSAC; exclude decode/load/save and geometry diagnostics',
        Success_at_10m=None,location_GT_available=False,source_hashes={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('loma_backend.py'),Path(__file__).with_name('lowlight.py')]},
        pair_manifest_sha256=digest(OUT/'PAIR_MANIFEST.json'))
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Resume source/protocol changed')
    else:dump(path,protocol)
    if not (OUT/'environment.json').exists():dump(OUT/'environment.json',dict(python=sys.executable,torch=torch.__version__,
        torch_cuda=torch.version.cuda,opencv=cv2.__version__,numpy=np.__version__,gpu=subprocess.check_output(['nvidia-smi'],text=True),command=sys.argv))
    images={}
    for pair in manifest['pairs']:
        for name in ['a','b']:
            path=ROOT/pair[name+'_path']
            if digest(path)!=pair[name+'_sha256']:raise ValueError('Original frame changed')
            images[str(path)]=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    def pair_images(pair):return images[str(ROOT/pair['a_path'])],images[str(ROOT/pair['b_path'])]
    backend=Backend();backend.set_precision(False)
    if not (OUT/'model.json').exists():dump(OUT/'model.json',backend.meta)
    if not (OUT/'dtype_audit.json').exists():
        audit={}
        def hook(name):
            def observe(module,args,result):
                if isinstance(result,torch.Tensor):
                    group='detector' if name.startswith('_detector') else 'descriptor' if name.startswith('_descriptor') else 'matcher'
                    key=group+':'+str(result.dtype);audit[key]=audit.get(key,0)+1
                    if result.dtype!=torch.float32:raise ValueError('FP32 compute dtype violated: '+name)
            return observe
        handles=[m.register_forward_hook(hook(n)) for n,m in backend.model.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]
        try:infer(backend,*pair_images(manifest['pairs'][0]),'none',20261110)
        finally:
            for h in handles:h.remove()
        dump(OUT/'dtype_audit.json',audit)
    for method in protocol['methods']:
        folder=OUT/method;folder.mkdir(exist_ok=True)
        if all((folder/(p['pair_id']+'.json')).exists() for p in manifest['pairs']):
            print('SKIP COMPLETE '+method,flush=True);continue
        print('START '+method,flush=True)
        for i in range(10):infer(backend,*pair_images(manifest['pairs'][i%5]),method,20261110+i%5)
        for i,pair in enumerate(manifest['pairs']):
            target=folder/(pair['pair_id']+'.json')
            if target.exists():continue
            a,b=pair_images(pair)
            r,p,q,mask=infer(backend,a,b,method,20261110+i)
            artifact=folder/(pair['pair_id']+f'_matches_{time.time_ns()}.npz')
            with artifact.open('xb') as f:np.savez_compressed(f,p=p,q=q,mask=mask if mask is not None else np.zeros((len(p),1),np.uint8))
            r.update(pair_id=pair['pair_id'],sequence=pair['sequence'],anchor=pair['anchor'],method=method,
                gap_frames=5,original_pair_mean_Y255=(brightness(a)+brightness(b))/2,
                correspondence_file=artifact.name,Success_at_10m=None)
            dump(target,r)
            print(json.dumps({k:r[k] for k in ['pair_id','method','matches','inliers','inlier_ratio','valid_homography','geometry_pass','mean_latency_ms']}),flush=True)
        print('COMPLETE '+method,flush=True)
    print('NAT MATCHING COMPLETE',flush=True)
