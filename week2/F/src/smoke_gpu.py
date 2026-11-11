"""Official LoMa 8-heading CUDA smoke. No accuracy claim before team scoring parity."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from loma_backend import Backend,ROOT
from lowlight import preprocess_uav

def infer(backend,uav,satellite,angles,lowlight='daylight',enhancement='none',seed=20261110,save_correspondences=False,input_scale=1.0):
    # No GT inputs. No feature cache: both images extracted for each heading.
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    uav=preprocess_uav(uav,lowlight,enhancement,seed)
    enhancement_ms=(time.perf_counter()-begin)*1000
    candidates=[];correspondences={}
    timings={'preprocess_ms':enhancement_ms,'feature_gpu_ms':0.,'matching_gpu_ms':0.,'geometry_ms':0.}
    events=[]
    for angle in angles:
        t=time.perf_counter()
        rotation=cv2.getRotationMatrix2D((uav.shape[1]/2,uav.shape[0]/2),angle,1.)
        rotated=cv2.warpAffine(uav,rotation,(uav.shape[1],uav.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        a=backend.preprocess(rotated,input_scale);b=backend.preprocess(satellite,input_scale)
        torch.cuda.synchronize()
        timings['preprocess_ms']+=(time.perf_counter()-t)*1000
        start=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();fa=backend.features(a);fb=backend.features(b);middle.record()
        p,q=backend.match(fa,fb,rotated.shape[:2],satellite.shape[:2]);end.record()
        events.append((start,middle,end))
        backend.check_features(fa);backend.check_features(fb)
        t=time.perf_counter();cv2.setRNGSeed(seed)
        H,mask=(None,None)
        if len(p)>=4:
            H,mask=cv2.findHomography(p.astype(np.float32),q.astype(np.float32),cv2.USAC_MAGSAC,3.,maxIters=10000,confidence=0.999)
        n=int(mask.sum()) if mask is not None else 0
        center=None
        if H is not None and np.isfinite(H).all() and n>=4:
            point=cv2.perspectiveTransform(np.array([[[uav.shape[1]/2,uav.shape[0]/2]]],dtype=np.float32),H)[0,0]
            if np.isfinite(point).all():center=point.tolist()
        timings['geometry_ms']+=(time.perf_counter()-t)*1000
        candidates.append({'angle':angle,'matches':len(p),'inliers':n,'center':center,'homography':H.tolist() if H is not None else None})
        if save_correspondences:
            correspondences[f'a{angle}_uav']=p
            correspondences[f'a{angle}_satellite']=q
        del a,b,fa,fb
    selected=max(candidates,key=lambda c:(c['inliers'] if c['center'] is not None else -1,c['matches']))
    torch.cuda.synchronize()
    elapsed=(time.perf_counter()-begin)*1000
    for start,middle,end in events:
        timings['feature_gpu_ms']+=start.elapsed_time(middle)
        timings['matching_gpu_ms']+=middle.elapsed_time(end)
    return {'selected':selected,'candidates':candidates,'e2e_ms':elapsed,**timings,
            '_correspondences':correspondences if save_correspondences else None,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--num-headings',type=int,choices=[4,8],default=8)
    p.add_argument('--lowlight',choices=['daylight','mild','severe'],default='daylight')
    p.add_argument('--enhance',choices=['none','gamma','clahe','gamma_clahe'],default='none')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:raise RuntimeError('GPU compute process already running: '+processes)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    backend=Backend();backend.set_precision(args.amp)
    env={'python':os.path.realpath(os.sys.executable),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
         'model':backend.meta,'opencv':cv2.__version__,'numpy':np.__version__,
         'pipeline':'standalone official adapter; team geometry/evaluation unverified',
         'gt_used_for_inference':False,'cache':False,'amp':args.amp,'num_headings':args.num_headings,
         'lowlight':args.lowlight,'enhance':args.enhance,'warmup_pairs':10,
         'resize':'nearest multiple14, antialias bilinear','rotation':'same canvas, black border; team parity unverified'}
    (args.out/'environment.json').write_text(json.dumps(env,indent=2))
    def images(k):
        result=[]
        for kind in ['uav','block']:
            bgr=cv2.imread(str(args.data/'3d_ho37bc'/f'k{k:03d}_{kind}.jpg'))
            if bgr is None:raise ValueError(f'Missing {k} {kind}')
            result.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        return result
    # Probe actual Conv/Linear compute dtypes outside measured inference.
    audit={}
    def hook(name):
        def check(module,inputs,output):
            if isinstance(output,torch.Tensor):
                group='detector' if name.startswith('_detector') else 'descriptor' if name.startswith('_descriptor') else 'matcher'
                key=group+':'+str(output.dtype);audit[key]=audit.get(key,0)+1
                if output.dtype!=backend.dtype:raise ValueError(f'Wrong dtype {name}: {output.dtype}')
        return check
    handles=[m.register_forward_hook(hook(n)) for n,m in backend.model.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]
    infer(backend,*images(0),[0])
    for h in handles:h.remove()
    (args.out/'dtype_audit.json').write_text(json.dumps(audit,indent=2))
    for i in range(10):infer(backend,*images(i%5),[0])
    rows=[]
    for k in range(5):
        result=infer(backend,*images(k),list(range(0,360,45 if args.num_headings==8 else 90)),args.lowlight,args.enhance,20261110+k)
        result.pop('_correspondences')
        result.update(k=k,status='ok' if result['selected']['center'] else 'no_homography')
        (args.out/f'k{k:03d}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        rows.append(result)
        print(json.dumps({k:result[k] for k in ['k','status','e2e_ms','peak_allocated_mib']}),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'n':5,'valid_homography':sum(r['status']=='ok' for r in rows),'mean_ms':float(np.mean([r['e2e_ms'] for r in rows])),
       'accuracy_evaluated':False,'reason':'team pipeline source/evaluation not available'},indent=2))

if __name__=='__main__':main()
