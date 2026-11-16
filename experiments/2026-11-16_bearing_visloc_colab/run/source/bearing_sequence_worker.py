
from pathlib import Path
import sys, json, time, random
import cv2, numpy as np, pandas as pd, torch
from bearing_adapter import BearingAdapter, sha256
from sequence_replay import load_segments, replay_one, summarize, ODO

class BearingLoc:
    def __init__(self, adapter, city): self.adapter,self.city,self.last=adapter,city,[]
    def __call__(self, crops, headings, blocks):
        output=[]; self.last=[]
        for crop,(bx,by) in zip(crops,blocks):
            block=self.city[by*256:by*256+512,bx*256:bx*256+512]
            if crop.shape!=(256,256,3) or block.shape!=(512,512,3):
                raise ValueError(f'Unexpected image shapes: {crop.shape}, {block.shape}')
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode(): pred=self.adapter.predict(crop,block)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            xy=np.array([pred['x'],pred['y']],float)
            if not pred['valid'] or not np.isfinite(xy).all(): raise ValueError('Invalid Bearing prediction')
            output.append(xy); self.last.append(dict(sec=time.perf_counter()-t))
        return np.array(output)

def main(config):
    out=Path(config['output']); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Output already completed; create a new RUN to rerun.')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    cv2.setRNGSeed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    adapter=BearingAdapter(Path(config['repo']),config['weights_dir'],'cuda:0')
    if adapter.metadata['checkpoint_sha256']!=config['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    groups=load_segments(config['data'],sets=config['sets'] or None)
    if config['smoke']:
        st=next(iter(groups)); groups={st:groups[st].head(3).copy()}
    logs=[]; rows=[]; t=time.perf_counter()
    for i,(st,g) in enumerate(groups.items()):
        city=cv2.cvtColor(cv2.imread(str(Path(config['data'])/'maps'/f'{st}.jpg')),cv2.COLOR_BGR2RGB)
        loc=BearingLoc(adapter,city)
        if i==0:
            crop=cv2.cvtColor(cv2.imread(str(Path(config['data'])/st/f'k{int(g.k.iloc[0]):03d}_uav.jpg')),cv2.COLOR_BGR2RGB)
            for _ in range(5): loc([crop],None,[(0,0)])
        log=[]
        replay_one(loc,g,config['data'],st,i,odo=ODO[config['odo']],seed=config['seed'],log=log)
        L=pd.DataFrame(log); scored=L.iloc[1:]; e=scored.err_m
        rows.append(dict(set=st,traj=i,n_steps=len(scored),length_m=float(g.dist_m.iloc[-1]),
            final_err_m=float(e.iloc[-1]),max_err_m=float(e.max()),med_err_m=float(e.median()),
            w10=float((e<=10).mean()),w20=float((e<=20).mean()),w32=float((e<=32).mean()),unknown=0.0))
        logs.extend(log)
        L.to_csv(out/'steps.csv',mode='w' if i==0 else 'a',header=i==0,index=False)
        pd.DataFrame(rows).to_csv(out/'segments.csv',index=False)
        print(f'{i+1}/{len(groups)} {st}: final {e.iloc[-1]:.1f} m',flush=True)
    steps=pd.DataFrame(logs); segs=pd.DataFrame(rows); result=summarize(steps,segs)
    scored=steps[steps.k!=steps.groupby('set').k.transform('min')]
    result.update(smoke=config['smoke'],seed=config['seed'],odo=config['odo'],
        known_start=True,wall_s=time.perf_counter()-t,mean_inference_s=float(scored.sec.mean()),
        latency_scope='preprocessing + H2D + forward; excludes disk IO; warmup5',**adapter.metadata)
    result['variant']='r0'
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame([result]).to_csv(out/'summary.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
