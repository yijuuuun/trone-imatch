
import sys, json, hashlib, os
from pathlib import Path
import torch
from benchmark_core import (IMatchAdapter, BearingAdapter, benchmark, save_json)
import pandas as pd
config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
name=config['name']
out=Path(config['output']); out.mkdir(parents=True,exist_ok=True)
signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
result=out/'summary.json'
if result.exists():
    saved=json.loads(result.read_text())
    if saved.get('config_signature') != signature:
        raise ValueError('Completed result uses a different protocol/weights/data. Start a new RUN.')
    print('Preserved completed result:',result,flush=True)
    sys.exit(0)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
manifest=pd.read_csv(config['queries'],dtype={'sample_id':str})
if config['smoke']: manifest=manifest.head(3).copy()
print(name,'loading pretrained model',flush=True)
if name=='Bearing-UAV':
    adapter=BearingAdapter(Path(config['repo']),config['weights_dir'],'cuda:0')
else:
    adapter=IMatchAdapter('cuda:0',config['angles'],config['stop_inliers'],config['checkpoint'])
summary=benchmark(adapter,manifest,out,config['audit'],'cuda:0',config['meters_per_unit'],
                  1 if config['smoke'] else config['warmup'],1 if config['smoke'] else config['repeats'])
summary.update(config_signature=signature,smoke=config['smoke'],latency_scope='pair preprocessing + H2D + forward + geometry; no gallery cache')
save_json(result,summary)
print(name,'completed',flush=True)
