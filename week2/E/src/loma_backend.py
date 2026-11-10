"""Official Full LoMa-B adapter; standalone, not a verified team imatch clone."""
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F

from runtime_paths import ROOT
sys.path.insert(0,str(ROOT/'vendor/LoMa-main/src'))

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Backend:
    def __init__(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        spec=json.loads((ROOT/'archive_reference/configs/weights.expected.json').read_text())
        loaded={}
        original=torch.hub.load_state_dict_from_url
        def cached(url,*args,**kwargs):
            name=url.rsplit('/',1)[-1]
            p=ROOT/'downloads/weights'/name
            if name not in spec or digest(p)!=spec[name]['sha256']:raise ValueError(name)
            loaded[name]={'sha256':digest(p),'bytes':p.stat().st_size,'url':url}
            return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        torch.hub.load_state_dict_from_url=cached
        try:
            from loma import LoMa,LoMaB
            self.model=LoMa(LoMaB(mp=False,compile=False)).float().eval()
            state=cached(spec['loma_B.pt']['url'])
            keys=self.model.load_state_dict(state,strict=True)
            del state
        finally:torch.hub.load_state_dict_from_url=original
        self.model.to('cuda:0')
        self.meta={'parameters':sum(p.numel() for p in self.model.parameters()),
                   'missing_keys':list(keys.missing_keys),'unexpected_keys':list(keys.unexpected_keys),
                   'weights':loaded,'device':str(next(self.model.parameters()).device),
                   'descriptor_note':'DeDoDe-G weights embedded in full checkpoint; separate DeDoDe download not required'}
        assert self.meta['parameters']==341523567
        self.set_precision(False)

    def set_precision(self,amp):
        self.amp=amp
        self.dtype=torch.float16 if amp else torch.float32
        self.model.cfg=replace(self.model.cfg,mp=amp)
        # Imported module globals and per-module settings must agree: outer AMP alone
        # does not override nested autocast or FrozenDINOv2's explicit input cast.
        for name,module in list(sys.modules.items()):
            if name=='loma' or name.startswith('loma.'):
                if hasattr(module,'device'):module.device=torch.device('cuda:0')
                if hasattr(module,'amp_dtype'):module.amp_dtype=self.dtype
        for module in self.model.modules():
            if hasattr(module,'amp'):module.amp=amp
            if hasattr(module,'amp_dtype'):module.amp_dtype=self.dtype
        self.model.float()

    @contextmanager
    def context(self):
        with torch.inference_mode(),torch.autocast('cuda',enabled=self.amp,dtype=torch.float16):yield

    def preprocess(self,rgb,input_scale=1.0):
        # VisMatch nearest multiple of 14, float32 bilinear interpolation.
        tensor=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float()[None].to('cuda:0')/255
        h,w=rgb.shape[:2]
        return F.interpolate(tensor,size=(max(14,round(h*input_scale/14)*14),max(14,round(w*input_scale/14)*14)),mode='bilinear',align_corners=False,antialias=True)

    def features(self,image):
        with self.context():
            k,d,_,_=self.model.detect_and_describe(image,2048)
        return k,d

    def match(self,a,b,shape_a,shape_b):
        from loma.loma import filter_matches,to_pixel_coords
        with self.context():
            scores=self.model(a[0],b[0],a[1],b[1])['scores']
            m,_,_,_=filter_matches(scores,0.1)
            valid=m[0]>=0
            # Normalized coordinates map directly to original shape (resize undone).
            p=to_pixel_coords(a[0][0][valid],*shape_a).float().cpu().numpy()-0.5
            q=to_pixel_coords(b[0][0][m[0][valid]],*shape_b).float().cpu().numpy()-0.5
            finite=bool(torch.isfinite(scores).all().item())
        if not finite or not(np.isfinite(p).all() and np.isfinite(q).all()):raise FloatingPointError('nonfinite match')
        good=((p>=0)&(p<[shape_a[1],shape_a[0]])&(q>=0)&(q<[shape_b[1],shape_b[0]])).all(1)
        return p[good],q[good]

    def check_features(self,features):
        for tensor in features:
            if not bool(torch.isfinite(tensor).all().item()):raise FloatingPointError('nonfinite feature')

if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    backend=Backend()
    target=ROOT/'model_load_audit.json'
    with target.open('x') as f:json.dump(backend.meta,f,indent=2)
    print(json.dumps(backend.meta,indent=2))
