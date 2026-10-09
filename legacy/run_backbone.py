# 대체 백본 비교 (학습 없음): 같은 dense 매칭에 특징만 교체. NY 1 km 32궤적, 주간 / L2(노이즈 시드 0).
import os, sys, time, gc
from unittest.mock import MagicMock
import numpy as np
for k, v in {'complex_': np.complex128, 'float_': np.float64, 'unicode_': np.str_, 'string_': np.bytes_}.items():
    if not hasattr(np, k): setattr(np, k, v)
sys.modules['imgaug'] = MagicMock(); sys.modules['imgaug.augmenters'] = MagicMock()
os.chdir('/content/bearinguav'); sys.path.insert(0, '/content/bearinguav')

import torch, torch.nn.functional as F, torchvision
from diag.diag_modules import DEV, UNSEEN, to_tensor, load_model, darken
from diag.diag_nav import load_rsi
from diag.diag_phase0 import make_traj, nav_run, wilson, mcnemar
from diag.diag_gluf import rot_tpl

OUT = '/content/drive/MyDrive/bearinguav_cache/nav_backbone_NY.txt'


class DarkWrap:
    """loc 앞에서 UAV 크롭만 어둡게 (위성 타일은 그대로)"""
    def __init__(self, loc, lvl, seed=0):
        self.loc, self.lvl, self.rng = loc, lvl, np.random.default_rng(seed)
    def __call__(self, crops, headings, blocks):
        return self.loc([darken(c, self.lvl, self.rng) for c in crops], headings, blocks)


class FeatLoc:
    def __init__(self, fe, st, img): self.fe, self.st, self.img, self.c = fe, st, img, {}
    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.c:
            sx, sy = bx*256, by*256
            B = self.fe(to_tensor(self.img[sy:sy+512, sx:sx+512]).unsqueeze(0).to(DEV))[0]
            mu = B.mean((1, 2), keepdim=True)
            self.c[(bx, by)] = (F.normalize(B - mu, dim=0), mu)
        return self.c[(bx, by)]
    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        tpl = torch.stack([to_tensor(rot_tpl(c, h)) for c, h in zip(crops, headings)])
        Uf = torch.cat([self.fe(tpl[k:k+64].to(DEV)) for k in range(0, len(tpl), 64)])
        hp = Uf.shape[-1]; out = []
        for j, (bx, by) in enumerate(blocks):
            Bf, mu = self.block(bx, by)
            U = F.normalize(Uf[j].unsqueeze(0) - mu.unsqueeze(0), dim=1)
            corr = F.conv2d(Bf.unsqueeze(0), U)[0] / (hp*hp)
            _, Hc, Wc = corr.shape
            iy, ix = divmod(int(corr.flatten().argmax()) % (Hc*Wc), Wc)
            out.append(((ix*self.st + hp*self.st/2 - 256)/128, (iy*self.st + hp*self.st/2 - 256)/128))
        return np.array(out)


def resnet_fe(name, upto):
    r = (torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) if name == 'resnet18'
         else torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)).eval().to(DEV)
    layers = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2] + ([r.layer3] if upto == 3 else [])
    return torch.nn.Sequential(*layers).eval()


def log(s):
    print(s, flush=True)
    with open(OUT, 'a') as f: f.write(s + '\n')


m2 = load_model('/content/bearinguav/Bearing_UAV/satellite_view')
img = load_rsi(UNSEEN['NY'])
starts, goals = make_traj(32, 1000, 2026)

cands = {  # 이름: (특징 추출기 생성 함수, stride)
    'vgg_conv4_3 (기준)': (lambda: m2.backbone, 8),
    'vgg_conv3_3':        (lambda: m2.backbone[:16], 4),
    'resnet18_layer2':    (lambda: resnet_fe('resnet18', 2), 8),
    'resnet18_layer3':    (lambda: resnet_fe('resnet18', 3), 16),
    'resnet50_layer2':    (lambda: resnet_fe('resnet50', 2), 8),
}

open(OUT, 'w').close()
log('# 대체 백본 비교 | NY 1 km, 32궤적, 시드 2026, compass, L2 노이즈 시드 0 | p는 vgg_conv4_3 대비 McNemar 정확검정')
log('backbone\tcond\tsucc\tCI95\tmed_err_m\tp_vs_ref\tsec')
ref = {}
for name, (mk, st) in cands.items():
    fe = mk()
    with torch.no_grad():  # stride 확인: 176 템플릿 → hp*st == 176 이어야 중심 보정이 맞음
        hp = fe(torch.zeros(1, 3, 176, 176, device=DEV)).shape[-1]
    assert hp * st == 176, (name, hp, st)
    for cond in ['day', 'L2']:
        loc = FeatLoc(fe, st, img)
        L = loc if cond == 'day' else DarkWrap(loc, 'L2', 0)
        t0 = time.time()
        succ, reason, err = nav_run(L, img, starts, goals, decode_px=128)
        dt = time.time() - t0
        if name.startswith('vgg_conv4_3'): ref[cond] = succ
        a, b, p = mcnemar(succ, ref[cond])
        log(f'{name}\t{cond}\t{succ.sum()}/32 ({100*succ.mean():.1f}%)\t{wilson(int(succ.sum()), 32)}\t'
            f'{np.median(err):.1f}\t+{a}/-{b} p={p:.3g}\t{dt:.0f}')
        del loc, L; gc.collect(); torch.cuda.empty_cache()
    del fe; gc.collect(); torch.cuda.empty_cache()
log('# done')
