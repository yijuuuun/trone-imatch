# (1) 지도 저장량 실측: NY 지도 전체를 타일 단위로 저장해 파일 크기를 잰다.
# (2) dense가 느린 원인: 상관 계산 구현별 시간 (conv2d 루프 / 배치 grouped conv / unfold+matmul / FFT).
import os, sys, time, gc
from unittest.mock import MagicMock
import numpy as np
for k, v in {'complex_': np.complex128, 'float_': np.float64, 'unicode_': np.str_, 'string_': np.bytes_}.items():
    if not hasattr(np, k): setattr(np, k, v)
sys.modules['imgaug'] = MagicMock(); sys.modules['imgaug.augmenters'] = MagicMock()
os.chdir('/content/bearinguav'); sys.path.insert(0, '/content/bearinguav')

import torch, torch.nn.functional as F
from diag.diag_modules import DEV, UNSEEN, to_tensor, load_model, enc
from diag.diag_nav import load_rsi
from diag.diag_phase0 import make_traj, Cropper, ModelLoc, DenseLoc
from diag.diag_gluf import rot_tpl

OUT = '/content/drive/MyDrive/bearinguav_cache/cost_NY.txt'
TMP = '/content/cost_tmp'; os.makedirs(TMP, exist_ok=True)


def log(s):
    print(s, flush=True)
    with open(OUT, 'a') as f: f.write(s + '\n')


def timeit(fn, n=50, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1000


m2 = load_model('/content/bearinguav/Bearing_UAV/satellite_view')
img = load_rsi(UNSEEN['NY'])
open(OUT, 'w').close()
H, W = img.shape[:2]; nt = (H // 256) * (W // 256)
log(f'# 비용 실측 | NY 지도 {W}x{H}px = {W*0.25/1000:.3f} x {H*0.25/1000:.3f} km, 256px 타일 {nt}장')

# ---------------- (1) 저장량
with torch.no_grad():
    dense, desc = [], []
    for y in range(0, H - 255, 256):
        for x in range(0, W - 255, 256):
            t = to_tensor(img[y:y+256, x:x+256]).unsqueeze(0).to(DEV)
            dense.append(m2.backbone(t)[0].cpu())
            desc.append(enc(m2, t)[0].cpu())
    dense, desc = torch.stack(dense), torch.stack(desc)
km2 = (W * 0.25 / 1000) * (H * 0.25 / 1000)
log(f'dense 타일 특징 shape {tuple(dense.shape[1:])}, Bearing 타일 설명자 shape {tuple(desc.shape[1:])}')
for name, t in [('dense fp32', dense), ('dense fp16', dense.half()), ('Bearing fp32', desc), ('Bearing fp16', desc.half())]:
    p = f'{TMP}/x.pt'; torch.save(t.contiguous(), p); sz = os.path.getsize(p)
    log(f'저장량\t{name}\t{sz/1e6:.2f} MB (지도 전체)\t{sz/1e6/km2:.2f} MB/km²\t타일당 {sz/nt/1e3:.1f} KB')
r = dense.numel() / desc.numel(); log(f'dense/Bearing 원소 수 비율: {r:.0f}배')
del dense, desc; gc.collect()

# ---------------- (2) 속도: 한 스텝(32샘플, 블록 각각 다름) 기준
starts, _ = make_traj(32, 1000, 2026); crop = Cropper(img)
crops = [crop(s[0], s[1], 90.0) for s in starts]; heads = np.full(32, 90.0)
blocks = [(int(np.clip(np.round((s[0]-256)/256), 0, 14)), int(np.clip(np.round((s[1]-256)/256), 0, 14))) for s in starts]
dl = DenseLoc(m2, img, compass=True); ml = ModelLoc(m2, img)
for b in blocks: dl.block(*b); ml.tc.block(*b)        # 블록 캐시 미리 채움 (이전 측정과 같은 조건)
tpl = torch.stack([to_tensor(rot_tpl(c, h)) for c, h in zip(crops, heads)]).to(DEV)
with torch.no_grad():
    Uf = m2.backbone(tpl)                                  # [32,512,22,22]
Bs = torch.stack([dl.block(*b)[0] for b in blocks]); mus = torch.stack([dl.block(*b)[1] for b in blocks])
Us = F.normalize(Uf - mus, dim=1); h = Us.shape[-1]; Hb = Bs.shape[-1]; Ho = Hb - h + 1

@torch.no_grad()
def corr_loop():   # 현재 구현: 샘플마다 conv2d 한 번 + argmax를 CPU로
    out = []
    for j in range(32):
        c = F.conv2d(Bs[j:j+1], Us[j:j+1])[0, 0]; out.append(int(c.flatten().argmax()))
    return out

@torch.no_grad()
def corr_loop_noargsync():   # 같은 루프, argmax를 GPU에 모아 마지막에 한 번만 동기화
    return torch.stack([F.conv2d(Bs[j:j+1], Us[j:j+1])[0, 0].flatten().argmax() for j in range(32)]).tolist()

@torch.no_grad()
def corr_group():  # 32개를 grouped conv 한 번으로
    c = F.conv2d(Bs.reshape(1, -1, Hb, Hb), Us, groups=32)[0]
    return c.flatten(1).argmax(1).tolist()

@torch.no_grad()
def corr_fft():
    n = Hb; Fb = torch.fft.rfft2(Bs, s=(n, n)); Fu = torch.fft.rfft2(Us, s=(n, n))
    c = torch.fft.irfft2((Fb * Fu.conj()).sum(1), s=(n, n))[:, :Ho, :Ho]
    return c.flatten(1).argmax(1).tolist()

ref = corr_loop()
log(f'\n# 속도 (한 스텝 = 32샘플, ms/샘플, Colab T4, 블록 특징 캐시 후) | 상관 출력 {Ho}x{Ho}, 커널 {h}x{h}')
log(f'(unfold+matmul은 생략: im2col 행렬이 샘플당 {512*h*h*Ho*Ho*4/1e9:.1f} GB로 메모리 이동이 상관 연산보다 훨씬 커서 의미 없음)')
log(f'백본(템플릿 176, 배치 32)\t{timeit(lambda: m2.backbone(tpl).sum().item())/32:.2f}')
for name, fn in [('상관: conv2d 루프 (현재)', corr_loop), ('상관: 루프, 동기화 1회', corr_loop_noargsync),
                 ('상관: grouped conv 1회', corr_group), ('상관: FFT', corr_fft)]:
    same = sum(a == b for a, b in zip(fn(), ref))
    log(f'{name}\t{timeit(fn)/32:.2f}\t(현재 구현과 argmax 일치 {same}/32)')
log(f'DenseLoc 전체 호출\t{timeit(lambda: dl(crops, heads, blocks), n=20)/32:.2f}')
log(f'ModelLoc(Bearing) 전체 호출\t{timeit(lambda: ml(crops, heads, blocks), n=20)/32:.2f}')
with torch.backends.cudnn.flags(enabled=True, benchmark=True):
    log(f'상관: conv2d 루프, cudnn.benchmark\t{timeit(corr_loop)/32:.2f}')
log('# done')
