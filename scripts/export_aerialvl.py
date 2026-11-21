#!/usr/bin/env python
"""AerialVL (He et al., RA-L 2024, CC-BY 4.0) VAL 시퀀스 → seq_eval과 같은 형식 (imatch/replay.py가 그대로 읽음).

두 번째 실제 비행 시험 데이터. 설정은 고르지 않는다: v1/R1의 픽셀 설정(블록 512 px, 드론 256 px, inlier 15, 4방향)을 그대로 쓴다.

  python scripts/export_aerialvl.py --out data/aerialvl          # HuggingFace에서 필요한 파일만 range 요청으로 받음

축척 (결과를 보기 전에 정함)
- 위성 지도는 0.96 m/px라 VisLoc(0.25 m/px)보다 거칠다. 드론 사진은 약 0.09~0.15 m/px, 짧은 변 1536 px ≈ 185 m.
- 그래서 장면 전체를 SCALE = 2.5배로 본다: 지도 0.625 m/px로 다시 샘플링, 드론 사진 가운데 160 m를 256 px로.
  파이프라인 픽셀 세계는 VisLoc과 같고, 파이프라인이 내는 "m"는 실제 m ÷ 2.5다 (samples.csv의 map_x/y, dist_m도
  파이프라인 m). 실제 m 오차 = err_m × 2.5. 160 m는 사진 짧은 변(약 185 m) 안에 들어가는 가장 큰 둥근 값.
- 걸음 간격: VisLoc 사진 간격 중앙값(약 80 m) × 2.5 = 200 m(실제). 구간: 실제 1,500~2,500 m (VisLoc 600~1,000 m × 2.5).
- 드론 축척(gsd): 연속 사진 쌍(8장 차이, 약 28 m)의 SIFT 이동량과 GPS 거리로 걸음마다 재고, 시퀀스 중앙값의
  0.6~1.6배 밖이면 중앙값을 쓴다. 고도계로 아는 값에 해당한다. 위성 지도·정답 매칭은 쓰지 않는다.
- 출발점을 50 m씩 밀어 시퀀스마다 4벌(offset)을 만든다(같은 경로, 다른 사진). 구간끼리 독립이 아니다.
- 사진 방향: AerialVL은 사진을 동쪽이 위가 되게 돌려 두었다. 파이프라인은 방향을 받지 않고 4방향을 다 시도한다.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.blocks import N_BLOCK, crop_block  # noqa: E402
from imatch.geometry import BLOCK_CENTER  # noqa: E402

SCALE = 2.5
PX_PER_M = 4.0 / SCALE            # 실제 m당 지도 px (0.625 m/px)
UAV_PX, UAV_M = 256, 160.0        # 실제 160 m → 256 px
STRIDE, MAP_PX = 256, 256 * (N_BLOCK + 1)
STEP_M, SEG_MAX, SEG_MIN, MARGIN_M = 200.0, 2500.0, 1500.0, 80.0
OFFSETS = (0.0, 50.0, 100.0, 150.0)
LAT_REF = 36.59

HF = 'https://huggingface.co/datasets/hmf21/AerialVL/resolve/main/images/VAL/VAL.zip.%03d'
HF_TREE = 'https://huggingface.co/api/datasets/hmf21/AerialVL/tree/main/images/VAL'


# ------------------------------------------------------------ 나뉜 zip을 HTTP range로 읽기

class Remote:
    def __init__(self, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        tp = cache / 'parts.json'
        if not tp.exists():
            tp.write_bytes(urllib.request.urlopen(HF_TREE, timeout=60).read())
        parts = sorted(json.loads(tp.read_text()), key=lambda p: p['path'])
        self.sizes = [p['size'] for p in parts]
        self.offs = np.concatenate([[0], np.cumsum(self.sizes)])
        ip = cache / 'index.tsv'
        if not ip.exists():
            z = zipfile.ZipFile(io.BufferedReader(_Raw(self), buffer_size=1 << 20))
            with open(ip, 'w') as f:
                for zi in z.infolist():
                    f.write(f'{zi.filename}\t{zi.header_offset}\t{zi.compress_size}\t{zi.compress_type}\n')
        self.index = {}
        for line in open(ip):
            n, a, b, c = line.rstrip('\n').split('\t')
            self.index[n] = (int(a), int(b), int(c))

    def get(self, pos, n):
        out = bytearray()
        while n > 0:
            i = int(np.searchsorted(self.offs, pos, side='right') - 1)
            a = pos - int(self.offs[i]); b = min(self.sizes[i] - 1, a + n - 1)
            for t in range(8):
                try:
                    req = urllib.request.Request(HF % (i + 1), headers={'Range': f'bytes={a}-{b}'})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        chunk = r.read()
                    break
                except Exception:
                    time.sleep(2 * (t + 1))
            else:
                raise RuntimeError(f'range {i} {a}-{b}')
            out += chunk; pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def read(self, name):
        off, csz, ctype = self.index[name]
        raw = self.get(off, 30 + len(name.encode()) + 512 + csz)
        assert raw[:4] == b'PK\x03\x04', name
        nl, xl = struct.unpack('<HH', raw[26:30])
        data = raw[30 + nl + xl:30 + nl + xl + csz]
        return data if ctype == 0 else zlib.decompress(data, -15)


class _Raw(io.RawIOBase):
    def __init__(self, r): self.r, self.pos, self.total = r, 0, int(r.offs[-1])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.total + o); return self.pos
    def readinto(self, b):
        n = min(len(b), self.total - self.pos)
        d = self.r.get(self.pos, n) if n > 0 else b''
        b[:len(d)] = d; self.pos += len(d); return len(d)


# ------------------------------------------------------------ 좌표

MX = 111412.84 * math.cos(math.radians(LAT_REF)) - 93.5 * math.cos(3 * math.radians(LAT_REF))
MY = 111132.92 - 559.82 * math.cos(2 * math.radians(LAT_REF)) + 1.175 * math.cos(4 * math.radians(LAT_REF))


def parse_map_name(n):
    p = n.split('@')
    return dict(lon0=float(p[2]), lat0=float(p[3]), lon1=float(p[4]), lat1=float(p[5]))


def frames_of(index):
    rows = []
    for n in index:
        p = n.split('/')
        if len(p) == 3 and p[2].endswith('.png'):
            t, lon, lat = p[2][1:-5].split('@')[:3]
            rows.append(dict(group=p[0], seq=p[1], t=int(t), lon=float(lon), lat=float(lat), name=n))
    df = pd.DataFrame(rows).sort_values(['seq', 't']).reset_index(drop=True)
    return df


def gray(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert('L'))


def pair_gsd(img1, img2, dist_m):
    sift = cv2.SIFT_create(4000)
    g1 = cv2.resize(gray(img1), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(gray(img2), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return np.nan
    good = [p[0] for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2 and p[0].distance < .75 * p[1].distance]
    if len(good) < 30:
        return np.nan
    p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3)
    if A is None or inl.sum() < 30 or abs(math.hypot(A[0, 0], A[1, 0]) - 1) > .05:
        return np.nan
    c = np.array([g1.shape[1] / 2, g1.shape[0] / 2, 1.])
    shift = np.linalg.norm(A @ c - c[:2]) * 2
    return dist_m / shift if shift > 20 else np.nan


def uav_crop(img_bytes, gsd):
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    w, h = im.size
    c = 2 * round(UAV_M / gsd / 2)
    if c <= min(w, h):
        l, t = (w - c) // 2, (h - c) // 2
        return im.crop((l, t, l + c, t + c)).resize((UAV_PX, UAV_PX), Image.Resampling.LANCZOS), c
    # 낮게 날아 사진 짧은 변이 160 m보다 작으면: 짧은 변만큼 잘라 축척을 지키고(0.625 m/px), 가장자리는 검은색
    c = min(w, h); l, t = (w - c) // 2, (h - c) // 2
    s = int(round(UAV_PX * c / (UAV_M / gsd)))
    small = im.crop((l, t, l + c, t + c)).resize((s, s), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (UAV_PX, UAV_PX))
    canvas.paste(small, ((UAV_PX - s) // 2, (UAV_PX - s) // 2))
    return canvas, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/aerialvl')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--quality', type=int, default=92)
    a = ap.parse_args()
    out = Path(a.out); cache = out / '_cache'
    R = Remote(cache)
    F = frames_of(R.index)

    # 지도: large_map을 0.625 m/px 평면으로 (원점 = 지도 왼쪽 위, X 동쪽, Y 남쪽)
    mp = cache / 'large_map.png'
    mname = [n for n in R.index if 'large_map' in n][0]
    g = parse_map_name(mname)
    if not mp.exists():
        Image.MAX_IMAGE_PIXELS = None
        src = np.asarray(Image.open(io.BytesIO(R.read(mname))).convert('RGB'))
        Wm, Hm = (g['lon1'] - g['lon0']) * MX, (g['lat0'] - g['lat1']) * MY
        W, H = int(Wm * PX_PER_M), int(Hm * PX_PER_M)
        cv2.imwrite(str(mp), cv2.cvtColor(cv2.resize(src, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR))
        print(f'지도 {src.shape[1]}x{src.shape[0]} ({Wm / src.shape[1]:.3f} m/px) → {W}x{H}', flush=True)
    big = cv2.cvtColor(cv2.imread(str(mp)), cv2.COLOR_BGR2RGB)
    F['X'] = (F.lon - g['lon0']) * MX * PX_PER_M
    F['Y'] = (g['lat0'] - F.lat) * MY * PX_PER_M

    # 걸음 고르기: 시퀀스마다 누적 거리 offset + 200 m 간격에 가장 가까운 사진
    picks = []
    for s, d in F.groupby('seq', sort=True):
        d = d.reset_index()
        cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(d.X), np.diff(d.Y)) / PX_PER_M)])
        for o in OFFSETS:
            for j, target in enumerate(np.arange(o, cum[-1], STEP_M)):
                i = int(np.argmin(np.abs(cum - target)))
                if abs(cum[i] - target) < 10 and i + 8 < len(d):
                    picks.append(dict(seq=s, off=o, j=j, i=i, cum=cum[i], name=d.name[i], partner=d.name[i + 8],
                                      X=d.X[i], Y=d.Y[i], pdist=float(np.hypot(d.X[i + 8] - d.X[i], d.Y[i + 8] - d.Y[i]) / PX_PER_M)))
    P = pd.DataFrame(picks)

    # 구간: offset마다 앞에서부터, 실제 1,500~2,500 m, 지도 창(4096 px) 안
    usable = MAP_PX - 2 * MARGIN_M * PX_PER_M
    segs = []
    for (s, o), d in P.groupby(['seq', 'off'], sort=True):
        d = d.sort_values('j'); idx = list(d.index); i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx):
                sub = P.loc[idx[i:j + 2]]
                if np.ptp(sub.X) > usable or np.ptp(sub.Y) > usable or P.cum[idx[j + 1]] - P.cum[idx[i]] > SEG_MAX:
                    break
                j += 1
            if P.cum[idx[j]] - P.cum[idx[i]] >= SEG_MIN:
                segs.append((s, o, idx[i:j + 1]))
            i = j + 1
    need = sorted(set(P.loc[[k for _, _, ks in segs for k in ks], 'name']) |
                  set(P.loc[[k for _, _, ks in segs for k in ks], 'partner']))
    print(f'시퀀스 {F.seq.nunique()}, 구간 {len(segs)}, 걸음 {sum(len(k) for *_, k in segs)}, 받을 사진 {len(need)}', flush=True)

    # 사진 받기 (cache/frames/에 원본 png 그대로)
    fd = cache / 'frames'; fd.mkdir(exist_ok=True)

    def get(n):
        p = fd / (n.replace('/', '__'))
        if not p.exists():
            b = R.read(n); tmp = p.with_suffix('.tmp'); tmp.write_bytes(b); tmp.rename(p)
        return p
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for k, _ in enumerate(ex.map(get, need)):
            if (k + 1) % 100 == 0:
                print(f'  {k + 1}/{len(need)} ({time.time() - t0:.0f}s)', flush=True)

    # 축척
    def gsd_of(r):
        return pair_gsd((fd / r['name'].replace('/', '__')).read_bytes(),
                        (fd / r['partner'].replace('/', '__')).read_bytes(), r['pdist'])
    used = sorted({k for *_, ks in segs for k in ks})
    with ThreadPoolExecutor(a.threads) as ex:
        P.loc[used, 'gsd_pair'] = list(ex.map(lambda k: gsd_of(P.loc[k]), used))
    med = P.groupby('seq').gsd_pair.median()
    P['gsd_med'] = P.seq.map(med)
    ok = (P.gsd_pair > .6 * P.gsd_med) & (P.gsd_pair < 1.6 * P.gsd_med)
    P['gsd'] = np.where(ok, P.gsd_pair, P.gsd_med)
    P['gsd_src'] = np.where(ok, 'pair', 'seq_median')

    # 내보내기
    d_out = out / 'seq_test'; (d_out / 'maps').mkdir(parents=True, exist_ok=True)
    rows, idx_rows = [], []
    seq_id = {s: i for i, s in enumerate(sorted(F.seq.unique()))}
    for n_seg, (s, o, ks) in enumerate(segs):
        sub = P.loc[ks]
        cx, cy = (sub.X.min() + sub.X.max()) / 2, (sub.Y.min() + sub.Y.max()) / 2
        X0 = int(round(min(max(cx - MAP_PX / 2, 0), big.shape[1] - MAP_PX)))
        Y0 = int(round(min(max(cy - MAP_PX / 2, 0), big.shape[0] - MAP_PX)))
        mx_, my_ = sub.X - X0, sub.Y - Y0
        m = MARGIN_M * PX_PER_M
        if mx_.min() < m or my_.min() < m or mx_.max() > MAP_PX - m or my_.max() > MAP_PX - m:
            print('  지도 창 밖, 건너뜀', s, o); continue
        city = big[Y0:Y0 + MAP_PX, X0:X0 + MAP_PX]
        name = f'avl_{seq_id[s]:02d}_o{int(o):03d}_s{n_seg:03d}'
        dd = d_out / name; dd.mkdir(exist_ok=True)
        cv2.imwrite(str(d_out / 'maps' / f'{name}.jpg'), cv2.cvtColor(city, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        c0 = sub.cum.iloc[0]
        for k, r in enumerate(sub.itertuples()):
            uav, cpx = uav_crop((fd / r.name.replace('/', '__')).read_bytes(), r.gsd)
            uav.save(dd / f'k{k:03d}_uav.jpg', quality=a.quality)
            xs, ys = r.X - X0, r.Y - Y0
            bx = int(np.clip(np.round((xs - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            by = int(np.clip(np.round((ys - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            cv2.imwrite(str(dd / f'k{k:03d}_block.jpg'), cv2.cvtColor(crop_block(city, bx, by), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append(dict(set=name, k=k, seq=s, offset_m=o, filename=r.name, lon=P.loc[ks].lon.iloc[k] if 'lon' in P else np.nan,
                             map_x=round(float(xs), 2), map_y=round(float(ys), 2),
                             dist_m=round((r.cum - c0) / SCALE, 2), dist_real_m=round(r.cum - c0, 2),
                             gsd=round(float(r.gsd), 4), gsd_src=r.gsd_src, uav_crop_px=cpx,
                             crop_m=round(cpx * float(r.gsd), 1), scale=SCALE))
            idx_rows.append(dict(set=name, k=k, block_x=bx, block_y=by))
    S = pd.DataFrame(rows).drop(columns=['lon'])
    S.to_csv(d_out / 'samples.csv', index=False)
    pd.DataFrame(idx_rows).to_csv(d_out / 'block_index.csv', index=False)
    (d_out / 'README.txt').write_text(__doc__)
    g2 = S.groupby('set')
    print(f'내보냄: {S.set.nunique()}구간, {len(S)}장 (출발 제외 {len(S) - S.set.nunique()}걸음), '
          f'구간 길이 실제 {g2.dist_real_m.max().min():.0f}~{g2.dist_real_m.max().max():.0f} m, '
          f'gsd 쌍 {np.mean(S.gsd_src == "pair") * 100:.0f}%, 160 m 못 채운 사진 {np.mean(S.crop_m < 159) * 100:.1f}%')


if __name__ == '__main__':
    main()
