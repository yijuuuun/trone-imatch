#!/usr/bin/env python
"""VPAIR (Schleiss et al., arXiv 2205.11567, 학술용) → seq_eval 형식. 세 번째 실제 비행 시험 데이터.

경비행기, 지면 위 300~400 m, 독일 본~아이펠 106 km 한 번 비행, 800x600 사진 2,706장(약 38 m 간격).
VPAIR에는 위성 지도가 없어서, 지도는 공개 항공 정사영상(NRW DOP, 없는 곳은 RLP DOP20 WMS)에서 받는다.
비행(2020-10)과 정사영상 촬영 시기는 다르다 (위성 지도와 같은 조건).

  python scripts/export_vpair.py --out data/vpair

규칙은 export_aerialvl.py와 같다 (결과를 보기 전에 정함):
- SCALE 2.5: 지도 0.625 m/px, 드론 사진 가운데 160 m → 256 px. v1 픽셀 설정 그대로.
- 걸음 간격 200 m(실제), 구간 1,500~2,500 m, 출발점 0/50/100/150 m 4벌. 드론 축척은 연속 사진 쌍(SIFT 이동량 ÷ GPS 거리).
VPAIR에만 있는 규칙:
- 렌즈 왜곡을 camera_calibration.yaml(radtan)로 편 뒤 자른다.
- 기운 사진(|roll| 또는 |pitch| > 5°, VisLoc의 tilted 기준과 같음)은 걸음 후보에서 뺀다. 지면 위 약 310 m에서 5°면 사진 가운데가
  GPS 위치에서 최대 약 27 m 벗어난다(3° 이하만 쓰면 사진 47%만 남아 구간이 17개뿐이라 5°로 정함, 항법 결과를 보기 전).
- 기운 사진 때문에 걸음이 빠지면 그 걸음만 건너뛴다(연속 3걸음 이상 빠지면 구간을 끊음).
"""
from __future__ import annotations

import argparse
import io
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
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from export_aerialvl import (MAP_PX, MARGIN_M, OFFSETS, PX_PER_M, SCALE, SEG_MAX, SEG_MIN, STEP_M, STRIDE,  # noqa: E402
                             UAV_M, UAV_PX, pair_gsd)
from imatch.blocks import N_BLOCK, crop_block  # noqa: E402
from imatch.geometry import BLOCK_CENTER  # noqa: E402

URL = 'https://ltc.cvg.cit.tum.de/vpair/vpair.zip'
TILT_MAX = 5.0
LAT_REF, LON0, LAT0 = 50.64, 6.70, 50.80          # 평면 원점 (지도 왼쪽 위)
MX = 111412.84 * math.cos(math.radians(LAT_REF)) - 93.5 * math.cos(3 * math.radians(LAT_REF))
MY = 111132.92 - 559.82 * math.cos(2 * math.radians(LAT_REF)) + 1.175 * math.cos(4 * math.radians(LAT_REF))
TILE = 2048                                         # 지도 캐시 타일 (px)
WMS = [('https://www.wms.nrw.de/geobasis/wms_nw_dop', 'nw_dop_rgb'),
       ('https://geo4.service24.rlp.de/wms/rp_dop20.fcgi', 'rp_dop20')]


def http(url, headers=None, tries=8):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=180) as r:
                return r.read()
        except Exception:
            time.sleep(3 * (t + 1))
    raise RuntimeError(url)


class Zip:
    def __init__(self, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        ip = cache / 'index.tsv'
        if not ip.exists():
            n = int(urllib.request.urlopen(urllib.request.Request(URL, method='HEAD')).headers['Content-Length'])
            z = zipfile.ZipFile(io.BufferedReader(_Raw(n), buffer_size=1 << 20))
            with open(ip, 'w') as f:
                for zi in z.infolist():
                    f.write(f'{zi.filename}\t{zi.header_offset}\t{zi.compress_size}\t{zi.compress_type}\n')
        self.index = {}
        for line in open(ip):
            n, a, b, c = line.rstrip('\n').split('\t'); self.index[n] = (int(a), int(b), int(c))

    def read(self, name):
        off, csz, ct = self.index[name]
        raw = http(URL, {'Range': f'bytes={off}-{off + 30 + len(name.encode()) + 512 + csz}'})
        nl, xl = struct.unpack('<HH', raw[26:30]); d = raw[30 + nl + xl:30 + nl + xl + csz]
        return d if ct == 0 else zlib.decompress(d, -15)


class _Raw(io.RawIOBase):
    def __init__(self, n): self.n, self.pos = n, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.n + o); return self.pos
    def readinto(self, b):
        k = min(len(b), self.n - self.pos)
        if k <= 0:
            return 0
        d = http(URL, {'Range': f'bytes={self.pos}-{self.pos + k - 1}'}); b[:len(d)] = d; self.pos += len(d); return len(d)


def tile(cache: Path, tx, ty):
    """지도 타일 (tx, ty): 평면 [tx*TILE, +TILE) px. NRW, 빈 곳은 RLP로 채움."""
    p = cache / 'tiles' / f'{tx}_{ty}.jpg'
    if p.exists():
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    p.parent.mkdir(parents=True, exist_ok=True)
    x0, y0 = tx * TILE / PX_PER_M, ty * TILE / PX_PER_M
    lon0, lon1 = LON0 + x0 / MX, LON0 + (x0 + TILE / PX_PER_M) / MX
    lat1, lat0 = LAT0 - y0 / MY, LAT0 - (y0 + TILE / PX_PER_M) / MY
    out = np.zeros((TILE, TILE, 3), np.uint8); hole = np.ones((TILE, TILE), bool)
    for url, layer in WMS:
        q = (f'{url}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:4326'
             f'&BBOX={lat0},{lon0},{lat1},{lon1}&WIDTH={TILE}&HEIGHT={TILE}&FORMAT=image/png&TRANSPARENT=TRUE')
        a = np.asarray(Image.open(io.BytesIO(http(q))).convert('RGBA'))
        ok = (a[..., 3] > 0) & hole
        out[ok] = a[..., :3][ok]; hole &= ~ok
        if not hole.any():
            break
    cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if hole.mean() > 0.01:
        print(f'  타일 {tx},{ty}: 지도 없는 곳 {hole.mean() * 100:.0f}%', flush=True)
    return out


def window(cache, X0, Y0):
    out = np.zeros((MAP_PX, MAP_PX, 3), np.uint8)
    for ty in range(Y0 // TILE, (Y0 + MAP_PX - 1) // TILE + 1):
        for tx in range(X0 // TILE, (X0 + MAP_PX - 1) // TILE + 1):
            t = tile(cache, tx, ty)
            gx0, gy0 = max(X0, tx * TILE), max(Y0, ty * TILE)
            gx1, gy1 = min(X0 + MAP_PX, tx * TILE + TILE), min(Y0 + MAP_PX, ty * TILE + TILE)
            out[gy0 - Y0:gy1 - Y0, gx0 - X0:gx1 - X0] = t[gy0 - ty * TILE:gy1 - ty * TILE, gx0 - tx * TILE:gx1 - tx * TILE]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/vpair')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--quality', type=int, default=92)
    a = ap.parse_args()
    out = Path(a.out); cache = out / '_cache'
    Z = Zip(cache)
    for n in ('poses.csv', 'camera_calibration.yaml', 'License.txt'):
        if not (cache / n).exists():
            (cache / n).write_bytes(Z.read(n))
    cal = yaml.safe_load((cache / 'camera_calibration.yaml').read_text())['cam0']
    fx, fy, cx, cy = cal['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]); D = np.array(cal['distortion_coeffs'])
    F = pd.read_csv(cache / 'poses.csv')
    F['X'] = (F.lon - LON0) * MX * PX_PER_M
    F['Y'] = (LAT0 - F.lat) * MY * PX_PER_M
    F['tilt'] = np.degrees(np.maximum(F.roll.abs(), F.pitch.abs()))
    F['level'] = F.tilt <= TILT_MAX
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(F.X), np.diff(F.Y)) / PX_PER_M)])
    F['cum'] = cum
    print(f'사진 {len(F)}, 비행 {cum[-1] / 1000:.1f} km, 기울기 {TILT_MAX}° 이하 {F.level.mean() * 100:.0f}%', flush=True)

    picks = []
    lv = np.where(F.level)[0]
    for o in OFFSETS:
        for j, target in enumerate(np.arange(o, cum[-1], STEP_M)):
            i = lv[np.argmin(np.abs(cum[lv] - target))]
            if abs(cum[i] - target) < 30 and i + 1 < len(F):
                picks.append(dict(off=o, j=j, i=int(i), cum=cum[i], X=F.X[i], Y=F.Y[i], name=F.filename[i],
                                  partner=F.filename[i + 1], pdist=float(cum[i + 1] - cum[i])))
    P = pd.DataFrame(picks)
    usable = MAP_PX - 2 * MARGIN_M * PX_PER_M
    segs = []
    for o, d in P.groupby('off', sort=True):
        idx = list(d.sort_values('j').index); i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx):
                if P.j[idx[j + 1]] > P.j[idx[j]] + 3:   # 기운 사진 때문에 연속 3걸음 이상 빠지면 구간을 끊는다
                    break
                sub = P.loc[idx[i:j + 2]]
                if np.ptp(sub.X) > usable or np.ptp(sub.Y) > usable or P.cum[idx[j + 1]] - P.cum[idx[i]] > SEG_MAX:
                    break
                j += 1
            if P.cum[idx[j]] - P.cum[idx[i]] >= SEG_MIN:
                segs.append((o, idx[i:j + 1]))
            i = j + 1
    used = sorted({k for _, ks in segs for k in ks})
    need = sorted(set(P.loc[used, 'name']) | set(P.loc[used, 'partner']))
    print(f'구간 {len(segs)}, 걸음 {sum(len(k) for _, k in segs)}, 받을 사진 {len(need)}', flush=True)

    fd = cache / 'queries'; fd.mkdir(exist_ok=True)

    def get(n):
        p = fd / n
        if not p.exists():
            b = Z.read(f'queries/{n}'); t = p.with_suffix('.tmp'); t.write_bytes(b); t.rename(p)
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for k, _ in enumerate(ex.map(get, need)):
            if (k + 1) % 200 == 0:
                print(f'  {k + 1}/{len(need)} ({time.time() - t0:.0f}s)', flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        P.loc[used, 'gsd_pair'] = list(ex.map(lambda k: pair_gsd((fd / P.name[k]).read_bytes(),
                                                                 (fd / P.partner[k]).read_bytes(), P.pdist[k]), used))
    med = float(P.gsd_pair.median())
    ok = (P.gsd_pair > .6 * med) & (P.gsd_pair < 1.6 * med)
    P['gsd'] = np.where(ok, P.gsd_pair, med); P['gsd_src'] = np.where(ok, 'pair', 'median')
    print(f'드론 축척 중앙값 {med:.3f} m/px (지면 위 약 {med * fx:.0f} m)', flush=True)

    d_out = out / 'seq_test'; (d_out / 'maps').mkdir(parents=True, exist_ok=True)
    rows, idx_rows = [], []
    m = MARGIN_M * PX_PER_M
    for n_seg, (o, ks) in enumerate(segs):
        sub = P.loc[ks]
        cxp, cyp = (sub.X.min() + sub.X.max()) / 2, (sub.Y.min() + sub.Y.max()) / 2
        X0, Y0 = int(round(cxp - MAP_PX / 2)), int(round(cyp - MAP_PX / 2))
        city = window(cache, X0, Y0)
        if (city.max(-1) == 0).mean() > 0.001:
            print('  지도 빈 곳, 건너뜀', n_seg); continue
        name = f'vpair_o{int(o):03d}_s{n_seg:03d}'
        dd = d_out / name; dd.mkdir(exist_ok=True)
        cv2.imwrite(str(d_out / 'maps' / f'{name}.jpg'), cv2.cvtColor(city, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        c0 = sub.cum.iloc[0]
        for k, r in enumerate(sub.itertuples()):
            img = cv2.undistort(cv2.imread(str(fd / r.name)), K, D)
            h, w = img.shape[:2]
            c = 2 * round(UAV_M / r.gsd / 2)
            if c <= min(h, w):
                crop = cv2.resize(img[(h - c) // 2:(h + c) // 2, (w - c) // 2:(w + c) // 2], (UAV_PX, UAV_PX), interpolation=cv2.INTER_AREA)
            else:   # 160 m를 못 채우면 축척을 지키고 가장자리 검은색
                c = min(h, w); s = int(round(UAV_PX * c / (UAV_M / r.gsd)))
                crop = np.zeros((UAV_PX, UAV_PX, 3), np.uint8)
                small = cv2.resize(img[(h - c) // 2:(h + c) // 2, (w - c) // 2:(w + c) // 2], (s, s), interpolation=cv2.INTER_AREA)
                crop[(UAV_PX - s) // 2:(UAV_PX - s) // 2 + s, (UAV_PX - s) // 2:(UAV_PX - s) // 2 + s] = small
            cv2.imwrite(str(dd / f'k{k:03d}_uav.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            xs, ys = r.X - X0, r.Y - Y0
            bx = int(np.clip(np.round((xs - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            by = int(np.clip(np.round((ys - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            cv2.imwrite(str(dd / f'k{k:03d}_block.jpg'), cv2.cvtColor(crop_block(city, bx, by), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append(dict(set=name, k=k, filename=r.name, offset_m=o, map_x=round(float(xs), 2), map_y=round(float(ys), 2),
                             dist_m=round((r.cum - c0) / SCALE, 2), dist_real_m=round(r.cum - c0, 2),
                             gsd=round(float(r.gsd), 4), gsd_src=r.gsd_src, uav_crop_px=c, crop_m=round(c * float(r.gsd), 1),
                             tilt_deg=round(float(F.tilt[r.i]), 2), scale=SCALE))
            idx_rows.append(dict(set=name, k=k, block_x=bx, block_y=by))
    S = pd.DataFrame(rows)
    S.to_csv(d_out / 'samples.csv', index=False)
    pd.DataFrame(idx_rows).to_csv(d_out / 'block_index.csv', index=False)
    (d_out / 'README.txt').write_text(__doc__)
    g = S.groupby('set')
    print(f'내보냄: {S.set.nunique()}구간, {len(S)}장 (출발 제외 {len(S) - S.set.nunique()}걸음), '
          f'구간 길이 실제 {g.dist_real_m.max().min():.0f}~{g.dist_real_m.max().max():.0f} m, gsd 쌍 {np.mean(S.gsd_src == "pair") * 100:.0f}%')


if __name__ == '__main__':
    main()
