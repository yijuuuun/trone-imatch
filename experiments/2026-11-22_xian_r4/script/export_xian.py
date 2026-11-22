#!/usr/bin/env python
r"""XIAN-VisLoc (real_world 01~03, 시안) → seq_eval과 같은 형식 (imatch/replay.py가 그대로 읽음).

세 번째 실제 비행 시험 데이터(지민). 설정은 고르지 않는다: v1/R1의 픽셀 설정(블록 512 px, 드론 256 px,
inlier 15, 4방향)을 그대로 쓴다. 아래 규칙은 결과를 보기 전에 정했다 (2026-11-22).

  python scripts/export_xian.py --src E:\R4_Datasets\XIAN-VisLoc --out data/xian --check 6   # 확인 그림만
  python scripts/export_xian.py --src E:\R4_Datasets\XIAN-VisLoc --out data/xian             # 내보내기

데이터
- 비행 01(300장, 3.9 km), 02(401장, 5.2 km), 03(221장, 2.8 km). 4K 사진(3840x2160), 사진 간격 약 13 m,
  고도 열 280~425 m. 위성 지도 satellite01.tif 한 장(25856x34560, 약 0.244 m/px).
- CSV 열: id, file_name, latitude, longitude, altitude. 방향(yaw) 열이 없다.

축척 (결과를 보기 전에 정함)
- 지도가 VisLoc과 거의 같은 해상도(0.244 vs 0.25 m/px)라 SCALE = 1. 지도는 0.25 m/px로 다시 샘플링,
  드론 사진 가운데 64 m를 256 px로 (VisLoc seq_eval과 같음). 파이프라인 m = 실제 m.
- 걸음 간격 80 m (VisLoc 사진 간격 중앙값), 구간 600~1,000 m (VisLoc과 같음).
- 출발점을 0 / 20 / 40 / 60 m 민 4벌 (AerialVL의 0/50/100/150 m ÷ 2.5, 걸음 간격의 1/4씩). 구간끼리 독립이 아니다.
- 드론 축척(gsd, m/px): 사진마다 앞뒤 ±3장 각각의 연속 사진 쌍(3장 차이, 약 39 m)의 SIFT 이동량 ÷ GPS 거리를
  재고 그 중앙값(희윤 VisLoc 방식을 사진마다 국소적으로). 쌍이 3개 미만이면 k × 고도 (k = 전체 쌍 gsd ÷ 고도의 중앙값).
  위성 지도·정답 매칭은 쓰지 않는다. (확인 그림에서 고도 열만으로는 축척이 최대 약 1.3배 틀려 이렇게 정함, 2026-11-22)
- VisLoc과 달리 고도 변화 구간을 빼지 않는다(축척을 사진마다 재므로).

사진 방향 (결과를 보기 전에 정함)
- 방향 열이 없어서 GPS 궤적 방향(앞뒤 2장, 약 52 m 사이)을 나침반 방향 대신 쓴다. 카메라가 기체 앞을 본다고 보고
  사진을 북쪽이 위가 되게 돌린다. 파이프라인이 4방향을 다 시도하므로 90° 배수 차이는 상관없고, 남는 오차는
  기체 옆미끄러짐(crab) 정도다.
- 돌리는 방향의 부호(+θ/−θ)는 --check 그림(지도와 나란히 놓은 사진)으로 내보내기 전에 한 번 정한다. 매칭 결과는 보지 않는다.
- 앞뒤 3장 방향 차이가 20°보다 크면 turning = True (replay 로그에만 남김, 걸음에서 빼지 않음).

판정 (결과를 보기 전에 정함) — 실행: run_visloc_nav, variants dr, r1, r4g0n5w20ac4
1. vio, seed 2026: R1 vs 추측 항법, 마지막 40 m 이내 McNemar.
2. vio, seed 2026: 경량 v1 vs R1, 마지막 40 m 이내 McNemar.
3. (주 판정) vio4, seed 2027~2030 합침: 경량 v1 vs R1, 마지막 40 m 이내 McNemar p < 0.05면
   "이동 오차가 클 때 R4가 도착을 유의하게 늘린다(9.1절)가 XIAN에서도 되풀이된다"로 보고. 구간 단위 부호 검정도 같이 적고,
   비행(3개) 단위 결과도 적는다. 아니면 그대로 보고한다.

한계: 비행 3개, 한 도시. 고도 열이 지면 기준인지 확실하지 않음(시내가 평지라 k × 고도로 둠). 방향을 GPS 궤적으로 대신함.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(1 << 40))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.blocks import N_BLOCK, crop_block  # noqa: E402
from imatch.geometry import BLOCK_CENTER  # noqa: E402

SCALE = 1.0
PX_PER_M = 4.0 / SCALE            # 실제 m당 지도 px (0.25 m/px)
UAV_PX, UAV_M = 256, 64.0         # 실제 64 m → 256 px
STRIDE, MAP_PX = 256, 256 * (N_BLOCK + 1)
STEP_M, SEG_MAX, SEG_MIN, MARGIN_M = 80.0, 1000.0, 600.0, 32.0
OFFSETS = (0.0, 20.0, 40.0, 60.0)
PARTNER = 3                       # gsd 쌍: 3장 뒤 (약 39 m)
GSD_WIN = 3                       # 축척: 앞뒤 3장의 쌍 중앙값
HEAD_W = 2                        # 방향: 앞뒤 2장
TURN_W, TURN_DEG = 3, 20.0
LAT_REF = 34.21

MX = 111412.84 * math.cos(math.radians(LAT_REF)) - 93.5 * math.cos(3 * math.radians(LAT_REF))
MY = 111132.92 - 559.82 * math.cos(2 * math.radians(LAT_REF)) + 1.175 * math.cos(4 * math.radians(LAT_REF))


# ------------------------------------------------------------ 읽기

def load_frames(src: Path) -> pd.DataFrame:
    rw = src / 'real_world'
    out = []
    for f in sorted(d for d in os.listdir(rw) if d.isdigit()):
        df = pd.read_csv(rw / f / f'{f}.csv').sort_values('id').reset_index(drop=True)
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'altitude': 'alt'})
        df['seq'] = f
        df['path'] = [str(rw / f / 'drone' / n) for n in df.file_name]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def load_map(src: Path):
    r = pd.read_csv(src / 'real_world' / 'satellite' / 'satellite_coordinates_range.csv').iloc[0]
    path = src / 'real_world' / 'satellite' / r.mapname
    # 3.6 GB 무압축 TIFF(RGBA)라 메모리에 다 올리지 않고 디스크에서 필요한 부분만 읽는다
    import tifffile
    img = tifffile.memmap(str(path), mode='r')
    g = dict(lon0=float(r.LT_lon_map), lat0=float(r.LT_lat_map), lon1=float(r.RB_lon_map), lat1=float(r.RB_lat_map))
    Ho, Wo = img.shape[:2]
    g['mpp_x'] = (g['lon1'] - g['lon0']) * MX / Wo
    g['mpp_y'] = (g['lat0'] - g['lat1']) * MY / Ho
    return img, g


def map_window(orig, g, X0, Y0, size):
    """파이프라인 px(0.25 m/px) 창 (X0, Y0, size) → 원본 지도에서 잘라 size x size로."""
    sx, sy = 1 / PX_PER_M / g['mpp_x'], 1 / PX_PER_M / g['mpp_y']   # 파이프라인 px → 원본 px
    x0, y0 = X0 * sx, Y0 * sy
    x1, y1 = (X0 + size) * sx, (Y0 + size) * sy
    a, b = int(math.floor(x0)), int(math.floor(y0))
    c, d = int(math.ceil(x1)), int(math.ceil(y1))
    H, W = orig.shape[:2]
    pad = np.zeros((d - b, c - a, 3), np.uint8)
    ya, yb, xa, xb = max(b, 0), min(d, H), max(a, 0), min(c, W)
    if yb > ya and xb > xa:
        pad[ya - b:yb - b, xa - a:xb - a] = orig[ya:yb, xa:xb, :3]
    # 소수 부분 보정까지 포함해 정확히 다시 샘플링
    M = np.array([[sx, 0, x0 - a], [0, sy, y0 - b]], np.float32)
    return cv2.warpAffine(pad, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


# ------------------------------------------------------------ 방향, 축척, 자르기

def headings(d: pd.DataFrame):
    E, N = d.X.to_numpy() / PX_PER_M, -d.Y.to_numpy() / PX_PER_M
    n = len(d)

    def course(i, j):
        i, j = max(0, i), min(n - 1, j)
        return math.degrees(math.atan2(E[j] - E[i], N[j] - N[i])) % 360

    head = np.array([course(i - HEAD_W, i + HEAD_W) for i in range(n)])
    back = np.array([course(i - TURN_W, i) for i in range(n)])
    fwd = np.array([course(i, i + TURN_W) for i in range(n)])
    diff = np.abs((fwd - back + 180) % 360 - 180)
    diff[0] = diff[-1] = 0.0   # 첫/끝 사진은 한쪽 방향이 없음
    return head, diff > TURN_DEG


def gray_half(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)


def pair_gsd(p1, p2, dist_m):
    sift = cv2.SIFT_create(4000)
    g1, g2 = gray_half(p1), gray_half(p2)
    k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return np.nan
    good = [p[0] for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2 and p[0].distance < .75 * p[1].distance]
    if len(good) < 30:
        return np.nan
    a = np.float32([k1[m.queryIdx].pt for m in good]); b = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3)
    if A is None or inl.sum() < 30 or abs(math.hypot(A[0, 0], A[1, 0]) - 1) > .05:
        return np.nan
    c = np.array([g1.shape[1] / 2, g1.shape[0] / 2, 1.])
    shift = np.linalg.norm(A @ c - c[:2]) * 2
    return dist_m / shift if shift > 20 else np.nan


def uav_crop(path, gsd, heading_deg, sign, crop_m=UAV_M):
    """사진 가운데를 북쪽이 위가 되게 돌려 64 m 정사각형으로 자르고 256 px로. sign = -1이면 시계 방향으로 θ."""
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    c = 2 * round(crop_m / gsd / 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), sign * heading_deg, 1.0)
    M[:, 2] += (c / 2 - w / 2, c / 2 - h / 2)
    out = cv2.warpAffine(im, M, (c, c), flags=cv2.INTER_LINEAR)
    a = math.radians(heading_deg)
    fits = c / 2 * (abs(math.cos(a)) + abs(math.sin(a))) <= min(w, h) / 2
    return cv2.resize(out, (UAV_PX, UAV_PX), interpolation=cv2.INTER_AREA), c, fits


def local_gsd(F, gis, threads):
    """전역 번호 gis 각각의 축척: 같은 비행 안 앞뒤 GSD_WIN장의 쌍 gsd 중앙값, 모자라면 k × 고도."""
    seq = F.seq.to_numpy(); X = F.X.to_numpy(); Y = F.Y.to_numpy(); path = F.path.to_numpy(); n = len(F)
    alt = F.alt.to_numpy()

    def ok(j):
        return 0 <= j and j + PARTNER < n and seq[j] == seq[j + PARTNER]
    need = sorted({j for gi in gis for j in range(gi - GSD_WIN, gi + GSD_WIN + 1) if ok(j) and seq[j] == seq[gi]})

    def one(j):
        return pair_gsd(path[j], path[j + PARTNER], float(np.hypot(X[j + PARTNER] - X[j], Y[j + PARTNER] - Y[j]) / PX_PER_M))
    with ThreadPoolExecutor(threads) as ex:
        val = dict(zip(need, ex.map(one, need)))
    v = np.array([val[j] for j in need], float)
    kk = float(np.nanmedian(v / alt[need]))
    out, src, npair = [], [], []
    for gi in gis:
        w = [val[j] for j in range(gi - GSD_WIN, gi + GSD_WIN + 1) if j in val and seq[j] == seq[gi] and np.isfinite(val[j])]
        if len(w) >= 3:
            out.append(float(np.median(w))); src.append('pair_win')
        else:
            out.append(kk * float(alt[gi])); src.append('alt')
        npair.append(len(w))
    return np.array(out), np.array(src), np.array(npair), kk, int(np.isfinite(v).sum()), len(need)


# ------------------------------------------------------------ 메인

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='data/xian')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--quality', type=int, default=92)
    ap.add_argument('--check', type=int, default=0, help='확인 그림 n장만 만들고 끝냄')
    ap.add_argument('--sign', type=int, default=-1, choices=(-1, 1), help='사진 돌리는 방향 (--check로 정함)')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    F = load_frames(src)
    orig, g = load_map(src)
    print(f'지도 {orig.shape[1]}x{orig.shape[0]}, {g["mpp_x"]:.4f} x {g["mpp_y"]:.4f} m/px', flush=True)
    F['X'] = (F.lon - g['lon0']) * MX * PX_PER_M
    F['Y'] = (g['lat0'] - F.lat) * MY * PX_PER_M
    W_full = (g['lon1'] - g['lon0']) * MX * PX_PER_M
    H_full = (g['lat0'] - g['lat1']) * MY * PX_PER_M
    parts = []
    for s, d in F.groupby('seq', sort=True):
        d = d.copy()
        d['heading'], d['turning'] = headings(d)
        parts.append(d)
    F = pd.concat(parts)

    # 걸음 고르기: 비행마다 누적 거리 offset + 80 m 간격에 가장 가까운 사진
    picks = []
    for s, d in F.groupby('seq', sort=True):
        d = d.reset_index()   # 'index' = F의 전역 번호
        cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(d.X), np.diff(d.Y)) / PX_PER_M)])
        for o in OFFSETS:
            for j, target in enumerate(np.arange(o, cum[-1], STEP_M)):
                i = int(np.argmin(np.abs(cum - target)))
                if abs(cum[i] - target) < 10 and i + PARTNER < len(d):
                    picks.append(dict(seq=s, off=o, j=j, i=i, gi=int(d['index'][i]), cum=cum[i], filename=d.file_name[i], path=d.path[i],
                                      partner=d.path[i + PARTNER], X=d.X[i], Y=d.Y[i], alt=d.alt[i],
                                      heading=d.heading[i], turning=bool(d.turning[i]),
                                      pdist=float(np.hypot(d.X[i + PARTNER] - d.X[i], d.Y[i + PARTNER] - d.Y[i]) / PX_PER_M)))
    P = pd.DataFrame(picks)

    if a.check:
        # 확인 그림: [드론 200 m (−θ로 돌림) | 정답 위치 지도 200 m | 지도 600 m, 빨간 네모 = 200 m]
        # 축척은 본 내보내기와 같은 방식: k = median(쌍 gsd / 고도), gsd = k × 고도 (확인용 사진들로 k를 냄)
        cd = out / 'check'; cd.mkdir(exist_ok=True)
        sel = P[(P.off == 0) & (~P.turning)]
        sel = sel.iloc[np.linspace(0, len(sel) - 1, a.check).round().astype(int)].copy()
        sel['gsd'], sel['gsd_src'], _, kk, nv, nn = local_gsd(F, sel.gi.tolist(), a.threads)
        print(f'확인용 k = {kk:.6f} (고도 300 m면 {kk * 300:.4f} m/px), 쌍 {nv}/{nn}', flush=True)
        W1, W3 = 200.0, 600.0
        sep = np.full((UAV_PX, 8, 3), 255, np.uint8)
        for n, r in enumerate(sel.itertuples()):
            gsd = r.gsd
            d_img, _, _ = uav_crop(r.path, gsd, r.heading, a.sign, crop_m=W1)
            n1, n3 = W1 * PX_PER_M, W3 * PX_PER_M
            m1 = cv2.resize(map_window(orig, g, r.X - n1 / 2, r.Y - n1 / 2, int(n1)), (UAV_PX, UAV_PX),
                            interpolation=cv2.INTER_AREA)
            m3 = np.ascontiguousarray(cv2.resize(map_window(orig, g, r.X - n3 / 2, r.Y - n3 / 2, int(n3)),
                                                 (UAV_PX, UAV_PX), interpolation=cv2.INTER_AREA))
            q = int(round(UAV_PX * W1 / W3 / 2)); c0 = UAV_PX // 2
            cv2.rectangle(m3, (c0 - q, c0 - q), (c0 + q, c0 + q), (255, 0, 0), 1)
            pane = np.concatenate([d_img, sep, m1, sep, m3], axis=1)
            fn = cd / f'wide_{n}_seq{r.seq}_head{int(r.heading):03d}.jpg'
            cv2.imwrite(str(fn), cv2.cvtColor(pane, cv2.COLOR_RGB2BGR))
            print(f'  {fn.name}  gsd={gsd:.4f} m/px ({r.gsd_src}), k×고도={kk * r.alt:.4f}  heading={r.heading:.0f}°', flush=True)
        print('1 = 드론 200 m, 2 = 정답 위치 지도 200 m, 3 = 지도 600 m (빨간 네모 = 2번 범위).')
        return

    # 구간: offset마다 앞에서부터, 600~1,000 m, 지도 창 안
    usable = MAP_PX - 2 * MARGIN_M * PX_PER_M
    segs = []
    for (s, o), d in P.groupby(['seq', 'off'], sort=True):
        idx = list(d.sort_values('j').index); i = 0
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
    used = sorted({k for *_, ks in segs for k in ks})
    print(f'비행 {F.seq.nunique()}, 구간 {len(segs)}, 걸음 {sum(len(k) for *_, k in segs)}, 사진 {len(used)}', flush=True)

    # 축척: 사진마다 앞뒤 3장의 쌍 gsd 중앙값 (모자라면 k × 고도)
    g_, src_, np_, kk, nv, nn = local_gsd(F, P.loc[used, 'gi'].tolist(), a.threads)
    P.loc[used, 'gsd'] = g_; P.loc[used, 'gsd_src'] = src_; P.loc[used, 'gsd_npair'] = np_
    print(f'축척: 쌍 {nv}/{nn} 성공, 쌍 중앙값 사용 {np.mean(src_ == "pair_win") * 100:.0f}%, k = {kk:.6f}, '
          f'gsd {np.min(g_):.4f}~{np.max(g_):.4f} (중앙값 {np.median(g_):.4f}) m/px', flush=True)

    # 내보내기
    d_out = out / 'seq_test'; (d_out / 'maps').mkdir(parents=True, exist_ok=True)
    rows, idx_rows = [], []
    for n_seg, (s, o, ks) in enumerate(segs):
        sub = P.loc[ks]
        cx, cy = (sub.X.min() + sub.X.max()) / 2, (sub.Y.min() + sub.Y.max()) / 2
        X0 = int(round(min(max(cx - MAP_PX / 2, 0), W_full - MAP_PX)))
        Y0 = int(round(min(max(cy - MAP_PX / 2, 0), H_full - MAP_PX)))
        mx_, my_ = sub.X - X0, sub.Y - Y0
        m = MARGIN_M * PX_PER_M
        if mx_.min() < m or my_.min() < m or mx_.max() > MAP_PX - m or my_.max() > MAP_PX - m:
            print('  지도 창 밖, 건너뜀', s, o); continue
        city = map_window(orig, g, X0, Y0, MAP_PX)
        name = f'xian_{s}_o{int(o):03d}_s{n_seg:03d}'
        dd = d_out / name; dd.mkdir(exist_ok=True)
        cv2.imwrite(str(d_out / 'maps' / f'{name}.jpg'), cv2.cvtColor(city, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        c0 = sub.cum.iloc[0]
        for k, r in enumerate(sub.itertuples()):
            uav, cpx, fits = uav_crop(r.path, r.gsd, r.heading, a.sign)
            cv2.imwrite(str(dd / f'k{k:03d}_uav.jpg'), cv2.cvtColor(uav, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            xs, ys = r.X - X0, r.Y - Y0
            bx = int(np.clip(np.round((xs - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            by = int(np.clip(np.round((ys - BLOCK_CENTER) / STRIDE), 0, N_BLOCK - 1))
            cv2.imwrite(str(dd / f'k{k:03d}_block.jpg'), cv2.cvtColor(crop_block(city, bx, by), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append(dict(set=name, k=k, seq=s, offset_m=o, filename=r.filename,
                             map_x=round(float(xs), 2), map_y=round(float(ys), 2),
                             dist_m=round((r.cum - c0) / SCALE, 2), dist_real_m=round(r.cum - c0, 2),
                             alt=r.alt, gsd=round(float(r.gsd), 4), gsd_src=r.gsd_src, gsd_npair=int(r.gsd_npair),
                             uav_crop_px=cpx, crop_m=round(cpx * float(r.gsd), 1), crop_fits=fits,
                             heading_deg=round(float(r.heading), 1), turning=r.turning, tilted=False, scale=SCALE))
            idx_rows.append(dict(set=name, k=k, block_x=bx, block_y=by))
        print(f'  {n_seg + 1}/{len(segs)} {name} ({len(sub)}장)', flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(d_out / 'samples.csv', index=False)
    pd.DataFrame(idx_rows).to_csv(d_out / 'block_index.csv', index=False)
    (d_out / 'README.txt').write_text(__doc__ + f'\n--sign {a.sign}, k = {kk:.6f}\n', encoding='utf-8')
    g2 = S.groupby('set')
    print(f'내보냄: {S.set.nunique()}구간, {len(S)}장 (출발 제외 {len(S) - S.set.nunique()}걸음), '
          f'구간 길이 {g2.dist_real_m.max().min():.0f}~{g2.dist_real_m.max().max():.0f} m, '
          f'turning {S.turning.mean() * 100:.1f}%, 돌려 자른 사진이 원본 안에 다 들어감 {S.crop_fits.mean() * 100:.0f}%')


if __name__ == '__main__':
    main()
