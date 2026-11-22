#!/usr/bin/env python
r"""XIAN-VisLoc 전체(시안 19 + 웨이난 2 비행, 지도 6장) 중 4K 비행 12개 → seq_eval과 같은 형식. 확인 실험용.

  python scripts/export_xian_full.py --src E:\R4_Datasets\XIAN_full\XIAN_Visloc --out E:\R4_Datasets\xianF_out --check 6
  python scripts/export_xian_full.py --src E:\R4_Datasets\XIAN_full\XIAN_Visloc --out E:\R4_Datasets\xianF_out

규칙 (결과 보기 전에 정함, 2026-11-22)
- 쓰는 비행: 4K(3840×2160) 사진인 시안 08·09·10·11·12·13·14·15·18·19, 웨이난 01·02.
  490×490 사진인 시안 01~07·16·17은 뺀다(원저자가 미리 자른 사진이라 축척·회전 처리를 알 수 없고 고도 열도 없음).
- 비행마다 사진이 가장 많이 들어가는 지도 한 장을 쓴다. 지도 밖 사진이 든 구간은 지도 창 검사에서 빠진다.
- 나머지 가공 규칙(축척 SCALE = 1, 0.25 m/px, 드론 64 m → 256 px, 걸음 80 m, 구간 600~1,000 m, 출발점 0/20/40/60 m,
  사진 방향 = GPS 궤적 방향으로 북쪽 위(시계 방향), 드론 축척 = 앞뒤 ±3장 쌍 gsd 중앙값)은 export_xian.py와 같다.
- 구간 이름: xianF_<지역 머리글자><비행>_o<출발점>_s<번호> (예: xianF_X08_o000_s000, xianF_W01_o020_s031).
- flights.csv에 비행 → 지도 대응을 남긴다(지도 어긋남 보정에 씀).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(1 << 40))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_xian as X  # noqa: E402  (같은 scripts 폴더)

USE = {'Xian': ['08', '09', '10', '11', '12', '13', '14', '15', '18', '19'], 'Weinan': ['01', '02']}


def mxy(lat):
    MX = 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat))
    MY = 111132.92 - 559.82 * math.cos(2 * math.radians(lat)) + 1.175 * math.cos(4 * math.radians(lat))
    return MX, MY


def open_tif(path):
    import tifffile
    try:
        return tifffile.memmap(str(path), mode='r')      # 무압축이면 디스크에서 필요한 부분만
    except Exception:
        return tifffile.imread(str(path))                # 압축이면 통째로 (지도 한 장 1 GB 안팎)


def load(src: Path):
    frames, maps = [], {}
    for region, fids in USE.items():
        rd = src / region
        R = pd.read_csv(rd / 'satellite' / 'satellite_coordinates_range.csv')
        for r in R.itertuples():
            maps[(region, r.mapname)] = dict(path=rd / 'satellite' / r.mapname, lon0=float(r.LT_lon_map),
                                             lat0=float(r.LT_lat_map), lon1=float(r.RB_lon_map),
                                             lat1=float(r.RB_lat_map), img=None)
        for fid in fids:
            df = pd.read_csv(rd / 'drone' / fid / f'{fid}.csv').sort_values('id').reset_index(drop=True)
            df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'altitude': 'alt'})
            if 'alt' not in df:
                df['alt'] = np.nan
            df['region'] = region
            df['seq'] = region[0] + fid
            df['path'] = [str(rd / 'drone' / fid / 'drone' / n) for n in df.file_name]
            # 사진이 가장 많이 들어가는 지도
            best, nbest = None, -1
            for (rg, mn), m in maps.items():
                if rg != region:
                    continue
                n = int(((df.lat <= m['lat0']) & (df.lat >= m['lat1']) & (df.lon >= m['lon0']) & (df.lon <= m['lon1'])).sum())
                if n > nbest:
                    best, nbest = (rg, mn), n
            df['map'] = best[1]
            df['in_map'] = nbest / len(df)
            frames.append(df)
    F = pd.concat(frames, ignore_index=True)
    return F, maps


def map_g(maps, key):
    m = maps[key]
    if m['img'] is None:
        m['img'] = open_tif(m['path'])
        Ho, Wo = m['img'].shape[:2]
        MX, MY = mxy((m['lat0'] + m['lat1']) / 2)
        m.update(MX=MX, MY=MY, mpp_x=(m['lon1'] - m['lon0']) * MX / Wo, mpp_y=(m['lat0'] - m['lat1']) * MY / Ho,
                 W_full=(m['lon1'] - m['lon0']) * MX * X.PX_PER_M, H_full=(m['lat0'] - m['lat1']) * MY * X.PX_PER_M)
        print(f"  지도 {key[0]}/{key[1]}: {Wo}x{Ho}, {m['mpp_x']:.4f} x {m['mpp_y']:.4f} m/px", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', default='data/xianF')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--quality', type=int, default=92)
    ap.add_argument('--check', type=int, default=0, help='확인 그림 n장만 만들고 끝냄')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    F, maps = load(src)
    # 비행별 지도 좌표(파이프라인 px, 지도 왼쪽 위 원점)
    xs, ys = np.zeros(len(F)), np.zeros(len(F))
    for (rg, mn), idx in F.groupby(['region', 'map']).groups.items():
        m = map_g(maps, (rg, mn))
        xs[idx] = (F.lon[idx] - m['lon0']) * m['MX'] * X.PX_PER_M
        ys[idx] = (m['lat0'] - F.lat[idx]) * m['MY'] * X.PX_PER_M
    F['X'], F['Y'] = xs, ys
    parts = []
    for s, d in F.groupby('seq', sort=True):
        d = d.copy(); d['heading'], d['turning'] = X.headings(d); parts.append(d)
    F = pd.concat(parts).reset_index(drop=True)   # local_gsd는 위치 번호를 씀
    fl = F.groupby('seq').agg(region=('region', 'first'), map=('map', 'first'), in_map=('in_map', 'first'),
                              photos=('path', 'size'), alt=('alt', 'median')).reset_index()
    print(fl.to_string(index=False), flush=True)

    picks = []
    for s, d in F.groupby('seq', sort=True):
        d = d.reset_index()
        cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(d.X), np.diff(d.Y)) / X.PX_PER_M)])
        for o in X.OFFSETS:
            for j, target in enumerate(np.arange(o, cum[-1], X.STEP_M)):
                i = int(np.argmin(np.abs(cum - target)))
                if abs(cum[i] - target) < 10 and i + X.PARTNER < len(d):
                    picks.append(dict(seq=s, region=d.region[i], map=d['map'][i], off=o, j=j, i=i, gi=int(d['index'][i]),
                                      cum=cum[i], filename=d.file_name[i], path=d.path[i], X=d.X[i], Y=d.Y[i],
                                      alt=d.alt[i], heading=d.heading[i], turning=bool(d.turning[i])))
    P = pd.DataFrame(picks)

    if a.check:
        cd = out / 'check'; cd.mkdir(exist_ok=True)
        sep = np.full((X.UAV_PX, 8, 3), 255, np.uint8)
        for s, d in P[(P.off == 0) & (~P.turning)].groupby('seq'):
            sel = d.iloc[np.linspace(0, len(d) - 1, min(a.check, len(d))).round().astype(int)].copy()
            g_, src_, _, kk, _, _ = X.local_gsd(F, sel.gi.tolist(), a.threads)
            m = maps[(sel.region.iloc[0], sel['map'].iloc[0])]
            for n, (r, gsd) in enumerate(zip(sel.itertuples(), g_)):
                d_img, _, _ = X.uav_crop(r.path, gsd, r.heading, -1, crop_m=200.0)
                n1, n3 = 200 * X.PX_PER_M, 600 * X.PX_PER_M
                m1 = cv2.resize(X.map_window(m['img'], m, r.X - n1 / 2, r.Y - n1 / 2, int(n1)), (X.UAV_PX, X.UAV_PX),
                                interpolation=cv2.INTER_AREA)
                m3 = np.ascontiguousarray(cv2.resize(X.map_window(m['img'], m, r.X - n3 / 2, r.Y - n3 / 2, int(n3)),
                                                     (X.UAV_PX, X.UAV_PX), interpolation=cv2.INTER_AREA))
                q = int(round(X.UAV_PX / 6)); c0 = X.UAV_PX // 2
                cv2.rectangle(m3, (c0 - q, c0 - q), (c0 + q, c0 + q), (255, 0, 0), 1)
                fn = cd / f'wide_{s}_{n}_head{int(r.heading):03d}.jpg'
                cv2.imwrite(str(fn), cv2.cvtColor(np.concatenate([d_img, sep, m1, sep, m3], axis=1), cv2.COLOR_RGB2BGR))
            print(f'  {s}: 확인 그림 {len(sel)}장, gsd {np.min(g_):.3f}~{np.max(g_):.3f}', flush=True)
        print('1 = 드론 200 m, 2 = 정답 위치 지도 200 m, 3 = 지도 600 m (빨간 네모 = 2번 범위).')
        return

    usable = X.MAP_PX - 2 * X.MARGIN_M * X.PX_PER_M
    segs = []
    for (s, o), d in P.groupby(['seq', 'off'], sort=True):
        idx = list(d.sort_values('j').index); i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx):
                sub = P.loc[idx[i:j + 2]]
                if np.ptp(sub.X) > usable or np.ptp(sub.Y) > usable or P.cum[idx[j + 1]] - P.cum[idx[i]] > X.SEG_MAX:
                    break
                j += 1
            if P.cum[idx[j]] - P.cum[idx[i]] >= X.SEG_MIN:
                segs.append((s, o, idx[i:j + 1]))
            i = j + 1
    used = sorted({k for *_, ks in segs for k in ks})
    print(f'비행 {P.seq.nunique()}, 구간 후보 {len(segs)}, 사진 {len(used)}', flush=True)
    g_, src_, np_, kk, nv, nn = X.local_gsd(F, P.loc[used, 'gi'].tolist(), a.threads)
    P.loc[used, 'gsd'] = g_; P.loc[used, 'gsd_src'] = src_; P.loc[used, 'gsd_npair'] = np_
    print(f'축척: 쌍 {nv}/{nn} 성공, 쌍 중앙값 사용 {np.mean(src_ == "pair_win") * 100:.0f}%, '
          f'gsd {np.nanmin(g_):.4f}~{np.nanmax(g_):.4f} (중앙값 {np.nanmedian(g_):.4f}) m/px', flush=True)

    d_out = out / 'seq_test'; (d_out / 'maps').mkdir(parents=True, exist_ok=True)
    rows, idx_rows, n_ok = [], [], 0
    for n_seg, (s, o, ks) in enumerate(segs):
        sub = P.loc[ks]
        if sub.gsd.isna().any():
            print('  축척 없음, 건너뜀', s, o); continue
        m = maps[(sub.region.iloc[0], sub['map'].iloc[0])]
        cx, cy = (sub.X.min() + sub.X.max()) / 2, (sub.Y.min() + sub.Y.max()) / 2
        X0 = int(round(min(max(cx - X.MAP_PX / 2, 0), m['W_full'] - X.MAP_PX)))
        Y0 = int(round(min(max(cy - X.MAP_PX / 2, 0), m['H_full'] - X.MAP_PX)))
        mx_, my_ = sub.X - X0, sub.Y - Y0
        mg = X.MARGIN_M * X.PX_PER_M
        if mx_.min() < mg or my_.min() < mg or mx_.max() > X.MAP_PX - mg or my_.max() > X.MAP_PX - mg:
            print('  지도 창 밖, 건너뜀', s, o); continue
        city = X.map_window(m['img'], m, X0, Y0, X.MAP_PX)
        name = f'xianF_{s}_o{int(o):03d}_s{n_seg:03d}'
        dd = d_out / name; dd.mkdir(exist_ok=True)
        cv2.imwrite(str(d_out / 'maps' / f'{name}.jpg'), cv2.cvtColor(city, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        c0 = sub.cum.iloc[0]
        for k, r in enumerate(sub.itertuples()):
            uav, cpx, fits = X.uav_crop(r.path, r.gsd, r.heading, -1)
            cv2.imwrite(str(dd / f'k{k:03d}_uav.jpg'), cv2.cvtColor(uav, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            xs_, ys_ = r.X - X0, r.Y - Y0
            bx = int(np.clip(np.round((xs_ - X.BLOCK_CENTER) / X.STRIDE), 0, X.N_BLOCK - 1))
            by = int(np.clip(np.round((ys_ - X.BLOCK_CENTER) / X.STRIDE), 0, X.N_BLOCK - 1))
            cv2.imwrite(str(dd / f'k{k:03d}_block.jpg'), cv2.cvtColor(X.crop_block(city, bx, by), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rows.append(dict(set=name, k=k, seq=s, region=r.region, map=r.map, offset_m=o, filename=r.filename,
                             map_x=round(float(xs_), 2), map_y=round(float(ys_), 2), dist_m=round(r.cum - c0, 2),
                             dist_real_m=round(r.cum - c0, 2), alt=r.alt, gsd=round(float(r.gsd), 4),
                             gsd_src=r.gsd_src, gsd_npair=int(r.gsd_npair), uav_crop_px=cpx,
                             crop_m=round(cpx * float(r.gsd), 1), crop_fits=fits, heading_deg=round(float(r.heading), 1),
                             turning=r.turning, tilted=False, scale=X.SCALE))
            idx_rows.append(dict(set=name, k=k, block_x=bx, block_y=by))
        n_ok += 1
        if n_ok % 20 == 0:
            print(f'  {n_ok}구간 ({n_seg + 1}/{len(segs)})', flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(d_out / 'samples.csv', index=False)
    pd.DataFrame(idx_rows).to_csv(d_out / 'block_index.csv', index=False)
    fl.rename(columns={'seq': 'flight'}).to_csv(d_out / 'flights.csv', index=False)
    (d_out / 'README.txt').write_text(__doc__ + '\n\n' + X.__doc__, encoding='utf-8')
    g2 = S.groupby('set')
    print(f'내보냄: {S.set.nunique()}구간, {len(S)}장 (출발 제외 {len(S) - S.set.nunique()}걸음), '
          f'구간 길이 {g2.dist_m.max().min():.0f}~{g2.dist_m.max().max():.0f} m, 비행별 구간 '
          f'{S.groupby("seq").set.nunique().to_dict()}', flush=True)


if __name__ == '__main__':
    main()
