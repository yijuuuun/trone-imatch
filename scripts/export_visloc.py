#!/usr/bin/env python
"""UAV-VisLoc 전체(비행 01~11) → data/ 형식. 한 장 세트와 비행 구간 세트를 만든다.

docs/07_교수님_피드백_1113.md 분담 1번. 원 Bearing-UAV, R1, R1 + R4, 다른 모델, 8/4방향이 모두 같은 파일을 쓴다.

  python scripts/export_visloc.py --root <UAV-VisLoc 폴더> --out data/visloc            # 1차 (보정 없음)
  python scripts/run.py --data data/visloc/single_calib --out outputs/visloc_calib --matchers loma --angles 8
  python scripts/visloc_calib.py outputs/visloc_calib/loma_s1.csv --data data/visloc/single_calib --out data/visloc_calib.csv
  python scripts/export_visloc.py --root <UAV-VisLoc 폴더> --out data/visloc_v2 --calib data/visloc_calib.csv   # 최종

결과 폴더
  single_calib/, single_eval/   한 장 세트. samples.csv + visloc_<비행>/k###_uav.jpg, k###_block.jpg
                                 보정값은 single_calib에서만 정하고 평가는 single_eval로만 한다
  seq_eval/                      비행 구간 세트 (평가 줄만). samples.csv, block_index.csv, maps/<set>.jpg
                                 set = visloc_<비행>_s##, k = 비행 순서, dist_m = 구간 시작부터 누적 거리
  frames.csv                     모든 사진의 판정(status), 표시(tilted, turning), 줄(line), 나눔(split)
  flights.csv                    비행별 축척(gsd)·heading 보정·위치 보정과 그 출처

가공 기준
- 위성 지도: 비행마다 지도 왼쪽 위를 원점으로 하는 0.25 m/px, 북쪽 위 평면으로 다시 샘플링한다.
  원 GeoTIFF 픽셀은 땅에서 정사각형이 아니다(동서 0.23~0.27, 남북 0.30 m). 09처럼 tif가 여러 장이면 이어 붙인다.
- 블록: 고정 격자(256 px 간격, 512 px 블록, imatch/blocks.py와 같은 규칙)에서 정답 위치를 담은 블록.
  GPS 중심으로 자르지 않는다 (정답이 항상 블록 중앙이면 "가운데" 답이 이긴다).
- 드론 영상: 가운데 64 m를 잘라 256 px (0.25 m/px). 64 m = 256 px × 0.25 m.
- 축척(gsd): EXIF가 없고 height가 해발 고도라 계산할 수 없다. 연속 사진 쌍의 SIFT 이동량과 GPS 거리로 비행마다 잰다.
  이 값은 드론 사진끼리만 써서 구하며 위성 지도나 정답 매칭을 쓰지 않는다. --calib로 덮어쓸 수 있다.
- heading: 정답에만 쓴다(파이프라인 입력 아님). heading_cw = Phi1 + head_off, theta = -heading_cw (imatch/geometry.py 규칙).
- 위치 보정: 정답 = GPS + 드론 진행 방향 기준 (along_m 앞, right_m 오른쪽). 보정 전 값은 x_norm_raw, y_norm_raw에 남긴다.
- 빼는 것: 자세 열 없음(07), 상승·하강(고도 중앙값 ±10 m 밖), 지도에 다 들어가지 않는 블록.
  기운 사진(|Omega, Kappa| > 5°)과 회전 중 사진은 빼지 않고 tilted, turning으로 표시만 한다.
- 나눔: 비행 줄(회전 사이 직선 구간) 단위. 각 비행의 앞쪽 --calib-frac 줄이 calib, 나머지가 eval.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.blocks import N_BLOCK, crop_block  # noqa: E402
from imatch.geometry import BLOCK_CENTER, PX_PER_UNIT  # noqa: E402

PX_PER_M = 4.0                       # 0.25 m/px
BLOCK_PX = 512
UAV_PX = 256
UAV_M = UAV_PX / PX_PER_M            # 64 m
STRIDE = 256                         # 이웃 블록 간격 (px)
MAP_PX = STRIDE * (N_BLOCK + 1)      # 4096 px = 1,024 m


def m_per_deg(lat):
    p = math.radians(lat)
    return (111412.84 * math.cos(p) - 93.5 * math.cos(3 * p),
            111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p))


def angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


# ---------------------------------------------------------------- 위성 지도

class Mosaic:
    """한 비행의 위성 tif(1장 또는 여러 장)를 0.25 m/px 평면 좌표 (X, Y)로 읽는다.

    X = 동쪽 거리(m) × 4, Y = 남쪽 거리(m) × 4, 원점은 지도 왼쪽 위. 위경도 ↔ (X, Y)는 비행 평균 위도의
    m/deg로 선형 변환한다 (10 km 안에서 축척 차이 0.1% 이하).
    """

    def __init__(self, flight_dir: Path, lat_ref: float):
        self.tifs = []
        for tp in sorted(flight_dir.glob('satellite*.tif')):
            with tifffile.TiffFile(str(tp)) as tif:
                pg = tif.pages[0]
                if not pg.is_tiled or int(pg.compression) != 1 or int(pg.samplesperpixel) != 3 \
                        or int(pg.bitspersample) != 8 or int(pg.planarconfig) != 1:
                    raise RuntimeError(f'{tp.name}: 압축 없는 8비트 RGB 타일 tif가 아님')
                sc, tie = pg.tags[33550].value, pg.tags[33922].value
                self.tifs.append(dict(
                    path=tp, w=int(pg.imagewidth), h=int(pg.imagelength), tw=int(pg.tilewidth), th=int(pg.tilelength),
                    off=np.asarray(pg.dataoffsets, np.int64), cnt=np.asarray(pg.databytecounts, np.int64),
                    sx=float(sc[0]), sy=float(sc[1]),
                    lon0=float(tie[3]) - float(tie[0]) * float(sc[0]), lat0=float(tie[4]) + float(tie[1]) * float(sc[1])))
        if not self.tifs:
            raise FileNotFoundError(f'{flight_dir}: satellite*.tif 없음')
        self.sx, self.sy = self.tifs[0]['sx'], self.tifs[0]['sy']
        for t in self.tifs:
            if abs(t['sx'] / self.sx - 1) > 1e-6 or abs(t['sy'] / self.sy - 1) > 1e-6:
                raise RuntimeError(f'{t["path"].name}: tif끼리 픽셀 크기가 다름')
        self.lon_o = min(t['lon0'] for t in self.tifs)
        self.lat_o = max(t['lat0'] for t in self.tifs)
        for t in self.tifs:
            t['ox'] = int(round((t['lon0'] - self.lon_o) / self.sx))
            t['oy'] = int(round((self.lat_o - t['lat0']) / self.sy))
        self.W = max(t['ox'] + t['w'] for t in self.tifs)
        self.H = max(t['oy'] + t['h'] for t in self.tifs)
        self.mx, self.my = m_per_deg(lat_ref)
        self.a = 1.0 / (PX_PER_M * self.mx * self.sx)   # 원 지도 px / 평면 px (x)
        self.b = 1.0 / (PX_PER_M * self.my * self.sy)   # (y)
        self.Wf, self.Hf = self.W / self.a, self.H / self.b
        self.src_m = (self.sx * self.mx, self.sy * self.my)

    def xy(self, lat, lon):
        return ((np.asarray(lon) - self.lon_o) * self.mx * PX_PER_M,
                (self.lat_o - np.asarray(lat)) * self.my * PX_PER_M)

    def _read(self, x0, y0, w, h):
        """원 지도 좌표 창 → (영상, 덮임 마스크). 지도 밖은 0."""
        out = np.zeros((h, w, 3), np.uint8)
        cov = np.zeros((h, w), np.uint8)
        for t in self.tifs:
            xa, ya = max(x0, t['ox']), max(y0, t['oy'])
            xb, yb = min(x0 + w, t['ox'] + t['w']), min(y0 + h, t['oy'] + t['h'])
            if xa >= xb or ya >= yb:
                continue
            lx0, ly0, lx1, ly1 = xa - t['ox'], ya - t['oy'], xb - t['ox'], yb - t['oy']   # tif 안 좌표
            tw, th = t['tw'], t['th']
            across = math.ceil(t['w'] / tw)
            with open(t['path'], 'rb') as fh:
                for ty in range(ly0 // th, (ly1 - 1) // th + 1):
                    for tx in range(lx0 // tw, (lx1 - 1) // tw + 1):
                        i = ty * across + tx
                        if t['cnt'][i] != tw * th * 3:
                            raise RuntimeError(f'{t["path"].name}: 타일 ({tx},{ty}) 크기 이상')
                        fh.seek(int(t['off'][i]))
                        tile = np.frombuffer(fh.read(int(t['cnt'][i])), np.uint8).reshape(th, tw, 3)
                        gx0, gy0 = max(lx0, tx * tw), max(ly0, ty * th)
                        gx1, gy1 = min(lx1, tx * tw + tw), min(ly1, ty * th + th)
                        dx, dy = gx0 + t['ox'] - x0, gy0 + t['oy'] - y0
                        out[dy:dy + gy1 - gy0, dx:dx + gx1 - gx0] = tile[gy0 - ty * th:gy1 - ty * th, gx0 - tx * tw:gx1 - tx * tw]
                        cov[dy:dy + gy1 - gy0, dx:dx + gx1 - gx0] = 1
        return out, cov

    def render(self, X0, Y0, w, h):
        """평면 [X0, X0+w) × [Y0, Y0+h)를 0.25 m/px RGB로. 지도가 다 덮지 못하면 None."""
        a, b = self.a, self.b
        if X0 < 0 or Y0 < 0 or X0 + w > self.Wf or Y0 + h > self.Hf:
            return None
        xs0, ys0 = int(math.floor(X0 * a)) - 3, int(math.floor(Y0 * b)) - 3
        ww, hh = int(math.ceil((X0 + w) * a)) + 3 - xs0, int(math.ceil((Y0 + h) * b)) + 3 - ys0
        src, cov = self._read(xs0, ys0, ww, hh)
        # 출력 픽셀 u(중심 u + 0.5) → 원 지도 연속 좌표 a (X0 + u + 0.5) → 원 지도 픽셀 인덱스 (- 0.5)
        M = np.array([[a, 0, a * (X0 + 0.5) - 0.5 - xs0], [0, b, b * (Y0 + 0.5) - 0.5 - ys0]])
        img = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        c = cv2.warpAffine(cov, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img if c.min() > 0 else None


# ---------------------------------------------------------------- 사진 표

def load_frames(flight_dir: Path, flight: str, height_tol: float, tilt_max: float, turn_deg: float):
    rows = list(csv.DictReader(open(flight_dir / f'{flight}.csv', encoding='utf-8-sig')))
    df = pd.DataFrame(rows)
    df['row'] = np.arange(len(df))
    for c in ['lat', 'lon', 'height', 'Omega', 'Kappa', 'Phi1', 'Phi2']:
        df[c] = pd.to_numeric(df[c], errors='coerce') if c in df else np.nan
    mx, my = m_per_deg(df.lat.mean())
    e, n = (df.lon - df.lon.iloc[0]).to_numpy() * mx, (df.lat - df.lat.iloc[0]).to_numpy() * my
    step = np.r_[np.nan, np.hypot(np.diff(e), np.diff(n))]
    df['step_m'] = step
    med_step = float(np.nanmedian(step))
    ge, gn = np.gradient(e), np.gradient(n)
    df['course'] = np.degrees(np.arctan2(ge, gn)) % 360
    prev = np.r_[df.course.iloc[0], df.course.to_numpy()[:-1]]
    nxt = np.r_[df.course.to_numpy()[1:], df.course.iloc[-1]]
    df['turning'] = np.abs(angdiff(nxt, prev)) > turn_deg
    df['tilt_deg'] = np.hypot(df.Omega, df.Kappa)
    df['tilted'] = df.tilt_deg > tilt_max
    df['gap'] = df.step_m > 2.5 * med_step
    h_med = float(np.nanmedian(df.height))
    status = np.full(len(df), 'ok', dtype=object)
    status[df.Phi1.isna().to_numpy()] = 'no_pose'
    status[(status == 'ok') & (np.abs(df.height - h_med) > height_tol).to_numpy()] = 'height'
    df['status'] = status
    # 줄: 회전·끊김이 끝나고 다시 직선이 시작될 때 새 줄
    new_line = (~df.turning & (df.turning.shift(1, fill_value=True) | df.gap)).to_numpy()
    df['line'] = np.cumsum(new_line)
    return df, med_step, h_med


def split_lines(df, frac):
    lines = sorted(df.loc[(df.status == 'ok') & ~df.turning, 'line'].unique())
    n_cal = max(1, math.ceil(frac * len(lines))) if lines else 0
    cal = set(lines[:n_cal])
    return np.where(df.line.isin(cal), 'calib', 'eval')


# ---------------------------------------------------------------- 축척

_sift = None


def _gray_small(path, width):
    with Image.open(path) as im:
        f = im.size[0] / width
        g = np.asarray(im.convert('L').resize((width, round(im.size[1] / f)), Image.Resampling.LANCZOS))
    return g, f


def estimate_gsd(drone_dir: Path, df, n_pairs=20, width=1000):
    """연속 사진 쌍: SIFT 이동량(px)과 GPS 거리(m)의 비. 원 사진 m/px의 중앙값을 낸다."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(4000)
    ok = (df.status == 'ok') & ~df.turning & ~df.tilted & ~df.gap
    med = float(np.nanmedian(df.step_m))
    cand = [i for i in range(len(df) - 1)
            if ok.iloc[i] and ok.iloc[i + 1] and not df.gap.iloc[i + 1]
            and 0.8 * med < df.step_m.iloc[i + 1] < 1.2 * med]
    if not cand:
        return np.nan, 0, np.nan
    pick = [cand[j] for j in np.unique(np.linspace(0, len(cand) - 1, min(n_pairs, len(cand))).round().astype(int))]
    vals = []
    bf = cv2.BFMatcher()
    for i in pick:
        g1, f = _gray_small(drone_dir / df.filename.iloc[i], width)
        g2, _ = _gray_small(drone_dir / df.filename.iloc[i + 1], width)
        k1, d1 = _sift.detectAndCompute(g1, None)
        k2, d2 = _sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            continue
        good = [p[0] for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 30:
            continue
        p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
        A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3)
        if A is None or inl.sum() < 30 or abs(math.hypot(A[0, 0], A[1, 0]) - 1) > 0.05:
            continue
        c = np.array([g1.shape[1] / 2, g1.shape[0] / 2, 1.0])
        shift_px = np.linalg.norm(A @ c - c[:2]) * f
        if shift_px > 1:
            vals.append(df.step_m.iloc[i + 1] / shift_px)
    if not vals:
        return np.nan, 0, np.nan
    v = np.array(vals)
    return float(np.median(v)), len(v), float(np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------- 정답

def gt_positions(mos, df, head_off, along, right):
    """(X, Y) 보정 전·후와 heading. 보정은 드론 진행 방향(heading) 기준."""
    Xr, Yr = mos.xy(df.lat.to_numpy(), df.lon.to_numpy())
    hcw = (df.Phi1.to_numpy() + head_off) % 360
    h = np.radians(hcw)
    dE = along * np.sin(h) + right * np.cos(h)
    dN = along * np.cos(h) - right * np.sin(h)
    return Xr, Yr, Xr + dE * PX_PER_M, Yr - dN * PX_PER_M, hcw


def block_of(X, Y, n_block=None):
    bx, by = np.round((np.asarray(X) - BLOCK_CENTER) / STRIDE), np.round((np.asarray(Y) - BLOCK_CENTER) / STRIDE)
    if n_block is not None:
        bx, by = np.clip(bx, 0, n_block - 1), np.clip(by, 0, n_block - 1)
    return bx.astype(int), by.astype(int)


def uav_crop(path, gsd):
    crop = 2 * round(UAV_M / gsd / 2)
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        if crop > min(w, h):
            raise ValueError(f'{path.name}: 자를 크기 {crop}px가 영상 {w}x{h}보다 큼 (gsd {gsd:.4f})')
        left, top = (w - crop) // 2, (h - crop) // 2
        return im.crop((left, top, left + crop, top + crop)).resize((UAV_PX, UAV_PX), Image.Resampling.LANCZOS), crop


def save_jpg(arr_or_img, path, quality):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.save(path, quality=quality)


# ---------------------------------------------------------------- 내보내기

def sample_row(set_name, k, fr, x, y, xr, yr, theta, cal, crop, extra=None):
    r = dict(set=set_name, k=k, row=int(fr.row), x_norm=round(float(x), 6), y_norm=round(float(y), 6),
             theta=round(float(theta), 4), uav_h=UAV_PX, uav_w=UAV_PX, block_h=BLOCK_PX, block_w=BLOCK_PX,
             x_norm_raw=round(float(xr), 6), y_norm_raw=round(float(yr), 6),
             flight=cal['flight'], filename=fr.filename, lat=fr.lat, lon=fr.lon, height=fr.height,
             Phi1=fr.Phi1, Phi2=fr.Phi2, heading_cw=round(float((-theta) % 360), 4),
             gsd=round(cal['gsd'], 5), uav_crop_px=crop, head_off=cal['head_off'], along_m=cal['along_m'],
             right_m=cal['right_m'], tilt_deg=round(float(fr.tilt_deg), 2), tilted=bool(fr.tilted),
             turning=bool(fr.turning), line=int(fr.line), split=fr.split)
    if extra:
        r.update(extra)
    return r


def export_single(out, flight, mos, drone_dir, df, cal, every, quality, rows_by_split):
    """한 장 세트: 비행 전체 격자에서 정답 블록을 자른다. 같은 비행 = 한 set."""
    set_name = f'visloc_{flight}'
    n_skip = 0
    for split in ('calib', 'eval'):
        sub = df[(df.status == 'ok') & (df.split == split)]
        sub = sub.iloc[::every]
        d = out / f'single_{split}' / set_name
        d.mkdir(parents=True, exist_ok=True)
        rows = rows_by_split[split]
        k = sum(1 for r in rows if r['set'] == set_name)
        for fr in sub.itertuples():
            bx, by = block_of(fr.X, fr.Y)
            block = mos.render(int(bx) * STRIDE, int(by) * STRIDE, BLOCK_PX, BLOCK_PX)
            if block is None:
                df.loc[fr.Index, 'status'] = 'edge'; n_skip += 1
                continue
            cx, cy = bx * STRIDE + BLOCK_CENTER, by * STRIDE + BLOCK_CENTER
            uav, crop = uav_crop(drone_dir / fr.filename, cal['gsd'])
            save_jpg(uav, d / f'k{k:03d}_uav.jpg', quality)
            save_jpg(block, d / f'k{k:03d}_block.jpg', quality)
            rows.append(sample_row(set_name, k, fr, (fr.X - cx) / PX_PER_UNIT, (fr.Y - cy) / PX_PER_UNIT,
                                   (fr.Xr - cx) / PX_PER_UNIT, (fr.Yr - cy) / PX_PER_UNIT, fr.theta, cal, crop))
            k += 1
    return n_skip


def find_segments(df, seg_len, seg_min, margin_px):
    """평가 줄의 연속 사진을 앞에서부터 겹치지 않게 자른 구간 [(시작, 끝) iloc]. 구간은 지도 한 장에 들어가야 한다."""
    usable = MAP_PX - 2 * margin_px
    ok = ((df.status == 'ok') & (df.split == 'eval')).to_numpy()
    gap = df.gap.to_numpy()
    X, Y = df.X.to_numpy(), df.Y.to_numpy()
    segs, i, n = [], 0, len(df)
    while i < n:
        if not ok[i]:
            i += 1; continue
        j, length = i, 0.0
        while j + 1 < n and ok[j + 1] and not gap[j + 1]:
            xs, ys = X[i:j + 2], Y[i:j + 2]
            step = math.hypot(X[j + 1] - X[j], Y[j + 1] - Y[j]) / PX_PER_M
            if np.ptp(xs) > usable or np.ptp(ys) > usable or length + step > seg_len:
                break
            length += step; j += 1
        if length >= seg_min:
            segs.append((i, j, length))
        i = j + 1
    return segs


def export_seq(out, flight, mos, drone_dir, df, cal, segs, quality, rows, idx_rows, margin_px):
    d_maps = out / 'seq_eval' / 'maps'
    d_maps.mkdir(parents=True, exist_ok=True)
    made = 0
    for s_i, (i, j, length) in enumerate(segs):
        sub = df.iloc[i:j + 1]
        cx, cy = (sub.X.min() + sub.X.max()) / 2, (sub.Y.min() + sub.Y.max()) / 2
        X0 = int(round(min(max(cx - MAP_PX / 2, 0), mos.Wf - MAP_PX)))
        Y0 = int(round(min(max(cy - MAP_PX / 2, 0), mos.Hf - MAP_PX)))
        mx_, my_ = sub.X - X0, sub.Y - Y0
        if (mx_.min() < margin_px or my_.min() < margin_px or mx_.max() > MAP_PX - margin_px
                or my_.max() > MAP_PX - margin_px):
            continue
        city = mos.render(X0, Y0, MAP_PX, MAP_PX)
        if city is None:
            continue
        set_name = f'visloc_{flight}_s{s_i:02d}'
        d = out / 'seq_eval' / set_name
        d.mkdir(parents=True, exist_ok=True)
        save_jpg(city, d_maps / f'{set_name}.jpg', quality)
        dist = 0.0
        prev = None
        for k, fr in enumerate(sub.itertuples()):
            Xs, Ys = fr.X - X0, fr.Y - Y0
            if prev is not None:
                dist += math.hypot(fr.X - prev[0], fr.Y - prev[1]) / PX_PER_M
            prev = (fr.X, fr.Y)
            bx, by = block_of(Xs, Ys, N_BLOCK)
            ccx, ccy = bx * STRIDE + BLOCK_CENTER, by * STRIDE + BLOCK_CENTER
            uav, crop = uav_crop(drone_dir / fr.filename, cal['gsd'])
            save_jpg(uav, d / f'k{k:03d}_uav.jpg', quality)
            save_jpg(crop_block(city, int(bx), int(by)), d / f'k{k:03d}_block.jpg', quality)
            rows.append(sample_row(set_name, k, fr, (Xs - ccx) / PX_PER_UNIT, (Ys - ccy) / PX_PER_UNIT,
                                   (fr.Xr - X0 - ccx) / PX_PER_UNIT, (fr.Yr - Y0 - ccy) / PX_PER_UNIT, fr.theta, cal,
                                   crop, dict(dist_m=round(dist, 2), map_x=round(float(Xs), 2), map_y=round(float(Ys), 2),
                                              map_x_raw=round(float(fr.Xr - X0), 2),
                                              map_y_raw=round(float(fr.Yr - Y0), 2), seg_len_m=round(length, 1))))
            idx_rows.append(dict(set=set_name, k=k, block_x=int(bx), block_y=int(by)))
        made += 1
    return made


def num(v, default):
    """calib 칸 값 → float. 비었거나 NaN이면 default."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def read_calib(path):
    if not path:
        return {}
    c = pd.read_csv(path, dtype={'flight': str})
    c['flight'] = c.flight.str.zfill(2)
    return {r.flight: r._asdict() for r in c.itertuples(index=False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='UAV-VisLoc 폴더 (01/ ... 11/, satellite_ coordinates_range.csv)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--flights', default='01,02,03,04,05,06,08,09,10,11', help='07은 자세 열이 없고 지도가 띠라 뺀다')
    ap.add_argument('--calib', default='', help='visloc_calib.py 출력 (flight, gsd, head_off, along_m, right_m). 없으면 보정 0')
    ap.add_argument('--every', type=int, default=3, help='한 장 세트: N장마다 1장 (연속 사진은 약 70%% 겹침)')
    ap.add_argument('--calib-frac', type=float, default=0.2, help='비행마다 앞쪽 이 비율의 줄을 보정용으로')
    ap.add_argument('--seg-len', type=float, default=1000, help='비행 구간 최대 길이 (m)')
    ap.add_argument('--seg-min', type=float, default=600, help='이보다 짧은 구간은 버림 (m)')
    ap.add_argument('--seg-per-flight', type=int, default=10, help='비행마다 구간 수 상한 (고르게 고름, 0 = 전부)')
    ap.add_argument('--margin-m', type=float, default=32, help='구간 지도 가장자리 여유 (m). 32 m면 |x_norm| <= 1')
    ap.add_argument('--height-tol', type=float, default=10)
    ap.add_argument('--tilt-max', type=float, default=5)
    ap.add_argument('--turn-deg', type=float, default=15)
    ap.add_argument('--gsd-pairs', type=int, default=20)
    ap.add_argument('--no-seq', action='store_true', help='비행 구간 세트를 만들지 않음')
    ap.add_argument('--quality', type=int, default=95)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = read_calib(a.calib)
    margin_px = a.margin_m * PX_PER_M
    single_rows = {'calib': [], 'eval': []}
    seq_rows, idx_rows, frame_tabs, flight_rows = [], [], [], []

    for flight in [f.strip().zfill(2) for f in a.flights.split(',') if f.strip()]:
        fdir = root / flight
        df, med_step, h_med = load_frames(fdir, flight, a.height_tol, a.tilt_max, a.turn_deg)
        if (df.status == 'no_pose').all():
            print(f'[{flight}] 자세 열 없음 → 건너뜀', flush=True)
            continue
        df['split'] = split_lines(df, a.calib_frac)
        mos = Mosaic(fdir, float(df.lat.mean()))

        c = calib.get(flight, {})
        gsd = num(c.get('gsd'), np.nan)
        gsd_src = 'calib'
        if not np.isfinite(gsd):
            gsd, n_ok, mad = estimate_gsd(fdir / 'drone', df, a.gsd_pairs)
            gsd_src = f'frame_pairs n={n_ok} mad={mad:.4f}'
            if not np.isfinite(gsd):
                print(f'[{flight}] 축척을 잴 수 없음 → 건너뜀 (--calib에 gsd를 넣으면 됨)', flush=True)
                continue
        cal = dict(flight=flight, gsd=gsd,
                   head_off=num(c.get('head_off'), 0.0), along_m=num(c.get('along_m'), 0.0),
                   right_m=num(c.get('right_m'), 0.0))

        Xr, Yr, X, Y, hcw = gt_positions(mos, df, cal['head_off'], cal['along_m'], cal['right_m'])
        df['Xr'], df['Yr'], df['X'], df['Y'] = Xr, Yr, X, Y
        df['theta'] = (-hcw) % 360
        print(f'[{flight}] {len(df)}장, ok {int((df.status == "ok").sum())}, 줄 {df.line.nunique()}, '
              f'gsd {gsd:.4f} ({gsd_src}), 원 지도 m/px {mos.src_m[0]:.3f} x {mos.src_m[1]:.3f}', flush=True)

        n_edge = export_single(out, flight, mos, fdir / 'drone', df, cal, a.every, a.quality, single_rows)
        n_seg = 0
        if not a.no_seq:
            segs = find_segments(df, a.seg_len, a.seg_min, margin_px)
            if a.seg_per_flight and len(segs) > a.seg_per_flight:
                pick = np.unique(np.linspace(0, len(segs) - 1, a.seg_per_flight).round().astype(int))
                segs = [segs[p] for p in pick]
            n_seg = export_seq(out, flight, mos, fdir / 'drone', df, cal, segs, a.quality, seq_rows, idx_rows, margin_px)
        print(f'[{flight}] 한 장 calib {sum(r["flight"] == flight for r in single_rows["calib"])}, '
              f'eval {sum(r["flight"] == flight for r in single_rows["eval"])}, 가장자리 제외 {n_edge}, 구간 {n_seg}', flush=True)

        df['flight'] = flight
        frame_tabs.append(df[['flight', 'row', 'filename', 'lat', 'lon', 'height', 'Phi1', 'Phi2', 'tilt_deg', 'tilted',
                              'turning', 'gap', 'step_m', 'line', 'split', 'status']])
        flight_rows.append(dict(**cal, gsd_source=gsd_src, height_med=h_med, step_med_m=round(med_step, 1),
                                src_m_e=round(mos.src_m[0], 4), src_m_n=round(mos.src_m[1], 4),
                                n_frames=len(df), n_ok=int((df.status == 'ok').sum()), n_lines=int(df.line.nunique()),
                                n_seg=n_seg))

    for split, rows in single_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(out / f'single_{split}' / 'samples.csv', index=False)
    if seq_rows:
        pd.DataFrame(seq_rows).to_csv(out / 'seq_eval' / 'samples.csv', index=False)
        pd.DataFrame(idx_rows).to_csv(out / 'seq_eval' / 'block_index.csv', index=False)
    if frame_tabs:
        pd.concat(frame_tabs).to_csv(out / 'frames.csv', index=False)
    pd.DataFrame(flight_rows).to_csv(out / 'flights.csv', index=False)
    (out / 'README.txt').write_text(
        'UAV-VisLoc export (scripts/export_visloc.py)\n'
        f'calib file: {a.calib or "(none: head_off = along_m = right_m = 0)"}\n'
        f'every={a.every} calib_frac={a.calib_frac} seg_len={a.seg_len} seg_min={a.seg_min} '
        f'seg_per_flight={a.seg_per_flight} margin_m={a.margin_m}\n\n'
        'single_calib/: decide calibration here only.  single_eval/: report numbers here only.\n'
        'seq_eval/: ordered frames per segment, maps/<set>.jpg is 4096 px = 1,024 m (imatch/blocks.py grid).\n'
        'x_norm/y_norm = corrected label, x_norm_raw/y_norm_raw = raw GPS.\n'
        'Limits: GPS/shutter timing not validated; UAV gsd from frame pairs, not camera calibration;\n'
        '        only the central 64 m of each drone photo is used.\n', encoding='utf-8')
    print('done:', out.resolve())


if __name__ == '__main__':
    main()
