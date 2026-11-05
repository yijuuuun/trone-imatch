"""닫힌 루프 1 km 항법. capstone1 diag/diag_phase0.py의 make_traj, Cropper, nav_run을 옮긴 것이다.

바뀐 점은 드론 영상을 frames(x, y, heading) → (영상, 영상의 실제 x, y)로 받는 것뿐이다.
- Crop2D: 기존 T5와 같다. 지도를 잘라 드론 영상으로 쓴다 (2D는 SIFT로도 풀리는 시험이라 동작 확인용)
- Nearest3D: 궤적 지점에서 가장 가까운 3D 렌더를 쓴다. 연속 프레임이 없어서 만든 근사다.
  렌더는 도시당 22,500장이고 궤적까지 거리 중앙값 3.0 m, 최대 약 11 m. heading은 렌더마다 무작위라
  비행 방향과 맞지 않는다. 오차는 렌더의 실제 위치로 잰다.
"""
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .blocks import crop_block
from .geometry import PX_PER_UNIT
from .pipeline import localize

PX = 4.0  # px per metre (0.25 m/px)
W = 384
META_3D = 'Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'


def make_traj(n_traj, dist_m, seed, n_block=15):
    rng = np.random.default_rng(seed)
    lo, hi = 300, 256 * (n_block + 1) - 300
    starts, goals = [], []
    while len(starts) < n_traj:
        s = rng.uniform(lo, hi, 2); a = rng.uniform(0, 2 * np.pi)
        g = s + dist_m * PX * np.array([np.cos(a), np.sin(a)])
        if (g > lo).all() and (g < hi).all():
            starts.append(s); goals.append(g)
    return np.array(starts), np.array(goals)


class Crop2D:
    """지도에서 드론 영상을 자른다 (capstone1 Cropper와 같은 결과)."""
    def __init__(self, city):
        self.pad = cv2.copyMakeBorder(city, W, W, W, W, cv2.BORDER_REFLECT)

    def __call__(self, cx, cy, theta):
        x0, y0 = int(cx) - W, int(cy) - W
        win = self.pad[y0 + W:y0 + 3 * W, x0 + W:x0 + 3 * W]
        lx, ly = cx - x0, cy - y0
        R = cv2.getRotationMatrix2D((lx, ly), -theta, 1)
        rot = cv2.warpAffine(win, R, (win.shape[1], win.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        c_x, c_y = int(lx), int(ly)
        return rot[c_y - 128:c_y + 128, c_x - 128:c_x + 128], cx, cy


class Nearest3D:
    """가장 가까운 3D 렌더를 돌려준다. capstone1의 Bearing_UAV_90K가 필요하다."""
    def __init__(self, capstone1, st):
        ho = re.search(r'(\d+bc)', st).group(1)
        df = pd.read_csv(os.path.join(capstone1, META_3D), usecols=['block_x', 'block_y', 'x_norm', 'y_norm', 'target_path'])
        df = df[df.target_path.str.contains(f'_{ho}_')]
        self.xy = np.c_[df.block_x * 256 + 256 + df.x_norm * PX_PER_UNIT, df.block_y * 256 + 256 + df.y_norm * PX_PER_UNIT]
        self.paths = [os.path.join(capstone1, p) for p in df.target_path]
        self.tree = cKDTree(self.xy)

    def __call__(self, cx, cy, theta):
        i = self.tree.query((cx, cy))[1]
        return cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB), self.xy[i, 0], self.xy[i, 1]


class MatchLoc:
    """localize를 nav_run의 loc(crops, headings, blocks) 형식으로 감싼다. 위치는 블록 단위 (1 = 128 px).

    추정이 없으면 블록 중심을 답한다 (imatch/metrics.py 규칙과 같음). 나침반은 쓰지 않으므로 headings는 무시한다.
    걸음별 기록은 self.last에 남고, nav_run의 log에 들어간다.
    """
    def __init__(self, matcher, city, scale=1.0, angles=(0.0,)):
        self.matcher, self.city, self.scale, self.angles, self.last = matcher, city, scale, angles, []

    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            recs, best = localize(self.matcher, crop, crop_block(self.city, bx, by), (0.0, 0.0, 0.0), self.scale, self.angles)
            self.last.append(dict(n_inl=best['n_inl'] if best else 0, found=int(best is not None), sec=sum(r['sec'] for r in recs)))
            out.append((best['px'], best['py']) if best else (0.0, 0.0))
        return np.array(out)


def nav_run(loc, frames, starts, goals, decode_px=128, oracle_block=False, step_m=15., arrive_m=20., n_block=15, log=None):
    """(성공 여부, 종료 이유, 궤적별 걸음 오차 중앙값 m). log: 걸음마다 기록을 덧붙일 목록."""
    n = len(starts); dist_m = np.linalg.norm(goals[0] - starts[0]) / PX
    r = starts.copy(); nom = r.copy()
    alive = np.ones(n, bool); success = np.zeros(n, bool); reason = np.array(['timeout'] * n, dtype=object)
    heading = np.full(n, 90.0); err = [[] for _ in range(n)]
    for t in range(int(3 * dist_m / step_m)):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        blocks, crops, cen, real = [], [], [], []
        for i in idx:
            ref = r[i] if oracle_block else nom[i]
            bx = int(np.clip(np.round((ref[0] - 256) / 256), 0, n_block - 1))
            by = int(np.clip(np.round((ref[1] - 256) / 256), 0, n_block - 1))
            blocks.append((bx, by)); cen.append(np.array([bx * 256 + 256, by * 256 + 256]))
            crop, fx, fy = frames(r[i, 0], r[i, 1], heading[i])
            crops.append(crop); real.append(np.array([fx, fy]))
        pos = loc(crops, heading[idx], blocks)
        for j, i in enumerate(idx):
            p = cen[j] + pos[j] * decode_px
            err[i].append(float(np.linalg.norm(p - real[j]) / PX))
            if log is not None:
                log.append(dict(traj=int(i), step=t, real_x=float(real[j][0]), real_y=float(real[j][1]), est_x=float(p[0]),
                                est_y=float(p[1]), block_x=blocks[j][0], block_y=blocks[j][1], err_m=err[i][-1],
                                outside_block=int(np.abs(real[j] - cen[j]).max() > 128),
                                **(loc.last[j] if getattr(loc, 'last', None) else {})))
            d = goals[i] - p; u = d / max(np.linalg.norm(d), 1e-6)
            heading[i] = np.degrees(np.arctan2(-u[1], u[0]))
            r[i] = r[i] + u * step_m * PX; nom[i] = p + u * step_m * PX
            real_d = np.linalg.norm(r[i] - goals[i]) / PX; pred_d = np.linalg.norm(p - goals[i]) / PX
            if real_d < arrive_m or pred_d < arrive_m:
                alive[i] = False; success[i] = real_d < 2 * arrive_m
                reason[i] = 'arrived_ok' if success[i] else 'arrived_pred_only'
            elif (r[i] < 200).any() or (r[i] > 256 * (n_block + 1) - 200).any():
                alive[i] = False; reason[i] = 'out_of_map'
    return success, reason, [float(np.median(e)) for e in err if e]
