"""닫힌 루프 1 km 항법. capstone1 diag/diag_phase0.py의 make_traj, Cropper, nav_run을 옮긴 것이다.

바뀐 점은 드론 영상을 frames(x, y, heading) → (영상, 영상의 실제 x, y)로 받는 것뿐이다.
- Crop2D: 기존 T5와 같다. 지도를 잘라 드론 영상으로 쓴다 (2D는 SIFT로도 풀리는 시험이라 동작 확인용)
- Nearest3D: 궤적 지점에서 가장 가까운 3D 렌더를 쓴다. 연속 프레임이 없어서 만든 근사다.
  렌더는 도시당 22,500장이고 궤적까지 거리 중앙값 3.0 m, 최대 약 11 m. heading은 렌더마다 무작위라
  비행 방향과 맞지 않는다. 오차는 렌더의 실제 위치로 잰다.
"""
import hashlib
import os
import re

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .blocks import N_BLOCK, crop_block, neighbors
from .geometry import PX_PER_UNIT, turn
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
    같은 (영상, 블록)은 다시 매칭하지 않는다 (self.cache). R0~R4를 한 프로세스에서 돌리면 겹치는 걸음을 아낀다.
    stop_inl > 0이면 각도 탐색을 inlier가 stop_inl 이상인 첫 각도에서 멈춘다 (localize 조기 종료).
    """
    def __init__(self, matcher, city, scale=1.0, angles=(0.0,), stop_inl=0):
        self.matcher, self.city, self.scale, self.angles, self.last, self.cache = matcher, city, scale, angles, [], {}
        self.stop_inl = stop_inl

    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            xy, rec = self.one(crop, bx, by)
            out.append(xy); self.last.append(rec)
        return np.array(out)

    def one(self, crop, bx, by, angles=None):
        """블록 하나: (위치(블록 단위), 기록). self.last는 건드리지 않는다 (Research가 nav_run 걸음 중간에 부른다).

        angles: 이번만 쓸 각도 목록 (None = self.angles). 기록의 angle = 고른 각도 (없으면 NaN).
        """
        angles = self.angles if angles is None else list(angles)
        key = (hashlib.md5(crop.tobytes()).hexdigest(), bx, by) + (() if angles is self.angles else tuple(angles))
        if key not in self.cache:
            recs, best = localize(self.matcher, crop, crop_block(self.city, bx, by), (0.0, 0.0, 0.0), self.scale, angles,
                                  stop_inl=self.stop_inl)
            self.cache[key] = ((best['px'], best['py']) if best else (0.0, 0.0),
                               dict(n_inl=best['n_inl'] if best else 0, found=int(best is not None), sec=sum(r['sec'] for r in recs),
                                    n_ang=len(recs), angle=float(best['angle']) if best else np.nan))
        return self.cache[key]


class NoMatch:
    """매칭하지 않는 기준선. 항상 블록 중심을 답하고 inlier 0을 돌려준다.

    reject_inl = 0이면 "항상 중앙", reject_inl > 0이면 모든 걸음이 "모름"이라 추측 항법만으로 난다.
    """
    def __call__(self, crops, headings, blocks):
        self.last = [dict(n_inl=0, found=0, sec=0.0) for _ in crops]
        return np.zeros((len(crops), 2))


class Research:
    """R3·R4: "모름"이면 추측 위치 주변 블록을 다시 매칭한다. nav_run의 recover로 넘긴다.

    R3: 추측 위치의 블록과 십자 이웃 4개 (⑥ 블록 선택의 후보 5개, experiments/2026-11-06_block_select).
    R4 (wide_after > 0): 연속 "모름"이 wide_after걸음 이상이면 3×3 9개.
    통과 = 그 블록 기준 추정이 ±inside 안 (inside 규칙)이고 inlier >= min_inl. 여러 개면 추측 위치에 가장 가까운 것, 없으면 None.
    gate_m > 0이면 십자 재탐색의 답도 추측 위치에서 gate_m m 안이어야 한다 (R2와 같은 움직임 검사).
    넓게 찾기는 추측 위치가 이미 밀렸다고 보고 움직임 검사를 하지 않는다. 대신 wide_inl > 0이면 넓게 찾기에서만
    inlier >= wide_inl을 요구한다 (R4 학습 도시에서 틀린 3×3 답은 inlier 중앙값 16, 맞은 답은 26).
    추측 위치의 블록은 이번 걸음에 이미 매칭했으므로 캐시에서 나온다. 부를 때마다 self.log에 한 줄 남긴다
    (rec_sec = 이번 걸음에 더 든 매칭 시간, 추측 위치의 블록 제외).

    경량화 (2026-11-18):
    - angle_memory: 이웃 블록은 각도를 다 돌리지 않고, 이 궤적에서 마지막으로 믿은 매칭의 각도 하나만 시도한다.
      그 각도는 nav_run / replay_one이 궤적마다 기억해서 self.hint_angle에 넣어 준다 (없으면 전체 각도).
      드론 방향은 걸음 사이에 거의 바뀌지 않는다는 가정. 추가 센서 없이 우리 매칭 결과만 쓴다.
    - wide_cool: 넓게 찾기(3×3)가 실패하면 wide_cool걸음 동안 십자만 하고 다시 3×3 (0 = 매 걸음 3×3, 예전과 같음).
    - yaw_correct: 기억한 각도에 그 뒤 기체가 돈 각도(self.hint_yaw, nav_run이 넣어 줌)를 더한다. 급선회 대비.
    """
    def __init__(self, match, min_inl=15, gate_m=0., wide_after=0, inside=1.1, wide_inl=0, angle_memory=False, wide_cool=0,
                 yaw_correct=False):
        self.match, self.min_inl, self.gate_m, self.wide_after, self.inside = match, min_inl, gate_m, wide_after, inside
        self.wide_inl, self.angle_memory, self.wide_cool, self.yaw_correct = wide_inl, angle_memory, wide_cool, yaw_correct
        self.hint_angle, self.hint_yaw = None, 0.0
        self.log = []

    def __call__(self, i, crop, guess, streak):
        wide = self.wide_after > 0 and streak >= self.wide_after and \
            (self.wide_cool == 0 or (streak - self.wide_after) % (self.wide_cool + 1) == 0)
        hint = self.hint_angle if self.angle_memory and self.hint_angle is not None and np.isfinite(self.hint_angle) else None
        if hint is not None and self.yaw_correct:
            # 기억한 뒤 기체가 돈 만큼(자세 센서·VIO의 회전량, 추측 항법이 쓰는 것과 같은 정보) 더한다
            hint = float((hint + self.hint_yaw) % 360)
        min_inl = max(self.min_inl, self.wide_inl) if wide else self.min_inl
        bx = int(np.clip(np.round((guess[0] - 256) / 256), 0, N_BLOCK - 1))
        by = int(np.clip(np.round((guess[1] - 256) / 256), 0, N_BLOCK - 1))
        cands = [(dx, dy) for dx, dy in neighbors(bx, by, 1) if wide or abs(dx) + abs(dy) <= 1]
        best, best_d, n_pass, sec = None, np.inf, 0, 0.
        for dx, dy in cands:
            # 추측 위치의 블록은 이번 걸음 본 매칭(전체 각도)이 캐시에 있다. 이웃 블록만 기억한 각도 하나로
            (x, y), rec = self.match.one(crop, bx + dx, by + dy,
                                         angles=[hint] if hint is not None and (dx, dy) != (0, 0) else None)
            sec += rec['sec'] if (dx, dy) != (0, 0) else 0.
            if not rec['found'] or rec['n_inl'] < min_inl or max(abs(x), abs(y)) > self.inside:
                continue
            p = np.array([(bx + dx) * 256 + 256, (by + dy) * 256 + 256]) + np.array([x, y]) * PX_PER_UNIT
            d = np.linalg.norm(p - guess) / PX
            if self.gate_m > 0 and not wide and d > self.gate_m:
                continue
            n_pass += 1
            if d < best_d:
                best, best_d, pick = p, d, (bx + dx, by + dy, rec['n_inl'])
        self.log.append(dict(traj=int(i), rec_wide=int(wide), rec_cand=len(cands), rec_pass=n_pass, rec_sec=sec,
                             rec_hint=hint if hint is not None else np.nan,
                             **(dict(zip(('rec_bx', 'rec_by', 'rec_n_inl'), pick)) if best is not None else {})))
        return best


ODO = {'none': None, 'vio': dict(bias_deg=3.0, head_deg=2.0, scale=0.05),
       # 민감도 분석용 (2026-11-21): vio의 모든 오차를 2배, 4배. 성능이 나쁜 VIO·센 바람 가정
       'vio2': dict(bias_deg=6.0, head_deg=4.0, scale=0.10), 'vio4': dict(bias_deg=12.0, head_deg=8.0, scale=0.20)}
"""이동 오차 (추측 항법이 공짜로 맞지 않게).

시뮬레이션의 드론은 명령한 대로 정확히 움직이므로, 오차가 없으면 영상을 안 쓰고 추측 항법만 해도 100% 도착한다.
'vio': 궤적마다 heading 치우침 N(0, 3°), 걸음마다 heading N(0, 2°), 걸음 길이 N(1, 5%).
1 km에서 heading 3° 치우침만으로 옆으로 약 52 m 밀린다. 드론은 이 오차를 모르고 명령한 이동량으로 추측한다.
"""


def nav_run(loc, frames, starts, goals, decode_px=128, oracle_block=False, step_m=15., arrive_m=20., n_block=15, log=None,
            odo=None, seed=0, reject_inl=0, gate_m=0., recover=None, turns=0):
    """(성공 여부, 종료 이유, 궤적별 걸음 오차 중앙값 m). log: 걸음마다 기록을 덧붙일 목록.

    복구 (docs/04 R0~R4). 기본값이면 R0이고 예전 결과와 같다.
    odo: ODO의 값. 실제 이동에 오차를 넣는다. 같은 seed면 모든 R에서 같은 오차가 나온다.
    reject_inl: 고른 답의 inlier가 이보다 작으면 "모름" (R1, 15).
    gate_m: 추정이 추측 위치(직전 위치 + 명령 이동량)에서 이보다 멀면 "모름" (R2).
    turns: > 0이면 급선회 시험. 궤적마다 경로의 1/3, 2/3 지점 근처(같은 seed면 같은 걸음)에서 기체가 ±90° 돈 것처럼
           드론 영상을 돌린다(누적). 기체의 회전량(yaw)은 자세 센서로 안다고 보고 recover.hint_yaw로 넘긴다.
           위치·이동은 바꾸지 않는다. 0이면 예전과 같다.
    recover: "모름"일 때 부르는 함수 recover(i, crop, guess_px, streak) → 지도 px 위치 또는 None (R3, R4 자리).
             None을 돌려주거나 recover가 없으면 추측 위치를 쓴다. streak = 이 궤적의 연속 "모름" 걸음 수 (이번 걸음 포함).
    """
    n = len(starts); dist_m = np.linalg.norm(goals[0] - starts[0]) / PX
    r = starts.copy(); nom = r.copy()
    alive = np.ones(n, bool); success = np.zeros(n, bool); reason = np.array(['timeout'] * n, dtype=object)
    heading = np.full(n, 90.0); err = [[] for _ in range(n)]; streak = np.zeros(n, int)
    good_angle = np.full(n, np.nan)  # 궤적마다 마지막으로 믿은 매칭의 각도 (Research angle_memory가 씀)
    T = int(3 * dist_m / step_m)
    good_yaw = np.zeros(n)  # 그때의 기체 회전량
    yaw = np.zeros((n, T + 1))
    if turns:
        trng = np.random.default_rng([seed, 9091])
        for i in range(n):
            for frac in (1 / 3, 2 / 3):  # 곧게 날면 dist_m / step_m 걸음에 도착한다. 그 1/3, 2/3 지점 + 0~5걸음
                t0 = int(frac * dist_m / step_m) + int(trng.integers(0, 6))
                yaw[i, t0:] += trng.choice([-90.0, 90.0])
    rng = np.random.default_rng(seed + 1)
    bias = rng.normal(0, odo['bias_deg'], n) if odo else np.zeros(n)
    dh = rng.normal(0, odo['head_deg'], (n, T)) if odo else np.zeros((n, T))
    ds = rng.normal(0, odo['scale'], (n, T)) if odo else np.zeros((n, T))
    for t in range(T):
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
            if yaw[i, t]:
                crop = turn(crop, -yaw[i, t])  # 기체가 yaw만큼 돈 영상 (매칭에 필요한 각도가 yaw만큼 바뀐다)
            crops.append(crop); real.append(np.array([fx, fy]))
        pos = loc(crops, heading[idx], blocks)
        last = getattr(loc, 'last', None) or [{} for _ in idx]
        for j, i in enumerate(idx):
            raw = cen[j] + pos[j] * decode_px
            guess = nom[i].copy()  # 드론이 아는 것만으로 추측한 이번 위치 (복사: 아래에서 nom[i]를 고친다)
            n_inl = last[j].get('n_inl', np.inf)
            unknown = (reject_inl > 0 and n_inl < reject_inl) or (gate_m > 0 and np.linalg.norm(raw - guess) / PX > gate_m)
            streak[i] = streak[i] + 1 if unknown else 0
            p, how = raw, 'match'
            if not unknown and np.isfinite(last[j].get('angle', np.nan)):
                good_angle[i], good_yaw[i] = last[j]['angle'], yaw[i, t]
            if unknown:
                if recover is not None:
                    recover.hint_angle, recover.hint_yaw = good_angle[i], yaw[i, t] - good_yaw[i]
                q = recover(i, crops[j], guess, streak[i]) if recover else None
                p, how = (np.asarray(q, float), 'recover') if q is not None else (guess, 'guess')
                if q is not None: streak[i] = 0
            err[i].append(float(np.linalg.norm(p - real[j]) / PX))
            if log is not None:
                log.append(dict(traj=int(i), step=t, real_x=float(real[j][0]), real_y=float(real[j][1]), est_x=float(p[0]),
                                est_y=float(p[1]), block_x=blocks[j][0], block_y=blocks[j][1], err_m=err[i][-1],
                                outside_block=int(np.abs(real[j] - cen[j]).max() > 128),
                                raw_err_m=float(np.linalg.norm(raw - real[j]) / PX), unknown=int(unknown), used=how,
                                yaw=float(yaw[i, t]), **last[j]))
            d = goals[i] - p; u = d / max(np.linalg.norm(d), 1e-6)
            heading[i] = np.degrees(np.arctan2(-u[1], u[0]))
            a = np.radians(bias[i] + dh[i, t]); c, s_ = np.cos(a), np.sin(a)
            move = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]]) * (1 + ds[i, t])
            r[i] = r[i] + move * step_m * PX; nom[i] = p + u * step_m * PX
            real_d = np.linalg.norm(r[i] - goals[i]) / PX; pred_d = np.linalg.norm(p - goals[i]) / PX
            if real_d < arrive_m or pred_d < arrive_m:
                alive[i] = False; success[i] = real_d < 2 * arrive_m
                reason[i] = 'arrived_ok' if success[i] else 'arrived_pred_only'
            elif (r[i] < 200).any() or (r[i] > 256 * (n_block + 1) - 200).any():
                alive[i] = False; reason[i] = 'out_of_map'
    return success, reason, [float(np.median(e)) for e in err if e]
