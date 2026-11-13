"""실제 비행 재생 항법 (UAV-VisLoc seq_eval). nav.py의 닫힌 루프 시뮬레이터와 따로 둔다.

실제 사진은 드론이 날아간 경로에서만 찍혀 있어, 추정대로 드론을 움직이고 그 자리 사진을 새로 가져올 수 없다
(nav.py nav_run의 frames). 그래서 찍힌 순서대로 사진을 주고, 매 걸음
  추측 위치 = 직전 추정 + 이동량(실제 이동 + VIO 오차),  블록 = 추측 위치를 담은 블록,  추정 = loc(사진, 블록)
으로 위치를 잇는다 (열린 루프). 정답 GPS는 사진 순서와 채점, 이동량 시뮬레이션에만 쓴다. 출발 위치는 정답으로 준다.

모든 방법이 같은 구간·출발점·이동 오차(같은 seed)·블록 고르기 규칙을 쓰므로 짝지어 비교할 수 있다.
loc은 nav_run과 같은 형식이다: loc(crops, headings, blocks) → (n, 2) 블록 단위 위치, 선택으로 loc.last = [{n_inl, ...}].
그래서 기준선(B1, CorrMatchNet 등)은 3D 항법에 꽂았던 loc을 지도만 바꿔 그대로 쓴다:

  from imatch.replay import run_segments, summarize
  steps, segs = run_segments(lambda city: B1Loc(model, city), 'data/visloc/seq_eval', odo='vio', seed=2026)
  # B1Loc: 3D 항법 때 b1_loc_sg를 만든 클래스. 구간마다 지도(city)만 바뀐다
  print(summarize(steps, segs))

nav_run과 다른 점
- 드론은 정답 경로를 그대로 지나간다. 그래서 "도착 성공" 대신 걸음별 오차와 마지막 오차로 잰다.
- 이동량 오차는 nav.ODO['vio']와 같은 분포를 실제 이동(사진 사이 GPS 변위)에 넣는다:
  구간마다 heading 치우침 N(0, 3°), 걸음마다 heading N(0, 2°), 걸음 길이 N(1, 5%).
- 걸음 간격이 60~130 m다 (3D는 15 m).
"""
import os

import cv2
import numpy as np
import pandas as pd

from .blocks import N_BLOCK
from .geometry import PX_PER_UNIT
from .nav import ODO, PX


def _rot(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def load_segments(data, sets=None, max_dist=None):
    """seq_eval/samples.csv → {set: 걸음 표(k 순서)}. max_dist(m)면 구간 시작부터 그 거리까지의 사진만."""
    S = pd.read_csv(os.path.join(data, 'samples.csv'))
    if sets:
        S = S[S.set.isin(sets)]
    if max_dist is not None:
        S = S[S.dist_m <= max_dist + 1e-6]
    return {st: g.sort_values('k').reset_index(drop=True) for st, g in S.groupby('set', sort=True)}


def replay_one(loc, g, data, st, i, odo=None, seed=0, reject_inl=0, gate_m=0., recover=None, oracle_block=False,
               n_block=N_BLOCK, log=None):
    """구간 하나. i = 구간 번호 (이동 오차 난수를 구간마다 고정하는 데 씀). 걸음마다 log에 dict를 덧붙인다."""
    real = g[['map_x', 'map_y']].to_numpy(float)
    rng = np.random.default_rng([seed + 1, i])
    n = len(g)
    bias = rng.normal(0, odo['bias_deg']) if odo else 0.0
    dh = rng.normal(0, odo['head_deg'], n) if odo else np.zeros(n)
    ds = rng.normal(0, odo['scale'], n) if odo else np.zeros(n)
    est, streak = real[0].copy(), 0
    for k in range(n):
        if k == 0:
            guess, course = real[0].copy(), 0.0
        else:
            d = _rot(real[k] - real[k - 1], bias + dh[k]) * (1 + ds[k])   # 드론이 아는 이동량 (지도 px, y 아래)
            guess = est + d
            course = float(np.degrees(np.arctan2(-d[1], d[0])))           # nav_run heading과 같은 규칙
        ref = real[k] if oracle_block else guess
        bx = int(np.clip(np.round((ref[0] - 256) / 256), 0, n_block - 1))
        by = int(np.clip(np.round((ref[1] - 256) / 256), 0, n_block - 1))
        cen = np.array([bx * 256 + 256, by * 256 + 256], float)
        crop = cv2.cvtColor(cv2.imread(os.path.join(data, st, f'k{int(g.k[k]):03d}_uav.jpg')), cv2.COLOR_BGR2RGB)
        pos = np.asarray(loc([crop], np.array([course]), [(bx, by)]), float)[0]
        last = (getattr(loc, 'last', None) or [{}])[0]
        raw = cen + pos * PX_PER_UNIT
        n_inl = last.get('n_inl', np.inf)
        unknown = (reject_inl > 0 and n_inl < reject_inl) or (gate_m > 0 and np.linalg.norm(raw - guess) / PX > gate_m)
        streak = streak + 1 if unknown else 0
        p, how = raw, 'match'
        if unknown:
            q = recover(i, crop, guess, streak) if recover else None
            p, how = (np.asarray(q, float), 'recover') if q is not None else (guess, 'guess')
            if q is not None:
                streak = 0
        est = p
        if log is not None:
            log.append(dict(set=st, traj=i, k=int(g.k[k]), dist_m=float(g.dist_m[k]), real_x=real[k, 0], real_y=real[k, 1],
                            guess_x=guess[0], guess_y=guess[1], est_x=p[0], est_y=p[1], block_x=bx, block_y=by,
                            err_m=float(np.linalg.norm(p - real[k]) / PX), raw_err_m=float(np.linalg.norm(raw - real[k]) / PX),
                            guess_err_m=float(np.linalg.norm(guess - real[k]) / PX),
                            outside_block=int(np.abs(real[k] - cen).max() > 128), unknown=int(unknown), used=how,
                            tilted=bool(g.tilted[k]) if 'tilted' in g else False,
                            turning=bool(g.turning[k]) if 'turning' in g else False, **last))
    return est


def run_segments(make_loc, data, sets=None, odo='vio', seed=2026, reject_inl=0, gate_m=0., make_recover=None,
                 oracle_block=False, max_dist=None, verbose=True):
    """모든 구간을 재생한다. make_loc(city) → loc, make_recover(loc) → recover 또는 None.

    반환: (걸음 표, 구간 표). 구간 표: 걸음 수, 길이, 마지막 오차, 걸음별 10/20/32 m 비율, 모름 비율.
    """
    odo = ODO[odo] if isinstance(odo, str) else odo
    segs = load_segments(data, sets, max_dist)
    steps, rows = [], []
    for i, (st, g) in enumerate(segs.items()):
        city = cv2.cvtColor(cv2.imread(os.path.join(data, 'maps', f'{st}.jpg')), cv2.COLOR_BGR2RGB)
        loc = make_loc(city)
        rec = make_recover(loc) if make_recover else None
        log = []
        replay_one(loc, g, data, st, i, odo=odo, seed=seed, reject_inl=reject_inl, gate_m=gate_m, recover=rec,
                   oracle_block=oracle_block, log=log)
        if rec is not None and getattr(rec, 'log', None):
            L = pd.DataFrame(log)
            R = pd.DataFrame(rec.log, index=L.index[L.unknown == 1])
            L = L.join(R.drop(columns='traj'))
            if 'sec' in L:
                L['sec'] += L.rec_sec.fillna(0)
            log = L.to_dict('records')
        steps += log
        e = np.array([s['err_m'] for s in log])
        rows.append(dict(set=st, traj=i, n_steps=len(log), length_m=float(g.dist_m.iloc[-1]), final_err_m=float(e[-1]),
                         max_err_m=float(e.max()), med_err_m=float(np.median(e)), w10=float((e <= 10).mean()),
                         w20=float((e <= 20).mean()), w32=float((e <= 32).mean()),
                         unknown=float(np.mean([s['unknown'] for s in log]))))
        if verbose and (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(segs)} 구간', flush=True)
    return pd.DataFrame(steps), pd.DataFrame(rows)


def summarize(steps, segs):
    """한 줄 요약 dict. 걸음 비율은 출발 걸음(k = 0, 위치를 줌)을 뺀다."""
    s = steps[steps.k != steps.groupby('set').k.transform('min')]
    return dict(segments=len(segs), steps=len(s), w10=round(float((s.err_m <= 10).mean() * 100), 1),
                w20=round(float((s.err_m <= 20).mean() * 100), 1), w32=round(float((s.err_m <= 32).mean() * 100), 1),
                med_err_m=round(float(s.err_m.median()), 1), final_med_m=round(float(segs.final_err_m.median()), 1),
                final_le40=round(float((segs.final_err_m <= 40).mean() * 100), 1),
                unknown=round(float(s.unknown.mean() * 100), 1), outside_block=round(float(s.outside_block.mean() * 100), 1))
