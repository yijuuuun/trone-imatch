"""항법 지표: run_nav.py의 _steps.csv에서 궤적·걸음 지표를 계산하고, 같은 궤적끼리 짝지어 비교한다.

  python scripts/nav_metrics.py experiments/2026-11-13_nav_recovery/Tokyo_3d_loma-vio-*_steps.csv --n-traj 32 --seed 7 --odo vio

성공은 같은 이름의 _traj.csv(run_nav.py가 남김)에서 읽는다. 없으면 걸음 기록과 seed로 실제 이동을 다시 계산한다.
지표
  성공       궤적 성공률. 첫 파일과 같은 궤적끼리 McNemar (나만 / 첫 파일만 성공)
  걸음 10 m  걸음 오차 10 m 이내 비율 (항법이 쓴 위치 기준)
  걸음 32 m  32 m 넘는 걸음 비율
  모름       "모름" 걸음 비율
  복귀       32 m 넘게 틀린 뒤 다시 10 m 안으로 돌아오기까지 걸음 수 (중앙값, 못 돌아온 구간은 뺌)과 돌아온 비율
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.metrics import mcnemar  # noqa: E402
from imatch.nav import ODO, PX, make_traj  # noqa: E402


def traj_success(f, n, seed, dist, odo):
    """궤적별 성공. _traj.csv가 있으면 읽고, 없으면 걸음 기록으로 nav_run의 실제 이동을 똑같이 다시 계산한다
    (출발점·목표는 make_traj, 이동 오차는 같은 seed라 결정적이다)."""
    tf = f.replace('_steps.csv', '_traj.csv')
    if os.path.exists(tf):
        return pd.read_csv(tf).set_index('traj').success.reindex(range(n), fill_value=0).astype(bool)
    starts, goals = make_traj(n, dist, seed)
    T = int(3 * (np.linalg.norm(goals[0] - starts[0]) / PX) / 15.)  # nav_run과 같은 식 (부동소수점까지 같아야 오차 난수가 같다)
    rng = np.random.default_rng(seed + 1); o = ODO[odo]
    bias = rng.normal(0, o['bias_deg'], n) if o else np.zeros(n)
    dh = rng.normal(0, o['head_deg'], (n, T)) if o else np.zeros((n, T))
    ds = rng.normal(0, o['scale'], (n, T)) if o else np.zeros((n, T))
    L = pd.read_csv(f).sort_values(['traj', 'step'])
    ok = np.zeros(n, bool)
    for i, g in L.groupby('traj'):
        r = starts[i].astype(float).copy()
        for t, p in zip(g.step, g[['est_x', 'est_y']].to_numpy()):
            d = goals[i] - p; u = d / max(np.linalg.norm(d), 1e-6)
            c, s_ = np.cos(np.radians(bias[i] + dh[i, t])), np.sin(np.radians(bias[i] + dh[i, t]))
            r = r + np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]]) * (1 + ds[i, t]) * 15. * PX
        real_d = np.linalg.norm(r - goals[i]) / PX; pred_d = np.linalg.norm(p - goals[i]) / PX
        ok[i] = (real_d < 20 or pred_d < 20) and real_d < 40  # 도착 판정 없이 끝난 궤적(지도 밖, 시간 초과)은 실패
    return pd.Series(ok)


def recovery(L):
    """32 m 넘는 구간마다 (복귀까지 걸음 수 또는 None)."""
    out = []
    for _, g in L.sort_values('step').groupby('traj'):
        e = g.err_m.to_numpy(); i = 0
        while i < len(e):
            if e[i] > 32:
                j = next((j for j in range(i + 1, len(e)) if e[j] < 10), None)
                out.append(None if j is None else j - i); i = len(e) if j is None else j
            else:
                i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs='+'); ap.add_argument('--n-traj', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--dist', type=float, default=1000)
    ap.add_argument('--odo', default='vio', choices=list(ODO), help='_traj.csv가 없을 때 다시 계산에 쓸 이동 오차')
    a = ap.parse_args()
    print('| 파일 | 성공 | vs 첫 파일 | 걸음 | 걸음 10 m | 걸음 32 m | 모름 | 복귀 걸음 (중앙값) | 돌아온 구간 | 걸음당 s |')
    print('| --- ' * 10 + '|')
    ref = None
    for f in a.steps:
        L = pd.read_csv(f)
        ok = traj_success(f, a.n_traj, a.seed, a.dist, a.odo).to_numpy()
        ref = ok if ref is None else ref
        rec = recovery(L); back = [r for r in rec if r is not None]
        name = re.sub(r'_steps\.csv$', '', os.path.basename(f))
        unk = L.unknown.mean() if 'unknown' in L else (L.n_inl < 15).mean() * np.nan
        print(f'| {name} | {ok.mean()*100:.1f}% ({ok.sum()}) | {mcnemar(ok, ref) if ok is not ref else "—"} | {len(L)} '
              f'| {(L.err_m < 10).mean()*100:.1f}% | {(L.err_m > 32).mean()*100:.1f}% | {unk*100:.1f}% '
              f'| {np.median(back) if back else float("nan"):.0f} | {len(back)}/{len(rec)} | {L.sec.mean():.2f} |')


if __name__ == '__main__':
    main()
