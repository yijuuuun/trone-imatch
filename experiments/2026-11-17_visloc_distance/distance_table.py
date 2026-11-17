"""UAV-VisLoc seq_eval 거리별 표: 같은 92구간을 100 m, 300 m 지점과 구간 끝에서 잘라 본다.

  python experiments/2026-11-17_visloc_distance/distance_table.py

재생 항법은 앞 걸음 결과가 뒤 사진과 상관없이 정해지므로, 구간을 끝까지 돌린 기록의 앞부분이
"그 거리까지만 난 비행"의 결과와 같다. 그래서 새로 돌리지 않고 기존 기록을 자른다.
  - LoMa(dr, r0, r1, r4g0n5w20): experiments/2026-11-16_visloc_nav_v1 (확정 v1 4방향 + 멈추기 15, #45)
  - Bearing-UAV(공개 사전학습): experiments/2026-11-16_bearing_visloc_colab (#42)
  다섯 조건 모두 출발 위치 GPS, seed 2026, 이동 오차 vio로 같은 구간·같은 이동 오차를 쓴다.
지점 d = 구간마다 누적 거리가 처음 d 이상이 되는 사진. 구간 끝 = 마지막 사진.
출력: 표(마크다운)와 distance_table.csv.
"""
import os
from math import comb

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SRC = {
    'dr': ('추측 항법만', f'{EXP}/2026-11-16_visloc_nav_v1/visloc_loma-dr_steps.csv'),
    'bearing': ('Bearing-UAV', f'{EXP}/2026-11-16_bearing_visloc_colab/run/Bearing-UAV/full/steps.csv'),
    'r0': ('LoMa R0 (항상 믿음)', f'{EXP}/2026-11-16_visloc_nav_v1/visloc_loma-r0_steps.csv'),
    'r1': ('R1', f'{EXP}/2026-11-16_visloc_nav_v1/visloc_loma-r1_steps.csv'),
    'r4g0n5w20': ('v1 (R1 + R4)', f'{EXP}/2026-11-16_visloc_nav_v1/visloc_loma-r4g0n5w20_steps.csv'),
}
POINTS = [(100, '100 m'), (300, '300 m'), (None, '1 km (구간 끝)')]
OK_M = 40  # 항법 성공 기준과 같은 40 m


def at(df, d):
    df = df.sort_values(['set', 'k'])
    if d is None:
        return df.groupby('set').tail(1).set_index('set')
    return df[df.dist_m >= d].groupby('set').head(1).set_index('set')


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def fp(p):
    return f'{p:.0e}' if p < 0.001 else ('1' if p >= 1 else f'{p:.2g}')


def main():
    L = {v: pd.read_csv(f) for v, (_, f) in SRC.items()}
    sets = sorted(L['r4g0n5w20'].set.unique())
    for v, df in L.items():
        assert sorted(df.set.unique()) == sets and len(df) == len(L['r4g0n5w20']), v
    rows = []
    for d, lab in POINTS:
        R = {v: at(df, d).reindex(sets) for v, df in L.items()}
        ok = {v: (R[v].err_m <= OK_M).values for v in R}
        real = R['dr'].dist_m
        print(f'\n### {lab}: 실제 누적 거리 중앙값 {real.median():.0f} m ({real.min():.0f}~{real.max():.0f}), '
              f'사진 순서 중앙값 {R["dr"].k.median():.0f}번째, {len(sets)}구간')
        print(f'| 조건 | {OK_M} m 이내 | v1 대비 (v1만 / 상대만) | 10 m 이내 | 20 m 이내 | 오차 중앙값 |')
        print('| --- | --- | --- | --- | --- | --- |')
        for v, (name, _) in SRC.items():
            e = R[v].err_m
            x, y, p = mcnemar(ok['r4g0n5w20'], ok[v])
            vs = '—' if v == 'r4g0n5w20' else f'{x} / {y}, p = {fp(p)}'
            print(f'| {name} | {ok[v].sum()} ({ok[v].mean() * 100:.1f}%) | {vs} | {(e <= 10).mean() * 100:.1f}% '
                  f'| {(e <= 20).mean() * 100:.1f}% | {e.median():.1f} m |')
            rows.append(dict(point=lab, real_dist_med=round(real.median(), 1), method=v, n=len(sets), ok40=int(ok[v].sum()),
                             v1_only=x, other_only=y, p=p, w10=(e <= 10).mean(), w20=(e <= 20).mean(), med_err_m=e.median()))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'distance_table.csv'), index=False)


if __name__ == '__main__':
    main()
