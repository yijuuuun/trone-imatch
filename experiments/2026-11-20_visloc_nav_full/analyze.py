"""UAV-VisLoc 긴 구간 재생 항법 표: R1 vs R1 + R4(v1 경량). README "미리 정한 것"의 지표만 낸다.

  python experiments/2026-11-20_visloc_nav_full/analyze.py [--run run/test]

입력: <run>/visloc_loma-cache-{r1,r4g0n5w20ac4}_steps.csv (scripts/run_visloc_nav.py 출력),
      있으면 <run>/visloc_loma-cache-dr_steps.csv (추측 항법만, 참고. CPU로 따로 돌림).
출력: 표(마크다운)와 <run>/checkpoints.csv, <run>/segments_table.csv.

- 지점 d = 구간마다 누적 거리가 처음 d 이상이 되는 사진. d에 못 미치는 구간은 그 지점에서 뺀다.
- 성공 = 그 지점 오차 40 m 이내 (1 km 거리별 비교 #46과 같은 기준).
- 길 잃음 = 오차 > 100 m인 걸음에 들어감, 되찾음 = 그 뒤 다시 40 m 이내. 끝까지 못 돌아온 거리 = 마지막 길 잃음의 시작.
- 실시간 여유 = 사진 간격 ÷ 19.5 m/s (#51과 같음). 걸음 시간 = 걸음마다 기록한 매칭 시간(재탐색 포함).
"""
import argparse
import os
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK_KM = [5, 10, 20, 50]
OK_M, LOST_M, SPEED = 40.0, 100.0, 19.5
NAMES = {'r1': 'R1', 'r4g0n5w20ac4': 'R1 + R4 (v1 경량)', 'dr': '추측 항법만 (참고)'}


def mcnemar(b, c):
    """양측 정확 McNemar p (b, c = 한쪽만 성공한 수)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def per_segment(steps):
    rows, cps = [], []
    for st, g in steps.groupby('set', sort=True):
        g = g.sort_values('k')
        mv = g[g.used != 'start']
        e, dist = g.err_m.to_numpy(), g.dist_m.to_numpy()
        n_lost = n_back = 0
        lost, lost_from, final_lost_km = False, None, np.nan
        for ei, di in zip(e, dist):
            if not lost and ei > LOST_M:
                lost, lost_from = True, di; n_lost += 1
            elif lost and ei <= OK_M:
                lost = False; n_back += 1
        if lost:
            final_lost_km = lost_from / 1000
        budget = np.diff(dist) / SPEED
        sec = mv.sec.fillna(0).to_numpy()
        rows.append(dict(set=st, length_km=round(dist[-1] / 1000, 1), steps=len(mv),
                         within40=float((mv.err_m <= OK_M).mean()), within20=float((mv.err_m <= 20).mean()),
                         unknown=float(mv.unknown.mean()), lost=n_lost, recovered=n_back, final_lost_km=final_lost_km,
                         end_err_m=float(e[-1]), sec_mean=float(sec.mean()), over_budget=float((sec > budget).mean())))
        for d in CHECK_KM + ['end']:
            if d == 'end':
                i = len(g) - 1
            else:
                hit = np.nonzero(dist >= d * 1000)[0]
                if not len(hit):
                    continue
                i = hit[0]
            cps.append(dict(set=st, point=d, dist_m=float(dist[i]), err_m=float(e[i]), ok=bool(e[i] <= OK_M)))
    return pd.DataFrame(rows), pd.DataFrame(cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(HERE, 'run', 'test'))
    ap.add_argument('--matcher', default='loma-cache')
    a = ap.parse_args()
    seg, cp = {}, {}
    for v in NAMES:
        p = os.path.join(a.run, f'visloc_{a.matcher}-{v}_steps.csv')
        if os.path.exists(p):
            seg[v], cp[v] = per_segment(pd.read_csv(p))
    if not seg:
        raise SystemExit(f'{a.run}에 결과 없음')
    points = CHECK_KM + ['end']
    label = lambda d: '구간 끝' if d == 'end' else f'{d} km'

    print('### 지점별 40 m 이내 (그 거리까지 간 구간 중)\n')
    print('| 조건 | ' + ' | '.join(label(d) for d in points) + ' |')
    print('| --- ' * (len(points) + 1) + '|')
    for v, c in cp.items():
        cells = []
        for d in points:
            x = c[c.point == d]
            cells.append(f'{int(x.ok.sum())} / {len(x)} ({x.ok.mean() * 100:.0f}%), 중앙값 {x.err_m.median():.0f} m' if len(x) else '-')
        print(f'| {NAMES[v]} | ' + ' | '.join(cells) + ' |')

    if 'r1' in cp and 'r4g0n5w20ac4' in cp:
        print('\n### R1 vs R1 + R4, 같은 구간 (둘 다 / R1만 / R1 + R4만 / 둘 다 실패, McNemar p)\n')
        print('| 지점 | 둘 다 | R1만 | R1 + R4만 | 둘 다 실패 | p |')
        print('| --- | --- | --- | --- | --- | --- |')
        for d in points:
            m = cp['r1'][cp['r1'].point == d].merge(cp['r4g0n5w20ac4'][cp['r4g0n5w20ac4'].point == d], on='set')
            if not len(m):
                continue
            b, c = int((m.ok_x & ~m.ok_y).sum()), int((~m.ok_x & m.ok_y).sum())
            print(f'| {label(d)} | {int((m.ok_x & m.ok_y).sum())} | {b} | {c} | {int((~m.ok_x & ~m.ok_y).sum())} '
                  f'| {mcnemar(b, c):.2g} |')
        # 구간별 40 m 이내 걸음 비율의 짝 비교 (2026-11-22_visloc_nav_full_vio4의 주 판정). 차이 0인 구간은 뺀다(zero_method='wilcox').
        w1, w4 = (seg[v].set_index('set').within40 for v in ('r1', 'r4g0n5w20ac4'))
        dd = (w4 - w1.reindex(w4.index)).dropna()
        p = wilcoxon(dd[dd != 0], zero_method='wilcox').pvalue if (dd != 0).sum() else 1.0
        print(f'\n구간별 40 m 이내 걸음 비율 (R1 + R4 − R1): 높음 {int((dd > 0).sum())}, 낮음 {int((dd < 0).sum())}, '
              f'같음 {int((dd == 0).sum())}, 차이 중앙값(같음 뺌) {dd[dd != 0].median() * 100:+.1f}%p, Wilcoxon 양측 p = {p:.2g}')

    print('\n### 구간별 (40 m 이내 걸음 비율, 길 잃음 → 되찾음, 끝까지 못 돌아온 거리)\n')
    sets = seg[next(iter(seg))][['set', 'length_km', 'steps']]
    print('| 구간 | 길이 | ' + ' | '.join(NAMES[v] for v in seg) + ' |')
    print('| --- ' * (len(seg) + 2) + '|')
    for _, s in sets.iterrows():
        cells = []
        for v in seg:
            r = seg[v][seg[v].set == s.set].iloc[0]
            tail = f', {r.final_lost_km:.1f} km부터 못 돌아옴' if np.isfinite(r.final_lost_km) else ''
            cells.append(f'{r.within40 * 100:.0f}%, {r.lost} → {r.recovered}{tail}')
        print(f'| {s.set} | {s.length_km} km | ' + ' | '.join(cells) + ' |')

    print('\n### 전체 (모든 걸음, 출발 제외)\n')
    print('| 조건 | 걸음 | 40 m 이내 | 20 m 이내 | 모름 | 길 잃음 → 되찾음 | 끝까지 못 돌아온 구간 | 걸음당 s | 실시간 넘는 걸음 |')
    print('| --- ' * 9 + '|')
    for v, s in seg.items():
        w = s.steps / s.steps.sum()
        print(f'| {NAMES[v]} | {int(s.steps.sum())} | {(s.within40 * w).sum() * 100:.1f}% | {(s.within20 * w).sum() * 100:.1f}% '
              f'| {(s.unknown * w).sum() * 100:.1f}% | {int(s.lost.sum())} → {int(s.recovered.sum())} '
              f'| {int(np.isfinite(s.final_lost_km).sum())} / {len(s)} | {(s.sec_mean * w).sum():.2f} '
              f'| {(s.over_budget * w).sum() * 100:.1f}% |')

    pd.concat([c.assign(variant=v) for v, c in cp.items()]).to_csv(os.path.join(a.run, 'checkpoints.csv'), index=False)
    pd.concat([s.assign(variant=v) for v, s in seg.items()]).to_csv(os.path.join(a.run, 'segments_table.csv'), index=False)


if __name__ == '__main__':
    main()
