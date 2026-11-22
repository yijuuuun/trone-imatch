#!/usr/bin/env python
"""XIAN 사후 분석: 위성 지도 좌표 어긋남(상수)을 정답에서 빼고 다시 채점한다. 다시 돌리지 않는다.

  python scripts/rescore_xian.py E:\\R4_Datasets\\xian_result\\out
  python scripts/rescore_xian.py <out> --flights <seq_test>/flights.csv --tag F   # 지도 여러 장: 같은 지도의 다른 비행으로 추정

배경: normal R1에서 매칭이 성공한 걸음의 (추정 − 정답)이 세 비행 모두 남쪽 30~41 m로 일정했고,
진행 방향 기준으로는 일정하지 않았다 → 위성 지도 좌표 어긋남으로 봄.
보정값: 채점할 비행을 뺀 나머지 두 비행의 normal R1 매칭 걸음 (추정 − 정답) 중앙값 (하나 빼고 추정).
정답 = GPS + 보정값으로 놓고 마지막 걸음 오차를 다시 잰다. 출발 위치는 GPS로 주어졌으므로 첫 걸음 어긋남은 남는다.
미리 정한 주 판정은 xian_judgement.md 그대로다. 이 파일은 사후 분석이다.
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

PX = 4.0
THRESH = (20, 40, 100)
R4 = 'r4g0n5w20ac4'


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(out, flights=None, tag=''):
    def load(name):
        s = pd.read_csv(os.path.join(out, f'{name}_steps.csv'))
        s['flight'] = s.set.str.split('_').str[1]
        return s

    # 보정값: 비행마다, 나머지 두 비행의 normal r1 매칭 걸음에서
    base = load('xian_normal_s2026-r1')
    m = base[base.used == 'match']
    fmap = {}
    if flights:
        FL = pd.read_csv(flights, dtype=str)
        fmap = dict(zip(FL.flight.astype(str), FL.region.astype(str) + '/' + FL['map'].astype(str)))
    off, how = {}, {}
    for fl in sorted(base.flight.unique()):
        same = [g for g, mp in fmap.items() if mp == fmap.get(fl) and g != fl]
        o = m[m.flight.isin(same)] if same else m[m.flight != fl]
        how[fl] = f"같은 지도 {','.join(same)}" if same else '나머지 모든 비행'
        off[fl] = np.array([np.median(o.est_x - o.real_x), np.median(o.est_y - o.real_y)])
    lines = [f'# XIAN{tag} 지도 좌표 어긋남 보정 후 다시 채점', '',
             '보정값 (채점할 비행을 뺀 비행들의 normal R1 매칭 걸음 (추정 − 정답) 중앙값, 동쪽 m, 남쪽 m):']
    for fl, v in off.items():
        lines.append(f'- 비행 {fl}: {v[0] / PX:.1f}, {v[1] / PX:.1f} ({how[fl]})')

    rows = []
    for f in sorted(glob.glob(os.path.join(out, 'xian_*_steps.csv'))):
        name = os.path.basename(f)[:-10]
        mt = re.fullmatch(r'xian_(normal|large)_s(\d+)-(.+)', name)
        if not mt:
            continue
        s = load(name)
        last = s.sort_values('k').groupby('set').tail(1)
        o = np.stack([off[fl] for fl in last.flight])
        err = np.hypot(last.est_x - last.real_x - o[:, 0], last.est_y - last.real_y - o[:, 1]) / PX
        rows.append(pd.DataFrame(dict(cond=mt.group(1), seed=int(mt.group(2)), variant=mt.group(3), set=last.set.values,
                                      flight=last.flight.values, err=err.values, err_raw=last.err_m.values)))
    S = pd.concat(rows, ignore_index=True)
    for t in THRESH:
        S[f'ok{t}'] = S.err <= t

    lines += ['', '## 표 (보정 후 마지막 오차 기준 도착 수)', '',
              '| 조건 | seed | variant | 구간 | 20 m | 40 m | 100 m | 보정 후 중앙값 | 보정 전 중앙값 |', '| --- ' * 9 + '|']
    for (c, sd, v), g in S.groupby(['cond', 'seed', 'variant']):
        lines.append(f'| {c} | {sd} | {v} | {len(g)} | {g.ok20.sum()} | {g.ok40.sum()} | {g.ok100.sum()} '
                     f'| {g.err.median():.1f} m | {g.err_raw.median():.1f} m |')

    def pair(df, va, vb, t):
        A = df[df.variant == va].set_index(['seed', 'set'])[f'ok{t}']
        B = df[df.variant == vb].set_index(['seed', 'set'])[f'ok{t}']
        A, B = A.align(B, join='inner')
        b, c = int((A & ~B).sum()), int((~A & B).sum())
        return int(A.sum()), int(B.sum()), b, c, mcnemar(b, c), len(A)

    N, L = S[S.cond == 'normal'], S[S.cond == 'large']
    lines += ['', '## 짝 비교 (보정 후)', '']
    for title, df, va, vb in [('normal: r1 vs dr', N, 'r1', 'dr'), ('normal: r4 vs r1', N, R4, 'r1'),
                              ('large 4 seed: r1 vs dr', L, 'r1', 'dr'), ('large 4 seed: r4 vs r1', L, R4, 'r1')]:
        lines.append(f'**{title}**')
        for t in THRESH:
            na, nb, b, c, p, n = pair(df, va, vb, t)
            lines.append(f'- {t} m: {va} {na}/{n}, {vb} {nb}/{n}, 한쪽만 {b} vs {c}, p = {p:.4g}')
        lines.append('')
    a4 = L[L.variant == R4].groupby('set').ok40.sum(); a1 = L[L.variant == 'r1'].groupby('set').ok40.sum()
    a4, a1 = a4.align(a1, join='inner')
    bt, wr = int((a4 > a1).sum()), int((a4 < a1).sum())
    lines.append(f'- large 구간 단위 부호 검정 (40 m): R4가 나은 구간 {bt}, 나쁜 구간 {wr}, p = {mcnemar(bt, wr):.4g}')
    text = '\n'.join(lines)
    open(os.path.join(out, f'xian{tag}_rescore.md'), 'w', encoding='utf-8').write(text)
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--flights', default=None); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main(a.out, a.flights, a.tag)
