#!/usr/bin/env python
"""XIAN-VisLoc 재생 항법: R1 vs R1 + R4(경량 v1). 다른 실험과 독립적으로 조건·판정을 이 파일 안에 정한다 (2026-11-22, 결과 보기 전).

  python scripts/run_xian_nav.py --data data/xian/seq_test --out outputs/xian_nav

데이터: scripts/export_xian.py가 만든 seq_test (README.txt에 가공 규칙).
방법 (팀 최종 v1 구성을 그대로 씀, imatch.nav)
  dr             매칭 없이 추측 항법만
  r1             inlier < 15이면 "모름" → 추측 위치
  r4g0n5w20ac4   r1 + 십자 재매칭(기억한 방향) + 연속 5걸음 "모름"이면 3×3 재탐색(inlier ≥ 20, 4걸음 간격)
매처 loma-cache, 4방향, 조기 종료 inlier 15 (v1 설정).

이동 오차 (imatch.replay가 받는 숫자 묶음으로 직접 정의. 다른 실험의 이름을 빌리지 않음)
  normal   구간마다 방향 치우침 N(0, 3°), 걸음마다 방향 N(0, 2°), 걸음 길이 N(1, 5%)   ← 보통 VIO 가정
  large    위의 4배: 12°, 8°, 20%                                                   ← 이동 센서 오차가 큰 경우
실행: normal seed 2026 한 번, large seed 2027·2028·2029·2030 네 번.

판정 (미리 정함). 도착 = 마지막 오차 ≤ 40 m. 20 m, 100 m도 같이 적는다.
  A. normal: r1 vs dr, McNemar(정확 이항) — 지도 매칭이 추측 항법보다 나은가
  B. normal: r4 vs r1, McNemar
  C. (주 판정) large 네 seed 합침: r4 vs r1, McNemar p < 0.05면 "이동 오차가 클 때 R4가 도착을 유의하게 늘린다"
     구간 단위 부호 검정(구간마다 도착한 seed 수 비교)과 비행(01·02·03) 단위 결과도 같이 적는다. 결과와 상관없이 모두 보고.
"""
import argparse
import hashlib
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.matchers import get_matcher  # noqa: E402
from imatch.nav import MatchLoc, NoMatch, Research  # noqa: E402
from imatch.replay import run_segments, summarize  # noqa: E402

ODO_X = {
    'normal': dict(bias_deg=3.0, head_deg=2.0, scale=0.05),
    'large': dict(bias_deg=12.0, head_deg=8.0, scale=0.20),
}
RUNS = [('normal', 2026), ('large', 2027), ('large', 2028), ('large', 2029), ('large', 2030)]
VARIANTS = ['dr', 'r1', 'r4g0n5w20ac4']
THRESH = (20, 40, 100)


def mcnemar(b, c):
    """정확 이항 McNemar 양측 p. b, c = 한쪽만 성공한 수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def make_research(loc, m, gate):
    return Research(loc, gate_m=gate, wide_after=int(m.group(4) or 0), wide_inl=int(m.group(5) or 0),
                    angle_memory=bool(m.group(6)), wide_cool=int(m.group(8) or 0), yaw_correct=bool(m.group(7)))


def run_all(a):
    matcher = get_matcher(a.matcher, a.device)
    angles = list(np.arange(0, 360, 360 / a.angles))
    cache = {}
    segs_all = []
    for cond, seed in RUNS:
        for v in VARIANTS:
            m = re.fullmatch(r'(r0|r1|r([234])g(\d+)(?:n(\d+)(?:w(\d+))?(a)?(y)?(?:c(\d+))?)?|center|dr)', v)
            gate = float(m.group(3) or 0)

            def make_loc(city, v=v):
                if v == 'dr':
                    return NoMatch()
                key = hashlib.md5(city.tobytes()).hexdigest()
                if key not in cache:
                    cache[key] = MatchLoc(matcher, city, angles=angles, stop_inl=a.stop_inl)
                return cache[key]
            make_rec = (lambda loc, m=m, gate=gate: make_research(loc, m, gate)) if m.group(2) in ('3', '4') else None
            t0 = time.time()
            steps, segs = run_segments(make_loc, a.data, odo=ODO_X[cond], seed=seed, reject_inl=15, gate_m=gate,
                                       make_recover=make_rec)
            name = f'xian_{cond}_s{seed}-{v}'
            steps.to_csv(os.path.join(a.out, f'{name}_steps.csv'), index=False)
            segs.to_csv(os.path.join(a.out, f'{name}_segs.csv'), index=False)
            s = summarize(steps, segs)
            mv = steps[steps.used != 'start']
            sec = float(mv.sec.fillna(0).mean()) if 'sec' in mv and mv.sec.notna().any() else 0.0
            print(f"| {cond} | {seed} | {v} | {s['segments']} | {s['final_med_m']} m | {s['final_le40']}% "
                  f"| {s['unknown']}% | {s['outside_block']}% | {sec:.2f} s | {time.time() - t0:.0f} s |", flush=True)
            segs = segs.assign(cond=cond, seed=seed, variant=v)
            segs_all.append(segs)
    S = pd.concat(segs_all, ignore_index=True)
    S.to_csv(os.path.join(a.out, 'xian_all_segs.csv'), index=False)
    return S


def judge(S, out):
    lines = ['# XIAN-VisLoc R1 vs R1 + R4 판정', '']
    S = S.copy()
    S['flight'] = S.set.str.split('_').str[1]
    for t in THRESH:
        S[f'ok{t}'] = S.final_err_m <= t

    def pair(df, va, vb, t):
        A = df[df.variant == va].set_index(['seed', 'set'])[f'ok{t}']
        B = df[df.variant == vb].set_index(['seed', 'set'])[f'ok{t}']
        A, B = A.align(B, join='inner')
        b, c = int((A & ~B).sum()), int((~A & B).sum())
        return int(A.sum()), int(B.sum()), b, c, mcnemar(b, c), len(A)

    lines += ['## 표 (마지막 오차 기준 도착 수)', '',
              '| 조건 | seed | variant | 구간 | 20 m | 40 m | 100 m | 마지막 오차 중앙값 |', '| --- ' * 8 + '|']
    for (cond, seed, v), g in S.groupby(['cond', 'seed', 'variant'], sort=False):
        lines.append(f'| {cond} | {seed} | {v} | {len(g)} | {g.ok20.sum()} | {g.ok40.sum()} | {g.ok100.sum()} '
                     f'| {g.final_err_m.median():.1f} m |')
    N = S[S.cond == 'normal']; L = S[S.cond == 'large']
    lines += ['', '## 판정 (McNemar 정확 이항, "한쪽만 성공" 수)', '']
    for title, df, va, vb in [('A. normal: r1 vs dr', N, 'r1', 'dr'),
                              ('B. normal: r4 vs r1', N, 'r4g0n5w20ac4', 'r1'),
                              ('C. (주) large 4 seed 합침: r4 vs r1', L, 'r4g0n5w20ac4', 'r1')]:
        lines.append(f'**{title}**')
        for t in THRESH:
            na, nb, b, c, p, n = pair(df, va, vb, t)
            lines.append(f'- {t} m: {va} {na}/{n}, {vb} {nb}/{n}, 한쪽만 {b} vs {c}, p = {p:.4g}')
        lines.append('')
    # 주 판정 문장
    na, nb, b, c, p, n = pair(L, 'r4g0n5w20ac4', 'r1', 40)
    verdict = ('통과: 이동 오차가 클 때 R4가 도착을 유의하게 늘린다 (XIAN에서도 되풀이됨)' if p < 0.05 and b > c
               else '기준 미달: XIAN에서는 이동 오차가 클 때도 R4의 유의한 도착 증가가 확인되지 않음')
    lines += [f'**주 판정 C (40 m): {b} vs {c}, p = {p:.4g} → {verdict}**', '']
    # 구간 단위 부호 검정
    R4 = L[L.variant == 'r4g0n5w20ac4'].groupby('set').ok40.sum()
    R1 = L[L.variant == 'r1'].groupby('set').ok40.sum()
    R4, R1 = R4.align(R1, join='inner')
    better, worse = int((R4 > R1).sum()), int((R4 < R1).sum())
    lines += [f'- 구간 단위 부호 검정 (large, 40 m, 구간마다 도착 seed 수): R4가 나은 구간 {better}, 나쁜 구간 {worse}, '
              f'같은 구간 {int((R4 == R1).sum())}, p = {mcnemar(better, worse):.4g}']
    # seed별, 비행별
    for seed, g in L.groupby('seed'):
        _, _, b2, c2, p2, _ = pair(g, 'r4g0n5w20ac4', 'r1', 40)
        lines.append(f'- seed {seed}: 한쪽만 {b2} vs {c2}')
    for fl, g in L.groupby('flight'):
        a1 = g[g.variant == 'r1']; a4 = g[g.variant == 'r4g0n5w20ac4']
        lines.append(f'- 비행 {fl} (large): 40 m 도착 r1 {a1.ok40.sum()}/{len(a1)}, r4 {a4.ok40.sum()}/{len(a4)}, '
                     f'마지막 오차 중앙값 r1 {a1.final_err_m.median():.1f} m, r4 {a4.final_err_m.median():.1f} m')
    lines += ['', '한계: 구간들은 같은 비행에서 출발점만 민 것이라 서로 독립이 아니다(비행 3개, 한 도시). '
              'large seed들은 같은 구간을 잡음만 바꿔 반복한 것이다. 정답 GPS와 사진 가운데가 수십 m 어긋날 수 있다(가공 README).']
    text = '\n'.join(lines)
    open(os.path.join(out, 'xian_judgement.md'), 'w', encoding='utf-8').write(text)
    print(text, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/xian/seq_test'); ap.add_argument('--out', required=True)
    ap.add_argument('--matcher', default='loma-cache'); ap.add_argument('--angles', type=int, default=4)
    ap.add_argument('--stop-inl', type=int, default=15); ap.add_argument('--device', default='cuda')
    ap.add_argument('--judge-only', action='store_true', help='xian_all_segs.csv로 판정만 다시')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.judge_only:
        S = pd.read_csv(os.path.join(a.out, 'xian_all_segs.csv'))
    else:
        print(f'XIAN 재생 항법: {a.data}, 매처 {a.matcher} {a.angles}방향, 조기 종료 {a.stop_inl}', flush=True)
        print('| 조건 | seed | variant | 구간 | 마지막 오차 중앙값 | 마지막 40 m 이내 | 모름 | 블록 밖 | 걸음당 | 경과 |')
        print('| --- ' * 10 + '|')
        S = run_all(a)
    judge(S, a.out)


if __name__ == '__main__':
    main()
