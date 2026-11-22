"""vio2 확인 실험 판정: 새 seed(2027~2029)의 구간별 40 m 이내 걸음 비율 차이를 구간마다 평균 내 Wilcoxon (README 2절 3번).

  python experiments/2026-11-22_visloc_nav_full_vio2_seeds/analyze_seeds.py

입력: run/s<seed>/visloc_loma-cache-{r1,r4g0n5w20ac4}_steps.csv, seed 2026은 ../2026-11-22_visloc_nav_full_vio2/run/test.
seed별 상세 표는 ../2026-11-20_visloc_nav_full/analyze.py --run run/s<seed>로 낸다.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('an', os.path.join(EXP, '2026-11-20_visloc_nav_full', 'analyze.py'))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

NEW = [2027, 2028, 2029]
RUNS = {s: os.path.join(HERE, 'run', f's{s}') for s in NEW}
RUNS[2026] = os.path.join(EXP, '2026-11-22_visloc_nav_full_vio2', 'run', 'test')


def seg_table(run, v):
    s, _ = an.per_segment(pd.read_csv(os.path.join(run, f'visloc_loma-cache-{v}_steps.csv')))
    return s.set_index('set')


def test(d):
    nz = d[d != 0]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    return f'높음 {int((d > 0).sum())}, 낮음 {int((d < 0).sum())}, 같음 {int((d == 0).sum())}, ' \
           f'차이 중앙값(같음 뺌) {nz.median() * 100:+.1f}%p, Wilcoxon 양측 p = {p:.2g}'


def main():
    rows, diffs = [], {}
    for s in sorted(RUNS):
        if not all(os.path.exists(os.path.join(RUNS[s], f'visloc_loma-cache-{v}_steps.csv')) for v in ('r1', 'r4g0n5w20ac4')):
            print(f'seed {s}: 결과 없음'); continue
        r1, r4 = seg_table(RUNS[s], 'r1'), seg_table(RUNS[s], 'r4g0n5w20ac4')
        diffs[s] = r4.within40 - r1.within40
        for v, t in (('R1', r1), ('R1 + R4', r4)):
            w = t.steps / t.steps.sum()
            rows.append(dict(seed=s, 조건=v, within40=(t.within40 * w).sum(), within20=(t.within20 * w).sum(),
                             못돌아옴=int(np.isfinite(t.final_lost_km).sum()), sec=(t.sec_mean * w).sum()))
    print('### seed별 전체 (모든 걸음)\n')
    print('| seed | 조건 | 40 m 이내 | 20 m 이내 | 끝까지 못 돌아온 구간 | 걸음당 s |')
    print('| --- | --- | --- | --- | --- | --- |')
    for r in rows:
        print(f"| {r['seed']}{' (가설)' if r['seed'] == 2026 else ''} | {r['조건']} | {r['within40'] * 100:.1f}% "
              f"| {r['within20'] * 100:.1f}% | {r['못돌아옴']} / 11 | {r['sec']:.2f} |")
    print('\n### seed별 구간 Wilcoxon (참고)\n')
    for s, d in diffs.items():
        print(f'- seed {s}: {test(d)}')
    new = [s for s in NEW if s in diffs]
    if new:
        D = pd.concat([diffs[s] for s in new], axis=1)
        print(f'\n### 주 판정: 새 seed {new}, 구간마다 차이 평균 (11개)\n')
        print('| 구간 | ' + ' | '.join(str(s) for s in new) + ' | 평균 |')
        print('| --- ' * (len(new) + 2) + '|')
        for st, r in D.iterrows():
            print(f'| {st} | ' + ' | '.join(f'{x * 100:+.1f}%p' for x in r) + f' | {r.mean() * 100:+.1f}%p |')
        print(f'\n**{test(D.mean(axis=1))}**' + ('' if len(new) == len(NEW) else f' (seed {len(new)}개뿐, 판정 아님)'))
        print(f'\n참고: seed × 구간 {D.size}개 Wilcoxon: {test(pd.Series(D.to_numpy().ravel()))}')
        if 2026 in diffs:
            A = pd.concat([diffs[s] for s in [2026] + new], axis=1).mean(axis=1)
            print(f'참고: 2026 포함 {len(new) + 1}개 seed 구간 평균: {test(A)}')


if __name__ == '__main__':
    main()
