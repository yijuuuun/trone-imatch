"""사진 한 장: "모름"(inlier < 15)이면 블록 가운데를 답하는 규칙(R1의 한 장 버전)을 기존 기록으로 채점한다. GPU 불필요.

  python experiments/2026-11-17_single_gate/single_gate.py

읽는 기록: 4방향 + 멈추기 15의 loma_s1.csv (개발용 #40, 시험용 #37), Bearing-UAV 예측(#33~35, 시험용만).
모두 팀 공통 규칙 imatch.metrics.errors(추정을 블록 가운데 영역 ±1로 clip, 추정 없으면 블록 가운데)로 채점한다.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from imatch.metrics import errors  # noqa: E402

TH = 15  # v1의 "모름" 기준. 이 실험 전에 정해져 있던 값
CALIB = 'experiments/2026-11-16_heading_dirs_visloc_calib/visloc-calib-4dir-stop/loma_s1.csv'
EVAL = 'experiments/2026-11-15_heading_dirs_visloc/visloc-4dir-stop/loma_s1.csv'
B1 = 'results/kaggle_2026-11-14/run_20261114T034717155134Z/Bearing-UAV/full/predictions.csv'


def pct(e):
    return ' / '.join(f'{100 * float((e <= t).mean()):.1f}%' for t in (10, 20, 32))


def pair(a, b):
    out = []
    for t in (10, 20, 32):
        x, y = int(((a <= t) & (b > t)).sum()), int(((b <= t) & (a > t)).sum())
        out.append(f'{x} vs {y}, p = {binomtest(x, x + y).pvalue:.2g}')
    return ' | '.join(out)


rows = []
for name, path in (('개발용 single_calib', CALIB), ('시험용 single_eval', EVAL)):
    R = pd.read_csv(os.path.join(ROOT, path))
    z = np.zeros(len(R))
    lo, _ = errors(R.pred_x, R.pred_y, R.gt_x, R.gt_y)
    cen, _ = errors(z, z, R.gt_x, R.gt_y)
    ok = ((R.n_inl.fillna(0) >= TH) & R.pred_x.notna()).to_numpy()
    gate = np.where(ok, lo, cen)
    m = {'블록 가운데': cen, 'LoMa 항상 답함': lo, f'LoMa + 모름이면 가운데 (inlier < {TH})': gate}
    if name.startswith('시험용'):
        B = pd.read_csv(os.path.join(ROOT, B1)); B = B[B.repeat == 0].copy()
        B['set'] = B.sample_id.str.split(':').str[0]; B['k'] = B.sample_id.str.split(':').str[1].astype(int)
        M = R.merge(B[['set', 'k', 'pred_x', 'pred_y']], on=['set', 'k'], suffixes=('', '_b'))
        assert len(M) == len(R)
        m['Bearing-UAV'], _ = errors(M.pred_x_b, M.pred_y_b, M.gt_x, M.gt_y)
    print(f'\n## {name} (n = {len(R)}, 답한 비율 {100 * ok.mean():.1f}%)\n')
    print('| 방법 | 10 m / 20 m / 32 m |\n| --- | --- |')
    for k, e in m.items():
        print(f'| {k} | {pct(e)} |'); rows.append(dict(data=name, method=k, n=len(R), w10=(e <= 10).mean(), w20=(e <= 20).mean(), w32=(e <= 32).mean()))
    print(f'| 답한 사진만: LoMa | {pct(lo[ok])} |\n| 답한 사진만: 블록 가운데 | {pct(cen[ok])} |')
    print('\n짝 비교 (한쪽만 성공, 10 m | 20 m | 32 m)\n')
    print('- 모름이면 가운데 vs 블록 가운데:', pair(gate, cen))
    print('- 모름이면 가운데 vs LoMa 항상 답함:', pair(gate, lo))
    if 'Bearing-UAV' in m:
        print('- 모름이면 가운데 vs Bearing-UAV:', pair(gate, m['Bearing-UAV']))
        print('- LoMa 항상 답함 vs Bearing-UAV:', pair(lo, m['Bearing-UAV']))
        print('- 블록 가운데 vs Bearing-UAV:', pair(cen, m['Bearing-UAV']))
pd.DataFrame(rows).round(4).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_gate.csv'), index=False)
