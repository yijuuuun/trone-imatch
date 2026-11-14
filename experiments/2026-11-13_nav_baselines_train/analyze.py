"""B1 학습 도시 항법을 v1과 같은 궤적끼리 비교하고, 걸음을 B1의 학습 영역(block_x <= 11)과 나머지로 나눈다.

  python experiments/2026-11-13_nav_baselines_train/analyze.py      # 저장소 루트에서
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from imatch.metrics import mcnemar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(ROOT, 'experiments', '2026-11-13_nav_recovery')


def load(p):
    L = pd.read_csv(p + '_steps.csv')
    T = pd.read_csv(p + '_traj.csv').set_index('traj').success.reindex(range(32), fill_value=0).astype(bool).to_numpy()
    return L, T


for city in ['Tokyo', 'Taipei']:
    runs = {'v1 (R1 + 멈추기)': f'{REC}/es/{city}_3d_loma-vio-es-r1', 'LoMa R0 (대처 없음)': f'{REC}/{city}_3d_loma-vio-r0',
            'B1 (SF 빠짐)': f'{HERE}/{city}_3d_b1-ho38-r0', 'B1 (Singapore 빠짐)': f'{HERE}/{city}_3d_b1-ho37-r0'}
    D = {k: load(v) for k, v in runs.items()}
    v1 = D['v1 (R1 + 멈추기)'][1]
    print(f'\n## {city}\n')
    print('| 방법 | 성공 | vs v1 (나만 / v1만) | 걸음 | 걸음 10 m 이내 | 걸음 32 m 넘음 | 블록 밖 걸음 | 걸음당 s |')
    print('| --- ' * 8 + '|')
    for k, (L, T) in D.items():
        print(f'| {k} | {T.mean()*100:.1f}% ({T.sum()}) | {mcnemar(T, v1) if T is not v1 else "—"} | {len(L)} '
              f'| {(L.err_m < 10).mean()*100:.1f}% | {(L.err_m > 32).mean()*100:.1f}% | {L.outside_block.mean()*100:.1f}% '
              f'| {L.sec.mean():.3f} |')
    print('\n| 방법 | 드론 위치 | 걸음 | 걸음 10 m 이내 | 걸음 32 m 넘음 | 맞는 블록을 고른 걸음의 32 m 넘음 |')
    print('| --- ' * 6 + '|')
    for k, (L, T) in D.items():
        bx = np.round((L.real_x - 256) / 256).clip(0, 14); by = np.round((L.real_y - 256) / 256).clip(0, 14)
        right = (L.block_x == bx) & (L.block_y == by)
        for name, msk in [('B1 학습 영역 (block_x ≤ 11)', bx <= 11), ('그 밖 (12–14)', bx >= 12)]:
            S, Sr = L[msk], L[msk & right]
            print(f'| {k} | {name} | {len(S)} | {(S.err_m < 10).mean()*100:.1f}% | {(S.err_m > 32).mean()*100:.1f}% '
                  f'| {(Sr.err_m > 32).mean()*100:.1f}% ({len(Sr)}걸음) |')
