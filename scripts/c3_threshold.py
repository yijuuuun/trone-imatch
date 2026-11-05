"""C3 거절 기준(inlier < T이면 모름)의 T를 바꿔 본다. 답 = inlier 1등 후보.

  python scripts/c3_threshold.py

T는 학습 도시(도쿄, 타이베이) 표로 정한다. 시험 도시 표는 참고로만 보여 주고, 이 표를 보고 T를 고치지 않는다.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from c3_train import RES, TEST_CITY, TRAIN_CITY, select, test_candidates  # noqa: E402

TS = [5, 10, 15, 20, 30, 50]


def table(g, title):
    """g: 샘플당 1행 (n_inl, err_m)."""
    wrong, big = (g.err_m >= 10).to_numpy(), (g.err_m > 32).to_numpy()
    print(f'\n### {title} ({len(g)}샘플, 틀린 답 {wrong.sum()}, 큰 오류 {big.sum()})\n')
    print('| T | 거절 비율 | 틀린 답 중 걸러냄 | 큰 오류(> 32 m) 중 걸러냄 | 거른 것 중 실제 틀림 | 버린 맞는 답 | 받아들인 답의 10 m 이내 |')
    print('| --- ' * 7 + '|')
    print(f'| 거절 없음 | 0% | — | — | — | — | {(~wrong).mean()*100:.1f}% |')
    for t in TS:
        rej = (g.n_inl < t).to_numpy()
        pct = lambda a, b: f'{a}/{b} ({a/max(b,1)*100:.0f}%)'
        print(f'| {"**15**" if t == 15 else t} | {rej.mean()*100:.1f}% | {pct((rej & wrong).sum(), wrong.sum())} '
              f'| {pct((rej & big).sum(), big.sum())} | {(rej & wrong).sum()/max(rej.sum(),1)*100:.0f}% '
              f'| {(rej & ~wrong).sum()} | {(~wrong)[~rej].mean()*100:.1f}% |')


def main():
    tr = pd.read_csv(os.path.join(RES, 'c3_train.csv'))
    s = select(tr, tr.n_inl.to_numpy())
    print('## 학습 도시 (T는 여기서 정한다)')
    table(s, '도쿄 + 타이베이')
    for st, city in TRAIN_CITY.items():
        table(s.loc[st], city)
    te = test_candidates()
    s = select(te, te.n_inl.to_numpy())
    print('\n## 시험 도시 (참고만. 이 표로 T를 고치지 않는다)')
    for st, city in TEST_CITY.items():
        table(s.loc[st], city)


if __name__ == '__main__':
    main()
