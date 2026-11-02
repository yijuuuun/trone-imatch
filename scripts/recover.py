"""복구 3단계: LoMa 답이 미덥지 않으면(inlier < 15, C3 규칙) 다른 매처(RoMa v2)의 답으로 바꾼다.

  python scripts/recover.py --kmin 100

규칙은 결과를 보기 전에 정했다. RoMa v2는 dense 매처라 inlier가 항상 수백 개 이상이어서
같은 기준(< 15)으로 다시 거를 수 없다. 그래서 바꾼 답은 그대로 받아들인다.
시간: LoMa 8방향은 모든 샘플, RoMa v2 8방향은 바꾼 샘플에만 든다고 보고 샘플당 평균을 계산한다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.metrics import errors, mcnemar  # noqa: E402

RES = os.path.join(ROOT, 'results')
CITY = {'3d_ho37bc': 'Singapore', '3d_ho38bc': 'SF'}
FEW = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmin', type=int, default=100)
    a = ap.parse_args()
    lo = pd.read_csv(os.path.join(RES, 'loma_8dir.csv')).set_index(['set', 'k'])
    ro = pd.read_csv(os.path.join(RES, 'romav2_8dir.csv')).set_index(['set', 'k']).reindex(lo.index)
    lab = pd.read_csv(os.path.join(RES, 'label_check_t1_3d.csv')).set_index(['set', 'k']).label_suspect.reindex(lo.index)
    keep = lo.index.get_level_values('k') >= a.kmin
    lo, ro, lab = lo[keep], ro[keep], lab[keep]

    e_lo, _ = errors(lo.pred_x, lo.pred_y, lo.gt_x, lo.gt_y)
    e_ro, _ = errors(ro.pred_x, ro.pred_y, ro.gt_x, ro.gt_y)
    swap = (lo.n_inl.fillna(0) < FEW).to_numpy()
    e_rec = np.where(swap, e_ro, e_lo)

    print(f'## 복구 3단계: LoMa inlier < {FEW}이면 RoMa v2 답으로 (k >= {a.kmin})\n')
    print('| 도시 | n | 바꾼 비율 | LoMa만 | RoMa v2만 | **LoMa + 복구** | 복구 vs LoMa (나만 / 걔만) | 큰 오류(> 32 m) LoMa → 복구 '
          '| 바꾼 샘플: LoMa → RoMa v2 10 m 이내 | 샘플당 시간 |')
    print('| --- ' * 10 + '|')
    for st, city in CITY.items():
        m = (lo.index.get_level_values('set') == st)
        for name, mm in [(city, m), (f'{city} (라벨 의심 제외)', m & ~lab.fillna(False).to_numpy())]:
            ok_lo, ok_ro, ok_rec = e_lo[mm] < 10, e_ro[mm] < 10, e_rec[mm] < 10
            s = swap[mm]
            t = lo.sec[mm].mean() + (ro.sec[mm] * s).mean()
            print(f'| {name} | {mm.sum()} | {s.mean()*100:.1f}% | {ok_lo.mean()*100:.1f}% | {ok_ro.mean()*100:.1f}% '
                  f'| **{ok_rec.mean()*100:.1f}%** | {mcnemar(ok_rec, ok_lo)} '
                  f'| {(e_lo[mm] > 32).sum()} → {(e_rec[mm] > 32).sum()} '
                  f'| {ok_lo[s].mean()*100:.0f}% → {ok_ro[s].mean()*100:.0f}% ({s.sum()}개) | {t:.1f} s |')


if __name__ == '__main__':
    main()
