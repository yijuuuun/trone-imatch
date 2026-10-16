"""결과 비교: scripts/run.py 출력 CSV를 기준선(B1 원 모델, B2, CorrMatchNet)과 LoMa 8방향 결과에
같은 샘플끼리 짝지어 비교한다. 규칙은 imatch/metrics.py.

  python scripts/evaluate.py outputs/loma_1dir/loma_s1.csv
  python scripts/evaluate.py a.csv b.csv --kmin 100        # 여러 개, k >= 100만

표의 '나만 / 걔만'은 10 m 이내 성공이 내 결과에만 / 상대에만 있는 샘플 수, p는 McNemar 정확 검정.
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
CITY = {'3d_ho38bc': 'SF', '3d_ho37bc': 'Singapore'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--kmin', type=int, default=0); ap.add_argument('--n', type=int, default=1000)
    a = ap.parse_args()
    base = pd.read_csv(os.path.join(RES, 'baselines_3d_k0-999.csv'))
    ref = pd.read_csv(os.path.join(RES, 'loma_8dir.csv'))
    for f in a.csv:
        R = pd.read_csv(f)
        print(f'\n## {f}\n')
        print('| 도시 | n | 10 m 이내 | 평균 오차 | 중앙값 | 실패 | heading 중앙값 | 1장당 s '
              '| vs B1 원모델 (나만 / 걔만) | vs B2 | vs CorrMatchNet | vs LoMa 8방향 |')
        print('| --- ' * 12 + '|')
        for st, city in CITY.items():
            r = R[(R.set == st) & (R.k >= a.kmin) & (R.k < a.n)].drop_duplicates('k').set_index('k').sort_index()
            if r.empty:
                continue
            e, fail = errors(r.pred_x, r.pred_y, r.gt_x, r.gt_y); ok = e < 10
            cols = []
            for m in ['B1_원모델', 'B2', 'CorrMatchNet']:
                b = base[(base.method == m) & (base.set == st)].set_index('k').err_m.reindex(r.index)
                cols.append(mcnemar(ok, b.to_numpy() < 10))
            lr = ref[ref.set == st].set_index('k').reindex(r.index)
            cols.append(mcnemar(ok, errors(lr.pred_x, lr.pred_y, lr.gt_x, lr.gt_y)[0] < 10))
            he = r.head_err.median() if 'head_err' in r else np.nan
            print(f"| {city} | {len(r)} | {ok.mean()*100:.1f}% | {e.mean():.2f} m | {np.median(e):.2f} m | {fail.mean()*100:.0f}% "
                  f"| {he:.1f}° | {r.sec.mean():.2f} | " + ' | '.join(cols) + ' |')


if __name__ == '__main__':
    main()
