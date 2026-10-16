"""다시 돌리지 않고 방향 수를 줄인 결과를 만든다: run.py의 *_angles.csv에서 일부 각도만 남기고
inlier가 가장 많은 각도를 다시 고른다. 결과는 run.py 출력과 같은 형식이라 evaluate.py에 바로 넣을 수 있다.

  python scripts/subset_angles.py results/loma_8dir_per_angle_k100-999.csv --angles 0 --out outputs/loma_1dir.csv
  python scripts/subset_angles.py results/loma_8dir_per_angle_k100-999.csv --angles 0,90,180,270 --out outputs/loma_4dir.csv
  python scripts/evaluate.py outputs/loma_1dir.csv --kmin 100
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.geometry import M_PER_UNIT, wrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('angles_csv'); ap.add_argument('--angles', required=True, help='쉼표로: 0 또는 0,180 ...')
    ap.add_argument('--samples', default=os.path.join(ROOT, 'data', 'im_t1_3d_1000', 'samples.csv'),
                    help='정답 위치·heading (gt) 파일')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    keep = [float(x) for x in a.angles.split(',')]
    A = pd.read_csv(a.angles_csv)
    A = A[np.isclose(A.angle.to_numpy()[:, None], keep).any(1)]
    S = pd.read_csv(a.samples).set_index(['set', 'k'])
    rows = []
    for (st, k), g in A.groupby(['set', 'k']):
        gt = S.loc[(st, k)]
        row = dict(matcher=g.matcher.iloc[0], scale=g.scale.iloc[0], set=st, k=k, gt_x=gt.x_norm, gt_y=gt.y_norm,
                   gt_theta=gt.theta, sec=g.sec.sum())
        ok = g.dropna(subset=['px'])
        if len(ok):
            b = ok.loc[ok.n_inl.idxmax()]
            row.update(pred_x=b.px, pred_y=b.py, pred_theta=b['head'], angle=b.angle, n_inl=b.n_inl, n_match=b.n_match)
        rows.append(row)
    R = pd.DataFrame(rows)
    R['err_m'] = np.hypot(R.pred_x.fillna(0) - R.gt_x, R.pred_y.fillna(0) - R.gt_y) * M_PER_UNIT
    R['head_err'] = wrap(R.pred_theta.fillna(0) - R.gt_theta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    R.to_csv(a.out, index=False)
    print(f'{len(R)} samples, angles {keep} -> {a.out}')


if __name__ == '__main__':
    main()
