"""정답 라벨 점검: 드론 영상을 정답 위치·heading으로 지도에 겹쳤을 때와 추정 위치로 겹쳤을 때의 밝기 상관(NCC)을 비교한다.

추정 쪽이 훨씬 잘 맞고(NCC >= 0.5) 정답 쪽은 거의 안 맞으면(< 0.2) 라벨이 틀렸을 가능성이 높다.
매처 결과가 맞는지와 무관하게 영상만으로 확인하는 방법이다. 축척은 1로 둔다.

  python scripts/check_labels.py results/loma_8dir.csv --data data/im_t1_3d_1000 --out results/label_check.csv
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.geometry import BLOCK_CENTER, PX_PER_UNIT, rot  # noqa: E402

PRED_MIN, GT_MAX = 0.5, 0.2


def ncc(uav, block, x, y, th):
    """정답 변환 block_px = C + R(th) (uav_px - 128)으로 지도를 드론 영상 좌표로 펴서 NCC."""
    C = np.array([BLOCK_CENTER + x * PX_PER_UNIT, BLOCK_CENTER + y * PX_PER_UNIT]); R = rot(th)
    h, w = uav.shape[:2]
    A = np.hstack([R, (C - R @ np.array([w / 2, h / 2]))[:, None]])
    Ai = cv2.invertAffineTransform(A)
    W = cv2.warpAffine(block, Ai, (w, h))
    m = cv2.warpAffine(np.ones(block.shape[:2], np.uint8), Ai, (w, h)) > 0
    if m.mean() < 0.3:
        return np.nan
    u = cv2.cvtColor(uav, cv2.COLOR_RGB2GRAY)[m].astype(float); v = cv2.cvtColor(W, cv2.COLOR_RGB2GRAY)[m].astype(float)
    u -= u.mean(); v -= v.mean()
    return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='run.py 출력 (pred_x, pred_y, pred_theta, gt_*)')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv).drop_duplicates(['set', 'k'])
    rows = []
    for r in d.itertuples():
        D = os.path.join(a.data, r.set)
        rd = lambda f: cv2.cvtColor(cv2.imread(os.path.join(D, f)), cv2.COLOR_BGR2RGB)
        U, B = rd(f'k{r.k:03d}_uav.jpg'), rd(f'k{r.k:03d}_block.jpg')
        s_pred = ncc(U, B, r.pred_x, r.pred_y, r.pred_theta) if np.isfinite(r.pred_x) else np.nan
        rows.append(dict(set=r.set, k=r.k, n_inl=r.n_inl, ncc_gt=ncc(U, B, r.gt_x, r.gt_y, r.gt_theta), ncc_pred=s_pred))
    R = pd.DataFrame(rows)
    R['label_suspect'] = (R.ncc_pred >= PRED_MIN) & (R.ncc_gt < GT_MAX)
    R.round(3).to_csv(a.out, index=False)
    print(R.groupby('set').agg(n=('k', 'size'), ncc_gt=('ncc_gt', 'median'), ncc_pred=('ncc_pred', 'median'),
                               label_suspect=('label_suspect', 'sum')).round(2).to_string())
    print('→', a.out)


if __name__ == '__main__':
    main()
