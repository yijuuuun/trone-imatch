"""C3 감지기 데이터: 각도별 기록(run.py의 *_angles.csv)을 후보 1개 = 1행의 특징 표로 바꾼다.

  python scripts/c3_features.py outputs/c3-train-tk/out/loma-16dir_s1_angles.csv ... \
      --samples data/im_train_3d_500/samples.csv --labels results/label_check_train.csv --out results/c3_train.csv

특징 (정답을 쓰지 않는 것만):
- n_inl, n_match, inl_ratio, scale_est        : 그 각도의 매칭 통계
- inl_rank, inl_rel                          : 같은 샘플의 다른 각도들과 비교한 순위, 1등 대비 비율
- agree_pos, agree_head                      : 다른 각도 추정 중 위치가 10 m 안 / heading이 20° 안인 비율 (각도 간 일치)
- in_block                                   : 추정 위치가 블록 안인가
good_match, good_inl은 정답으로 채점한 값이라 특징에 넣지 않는다.

라벨: ok = 그 후보의 위치 오차 < 10 m (imatch/metrics.py 규칙). --labels를 주면 label_suspect 샘플은 뺀다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.geometry import M_PER_UNIT  # noqa: E402
from imatch.metrics import errors  # noqa: E402

FEATURES = ['n_inl', 'n_match', 'inl_ratio', 'scale_est', 'inl_rank', 'inl_rel', 'agree_pos', 'agree_head', 'in_block']


def features(pa):
    pa = pa.copy()
    pa['n_inl'] = pa.n_inl.fillna(0); pa['n_match'] = pa.n_match.fillna(0)
    pa['inl_ratio'] = pa.n_inl / pa.n_match.clip(lower=1)
    g = pa.groupby(['set', 'k'])
    pa['inl_rank'] = g.n_inl.rank(ascending=False, method='first')
    pa['inl_rel'] = pa.n_inl / g.n_inl.transform('max').clip(lower=1)
    pa['in_block'] = ((pa.px.abs() <= 1.1) & (pa.py.abs() <= 1.1)).astype(float)
    pa['agree_pos'] = np.nan; pa['agree_head'] = np.nan
    for _, idx in g.groups.items():
        s = pa.loc[idx]
        P = s[['px', 'py']].to_numpy(float); H = s['head'].to_numpy(float)
        dpos = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1)) * M_PER_UNIT
        dh = np.abs((H[:, None] - H[None, :] + 180) % 360 - 180)
        valid = np.isfinite(P).all(1)
        n_other = max(valid.sum() - 1, 1)
        pa.loc[idx, 'agree_pos'] = (((dpos < 10) & valid[None, :]).sum(1) - 1) / n_other
        pa.loc[idx, 'agree_head'] = (((dh < 20) & valid[None, :]).sum(1) - 1) / n_other
    pa.loc[~np.isfinite(pa.px), ['agree_pos', 'agree_head', 'in_block']] = 0
    return pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('angles', nargs='+')
    ap.add_argument('--samples', required=True, help='정답이 든 samples.csv (set, k, x_norm, y_norm, theta)')
    ap.add_argument('--labels', default='', help='check_labels.py 출력. label_suspect 샘플을 뺀다')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pa = pd.concat([pd.read_csv(p) for p in a.angles], ignore_index=True).drop_duplicates(['set', 'k', 'angle'])
    gt = pd.read_csv(a.samples)[['set', 'k', 'x_norm', 'y_norm', 'theta']]
    pa = pa.merge(gt, on=['set', 'k'])
    if a.labels:
        L = pd.read_csv(a.labels)
        bad = L[L.label_suspect][['set', 'k']].assign(_bad=True)
        pa = pa.merge(bad, on=['set', 'k'], how='left')
        print('라벨 의심으로 뺀 샘플', int(bad.shape[0])); pa = pa[pa._bad.isna()].drop(columns='_bad')
    pa = features(pa)
    pa['err_m'], _ = errors(pa.px, pa.py, pa.x_norm, pa.y_norm)
    pa['ok'] = pa.err_m < 10
    cols = ['set', 'k', 'angle'] + FEATURES + ['px', 'py', 'head', 'err_m', 'ok']
    pa[cols].to_csv(a.out, index=False)
    s = pa.groupby('set').agg(samples=('k', 'nunique'), candidates=('k', 'size'), ok=('ok', 'mean'))
    s['any_ok'] = pa.groupby(['set', 'k']).ok.any().groupby('set').mean()
    print(s.round(3).to_string()); print('→', a.out)


if __name__ == '__main__':
    main()
