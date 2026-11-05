"""⑥ 블록 선택: run.py --block-radius 1의 _angles.csv에서 후보 블록을 고른다. 다시 돌리지 않고 규칙을 바꿔 볼 수 있다.

  python scripts/select_block.py outputs/blk3/loma_s1_angles.csv --kmin 100 --out outputs/blk3_select

규칙 (--rules)
  inlier   inlier가 가장 많은 후보
  inside   추정이 자기 블록 밖으로 0.1단위 넘게 나간 후보를 버리고, 남은 것 중 inlier가 가장 많은 후보
후보 수 (--blocks): 9 = 3x3, 5 = 상하좌우 + 가운데, 1 = 가운데(정답 블록)만

출력: 표(stdout)와 --out 폴더의 <규칙>_b<후보 수>.csv. pred_x, pred_y는 고른 블록 기준 좌표이고 dx, dy가 고른 블록이다.
이웃 블록 결과는 clip 규칙이 달라서 evaluate.py에 넣으면 안 된다 (imatch/metrics.py errors_chosen_block).
결과를 보고 정한 규칙은 --kmin 100으로만 평가한다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.geometry import wrap  # noqa: E402
from imatch.metrics import errors_chosen_block, mcnemar  # noqa: E402

CITY = {'3d_ho38bc': 'SF', '3d_ho37bc': 'Singapore'}
KEEP = {9: lambda c: np.ones(len(c), bool), 5: lambda c: c.dx.abs() + c.dy.abs() <= 1, 1: lambda c: (c.dx == 0) & (c.dy == 0)}
OUTSIDE = 1.1  # imatch/metrics.py의 실패 기준


def select(cands, samples, rule, blocks):
    """샘플당 1행: 고른 후보. 추정이 하나도 없으면 pred_x 등이 비어 있다."""
    c = cands[KEEP[blocks](cands)]
    sec = c.groupby(['set', 'k']).agg(sec=('sec', 'sum'), n_blocks=('sec', 'size'))
    ok = c.dropna(subset=['px'])
    if rule == 'inside':
        ok = ok[(ok.px.abs() <= OUTSIDE) & (ok.py.abs() <= OUTSIDE)]
    best = ok.loc[ok.groupby(['set', 'k']).n_inl.idxmax()].set_index(['set', 'k'])
    best = best[['dx', 'dy', 'px', 'py', 'head', 'angle', 'n_inl', 'n_match', 'scale_est']]
    R = samples.join(sec, how='inner').join(best.rename(columns=dict(px='pred_x', py='pred_y', head='pred_theta')))
    R['err_m'], R['fail'] = errors_chosen_block(R.pred_x, R.pred_y, R.dx, R.dy, R.gt_x, R.gt_y)
    R['head_err'] = wrap(R.pred_theta.fillna(0) - R.gt_theta)
    return R.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('angles_csv', nargs='+')
    ap.add_argument('--rules', default='inlier,inside'); ap.add_argument('--blocks', default='1,5,9')
    ap.add_argument('--kmin', type=int, default=0); ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--samples', default=os.path.join(ROOT, 'data', 'im_t1_3d_1000', 'samples.csv'))
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    A = pd.concat([pd.read_csv(f) for f in a.angles_csv], ignore_index=True)
    A = A[(A.k >= a.kmin) & (A.k < a.n)]
    if 'px' not in A:
        sys.exit('추정이 하나도 없다. 매처가 짝을 못 찾았는지 summary.csv의 found를 확인할 것')
    # 후보마다 inlier가 가장 많은 각도를 먼저 고른다. sec는 그 후보의 모든 각도 합
    sec = A.groupby(['set', 'k', 'dx', 'dy']).sec.sum()
    ok = A.dropna(subset=['px'])
    C = ok.loc[ok.groupby(['set', 'k', 'dx', 'dy']).n_inl.idxmax()].drop(columns='sec').set_index(['set', 'k', 'dx', 'dy'])
    C = C.reindex(sec.index).assign(sec=sec).reset_index()
    S = pd.read_csv(a.samples).rename(columns=dict(x_norm='gt_x', y_norm='gt_y', theta='gt_theta'))
    S = S.set_index(['set', 'k'])[['gt_x', 'gt_y', 'gt_theta']]
    if a.out: os.makedirs(a.out, exist_ok=True)
    print('| 도시 | 규칙 | 후보 | n | 정답 블록 고름 | 10 m 이내 | 평균 오차 | 실패 | 블록당 s | 1장당 s | vs 정답 블록 (나만 / 걔만) |')
    print('| --- ' * 11 + '|')
    ref = select(C, S, 'inlier', 1).set_index(['set', 'k'])
    for st, city in CITY.items():
        if st not in C.set.values: continue
        ok_ref = ref.loc[st].err_m.sort_index().to_numpy() < 10
        for rule in a.rules.split(','):
            for blocks in [int(b) for b in a.blocks.split(',')]:
                R = select(C, S, rule, blocks)
                if a.out and st == next(s for s in CITY if s in C.set.values):
                    R.to_csv(os.path.join(a.out, f'{rule}_b{blocks}.csv'), index=False)
                r = R[R.set == st].sort_values('k')
                ok_mine = r.err_m.to_numpy() < 10
                picked = (r.dx == 0) & (r.dy == 0)
                print(f'| {city} | {rule} | {blocks} | {len(r)} | {picked.mean()*100:.1f}% | {ok_mine.mean()*100:.1f}% | {r.err_m.mean():.2f} m '
                      f'| {r.fail.mean()*100:.0f}% | {(r.sec / r.n_blocks).mean():.2f} | {r.sec.mean():.2f} | {mcnemar(ok_mine, ok_ref)} |')


if __name__ == '__main__':
    main()
