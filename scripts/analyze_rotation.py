"""① 회전 탐색 분석과 실패 원인 태그.

  python scripts/analyze_rotation.py                                   # 기존 8방향만
  python scripts/analyze_rotation.py --extra outputs/off22/loma-off22_s1_angles.csv   # 22.5° 사이 방향 추가 = 16방향

출력
- 표준 출력: 도시별 표 (inlier 1등 각도, 정답에 가장 가까운 각도를 골랐을 때의 상한, 1·2등 inlier 비)
- results/failure_tags.csv: 10 m 넘은 샘플마다 원인 태그 (2주차 C3 학습 데이터)

각도별 기록은 k = 100~999에만 있으므로 각도 관련 숫자는 그 범위에서 센다. 규칙은 imatch/metrics.py.
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
FEW_INL = 30  # 짝 부족 기준: 성공 샘플 inlier 중앙값 70개, 실패 14개(Singapore) 사이


def wrap(a):
    return np.abs((np.asarray(a) + 180) % 360 - 180)


def per_angle(paths, gt):
    """각도별 기록을 합치고 각도마다 오차를 붙인다."""
    pa = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    pa = pa.drop_duplicates(['set', 'k', 'angle']).merge(gt, on=['set', 'k'])
    pa['n_inl'] = pa.n_inl.fillna(0)
    pa['e'], pa['fail'] = errors(pa.px, pa.py, pa.gt_x, pa.gt_y)
    pa['he'] = wrap(pa['head'] - pa.gt_theta)
    return pa


def pick(pa):
    """샘플마다 inlier 1등, 2등, 오차가 가장 작은 각도(상한)."""
    r = pa.sort_values(['set', 'k', 'n_inl'], ascending=[True, True, False])
    top = r.groupby(['set', 'k']).head(1).set_index(['set', 'k'])
    sec = r.groupby(['set', 'k']).nth(1).set_index(['set', 'k'])
    orc = pa.sort_values('e').groupby(['set', 'k']).head(1).set_index(['set', 'k'])
    return top, sec.reindex(top.index), orc.reindex(top.index)


def table(pa, label):
    top, sec, orc = pick(pa)
    print(f'\n## {label} (k = 100~999)\n')
    print('| 도시 | n | inlier 1등 10 m 이내 | 상한 (가장 좋은 각도) | 다른 각도면 맞았던 실패 | 어떤 각도도 안 됨 '
          '| 2등/1등 inlier (성공 / 실패) |')
    print('| --- ' * 7 + '|')
    for st, city in CITY.items():
        t, s, o = top.loc[st], sec.loc[st], orc.loc[st]
        ok, ok_o = t.e < 10, o.e < 10
        ratio = s.n_inl / t.n_inl.clip(lower=1)
        print(f'| {city} | {len(t)} | {ok.mean()*100:.1f}% | {ok_o.mean()*100:.1f}% | {int((~ok & ok_o).sum())} '
              f'| {int((~ok_o).sum())} | {ratio[ok].median():.2f} / {ratio[~ok].median():.2f} |')
    return top, orc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extra', nargs='*', default=[], help='추가 방향의 *_angles.csv')
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(RES, 'loma_8dir.csv'))
    gt = d[['set', 'k', 'gt_x', 'gt_y', 'gt_theta']]
    base = [os.path.join(RES, 'loma_8dir_per_angle_k100-999.csv')]
    pa8 = per_angle(base, gt)
    top8, orc8 = table(pa8, '8방향')

    if a.extra:
        pa16 = per_angle(base + a.extra, gt)
        top16, _ = table(pa16, '16방향 (8방향 + 22.5° 사이)')
        print('\n| 도시 | 16방향 vs 8방향 (16만 성공 / 8만 성공) | 8방향 회전 오류 중 16방향으로 고쳐진 수 |')
        print('| --- | --- | --- |')
        for st, city in CITY.items():
            t8, t16 = top8.loc[st], top16.loc[st]
            rot = (t8.e >= 10) & (t8.he > 45)
            print(f'| {city} | {mcnemar((t16.e < 10).to_numpy(), (t8.e < 10).to_numpy())} '
                  f'| {int((rot & (t16.e < 10)).sum())} / {int(rot.sum())} |')

    # 실패 원인 태그: 8방향 최종 결과(k = 0~999) 기준. 우선순위 블록 밖 > 회전 > 짝 부족 > 기타
    e, fail = errors(d.pred_x, d.pred_y, d.gt_x, d.gt_y)
    he = wrap(d.pred_theta.fillna(0) - d.gt_theta)
    n_inl = d.n_inl.fillna(0)
    tag = np.select([fail, he > 45, n_inl < FEW_INL], ['블록 밖', '회전', '짝 부족'], '기타')
    T = d[['set', 'k', 'angle', 'n_inl', 'n_match', 'scale_est']].assign(
        city=d.set.map(CITY), err_m=e.round(2), head_err=he.round(1), tag=tag)
    T = T[e >= 10]
    n_ok = (pa8.assign(o=pa8.e < 10).groupby(['set', 'k']).o.sum())
    T = T.join(n_ok.rename('angles_ok_of8'), on=['set', 'k'])  # k < 100은 비어 있음
    T.to_csv(os.path.join(RES, 'failure_tags.csv'), index=False)
    print('\n## 실패 원인 태그 (8방향 결과, k = 0~999, 10 m 넘은 샘플)\n')
    X = pd.crosstab(T.city, T.tag, margins=True, margins_name='합계')
    print('| 도시 | ' + ' | '.join(X.columns) + ' |'); print('| --- ' * (len(X.columns) + 1) + '|')
    for city, row in X.iterrows():
        print(f'| {city} | ' + ' | '.join(str(v) for v in row) + ' |')
    print(f'\n→ {os.path.relpath(os.path.join(RES, "failure_tags.csv"), ROOT)} ({len(T)}행)')


if __name__ == '__main__':
    main()
