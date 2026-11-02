"""C3 감지기: 후보(샘플 × 각도)마다 "맞을 확률"을 내는 작은 분류기. 점수 1등 후보를 답으로 고른다.

  python scripts/c3_train.py                 # 학습 도시끼리 교차 검증만 (도쿄 → 타이베이, 타이베이 → 도쿄)
  python scripts/c3_train.py --model gb --test   # 고른 설정으로 두 학습 도시 전체를 학습, 시험 도시에서 한 번 평가

시험 도시(SF, Singapore) 평가는 설정을 확정한 뒤 한 번만 한다. 결과를 보고 설정을 다시 고치지 않는다.
시험 도시 후보: 8방향(results/loma_8dir_per_angle_k100-999.csv) + 22.5° 사이 8방향(experiments/2026-11-06_rotation/) = 16방향.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from c3_features import FEATURES, features  # noqa: E402
from imatch.metrics import errors, mcnemar  # noqa: E402

RES = os.path.join(ROOT, 'results')
TRAIN_CITY = {'3d_ho34bc': '도쿄', '3d_ho36bc': '타이베이'}
TEST_CITY = {'3d_ho37bc': 'Singapore', '3d_ho38bc': 'SF'}
ROT = os.path.join(ROOT, 'experiments', '2026-11-06_rotation')
FEW = 15  # 거절 기준 inlier 수 (학습 도시에서 정함)


def make(kind):
    if kind == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def select(df, score):
    """샘플마다 score 1등 후보."""
    return df.assign(_s=score).sort_values('_s', ascending=False).groupby(['set', 'k']).head(1).set_index(['set', 'k']).sort_index()


def report(df, score, label):
    """한 도시 묶음에 대해: score로 고른 답 vs inlier 1등, 상한, 후보 AUC."""
    sel, inl = select(df, score), select(df, df.n_inl.to_numpy())
    best = df.groupby(['set', 'k']).ok.any()
    rows = []
    for st, g in sel.groupby(level=0):
        a, b = g.ok.to_numpy(), inl.loc[st].ok.reindex(g.index.get_level_values(1)).to_numpy()
        d = df[df.set == st]
        auc = roc_auc_score(d.ok, score[df.set.to_numpy() == st]) if d.ok.nunique() == 2 else np.nan
        rows.append(f"| {label.get(st, st)} | {len(g)} | {b.mean()*100:.1f}% | **{a.mean()*100:.1f}%** | {best.loc[st].mean()*100:.1f}% "
                    f"| {mcnemar(a, b)} | {auc:.3f} |")
    print('| 도시 | 샘플 | inlier 1등 | 감지기 1등 | 상한 | 감지기만 / inlier만 성공 | 후보 AUC |')
    print('| --- ' * 7 + '|')
    print('\n'.join(rows))
    return sel


def test_candidates():
    pa = pd.concat([pd.read_csv(os.path.join(RES, 'loma_8dir_per_angle_k100-999.csv'))] +
                   [pd.read_csv(os.path.join(ROT, f'loma_off22_{c}_angles.csv')) for c in ['sg', 'sf']], ignore_index=True)
    pa = pa.drop_duplicates(['set', 'k', 'angle'])
    gt = pd.read_csv(os.path.join(ROOT, 'data', 'im_t1_3d_1000', 'samples.csv'))[['set', 'k', 'x_norm', 'y_norm', 'theta']]
    pa = features(pa.merge(gt, on=['set', 'k']))
    pa['err_m'], _ = errors(pa.px, pa.py, pa.x_norm, pa.y_norm)
    pa['ok'] = pa.err_m < 10
    return pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='', help='lr 또는 gb. 비우면 교차 검증에서 둘 다 본다')
    ap.add_argument('--test', action='store_true', help='시험 도시에서 한 번 평가 (설정 확정 후)')
    a = ap.parse_args()
    tr = pd.read_csv(os.path.join(RES, 'c3_train.csv'))
    X, y = tr[FEATURES].fillna(0).to_numpy(), tr.ok.to_numpy()

    if not a.test:
        for kind in ([a.model] if a.model else ['lr', 'gb']):
            print(f'\n## 교차 검증: {kind} (한 학습 도시로 학습, 다른 학습 도시로 확인)\n')
            score = np.zeros(len(tr))
            for st in TRAIN_CITY:
                m = make(kind).fit(X[tr.set != st], y[tr.set != st])
                score[tr.set == st] = m.predict_proba(X[tr.set == st])[:, 1]
            report(tr, score, TRAIN_CITY)
        return

    kind = a.model or 'gb'
    m = make(kind).fit(X, y)
    pickle.dump(m, open(os.path.join(RES, f'c3_{kind}.pkl'), 'wb'))
    te = test_candidates()
    score = m.predict_proba(te[FEATURES].fillna(0).to_numpy())[:, 1]
    print(f'\n## 시험 도시 (k = 100~999, 16방향 후보, 모델 {kind}, 두 학습 도시 전체로 학습)\n')
    sel = report(te, score, TEST_CITY)
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    te.assign(score=score).to_csv(os.path.join(ROOT, 'outputs', f'c3_test_scores_{kind}.csv'), index=False)  # 9 MB라 git에 안 올림

    # 라벨 의심 샘플 제외
    L = pd.read_csv(os.path.join(RES, 'label_check_t1_3d.csv'))
    bad = set(map(tuple, L[L.label_suspect][['set', 'k']].to_numpy()))
    keep = ~np.array([(s, k) in bad for s, k in zip(te.set, te.k)])
    print('\n### 라벨 의심 샘플 제외\n')
    report(te[keep].reset_index(drop=True), score[keep], TEST_CITY)

    # 거절: 고른 답이 미덥지 않으면 "모름". 기준은 학습 도시에서 정했다 (inlier < 15, 감지기 점수 < 0.5)
    inl = select(te, te.n_inl.to_numpy())
    print('\n### 거절 (답 = inlier 1등 후보. 큰 오류 = 32 m 넘음)\n')
    print('| 도시 | 기준 | 거절 비율 | 틀린 답 중 걸러낸 비율 | 큰 오류 중 걸러낸 비율 | 거절한 것 중 실제 틀린 비율 | 받아들인 답의 10 m 이내 |')
    print('| --- ' * 7 + '|')
    for st, city in TEST_CITY.items():
        g = inl.loc[st]
        sg = te.assign(_s=score).set_index(['set', 'k', 'angle']).loc[st]._s
        g_s = np.array([sg.loc[(k, ang)] for k, ang in zip(g.index, g.angle)])
        wrong, big = ~g.ok.to_numpy(), (g.err_m > 32).to_numpy()
        for name, rej in [(f'inlier < {FEW}', (g.n_inl < FEW).to_numpy()), (f'{kind} 점수 < 0.5', g_s < 0.5)]:
            print(f'| {city} | {name} | {rej.mean()*100:.1f}% | {(rej & wrong).sum()}/{wrong.sum()} ({(rej & wrong).sum()/max(wrong.sum(),1)*100:.0f}%) '
                  f'| {(rej & big).sum()}/{big.sum()} ({(rej & big).sum()/max(big.sum(),1)*100:.0f}%) '
                  f'| {(rej & wrong).sum()/max(rej.sum(),1)*100:.0f}% | {(~wrong)[~rej].mean()*100:.1f}% |')


if __name__ == '__main__':
    main()
