"""두 매처 조합: 샘플마다 MAGSAC++ inlier 비율(inlier / 대응점)이 높은 매처의 추정을 쓴다 (정답은 안 씀).
inlier '개수'는 비교할 수 없다: RoMa v2는 대응점을 2,048개씩 뽑아 개수만으로는 스모크 100%를 이겼다.
이 규칙은 스모크(k < 100)를 보고 정했으므로 k >= 100으로만 평가한다.

  python scripts/ensemble.py results/loma_8dir.csv results/romav2_8dir.csv --out outputs/ensemble.csv
  python scripts/evaluate.py outputs/ensemble.csv --kmin 100
결과(2026-10-16): Singapore 76% (LoMa 단독 75%, 900샘플) → 효과 없음. 둘 중 하나라도 맞힌 비율은 82%.
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--out', required=True)
    x = ap.parse_args()
    A, B = (pd.read_csv(f).drop_duplicates(['set', 'k']).set_index(['set', 'k']) for f in (x.a, x.b))
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    ratio = lambda d: (d.n_inl / d.n_match).fillna(0)
    pick_b = ratio(B) > ratio(A)
    E = A.copy(); E.loc[pick_b] = B.loc[pick_b]
    E['matcher'] = 'ensemble'; E['picked'] = pick_b.map({True: B.matcher.iloc[0], False: A.matcher.iloc[0]})
    E['sec'] = A.sec + B.sec
    os.makedirs(os.path.dirname(os.path.abspath(x.out)), exist_ok=True)
    E.reset_index().to_csv(x.out, index=False)
    print(f'{len(E)} samples, {B.matcher.iloc[0]} picked {pick_b.mean() * 100:.0f}% -> {x.out}')


if __name__ == '__main__':
    main()
