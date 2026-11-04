"""Validate historical tag membership and reconstruct historical 0-degree results (no inference)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, SET, sha256, write_json
sys.path.insert(0, str(ROOT))
from imatch.geometry import wrap
from imatch.metrics import errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    tags = pd.read_csv(ROOT / 'results/failure_tags.csv')
    old = pd.read_csv(ROOT / 'results/loma_8dir.csv')
    if tags.duplicated(['set', 'k']).any() or old.duplicated(['set', 'k']).any():
        raise ValueError('Duplicate historical keys')
    indexed = old.set_index(['set', 'k'])
    expected = indexed.reindex(pd.MultiIndex.from_frame(tags[['set', 'k']]))
    if expected.gt_x.isna().any():
        raise ValueError('Unknown tagged sample')
    e, fail = errors(expected.pred_x, expected.pred_y, expected.gt_x, expected.gt_y)
    he = wrap(expected.pred_theta.fillna(0)-expected.gt_theta)
    recomputed = np.select([fail, he > 45, expected.n_inl.fillna(0) < 30], ['블록 밖', '회전', '짝 부족'], '기타')
    if not np.array_equal(tags.tag.to_numpy(), recomputed) or (e < 10).any():
        raise ValueError('Historical tags disagree with documented priority/rules')
    e_all = errors(old.pred_x, old.pred_y, old.gt_x, old.gt_y)[0]
    all_failed = set(map(tuple, old.loc[e_all >= 10, ['set', 'k']].to_numpy()))
    if all_failed != set(map(tuple, tags[['set', 'k']].to_numpy())):
        raise ValueError('Historical tag coverage is incomplete')
    if tags.loc[tags['set'].eq(SET), 'city'].ne('Singapore').any():
        raise ValueError('city/set mismatch')
    counts = []
    for split, lo, hi in [('tune', 0, 100), ('final', 100, 1000), ('all', 0, 1000)]:
        g = tags[tags['set'].eq(SET) & tags.k.between(lo, hi-1)]
        for tag in ['블록 밖', '짝 부족', '회전', '기타']:
            counts.append({'split': split, 'kmin': lo, 'kmax': hi-1, 'tag': tag, 'n': int(g.tag.eq(tag).sum())})
        counts.append({'split': split, 'kmin': lo, 'kmax': hi-1, 'tag': '블록 밖+짝 부족',
                       'n': int(g.tag.isin(['블록 밖', '짝 부족']).sum())})
    angles = pd.read_csv(ROOT / 'results/loma_8dir_per_angle_k100-999.csv')
    one = angles[angles['set'].eq(SET) & angles.angle.eq(0)].copy()
    if sorted(one.k) != list(range(100, 1000)):
        raise ValueError('Historical zero-degree samples missing or duplicated')
    one = one.merge(old[['set', 'k', 'gt_x', 'gt_y', 'gt_theta']], on=['set', 'k'], validate='one_to_one')
    # Angle CSV may contain px even when n_inl<4: use the same eligibility as pipeline.localize.
    valid = one.n_inl.ge(4) & one.px.notna() & one.py.notna()
    one['pred_x'], one['pred_y'] = one.px.where(valid), one.py.where(valid)
    e1, _ = errors(one.pred_x, one.pred_y, one.gt_x, one.gt_y)
    a.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(counts).to_csv(a.out / 'historical_tag_counts.csv', index=False, mode='x')
    write_json(a.out / 'historical_audit.json', {
        'origin': 'existing 8-direction CSV and 0-degree candidates; NOT new E3/E4 GPU inference',
        'historical_1dir_final': {'n': len(one), 'success_pct': float((e1 < 10).mean()*100),
                                 'error_mean_m': float(e1.mean()), 'error_median_m': float(np.median(e1))},
        'tag_counts': counts, 'tag_rules_verified': True,
        'sha256': {f: sha256(ROOT / 'results' / f) for f in ['failure_tags.csv', 'loma_8dir.csv', 'loma_8dir_per_angle_k100-999.csv']}})
    print(pd.DataFrame(counts).to_string(index=False))
    print('Historical 1dir k100..999:', len(one), f'{(e1 < 10).mean()*100:.3f}%', 'NOT a fresh baseline')


if __name__ == '__main__':
    main()
