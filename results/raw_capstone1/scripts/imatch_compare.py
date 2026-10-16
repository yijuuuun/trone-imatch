# Image-matching flow (IM_): paired comparison of every matcher run in diag/out/imatch/ against the harness baselines
# (B2, CorrMatchNet vision-only = P2Bv3_img16b2) on the same T1 samples (3D, held-out SF / Singapore, k < n).
# Same statistics as diag/paired_compare.py: share within 10 m with an exact McNemar test (baseline-only /
# matcher-only successes), MLE difference (matcher - baseline) with a 95 % bootstrap CI (2,000 resamples, seed 0).
#   python diag/imatch_compare.py                  -> prints a markdown table, writes diag/out/imatch/compare.csv
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest

SETS = {'3d_ho38bc': 'SF', '3d_ho37bc': 'Singapore'}
BASES = {'B2': 'B2', 'CorrMatchNet': 'P2Bv3_img16b2'}


def base(harness, tag, st, n):
    d = pd.read_csv(f'{harness}/samples/{tag}_{st}.csv')
    d = d[(d.test == 'T1') & (d.cond.fillna('clean') == 'clean') & (d.k < n)]
    return d.drop_duplicates('k').set_index('k')[['row', 'err_m', 'head_err']]


def stats(ea, eb, rng):
    oa, ob = ea < 10, eb < 10
    n_a, n_b = int((oa & ~ob).sum()), int((~oa & ob).sum())
    diff = eb - ea
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    return dict(dMLE=diff.mean(), dMLE_lo=np.percentile(boot, 2.5), dMLE_hi=np.percentile(boot, 97.5),
                base_only=n_a, matcher_only=n_b, p=binomtest(n_b, n_a + n_b).pvalue if n_a + n_b else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='diag/out/imatch'); ap.add_argument('--harness', default='diag/out/harness')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--kmin', type=int, default=0, help='evaluate k in [kmin, n) only')
    ap.add_argument('--runs', default='', help='only runs whose path contains this text')
    a = ap.parse_args()
    samples = pd.read_csv(f'{a.root}/samples.csv')
    if os.path.exists(f'{a.root}/samples_1000.csv'):
        samples = pd.concat([samples, pd.read_csv(f'{a.root}/samples_1000.csv')]).drop_duplicates(['set', 'k'])
    runs = [f for f in glob.glob(f'{a.root}/**/*.csv', recursive=True)
            if re.search(r'_s[\d.]+\.csv$', f) and a.runs in f]
    out = []
    for st, city in SETS.items():
        B = {name: base(a.harness, tag, st, a.n) for name, tag in BASES.items()}
        B = {name: b[b.index >= a.kmin] for name, b in B.items()}
        rows = samples[(samples.set == st) & (samples.k >= a.kmin) & (samples.k < a.n)].set_index('k').row
        for name, b in B.items():
            assert (b.row.reindex(rows.index) == rows).all(), f'{name} {st}: sample rows differ from the export'
        for f in sorted(runs):
            r = pd.read_csv(f)
            r = r[(r.set == st) & (r.k < a.n) & (r.k >= a.kmin)].set_index('k').sort_index()
            if r.empty:
                continue
            idx = r.index
            # rule fixed before the deep-matcher results (2026-10-15): the task puts the UAV inside the 2x2 block,
            # and B2 / CorrMatchNet can only answer inside it, so an estimate is clipped into the block
            # (a first version sent out-of-block estimates to the centre; on 3D SF 7 of 9 of them were correct
            # estimates of near-edge ground truth just past the border, so it was dropped).
            # No estimate -> block centre. 'fail' = no estimate or more than 0.1 unit (3.2 m) outside the block.
            px, py = r.pred_x.to_numpy(float), r.pred_y.to_numpy(float)
            gx, gy = r.gt_x.to_numpy(float), r.gt_y.to_numpy(float)
            found = np.isfinite(px) & np.isfinite(py)
            fail = ~found | (np.abs(np.nan_to_num(px)) > 1.1) | (np.abs(np.nan_to_num(py)) > 1.1)
            raw = np.hypot(np.nan_to_num(px) - gx, np.nan_to_num(py) - gy) * 32
            cx, cy = np.clip(np.nan_to_num(px), -1, 1), np.clip(np.nan_to_num(py), -1, 1)
            em = np.hypot(cx - gx, cy - gy) * 32
            hm = r.head_err.to_numpy()
            rec = dict(city=city, run=os.path.relpath(f, a.root)[:-4], n=len(r), fail=fail.mean() * 100,
                       MLE=em.mean(), MLE_raw=raw.mean(), LSR10_raw=(raw < 10).mean() * 100,
                       medLE=np.median(em), LSR10=(em < 10).mean() * 100,
                       big32=(em > 32).mean() * 100, MHE=hm.mean(), medHE=np.median(hm),
                       inl_med=r.n_inl.fillna(0).median() if 'n_inl' in r else np.nan,
                       good_inl=r.good_inl.mean() * 100 if 'good_inl' in r else np.nan,
                       sec=r.sec.mean() if 'sec' in r else np.nan)
            for name, b in B.items():
                eb = b.err_m.reindex(idx).to_numpy()
                s = stats(eb, em, np.random.default_rng(0))
                rec.update({f'{name}_LSR10': (eb < 10).mean() * 100, f'{name}_MLE': eb.mean(),
                            f'vs{name}_dMLE': s['dMLE'], f'vs{name}_CI': f"[{s['dMLE_lo']:.2f}, {s['dMLE_hi']:.2f}]",
                            f'vs{name}_wins': f"{s['matcher_only']} / {s['base_only']}", f'vs{name}_p': s['p']})
            out.append(rec)
    D = pd.DataFrame(out)
    D.to_csv(f'{a.root}/compare' + (f"_{a.runs.strip('/').replace('/', '_')}" if a.runs else '') + f'_k{a.kmin}-{a.n}.csv', index=False)
    for city, g in D.groupby('city', sort=False):
        b = g.loc[g.n.idxmax()]
        print(f"\n### {city} (3D, n = {int(b.n)})  B2: {b.B2_MLE:.2f} m / {b.B2_LSR10:.0f}%  "
              f"CorrMatchNet: {b.CorrMatchNet_MLE:.2f} m / {b.CorrMatchNet_LSR10:.0f}%\n")
        print('| run | 실패 (추정 없음 · 블록 밖 3 m+) | MLE (원래) | 중앙값 | 10 m 이내 (원래) | 32 m 초과 | heading 중앙값 | inlier 중앙값 | s/샘플 '
              '| vs B2: ΔMLE [95% CI] | vs B2: 이김 / 짐, p | vs CMN: ΔMLE [95% CI] | vs CMN: 이김 / 짐, p |')
        print('| --- ' * 13 + '|')
        for _, r in g.iterrows():
            print(f"| {r.run} | {r.fail:.0f}% | {r.MLE:.2f} m ({r.MLE_raw:.2f}) | {r.medLE:.2f} m | {r.LSR10:.0f}% ({r.LSR10_raw:.0f}%) "
                  f"| {r.big32:.0f}% | {r.medHE:.1f}° "
                  f"| {r.inl_med:.0f} | {r.sec:.2f} | {r.vsB2_dMLE:+.2f} {r.vsB2_CI} | {r.vsB2_wins}, p = {r.vsB2_p:.3g} "
                  f"| {r.vsCorrMatchNet_dMLE:+.2f} {r.vsCorrMatchNet_CI} | {r.vsCorrMatchNet_wins}, p = {r.vsCorrMatchNet_p:.3g} |")


if __name__ == '__main__':
    main()
