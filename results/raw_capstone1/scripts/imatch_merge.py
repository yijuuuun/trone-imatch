# Image-matching flow (IM_), stage 2: join the per-sample CSVs of one matcher from several runs (smoke k < 100,
# stage-2 Kaggle parts k >= 100) into diag/out/imatch/stage2/<name>_s1.csv, and build the two-matcher ensemble:
# per sample, keep the estimate of the matcher with the higher MAGSAC++ inlier ratio (inliers / matches; no ground
# truth involved). Raw inlier counts are not comparable: RoMa v2 samples 2,048 dense matches and won 100 % of the
# smoke samples on counts alone. The ratio rule was set after the smoke run (k < 100), so the ensemble is judged
# on k >= 100 only.
#   python diag/imatch_merge.py
import glob, os
import pandas as pd

ROOT = 'diag/out/imatch'
OUT = f'{ROOT}/stage2'


def gather(name):
    fs = [f for f in glob.glob(f'{ROOT}/**/{name}_s1.csv', recursive=True) if not f.startswith(OUT + '/')]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.set.str.startswith('3d')].drop_duplicates(['set', 'k'], keep='last').sort_values(['set', 'k'])
    print(name, len(fs), 'files', d.groupby('set').size().to_dict())
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    D = {}
    for name in ['loma', 'romav2']:
        D[name] = gather(name)
        D[name].to_csv(f'{OUT}/{name}_s1.csv', index=False)
    a, b = (D[n].set_index(['set', 'k']) for n in ['loma', 'romav2'])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    ratio = lambda d: (d.n_inl / d.n_match).fillna(0)
    pick_b = ratio(b) > ratio(a)
    e = a.copy(); e.loc[pick_b] = b.loc[pick_b]
    e['matcher'] = 'ens-ratio'; e['picked'] = pick_b.map({True: 'romav2', False: 'loma'})
    e['sec'] = a.sec + b.sec
    e.reset_index().to_csv(f'{OUT}/ens-ratio_s1.csv', index=False)
    print('ensemble', len(e), 'romav2 picked %.0f%%' % (pick_b.mean() * 100))


if __name__ == '__main__':
    main()
