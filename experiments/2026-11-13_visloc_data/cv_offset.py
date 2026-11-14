"""좌표 보정 교차검증 (single_calib 안에서 비행 줄 홀짝으로 2겹).

  python experiments/2026-11-13_visloc_data/cv_offset.py data/visloc/single_calib/samples.csv experiments/2026-11-13_visloc_data/loma_calib_s1.csv

Cross-validate the label correction inside single_calib (2-fold by line parity).

Models: none / body-frame (along, right) / map-frame (E, N). Heading is not refit.
Uses only confident samples (n_inl >= 15) for fitting; scores all samples with the team rule.
"""
import sys
import numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from imatch.metrics import errors

S = pd.read_csv(sys.argv[1], dtype={'flight': str}); P = pd.read_csv(sys.argv[2])
R = S.merge(P.drop(columns=['gt_x', 'gt_y', 'gt_theta']), on=['set', 'k'])
R['dE'] = (R.pred_x - R.x_norm) * 32; R['dN'] = -(R.pred_y - R.y_norm) * 32
h = np.radians(R.heading_cw)
R['al'] = R.dE * np.sin(h) + R.dN * np.cos(h); R['ri'] = R.dE * np.cos(h) - R.dN * np.sin(h)
R['conf'] = (R.n_inl.fillna(0) >= 15) & (np.hypot(R.dE, R.dN) < 60)
R['fold'] = R.line % 2


def score(g, E_off, N_off):
    # corrected GT in metres = raw GT + offset; compare pred (unchanged) to it
    gx = g.x_norm + E_off / 32; gy = g.y_norm - N_off / 32
    e, fail = errors(g.pred_x, g.pred_y, gx, gy, lim=2.0)  # lim 2: corrected GT can leave the block slightly
    ok = ~fail
    return ok & (e <= 10), ok & (e <= 20)


rows = []
print('flight | conf n | fwd/back along (body test) | E,N by direction')
for fl, g in R.groupby('flight'):
    c = g[g.conf]
    hd = c.heading_cw.values
    if len(c) >= 6:
        ref = np.median(hd)
        fwd = np.abs(((hd - ref) + 180) % 360 - 180) < 90
        def med(x, m): return f'{np.median(x[m]):+5.1f}' if m.sum() else '  -  '
        print(f'{fl} | {len(c):3d} | along fwd {med(c.al.values, fwd)} back {med(c.al.values, ~fwd)} | '
              f'E fwd {med(c.dE.values, fwd)} back {med(c.dE.values, ~fwd)}  N fwd {med(c.dN.values, fwd)} back {med(c.dN.values, ~fwd)}  (n {fwd.sum()}/{(~fwd).sum()})')
    for fold in (0, 1):
        fit, test = g[(g.fold != fold) & g.conf], g[g.fold == fold]
        if len(fit) < 5 or not len(test):
            continue
        a, r = np.median(fit.al), np.median(fit.ri)
        th = np.radians(test.heading_cw)
        Eb, Nb = a * np.sin(th) + r * np.cos(th), a * np.cos(th) - r * np.sin(th)
        Em, Nm = np.median(fit.dE), np.median(fit.dN)
        for name, (E, N) in {'none': (0 * th, 0 * th), 'body': (Eb, Nb), 'map': (Em + 0 * th, Nm + 0 * th)}.items():
            s10, s20 = score(test, np.asarray(E), np.asarray(N))
            rows.append(dict(flight=fl, model=name, n=len(test), s10=s10.sum(), s20=s20.sum()))
T = pd.DataFrame(rows).groupby(['flight', 'model'])[['n', 's10', 's20']].sum()
T['10m'] = (T.s10 / T.n * 100).round(1); T['20m'] = (T.s20 / T.n * 100).round(1)
print(T[['n', '10m', '20m']].unstack('model').to_string())
A = pd.DataFrame(rows).groupby('model')[['n', 's10', 's20']].sum()
print('\nALL (held-out folds):'); print((A[['s10', 's20']].div(A.n, axis=0) * 100).round(1).assign(n=A.n).to_string())
