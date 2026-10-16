# 1-d: inside GLUF, is the culprit the frozen VGG16 backbone or the trained CSMG aggregation?
# Same frozen conv4_3 maps -> (trained) CSMG vs (training-free) pooling vs (training-free) dense correlation.
import os, sys, json, argparse
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from torch.utils.data import random_split

sys.path.insert(0, os.getcwd())
from diag.diag_modules import (DEV, M_PER_UNIT, SEEN, UNSEEN, to_tensor, imread_rgb, darken, load_model, head,
                               gen_samples, gt_quadrant)
from diag.diag_nav import load_rsi

XY_SLOTS = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])  # p1..p4 as (x right, y down)
ANGLES = np.arange(0, 360, 22.5)
TPL = 176  # template side: largest square inside the rotated 256 patch (no reflected border content)


def wrap(d):
    return np.abs((d + 180) % 360 - 180)


@torch.no_grad()
def global_descs(m, x):
    """x [B,3,256,256] -> dict of L2-normalised global descriptors built on the same frozen conv4_3 map."""
    bb = m.backbone(x)
    _, _, dflat, xnl = m.csmg(bb)
    return {'CSMG(trained)': dflat,
            'CSMG_inner_GAP(trained conv+NL)': F.normalize(xnl.mean((2, 3)), dim=1),
            'VGG_GAP(free)': F.normalize(bb.mean((2, 3)), dim=1),
            'VGG_GeM(free)': F.normalize(bb.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1 / 3), dim=1)}


def rot_tpl(rgb, a):
    R = cv2.getRotationMatrix2D((128, 128), a, 1)
    r = cv2.warpAffine(rgb, R, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    o = (256 - TPL) // 2
    return r[o:o + TPL, o:o + TPL]


@torch.no_grad()
def dense_match(m, tiles_rgb, uav_rgb, angles, centered):
    """Normalised cross-correlation of conv4_3 maps. Returns (x_norm, y_norm, best_angle, peak)."""
    block = np.vstack([np.hstack([tiles_rgb[0], tiles_rgb[1]]), np.hstack([tiles_rgb[2], tiles_rgb[3]])])
    Bf = m.backbone(to_tensor(block).unsqueeze(0).to(DEV))[0]                     # [C,64,64]
    Uf = m.backbone(torch.stack([to_tensor(rot_tpl(uav_rgb, a)) for a in angles]).to(DEV))  # [A,C,22,22]
    if centered:
        mu = Bf.mean((1, 2), keepdim=True); Bf = Bf - mu; Uf = Uf - mu.unsqueeze(0)
    Bf = F.normalize(Bf, dim=0); Uf = F.normalize(Uf, dim=1)
    h = Uf.shape[-1]
    corr = F.conv2d(Bf.unsqueeze(0), Uf)[0] / (h * h)                           # [A,H',W']
    A, Hc, Wc = corr.shape
    idx = int(corr.flatten().argmax()); ai, rem = divmod(idx, Hc * Wc); iy, ix = divmod(rem, Wc)
    stride = 512 / Bf.shape[-1]
    cx, cy = ix * stride + TPL / 2, iy * stride + TPL / 2
    return (cx - 256) / 128, (cy - 256) / 128, angles[ai], float(corr.max())


def eval_set(m, samples, tau_by_method=None):
    """samples: list of (tiles_rgb[4], uav_rgb, (xn,yn), theta_deg)."""
    G = {}; dense = {'dense_search16': [], 'dense_search16_centered': [], 'dense_knownheading_centered': []}
    gts, ths, model_pos = [], [], []
    for i in range(0, len(samples), 16):
        ch = samples[i:i + 16]
        tiles = torch.stack([torch.stack([to_tensor(t) for t in s[0]]) for s in ch]).to(DEV); B = tiles.size(0)
        uav = torch.stack([to_tensor(s[1]) for s in ch]).to(DEV)
        dt, du = global_descs(m, tiles.flatten(0, 1)), global_descs(m, uav)
        for k in dt:
            G.setdefault(k, []).append(F.cosine_similarity(du[k].unsqueeze(1), dt[k].view(B, 4, -1), -1))
        model_pos.append(head(m, du['CSMG(trained)'], dt['CSMG(trained)'].view(B, 4, -1))[0])
        for s in ch:
            gts.append(s[2]); ths.append(s[3])
            dense['dense_search16'].append(dense_match(m, s[0], s[1], ANGLES, False))
            dense['dense_search16_centered'].append(dense_match(m, s[0], s[1], ANGLES, True))
            dense['dense_knownheading_centered'].append(dense_match(m, s[0], s[1], [s[3]], True))
    gt = torch.tensor(gts, dtype=torch.float32, device=DEV); th = np.array(ths); q = gt_quadrant(gt)
    gn = gt.cpu().numpy()
    out = {}
    mp = torch.cat(model_pos)
    out['full_model(ref)'] = dict(MLE=round(float((mp - gt).norm(dim=1).mean() * M_PER_UNIT), 2))
    for k, L in G.items():
        cos = torch.cat(L)
        r = dict(retr_top1=round(float((cos.argmax(1) == q).float().mean() * 100), 1),
                 cos_spread=round(float(cos.std(1).mean()), 4))
        if tau_by_method:
            sp = (torch.softmax(cos / tau_by_method[k], 1).unsqueeze(2) * XY_SLOTS.to(DEV)).sum(1).cpu().numpy()
            r['softpos_corr'] = round(float(np.mean([np.corrcoef(sp[:, j], gn[:, j])[0, 1] for j in range(2)])), 3)
        out[k] = r
        out[k]['_cos'] = cos  # kept for tau fitting, stripped before saving
    for k, L in dense.items():
        P = np.array([(x, y) for x, y, _, _ in L]); A = np.array([a for _, _, a, _ in L])
        le = np.linalg.norm(P - gn, axis=1) * M_PER_UNIT
        r = dict(retr_top1=round(float(((P[:, 0] > 0) + 2 * (P[:, 1] > 0) == q.cpu().numpy()).mean() * 100), 1),
                 MLE=round(float(le.mean()), 2), medLE=round(float(np.median(le)), 2),
                 LSR10=round(float((le < 10).mean() * 100), 1))
        if 'search' in k:
            r['MHE'] = round(float(wrap(A - th).mean()), 2); r['MHE_signflip_check'] = round(float(wrap(-A - th).mean()), 2)
        out[k] = r
    return out, gt


def fit_tau(per_city):
    """Per method, pick the softmax temperature maximising soft-position/GT correlation on SEEN-day cities."""
    taus = {}
    for k in ['CSMG(trained)', 'CSMG_inner_GAP(trained conv+NL)', 'VGG_GAP(free)', 'VGG_GeM(free)']:
        cos = torch.cat([o[k]['_cos'] for o, _ in per_city]); gn = torch.cat([g for _, g in per_city]).cpu().numpy()
        best = None
        for t in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            sp = (torch.softmax(cos / t, 1).unsqueeze(2) * XY_SLOTS.to(DEV)).sum(1).cpu().numpy()
            c = np.mean([np.corrcoef(sp[:, j], gn[:, j])[0, 1] for j in range(2)])
            if best is None or c > best[1]:
                best = (t, c)
        taus[k] = best[0]
    return taus


def gen_rgb(img, n, rng, dark=None):
    S = []
    for tiles, tgt, xy, cs in gen_samples(img, n, rng):
        th = float(np.degrees(np.arctan2(cs[1], cs[0])))
        S.append((tiles, darken(tgt, dark, rng) if dark else tgt, xy, th))
    return S


def seen_3d(n, rng, dark=None):
    df = pd.read_csv('Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv')
    N = len(df); ntr, nva = int(.85 * N), int(.05 * N)
    te = np.array(random_split(range(N), [ntr, nva, N - ntr - nva], generator=torch.Generator().manual_seed(42))[2].indices)
    S = []
    for i in rng.choice(te, n, replace=False):
        r = df.iloc[i]; u = imread_rgb(r.target_path)
        S.append(([imread_rgb(r[c]) for c in ['p1_path', 'p2_path', 'p3_path', 'p4_path']],
                  darken(u, dark, rng) if dark else u, (r.x_norm, r.y_norm), float(r.theta)))
    return S


def clean(o):
    return {k: {kk: vv for kk, vv in v.items() if kk != '_cos'} for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--out', default='diag/out')
    a = ap.parse_args()
    m2, m3 = load_model('Bearing_UAV/satellite_view'), load_model('Bearing_UAV/cross_view')
    rng = np.random.default_rng(5)
    imgs = {**{f'seen:{k}': load_rsi(v) for k, v in SEEN.items()}, **{f'unseen:{k}': load_rsi(v) for k, v in UNSEEN.items()}}
    sets = {f'{k}|day': gen_rgb(im, a.n, rng) for k, im in imgs.items()}
    for k in ['seen:34bc', 'seen:38bc']:
        for d in ['L2', 'L3']:
            sets[f'{k}|{d}'] = gen_rgb(imgs[k], a.n, rng, dark=d)
    # 2D sets use the satellite-view model; real UAV (3D) sets use the cross-view model
    raw = {}
    for k, S in sets.items():
        raw[k] = eval_set(m2, S); print('done', k, flush=True)
    for d in [None, 'L2', 'L3']:
        k = f'seen3D(real UAV,crossview model)|{d or "day"}'
        raw[k] = eval_set(m3, seen_3d(a.n, rng, d)); print('done', k, flush=True)
    taus2 = fit_tau([raw[k] for k in raw if k.startswith('seen:') and k.endswith('|day')])
    taus3 = fit_tau([raw['seen3D(real UAV,crossview model)|day']])
    R = {'tau_fitted_on_seen_day': {'2d': taus2, '3d': taus3}}
    for k, (o, gt) in raw.items():
        taus = taus3 if k.startswith('seen3D') else taus2
        gn = gt.cpu().numpy()
        for mk, t in taus.items():
            sp = (torch.softmax(o[mk]['_cos'] / t, 1).unsqueeze(2) * XY_SLOTS.to(DEV)).sum(1).cpu().numpy()
            o[mk]['softpos_corr'] = round(float(np.mean([np.corrcoef(sp[:, j], gn[:, j])[0, 1] for j in range(2)])), 3)
        R[k] = clean(o)
        print(k, json.dumps(R[k]), flush=True)
    json.dump(R, open(f'{a.out}/diag_gluf.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
