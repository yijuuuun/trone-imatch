# Phase 0: (1) training-free VGG dense matching (B2) inside the closed-loop nav simulator,
#          (2) NY decode 128 px vs 256 px re-measured on IDENTICAL trajectories (fixed seed, more trajectories).
# Every method/condition of a city shares the same start/goal pairs, so success can be compared per trajectory.
import os, sys, json, time, argparse
import numpy as np, cv2, torch, torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from diag.diag_modules import DEV, SEEN, UNSEEN, to_tensor, load_model, enc, head
from diag.diag_nav import load_rsi, TileCache
from diag.diag_gluf import rot_tpl, TPL, ANGLES

PX = 4.0  # px per metre (0.25 m/px)
W = 384


class Cropper:
    """diag_modules.crop_rot with the reflect padding done once per map instead of per crop (identical output)."""
    def __init__(self, img):
        self.pad = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_REFLECT)

    def __call__(self, cx, cy, theta):
        x0, y0 = int(cx) - W, int(cy) - W
        win = self.pad[y0 + W:y0 + 3 * W, x0 + W:x0 + 3 * W]
        lx, ly = cx - x0, cy - y0
        R = cv2.getRotationMatrix2D((lx, ly), -theta, 1)
        rot = cv2.warpAffine(win, R, (win.shape[1], win.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        c_x, c_y = int(lx), int(ly)
        return rot[c_y - 128:c_y + 128, c_x - 128:c_x + 128]


def make_traj(n_traj, dist_m, seed, n_block=15):
    rng = np.random.default_rng(seed)
    lo, hi = 300, 256 * (n_block + 1) - 300
    starts, goals = [], []
    while len(starts) < n_traj:
        s = rng.uniform(lo, hi, 2); a = rng.uniform(0, 2 * np.pi)
        g = s + dist_m * PX * np.array([np.cos(a), np.sin(a)])
        if (g > lo).all() and (g < hi).all():
            starts.append(s); goals.append(g)
    return np.array(starts), np.array(goals)


class ModelLoc:
    """Bearing-UAV forward (GLUF -> RCE/CA/PSG -> regression). Returns label units (1 = 128 px)."""
    def __init__(self, m, img):
        self.m, self.tc = m, TileCache(m, img)

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        nb = torch.stack([self.tc.block(bx, by) for bx, by in blocks])
        fi = enc(self.m, torch.stack([to_tensor(c) for c in crops]).to(DEV))
        return head(self.m, fi, nb)[0].cpu().numpy()


class DenseLoc:
    """B2: frozen VGG16 conv4_3 normalised cross-correlation over the 2x2-tile block (same as diag_gluf.dense_match,
    centred), with the block feature map cached. compass=True uses the known flight heading (1 template);
    otherwise a 16-angle search. Returns label units (1 = 128 px)."""
    def __init__(self, m, img, compass):
        self.m, self.img, self.compass, self.c = m, img, compass, {}

    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.c:
            sx, sy = bx * 256, by * 256
            Bf = self.m.backbone(to_tensor(self.img[sy:sy + 512, sx:sx + 512]).unsqueeze(0).to(DEV))[0]
            mu = Bf.mean((1, 2), keepdim=True)
            self.c[(bx, by)] = (F.normalize(Bf - mu, dim=0), mu)
        return self.c[(bx, by)]

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        angs = [[h] for h in headings] if self.compass else [list(ANGLES)] * len(crops)
        tpl = torch.stack([to_tensor(rot_tpl(c, a)) for c, al in zip(crops, angs) for a in al])
        Uf = torch.cat([self.m.backbone(tpl[k:k + 64].to(DEV)) for k in range(0, len(tpl), 64)])
        A = len(angs[0]); Uf = Uf.view(len(crops), A, *Uf.shape[1:])
        out = []
        for j, (bx, by) in enumerate(blocks):
            Bf, mu = self.block(bx, by)
            U = F.normalize(Uf[j] - mu.unsqueeze(0), dim=1); h = U.shape[-1]
            corr = F.conv2d(Bf.unsqueeze(0), U)[0] / (h * h)
            _, Hc, Wc = corr.shape
            idx = int(corr.flatten().argmax()); rem = idx % (Hc * Wc); iy, ix = divmod(rem, Wc)
            stride = 512 / Bf.shape[-1]
            out.append(((ix * stride + TPL / 2 - 256) / 128, (iy * stride + TPL / 2 - 256) / 128))
        return np.array(out)


def nav_run(loc, img, starts, goals, decode_px=128, oracle_block=False, step_m=15., arrive_m=20., n_block=15):
    """Same closed loop as diag_nav.nav_sim / diag_nav_trace.nav_trace (mirrors naver/runners/nav.py)."""
    n = len(starts); dist_m = np.linalg.norm(goals[0] - starts[0]) / PX
    crop_rot = Cropper(img)
    r = starts.copy(); nom = r.copy()
    alive = np.ones(n, bool); success = np.zeros(n, bool); reason = np.array(['timeout'] * n, dtype=object)
    heading = np.full(n, 90.0); err = [[] for _ in range(n)]
    for t in range(int(3 * dist_m / step_m)):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        blocks, crops, cen = [], [], []
        for i in idx:
            ref = r[i] if oracle_block else nom[i]
            bx = int(np.clip(np.round((ref[0] - 256) / 256), 0, n_block - 1))
            by = int(np.clip(np.round((ref[1] - 256) / 256), 0, n_block - 1))
            blocks.append((bx, by)); cen.append(np.array([bx * 256 + 256, by * 256 + 256]))
            crops.append(crop_rot(r[i, 0], r[i, 1], heading[i]))
        pos = loc(crops, heading[idx], blocks)
        for j, i in enumerate(idx):
            p = cen[j] + pos[j] * decode_px
            err[i].append(float(np.linalg.norm(p - r[i]) / PX))
            d = goals[i] - p; u = d / max(np.linalg.norm(d), 1e-6)
            heading[i] = np.degrees(np.arctan2(-u[1], u[0]))
            r[i] = r[i] + u * step_m * PX; nom[i] = p + u * step_m * PX
            real_d = np.linalg.norm(r[i] - goals[i]) / PX; pred_d = np.linalg.norm(p - goals[i]) / PX
            if real_d < arrive_m or pred_d < arrive_m:
                alive[i] = False; success[i] = real_d < 2 * arrive_m
                reason[i] = 'arrived_ok' if success[i] else 'arrived_pred_only'
            elif (r[i] < 200).any() or (r[i] > 256 * (n_block + 1) - 200).any():
                alive[i] = False; reason[i] = 'out_of_map'
    return success, reason, [float(np.median(e)) for e in err if e]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired success vectors. Returns (a-only wins, b-only wins, p)."""
    from math import comb
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_traj', type=int, default=96)
    ap.add_argument('--dist', type=float, default=1000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--cities', default='seen:Tokyo,seen:SF,unseen:NY,unseen:Paris,unseen:London,unseen:Birmingham,unseen:Berlin')
    ap.add_argument('--methods', default='model_dec128,model_dec256,model_dec128_oracle,B2_compass,B2_search16')
    ap.add_argument('--out', default='diag/out/diag_phase0.json')
    a = ap.parse_args()
    m2 = load_model('Bearing_UAV/satellite_view')
    pre = {'seen:Tokyo': SEEN['34bc'], 'seen:SF': SEEN['38bc'], **{f'unseen:{k}': v for k, v in UNSEEN.items()}}
    R = json.load(open(a.out)) if os.path.exists(a.out) else {}
    R['_config'] = dict(n_traj=a.n_traj, dist_m=a.dist, seed=a.seed, view='2D (satellite crop)',
                        model='Bearing_UAV/satellite_view (B0, leaky public weights)')
    for city in a.cities.split(','):
        img = load_rsi(pre[city])
        starts, goals = make_traj(a.n_traj, a.dist, a.seed)  # same pairs for every method of this city
        S = {}
        for meth in a.methods.split(','):
            k = f'{city}|{meth}'
            t0 = time.time()
            if meth.startswith('model'):
                loc = ModelLoc(m2, img); dec = 256 if 'dec256' in meth else 128
                succ, rs, e = nav_run(loc, img, starts, goals, decode_px=dec, oracle_block='oracle' in meth)
            else:
                loc = DenseLoc(m2, img, compass=(meth == 'B2_compass'))
                succ, rs, e = nav_run(loc, img, starts, goals, decode_px=128)
            S[meth] = succ
            R[k] = dict(success_pct=round(float(succ.mean() * 100), 1), ci95=wilson(int(succ.sum()), len(succ)),
                        reasons={x: int((rs == x).sum()) for x in np.unique(rs)},
                        per_step_loc_err_median_m=round(float(np.median(e)), 2),
                        sec=round(time.time() - t0, 1), success=succ.astype(int).tolist())
            print(k, {kk: v for kk, v in R[k].items() if kk != 'success'}, flush=True)
            json.dump(R, open(a.out, 'w'), indent=1)
        for x, y in [('model_dec128', 'model_dec256'), ('B2_compass', 'model_dec128'), ('B2_compass', 'model_dec256'),
                     ('B2_search16', 'model_dec128')]:
            if x in S and y in S:
                R[f'{city}|paired:{x}_vs_{y}'] = dict(zip(['x_only', 'y_only', 'p_mcnemar'], mcnemar(S[x], S[y])))
                print(city, x, 'vs', y, R[f'{city}|paired:{x}_vs_{y}'], flush=True)
        json.dump(R, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
