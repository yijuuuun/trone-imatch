# Follow-up diagnosis: heading shortcut (map rotation), out-of-block response, closed-loop long-distance nav.
import os, sys, json, glob, argparse
import numpy as np, cv2, torch, torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from diag.diag_modules import (DEV, M_PER_UNIT, SEEN, UNSEEN, DARK, to_tensor, darken, load_model, enc, head,
                               errs, summ, crop_rot, gen_samples, eval_generated, gt_quadrant)


def load_rsi(pre):
    f = [p for p in glob.glob(f'Bearing_UAV_90K/city_rsi/{pre}*_city.jpg') if '6144' not in p][0]
    return cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- (a) heading shortcut: rotate whole map
@torch.no_grad()
def rotation_test(models, n, rng):
    """Rotating the map rotates tiles AND UAV view together; the UAV heading *relative to the map* (the label)
    is unchanged in definition. A model that measures heading against the reference tiles is unaffected;
    a model that infers absolute heading from UAV-image priors (shadows, street grid) breaks."""
    out = {}
    for city in ['34bc', '38bc']:
        img = load_rsi(SEEN[city])
        for k in [0, 1, 2]:
            S = gen_samples(np.ascontiguousarray(np.rot90(img, k)), n, rng)
            out[f'{city}_rot{90 * k}'] = {mn: eval_generated(m, S) for mn, m in models.items()}
            print(city, k, out[f'{city}_rot{90 * k}'], flush=True)
    return out


# ---------------------------------------------------------------- (b) out-of-block response
@torch.no_grad()
def out_of_block(m, img, n, rng):
    """UAV centre placed outside the trained [-1,1] window (but still inside the map)."""
    res = {}
    for lo, hi in [(0, 1), (1, 1.5), (1.5, 2.5)]:
        P, G, MX = [], [], []
        for _ in range(n // 32):
            tiles_b, tgt_b, gts = [], [], []
            for _ in range(32):
                bx, by = rng.integers(2, 12, 2)
                r = rng.uniform(lo, hi); a = rng.uniform(0, 2 * np.pi)
                xn, yn = np.clip(r * np.cos(a), -2.5, 2.5), np.clip(r * np.sin(a), -2.5, 2.5)
                sx, sy = bx * 256, by * 256
                tiles_b.append(torch.stack([to_tensor(img[sy + dy * 256:sy + (dy + 1) * 256, sx + dx * 256:sx + (dx + 1) * 256])
                                            for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]]))
                tgt_b.append(to_tensor(crop_rot(img, sx + 256 + xn * 128, sy + 256 + yn * 128, rng.uniform(-180, 180))))
                gts.append((xn, yn))
            tiles = torch.stack(tiles_b).to(DEV); B = tiles.size(0)
            nb = enc(m, tiles.flatten(0, 1)).view(B, 4, -1); fi = enc(m, torch.stack(tgt_b).to(DEV))
            pos, *_ = head(m, fi, nb)
            P.append(pos); G.append(torch.tensor(gts, dtype=torch.float32, device=DEV))
            MX.append(F.cosine_similarity(fi.unsqueeze(1), nb, -1).max(1).values)
        P, G, MX = torch.cat(P), torch.cat(G), torch.cat(MX)
        res[f'|gt|in[{lo},{hi})'] = dict(
            MLE=round(float((P - G).norm(dim=1).mean() * M_PER_UNIT), 2),
            pred_norm_mean=round(float(P.norm(dim=1).mean()), 3), gt_norm_mean=round(float(G.norm(dim=1).mean()), 3),
            max_cos_to_tiles=round(float(MX.mean()), 3))
    return res


# ---------------------------------------------------------------- (c) closed-loop navigation (mirrors naver/runners/nav.py)
class TileCache:
    def __init__(self, m, img):
        self.m, self.img, self.c = m, img, {}

    @torch.no_grad()
    def block(self, bx, by):
        out = []
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            k = (bx + dx, by + dy)
            if k not in self.c:
                t = self.img[k[1] * 256:(k[1] + 1) * 256, k[0] * 256:(k[0] + 1) * 256]
                self.c[k] = enc(self.m, to_tensor(t).unsqueeze(0).to(DEV))[0]
            out.append(self.c[k])
        return torch.stack(out)


@torch.no_grad()
def nav_sim(m, img, n_traj, dist_m, rng, decode_px=128, dark=None, step_m=15., arrive_m=20., n_block=15):
    """Pixel frame, 0.25 m/px. Real pos r, nominal n; block = nearest block centre to n (nav.get_nearest_block);
    target crop at r with heading = flight direction; p = centre + pred*decode_px (nav uses 256, labels 128);
    next: dir from p to goal, r += step, n = p + step (nav.py lines 790-798)."""
    tc = TileCache(m, img); px = 4.0  # px per metre
    lo, hi = 300, 256 * (n_block + 1) - 300
    starts, goals = [], []
    while len(starts) < n_traj:
        s = rng.uniform(lo, hi, 2); a = rng.uniform(0, 2 * np.pi)
        g = s + dist_m * px * np.array([np.cos(a), np.sin(a)])
        if (g > lo).all() and (g < hi).all():
            starts.append(s); goals.append(g)
    r = np.array(starts); nom = r.copy(); goal = np.array(goals)
    alive = np.ones(n_traj, bool); success = np.zeros(n_traj, bool); first_fail = np.full(n_traj, -1)
    loc_err = [[] for _ in range(n_traj)]
    heading = np.full(n_traj, 90.0)
    max_steps = int(3 * dist_m / step_m)
    for t in range(max_steps):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        nbs, tg, cen = [], [], []
        for i in idx:
            bx = int(np.clip(np.round((nom[i, 0] - 256) / 256), 0, n_block - 1))
            by = int(np.clip(np.round((nom[i, 1] - 256) / 256), 0, n_block - 1))
            c = np.array([bx * 256 + 256, by * 256 + 256]); cen.append(c)
            nbs.append(tc.block(bx, by))
            # UAV view rotation: heading in CCS (counter-clockwise from +x, y up) -> theta convention of crop_target_patch
            crop = crop_rot(img, r[i, 0], r[i, 1], heading[i])
            if dark:
                crop = darken(crop, dark, rng)
            tg.append(to_tensor(crop))
            if first_fail[i] < 0 and np.abs(r[i] - c).max() > 128:
                first_fail[i] = t  # UAV outside trained [-1,1] window of the selected block
        fi = enc(m, torch.stack(tg).to(DEV)); nb = torch.stack(nbs)
        pos, *_ = head(m, fi, nb)
        pos = pos.cpu().numpy()
        for j, i in enumerate(idx):
            p = cen[j] + pos[j] * decode_px
            loc_err[i].append(float(np.linalg.norm(p - r[i]) / px))
            d = goal[i] - p; dn = np.linalg.norm(d)
            u = d / max(dn, 1e-6)
            heading[i] = np.degrees(np.arctan2(-u[1], u[0]))  # image y is down
            r[i] = r[i] + u * step_m * px; nom[i] = p + u * step_m * px
            if np.linalg.norm(r[i] - goal[i]) / px < arrive_m or np.linalg.norm(p - goal[i]) / px < arrive_m:
                alive[i] = False; success[i] = np.linalg.norm(r[i] - goal[i]) / px < 2 * arrive_m
            elif (r[i] < 200).any() or (r[i] > 256 * (n_block + 1) - 200).any():
                alive[i] = False
    med_err = [np.median(e) for e in loc_err if e]
    return dict(success_pct=round(float(success.mean() * 100), 1),
                capture_lost_pct=round(float((first_fail >= 0).mean() * 100), 1),
                median_step_of_capture_loss=(int(np.median(first_fail[first_fail >= 0])) if (first_fail >= 0).any() else None),
                per_step_loc_err_median_m=round(float(np.median(med_err)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--n_traj', type=int, default=32)
    ap.add_argument('--out', default='diag/out')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(11)
    m2 = load_model('Bearing_UAV/satellite_view'); m3 = load_model('Bearing_UAV/cross_view')
    R = {}
    R['a_rotation_test'] = rotation_test({'satview_model': m2, 'crossview_model': m3}, a.n, rng)
    R['b_out_of_block_satview'] = out_of_block(m2, load_rsi(SEEN['34bc']), a.n, rng)
    print(R['b_out_of_block_satview'], flush=True)
    N = {}
    for dist in [300, 1000]:
        for name, pre, conds in [('seen:Tokyo', SEEN['34bc'], [(128, None), (256, None), (128, 'L2')]),
                                 ('seen:SF', SEEN['38bc'], [(128, None), (256, None), (128, 'L2')]),
                                 ('unseen:NY', UNSEEN['NY'], [(128, None), (256, None)]),
                                 ('unseen:Paris', UNSEEN['Paris'], [(128, None)]),
                                 ('unseen:Berlin', UNSEEN['Berlin'], [(128, None)])]:
            img = load_rsi(pre)
            for dec, dark in conds:
                k = f'{name}|{dist}m|decode{dec}px|{dark or "day"}'
                N[k] = nav_sim(m2, img, a.n_traj, dist, rng, decode_px=dec, dark=dark)
                print(k, N[k], flush=True)
    R['c_nav_sim_satview'] = N
    json.dump(R, open(f'{a.out}/diag_nav_results.json', 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
