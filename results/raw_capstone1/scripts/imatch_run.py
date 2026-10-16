# Image-matching flow (IM_): off-the-shelf matcher + MAGSAC++ homography -> position and heading, no training.
# Reads the files of diag/imatch_export.py, so it runs anywhere (Kaggle) without this repo.
#
# Per sample: the UAV view is turned by each search angle a (CCW, cv2 convention), matched to the 512 px block,
# and the homography H (turned UAV -> block) is fitted with MAGSAC++. The angle with the most inliers wins.
#   position  H maps the UAV centre into the block: x_norm = (bx - 256) / 128
#   heading   linear part L = H_lin @ R(a) maps original UAV px to block px; heading = atan2(L01, L00)
# Ground truth (diag_modules.crop_rot): block_px = C + R(theta) (uav_px - 128), R(t) = [[cos t, sin t], [-sin t, cos t]],
# so the same formula on the true transform gives theta. Matches are scored against this transform (exact for 2D,
# approximate for 3D renders, where buildings break the plane).
#
#   python diag/imatch_run.py --data <dir> --matchers sift-lightglue,loma,romav2 --scales 1,2 --angles 8 --out <dir>
import os, time, json, argparse
import numpy as np, pandas as pd, cv2

M_PER_PX = 0.25  # one block pixel; x_norm unit = 128 px = 32 m


def rot(a):
    t = np.deg2rad(a); c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def turn(img, a):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def wrap(d):
    return np.abs((np.asarray(d) + 180) % 360 - 180)


class CvSift:
    """Local fallback: OpenCV SIFT + Lowe ratio + MAGSAC++ (same RANSAC settings as vismatch)."""
    name = 'cv-sift'

    def __init__(self):
        self.sift = cv2.SIFT_create(4096); self.bf = cv2.BFMatcher()

    def __call__(self, im0, im1):
        k0, d0 = self.sift.detectAndCompute(cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY), None)
        k1, d1 = self.sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), None)
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), None, np.zeros(0, bool)
        mm = [m for m, n in self.bf.knnMatch(d0, d1, k=2) if m.distance < 0.8 * n.distance]
        p0 = np.float32([k0[m.queryIdx].pt for m in mm]).reshape(-1, 2)
        p1 = np.float32([k1[m.trainIdx].pt for m in mm]).reshape(-1, 2)
        return magsac(p0, p1)


RANSAC_PX = 3.0  # MAGSAC++ threshold (px); --ransac-px


def magsac(p0, p1):
    if len(p0) < 4:
        return p0, p1, None, np.zeros(len(p0), bool)
    H, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=2000, confidence=0.995)
    return p0, p1, H, (mask[:, 0].astype(bool) if mask is not None else np.zeros(len(p0), bool))


class VggMNN:
    """Ablation 'keep the Bearing-UAV backbone': the frozen VGG16 conv4_3 of the Bearing-UAV checkpoint (the same maps
    B2 correlates) used as a dense descriptor. Every feature cell (stride 8, centre at 8 i + 4) is a point; points are
    paired by mutual nearest neighbour on cosine similarity (descriptors centred on the block mean, as B2), then go
    through the same MAGSAC++ / homography back end as the off-the-shelf matchers. Cells whose centre falls in the
    black corners of the turned view are dropped. Local only (needs this repo)."""
    name = 'vgg-mnn'

    def __init__(self, ckpt='Bearing_UAV/cross_view'):
        import sys, torch
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from diag.diag_modules import load_model, to_tensor, DEV
        self.torch, self.tt, self.dev = torch, to_tensor, DEV
        self.bb = load_model(ckpt).backbone

    def _feat(self, im):
        with self.torch.no_grad():
            f = self.bb(self.tt(im).unsqueeze(0).to(self.dev))[0]          # [C, h, w]
        C, h, w = f.shape; st = im.shape[0] / h
        yy, xx = np.mgrid[0:h, 0:w]
        pts = np.stack([xx.ravel() * st + st / 2, yy.ravel() * st + st / 2], 1).astype(np.float32)
        return f.reshape(C, -1).T, pts

    def __call__(self, im0, im1):
        F = self.torch.nn.functional
        d0, p0 = self._feat(im0); d1, p1 = self._feat(im1)
        valid = im0[np.clip(p0[:, 1].astype(int), 0, im0.shape[0] - 1), np.clip(p0[:, 0].astype(int), 0, im0.shape[1] - 1)].sum(1) > 0
        d0, p0 = d0[self.torch.from_numpy(valid)], p0[valid]
        mu = d1.mean(0, keepdim=True)
        d0, d1 = F.normalize(d0 - mu, dim=1), F.normalize(d1 - mu, dim=1)
        S = d0 @ d1.T
        j = S.argmax(1); i = S.argmax(0)
        mutual = (i[j] == self.torch.arange(len(j), device=j.device)).cpu().numpy()
        j = j.cpu().numpy()
        return magsac(p0[mutual], p1[j[mutual]])


class VisMatch:
    def __init__(self, name, device):
        import torch
        from vismatch import get_matcher
        self.name, self.torch = name, torch
        self.m = get_matcher(name, device=device)
        if name == 'romav2':
            # default 'precise' upsamples to 800 / 1280 px both ways under forced bf16 autocast: on a T4 (no bf16)
            # it ran out of memory and stalled. Our images are 256 / 512 px, so use RoMa v2's 'base' preset
            # (640 px, one direction) in fp32.
            from vismatch.im_models import romav2 as rv2
            rv2._disable_autocast()
            self.m.romav2_model.apply_setting(os.environ.get('ROMAV2_SETTING', 'base'))
            self.name = f"romav2-{os.environ.get('ROMAV2_SETTING', 'base')}"

    def __call__(self, im0, im1):
        t = lambda im: self.torch.from_numpy(im).permute(2, 0, 1).float() / 255.
        r = self.m(t(im0), t(im1))
        p0, p1 = r['matched_kpts0'], r['matched_kpts1']
        # recompute the inlier mask on the same points (vismatch returns inlier points, not a mask)
        return magsac(np.float32(p0).reshape(-1, 2), np.float32(p1).reshape(-1, 2))


def run_one(matcher, uav, block, gt, scale, angles, px_tol=8.0):
    """Returns the per-angle records and the chosen (most inliers) one. Coordinates in base (512 block) pixels."""
    hu, wu = uav.shape[:2]; cu = np.array([wu / 2, hu / 2])
    U = cv2.resize(uav, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else uav
    B = cv2.resize(block, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else block
    xn, yn, th = gt; C = np.array([256 + xn * 128, 256 + yn * 128]); Rg = rot(th)
    recs = []
    for a in angles:
        t0 = time.time()
        p0, p1, H, inl = matcher(turn(U, a), B)
        dt = time.time() - t0
        p0 = np.asarray(p0, np.float64) / scale; p1 = np.asarray(p1, np.float64) / scale
        # back to original UAV px: p_turned = cu + R(a) (p - cu)  ->  p = cu + R(a)^T (p_turned - cu)
        po = (p0 - cu) @ rot(a) + cu if len(p0) else p0
        good = np.linalg.norm(C + (po - cu) @ Rg.T - p1, axis=1) < px_tol if len(p0) else np.zeros(0, bool)
        r = dict(angle=a, n_match=int(len(p0)), n_inl=int(inl.sum()), sec=dt,
                 good_match=float(good.mean()) if len(good) else np.nan,
                 good_inl=float(good[inl].mean()) if inl.sum() else np.nan)
        if H is not None and inl.sum() >= 4:
            S = np.diag([scale, scale, 1.]); Hb = np.linalg.inv(S) @ H @ S        # turned UAV (base px) -> block (base px)
            h = Hb @ np.array([cu[0], cu[1], 1.]); c = h[:2] / h[2]               # turning keeps the centre
            L = (Hb[:2, :2] - np.outer(c, Hb[2, :2])) / h[2] @ rot(a)               # local Jacobian at the centre
            r.update(px=float((c[0] - 256) / 128), py=float((c[1] - 256) / 128),
                     head=float(np.degrees(np.arctan2(L[0, 1], L[0, 0]))), scale_est=float(np.sqrt(abs(np.linalg.det(L)))))
        recs.append(r)
    ok = [r for r in recs if 'px' in r]
    best = max(ok, key=lambda r: r['n_inl']) if ok else None
    return recs, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--matchers', default='cv-sift'); ap.add_argument('--scales', default='1')
    ap.add_argument('--angles', type=int, default=8); ap.add_argument('--sets', default='')
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--kmin', type=int, default=0); ap.add_argument('--device', default='cuda')
    ap.add_argument('--ransac-px', type=float, default=3.0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    global RANSAC_PX; RANSAC_PX = a.ransac_px
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.data, 'samples.csv'))
    if a.sets: df = df[df.set.isin(a.sets.split(','))]
    if a.n: df = df[df.k < a.n]
    if a.kmin: df = df[df.k >= a.kmin]
    angles = list(np.arange(0, 360, 360 / a.angles))
    summ = []
    for mname in a.matchers.split(','):
        try:
            matcher = CvSift() if mname == 'cv-sift' else VggMNN() if mname == 'vgg-mnn' else VisMatch(mname, a.device)
        except Exception as e:
            print('LOAD FAIL', mname, repr(e), flush=True); summ.append(dict(matcher=mname, error=repr(e)[:300])); continue
        for scale in [float(s) for s in a.scales.split(',')]:
            rows, per_angle = [], []
            for s in df.itertuples():
                d = os.path.join(a.data, s.set)
                rd = lambda f: cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2RGB)
                try:
                    recs, best = run_one(matcher, rd(f'k{s.k:03d}_uav.jpg'), rd(f'k{s.k:03d}_block.jpg'),
                                         (s.x_norm, s.y_norm, s.theta), scale, angles)
                except Exception as e:
                    print('RUN FAIL', mname, scale, s.set, s.k, repr(e)[:200], flush=True); recs, best = [], None
                for r in recs: per_angle.append(dict(matcher=mname, scale=scale, set=s.set, k=s.k, **r))
                row = dict(matcher=mname, scale=scale, set=s.set, k=s.k, gt_x=s.x_norm, gt_y=s.y_norm, gt_theta=s.theta,
                           sec=sum(r['sec'] for r in recs))
                if best:
                    row.update(pred_x=best['px'], pred_y=best['py'], pred_theta=best['head'], angle=best['angle'],
                               n_inl=best['n_inl'], n_match=best['n_match'], good_match=best['good_match'],
                               good_inl=best['good_inl'], scale_est=best['scale_est'])
                rows.append(row)
                if len(rows) % 20 == 0:
                    print(f'  {mname} s{scale:g} {len(rows)}/{len(df)} samples, {row["sec"]:.2f} s/sample', flush=True)
            R = pd.DataFrame(rows)
            if 'pred_x' not in R: R['pred_x'] = np.nan; R['pred_y'] = np.nan; R['pred_theta'] = np.nan; R['n_inl'] = 0
            # no estimate -> predict the block centre (same fallback as a blind guess)
            px, py = R.pred_x.fillna(0), R.pred_y.fillna(0)
            R['err_m'] = np.hypot(px - R.gt_x, py - R.gt_y) * 128 * M_PER_PX
            R['head_err'] = wrap(R.pred_theta.fillna(0) - R.gt_theta)
            tag = f'{mname}{a.tag}_s{scale:g}'
            R.to_csv(os.path.join(a.out, f'{tag}.csv'), index=False)
            pd.DataFrame(per_angle).to_csv(os.path.join(a.out, f'{tag}_angles.csv'), index=False)
            for st, g in R.groupby('set'):
                summ.append(dict(matcher=mname, scale=scale, set=st, n=len(g), found=round(float(g.pred_x.notna().mean() * 100), 1),
                                 MLE=round(float(g.err_m.mean()), 2), medLE=round(float(g.err_m.median()), 2),
                                 LSR10=round(float((g.err_m < 10).mean() * 100), 1), MHE=round(float(g.head_err.mean()), 2),
                                 medHE=round(float(g.head_err.median()), 2), inl_med=float(g.n_inl.fillna(0).median()),
                                 good_inl=round(float(g.get('good_inl', pd.Series(dtype=float)).mean() * 100), 1),
                                 good_match=round(float(g.get('good_match', pd.Series(dtype=float)).mean() * 100), 1),
                                 sec_per_sample=round(float(g.sec.mean()), 3)))
                print(summ[-1], flush=True)
            pd.DataFrame(summ).to_csv(os.path.join(a.out, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
