# 1-c: does "low-light failure is mainly NOISE (not brightness)" survive a physically-based synthesis?
# Unprocess (Brooks et al. 2019 style) -> raw low-light capture (Poisson shot + Gaussian read noise) -> ISP.
# Three camera strategies for darkness: ISO gain (noise), long exposure (motion blur), auto-exposure (both).
import os, sys, json, argparse
import numpy as np, cv2, torch, torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from diag.diag_modules import DEV, M_PER_UNIT, SEEN, to_tensor, darken, load_model, head, gen_samples, gt_quadrant
from diag.diag_gluf import global_descs, dense_match, seen_3d
from diag.diag_nav import load_rsi

CAM2RGB = np.array([[1.80, -0.60, -0.20], [-0.25, 1.65, -0.40], [0.00, -0.60, 1.60]], np.float32)  # rows sum to 1
RGB2CAM = np.linalg.inv(CAM2RGB).astype(np.float32)
WB = np.array([2.0, 1.0, 1.7], np.float32)  # red, green, blue gains
FULL_WELL, READ_E, ADC_BITS = 6000.0, 3.0, 12
BASE_EXPOSURE_S, MAX_EXPOSURE_S, MAX_GAIN = 1 / 500, 1 / 30, 16.0
SPEED_MPS, GSD = 10.0, 0.25


SENSOR_SCALE = 2  # sensor has 2x the output resolution (camera demosaics, then downsamples)


def unprocess_linear(rgb8):
    """sRGB uint8 -> linear camera RGB at sensor resolution (scene light, before the colour filter array)."""
    rgb8 = cv2.resize(rgb8, None, fx=SENSOR_SCALE, fy=SENSOR_SCALE, interpolation=cv2.INTER_CUBIC)
    y = rgb8.astype(np.float32) / 255.
    x = 0.5 - np.sin(np.arcsin(np.clip(1 - 2 * y, -1, 1)) / 3)      # inverse smoothstep tone curve
    x = np.clip(x, 1e-8, 1) ** 2.2                                    # inverse gamma
    x = np.clip(x @ RGB2CAM.T, 0, 1)                                  # inverse colour correction
    return np.clip(x / WB, 0, 1)                                      # inverse white balance


def mosaic(x):
    H, W, _ = x.shape
    raw = np.empty((H, W), np.float32)                                # RGGB mosaic
    raw[0::2, 0::2] = x[0::2, 0::2, 0]; raw[0::2, 1::2] = x[0::2, 1::2, 1]
    raw[1::2, 0::2] = x[1::2, 0::2, 1]; raw[1::2, 1::2] = x[1::2, 1::2, 2]
    return raw


def isp(raw):
    r16 = (np.clip(raw, 0, 1) * 65535).astype(np.uint16)
    x = cv2.cvtColor(r16, cv2.COLOR_BayerBG2RGB).astype(np.float32) / 65535  # OpenCV names RGGB as BayerBG
    x = np.clip(x * WB, 0, 1)
    x = np.clip(x @ CAM2RGB.T, 0, 1)
    x = x ** (1 / 2.2)
    x = 3 * x ** 2 - 2 * x ** 3                                       # smoothstep tone curve
    x = cv2.resize(x, None, fx=1 / SENSOR_SCALE, fy=1 / SENSOR_SCALE, interpolation=cv2.INTER_AREA)
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def unprocess(rgb8):
    return mosaic(unprocess_linear(rgb8))


def motion_blur(img, length_px, angle_deg):
    """Applied to scene light (linear RGB) BEFORE the colour filter array - blurring the mosaic mixes R/G/B samples."""
    if length_px < 1:
        return img
    L = int(np.ceil(length_px)) | 1
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2 - .5, L / 2 - .5), angle_deg, 1), (L, L))
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def capture(rgb8, light, strategy, rng, noise=True):
    """light: fraction of base-exposure photons (1 = daylight base). Returns sRGB uint8 as the camera would output."""
    scene = unprocess_linear(rgb8)
    if strategy == 'iso':
        exp_gain, gain = 1.0, 1.0 / light
    elif strategy == 'exposure':
        exp_gain, gain = 1.0 / light, 1.0
    elif strategy == 'auto':
        exp_gain = min(1.0 / light, MAX_EXPOSURE_S / BASE_EXPOSURE_S)
        gain = min(1.0 / (light * exp_gain), MAX_GAIN)                # beyond this the image stays dark
    else:
        raise ValueError(strategy)
    blur_px = SPEED_MPS * BASE_EXPOSURE_S * exp_gain / GSD * SENSOR_SCALE  # in sensor pixels
    raw = mosaic(motion_blur(scene, blur_px, rng.uniform(0, 180)))
    electrons = raw * light * exp_gain * FULL_WELL
    if noise:
        electrons = rng.poisson(np.clip(electrons, 0, None)).astype(np.float32) + rng.normal(0, READ_E, raw.shape).astype(np.float32)
    levels = 2 ** ADC_BITS - 1
    digital = np.round(np.clip(electrons / FULL_WELL * gain, 0, 1) * levels) / levels  # analog gain then ADC
    return isp(digital)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


CONDS = [('clean', None), ('camera_only(light1,NOnoise)', ('iso', 1.0, False))] + \
        [(f'simple_{l}', ('simple', l)) for l in ['L1', 'L2', 'L3']] + \
        [(f'iso_1/{d}', ('iso', 1 / d, True)) for d in [4, 16, 64, 256]] + \
        [(f'iso_1/{d}_NOnoise', ('iso', 1 / d, False)) for d in [64, 256]] + \
        [(f'exposure_1/{d}(blur)', ('exposure', 1 / d, True)) for d in [16, 64, 256]] + \
        [('exposure_1/64(blur)_NOnoise', ('exposure', 1 / 64, False))] + \
        [(f'auto_1/{d}', ('auto', 1 / d, True)) for d in [16, 64, 256]] + \
        [('auto_1/256_NOnoise', ('auto', 1 / 256, False))]


def degrade(rgb, spec, rng):
    if spec is None:
        return rgb
    if spec[0] == 'simple':
        return darken(rgb, spec[1], rng)
    return capture(rgb, spec[1], spec[0], rng, noise=spec[2])


@torch.no_grad()
def evaluate(m, base, rng, only=None):
    """base: list of (tiles_rgb[4], uav_rgb, (xn,yn), theta). Light-weight: model head + CSMG retrieval + dense(known heading)."""
    R = {}
    conds = [c for c in CONDS if not only or any(o in c[0] for o in only)]
    gt = torch.tensor([s[2] for s in base], dtype=torch.float32, device=DEV); q = gt_quadrant(gt); gn = gt.cpu().numpy()
    th = np.array([s[3] for s in base])
    nb = []
    for i in range(0, len(base), 16):
        t = torch.stack([torch.stack([to_tensor(x) for x in s[0]]) for s in base[i:i + 16]]).to(DEV)
        nb.append(global_descs(m, t.flatten(0, 1))['CSMG(trained)'].view(t.size(0), 4, -1))
    nb = torch.cat(nb)
    for name, spec in conds:
        uavs = [degrade(s[1], spec, rng) for s in base]
        fi = torch.cat([global_descs(m, torch.stack([to_tensor(u) for u in uavs[i:i + 32]]).to(DEV))['CSMG(trained)']
                        for i in range(0, len(uavs), 32)])
        pos, dr, *_ = head(m, fi, nb)
        le = (pos - gt).norm(dim=1) * M_PER_UNIT
        a = torch.atan2(dr[:, 1], dr[:, 0]).cpu().numpy() - np.deg2rad(th)
        he = np.degrees(np.abs(np.arctan2(np.sin(a), np.cos(a))))
        cos = F.cosine_similarity(fi.unsqueeze(1), nb, -1)
        D = np.array([dense_match(m, s[0], u, [s[3]], False)[:2] for s, u in zip(base, uavs)])
        dle = np.linalg.norm(D - gn, axis=1) * M_PER_UNIT
        R[name] = dict(
            img_mean=round(float(np.mean([u.mean() for u in uavs])), 1),
            PSNR=round(float(np.mean([psnr(u, s[1]) for s, u in zip(base, uavs)])), 2),
            model_MLE=round(float(le.mean()), 2), model_MHE=round(float(he.mean()), 2),
            CSMG_retr=round(float((cos.argmax(1) == q).float().mean() * 100), 1),
            dense_retr=round(float((((D[:, 0] > 0) + 2 * (D[:, 1] > 0)) == q.cpu().numpy()).mean() * 100), 1),
            dense_MLE=round(float(dle.mean()), 2), dense_medLE=round(float(np.median(dle)), 2),
            dense_LSR10=round(float((dle < 10).mean() * 100), 1))
        print(name, json.dumps(R[name]), flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--out', default='diag/out')
    ap.add_argument('--save_examples', default='diag/out/lowlight_examples.jpg')
    ap.add_argument('--only', nargs='*', default=None, help='run only conditions whose name contains one of these')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    m2, m3 = load_model('Bearing_UAV/satellite_view'), load_model('Bearing_UAV/cross_view')
    base3d = seen_3d(a.n, rng)
    base2d = []
    for c in ['34bc', '38bc']:
        for tiles, tgt, xy, cs in gen_samples(load_rsi(SEEN[c]), a.n // 2, rng):
            base2d.append((tiles, tgt, xy, float(np.degrees(np.arctan2(cs[1], cs[0])))))
    # visual sanity sheet: one 3D UAV image under every condition
    ex = base3d[0][1]; r2 = np.random.default_rng(0)
    tiles_img = [cv2.putText(cv2.cvtColor(degrade(ex, s, r2), cv2.COLOR_RGB2BGR), n[:18], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                 for n, s in CONDS]
    while len(tiles_img) % 4:
        tiles_img.append(np.zeros_like(tiles_img[0]))
    cv2.imwrite(a.save_examples, np.vstack([np.hstack(tiles_img[i:i + 4]) for i in range(0, len(tiles_img), 4)]))
    R = {'config': dict(full_well_e=FULL_WELL, read_noise_e=READ_E, adc_bits=ADC_BITS, base_exposure_s=BASE_EXPOSURE_S,
                        max_exposure_s=MAX_EXPOSURE_S, max_gain=MAX_GAIN, speed_mps=SPEED_MPS)}
    print('=== 3D real UAV / cross-view model ===', flush=True)
    R['3d_crossview'] = evaluate(m3, base3d, rng, a.only)
    print('=== 2D satellite crop / satellite-view model ===', flush=True)
    R['2d_satview'] = evaluate(m2, base2d, rng, a.only)
    json.dump(R, open(f'{a.out}/diag_lowlight{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
