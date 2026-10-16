# Module-level diagnosis of Bearing-UAV (inference only, no retraining).
#   D0 split leakage | D1 module interventions | D2 GLUF retrieval
#   D3 low-light decomposition | D4 unseen-city generalization | D5 latency
import os, sys, json, time, glob, argparse
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.getcwd())
from cvphr.models.posaglreg.models import load_config_and_model

DEV = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')  # 로컬 Mac용 수정
M_PER_UNIT = 128 * 0.25  # x_norm unit -> meters (UNI_PIXEL * GSD)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SEEN = {'34bc': '35.67091338738739_139.69289911300856', '36bc': '25.030947387387386_121.51462868800057',
        '37bc': '1.2897673873873876_103.84197619336068', '38bc': '37.75538738738739_-122.4533351740761'}
UNSEEN = {'NY': '40.754754987387386_-73.95458253812376', 'Paris': '48.85538738738739_2.319377605520954',
          'London': '51.49878638738739_-0.126930697644793', 'Birmingham': '52.472222222222_-1.904166666667',
          'Berlin': '52.51263958738739_13.394127089285984'}
DARK = {  # gain, gamma, shot-noise k, read-noise sigma
    'L1': (0.50, 1.3, 0.002, 0.005), 'L2': (0.25, 1.6, 0.004, 0.010), 'L3': (0.10, 2.0, 0.006, 0.015)}


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def imread_rgb(p):
    im = cv2.imread(p); assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def darken(rgb, lvl, rng, noise=True):
    g, gam, ks, sr = DARK[lvl]
    x = g * (rgb.astype(np.float32) / 255.) ** gam
    if noise:
        x = x + rng.normal(size=x.shape).astype(np.float32) * np.sqrt(x * ks + sr ** 2)
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)  # 8-bit quantization


def autogain(rgb, target_mean):
    m = rgb.reshape(-1, 3).mean(0) + 1e-3
    return np.clip(rgb.astype(np.float32) * (target_mean / m), 0, 255).astype(np.uint8)


def load_model(d):
    cls, kw = load_config_and_model(d)
    m = cls(**kw)
    m.load_state_dict(torch.load(f'{d}/best_model.pth', map_location='cpu')['model_state_dict'])
    return m.to(DEV).eval()


@torch.no_grad()
def enc(m, x, stats=False):
    """GLUF = VGG16(conv4_3) + CSMG. Bypasses JointNet's per-sample numpy centroid loop (unused in forward)."""
    bb = m.backbone(x)
    sim, d, dflat, _ = m.csmg(bb)
    if not stats:
        return dflat
    assign = F.normalize(sim, p=1, dim=1).mean(-1)  # cluster usage [B,K]
    return dflat, bb, assign


@torch.no_grad()
def head(m, fi, nb, ctx_mode='normal', psg_mode='normal', ctx_mean=None):
    """RCE -> CA -> PSG -> regression, re-implemented to expose internals (checked == model.forward)."""
    B = fi.size(0)
    kc = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], device=fi.device)
    ce = m.coord_encoder(kc).unsqueeze(0).expand(B, -1, -1)
    psg = m.sim_pos_prior(fi, nb)
    nbk = nb + ce if m.add_patch_coord else nb
    ca = m.neighbors_cross_attn
    q = ca.q_proj(fi).unsqueeze(1); k = ca.k_proj(nbk); v = ca.v_proj(nbk)
    attn = torch.softmax(q @ k.transpose(1, 2) / np.sqrt(q.size(-1)), -1)
    ctx = (attn @ v).squeeze(1)
    if ctx_mode == 'mean':
        ctx = ctx_mean.expand(B, -1)
    if psg_mode == 'zero':
        psg = torch.zeros_like(psg)
    comb = torch.cat([fi, ctx], 1)
    pos = m.pos_regressor(torch.cat([comb, psg], 1)); dr = m.dir_regressor(comb)
    return pos, dr, attn.squeeze(1), psg, ctx, ce


def errs(pos, dr, gt_xy, gt_cs):
    le = (pos - gt_xy).norm(dim=1) * M_PER_UNIT
    a = torch.atan2(dr[:, 1], dr[:, 0]) - torch.atan2(gt_cs[:, 1], gt_cs[:, 0])
    he = torch.rad2deg(torch.atan2(torch.sin(a), torch.cos(a)).abs())
    return le, he


def summ(le, he):
    le, he = le.cpu().numpy(), he.cpu().numpy()
    return dict(MLE=round(float(le.mean()), 2), medLE=round(float(np.median(le)), 2),
                LSR10=round(float((le < 10).mean() * 100), 1), MHE=round(float(he.mean()), 2),
                medHE=round(float(np.median(he)), 2))


def gt_quadrant(xy):  # slot order p1..p4 = base_00(TL), base_10(TR), base_01(BL), base_11(BR); y_norm points down
    return ((xy[:, 0] > 0).long() + 2 * (xy[:, 1] > 0).long())


# ---------------------------------------------------------------- seen-city test split
class SeenSet(Dataset):
    def __init__(self, df, uav_mean):
        self.df, self.uav_mean = df.reset_index(drop=True), uav_mean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]; rng = np.random.default_rng(1000 + i)
        tiles = torch.stack([to_tensor(imread_rgb(r[c])) for c in ['p1_path', 'p2_path', 'p3_path', 'p4_path']])
        uav = imread_rgb(r['target_path'])
        sat = imread_rgb(r['target_path'].replace('/uav_', '/sat_').replace('_v3d.jpg', '.jpg'))
        v = {'normal': uav}
        for l in DARK:
            v[l] = darken(uav, l, rng)
        v['L3_nonoise'] = darken(uav, 'L3', rng, noise=False)
        v['L3_autogain'] = autogain(v['L3'], self.uav_mean)
        v['L3_nonoise_autogain'] = autogain(v['L3_nonoise'], self.uav_mean)
        out = {k: to_tensor(im) for k, im in v.items()}
        out['sat2d'] = to_tensor(sat)
        out['tiles'] = tiles
        out['xy'] = torch.tensor([r['x_norm'], r['y_norm']], dtype=torch.float32)
        out['cs'] = torch.tensor([r['x_cosa'], r['y_sina']], dtype=torch.float32)
        out['city'] = r['target_path'].split('/')[2]
        out['blk'] = int(r['block_x']) * 100 + int(r['block_y'])
        return out


# ---------------------------------------------------------------- generated samples from RSI (2D sat-view)
def crop_rot(img, cx, cy, theta):
    """Same as naver.runners.visnav.crop_target_patch, on a local window for speed."""
    W = 384; x0, y0 = int(cx) - W, int(cy) - W
    pad = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_REFLECT)
    win = pad[y0 + W:y0 + 3 * W, x0 + W:x0 + 3 * W]
    lx, ly = cx - x0, cy - y0
    R = cv2.getRotationMatrix2D((lx, ly), -theta, 1)
    rot = cv2.warpAffine(win, R, (win.shape[1], win.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    c_x, c_y = int(lx), int(ly)
    return rot[c_y - 128:c_y + 128, c_x - 128:c_x + 128]


def gen_samples(rsi_img, n, rng, n_block=15):
    S = []
    for _ in range(n):
        bx, by = rng.integers(0, n_block, 2)
        xn, yn = rng.uniform(-1, 1, 2); th = rng.uniform(-180, 180)
        sx, sy = bx * 256, by * 256
        tiles = [rsi_img[sy + dy * 256:sy + (dy + 1) * 256, sx + dx * 256:sx + (dx + 1) * 256]
                 for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]]
        cx, cy = sx + 256 + xn * 128, sy + 256 + yn * 128
        tgt = crop_rot(rsi_img, cx, cy, th)
        S.append((tiles, tgt, (xn, yn), (np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)))))
    return S


@torch.no_grad()
def eval_generated(m, S, bs=32):
    P, D, G, C, ACC = [], [], [], [], []
    for i in range(0, len(S), bs):
        ch = S[i:i + bs]
        tiles = torch.stack([torch.stack([to_tensor(t) for t in s[0]]) for s in ch]).to(DEV)
        tgt = torch.stack([to_tensor(s[1]) for s in ch]).to(DEV)
        B = tiles.size(0)
        nb = enc(m, tiles.flatten(0, 1)).view(B, 4, -1); fi = enc(m, tgt)
        pos, dr, *_ = head(m, fi, nb)
        xy = torch.tensor([s[2] for s in ch], dtype=torch.float32, device=DEV)
        P.append(pos); D.append(dr); G.append(xy)
        C.append(torch.tensor([s[3] for s in ch], dtype=torch.float32, device=DEV))
        ACC.append(((F.cosine_similarity(fi.unsqueeze(1), nb, -1).argmax(1)) == gt_quadrant(xy)).float())
    P, D, G, C, ACC = map(torch.cat, (P, D, G, C, ACC))
    le, he = errs(P, D, G, C)
    s = summ(le, he); s['retr_acc'] = round(float(ACC.mean() * 100), 1)
    s['pred_std'] = round(float(P.std(0).mean()), 3)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--n_gen', type=int, default=300)
    ap.add_argument('--out', default='diag/out')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = {}
    torch.manual_seed(0)

    # ---------------- D0: split leakage (same split as par_dataloader, seed 42)
    df = pd.read_csv('Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv')
    n = len(df); ntr = int(.85 * n); nva = int(.05 * n)
    tr, va, te = random_split(range(n), [ntr, nva, n - ntr - nva], generator=torch.Generator().manual_seed(42))
    tr, te = np.array(tr.indices), np.array(te.indices)
    key = df['p1_path'].str.split('/').str[2] + '_' + df['block_x'].astype(str) + '_' + df['block_y'].astype(str)
    tr_blocks = set(key.iloc[tr]); tr_tiles = set(pd.concat([df.iloc[tr][c] for c in ['p1_path', 'p2_path', 'p3_path', 'p4_path']]))
    te_tiles = pd.concat([df.iloc[te][c] for c in ['p1_path', 'p2_path', 'p3_path', 'p4_path']])
    # nearest train sample (same block) distance in meters
    trdf = df.iloc[tr]; g = trdf.groupby(key.iloc[tr])
    cent = {k: v[['x_norm', 'y_norm']].values for k, v in g}
    nn_d = [np.min(np.linalg.norm(cent[key.iloc[i]] - df.iloc[i][['x_norm', 'y_norm']].values.astype(float), axis=1)) * M_PER_UNIT
            for i in te[:2000]]
    R['D0_leakage'] = dict(n_train=len(tr), n_test=len(te),
                           test_blocks_seen_in_train_pct=round(100 * key.iloc[te].isin(tr_blocks).mean(), 2),
                           test_tiles_seen_in_train_pct=round(100 * te_tiles.isin(tr_tiles).mean(), 2),
                           nearest_train_uav_same_block_m_median=round(float(np.median(nn_d)), 2),
                           nearest_train_uav_same_block_m_p90=round(float(np.percentile(nn_d, 90)), 2))
    print(json.dumps(R['D0_leakage'], indent=1), flush=True)

    # ---------------- load models
    m3 = load_model('Bearing_UAV/cross_view'); m2 = load_model('Bearing_UAV/satellite_view')

    # ---------------- seen-city test subset: extract features once
    rng = np.random.default_rng(0)
    sub = df.iloc[rng.choice(te, a.n, replace=False)]
    uav_mean = np.mean([imread_rgb(p).reshape(-1, 3).mean(0) for p in sub['target_path'].iloc[:200]], 0)
    dl = DataLoader(SeenSet(sub, uav_mean), batch_size=24, num_workers=8)
    conds = ['normal', 'L1', 'L2', 'L3', 'L3_nonoise', 'L3_autogain', 'L3_nonoise_autogain']
    F3 = {c: [] for c in conds + ['sat2d']}; NB3, NB2, F2sat = [], [], []
    XY, CS, CITY, BLK = [], [], [], []
    st = {c: dict(bb_cos=[], bb_mean=[], bb_zero=[], assign=[]) for c in conds}
    chk = None
    for bi, b in enumerate(dl):
        B = b['tiles'].size(0); tiles = b['tiles'].to(DEV).flatten(0, 1)
        NB3.append(enc(m3, tiles).view(B, 4, -1)); NB2.append(enc(m2, tiles).view(B, 4, -1))
        F2sat.append(enc(m2, b['sat2d'].to(DEV))); F3['sat2d'].append(enc(m3, b['sat2d'].to(DEV)))
        bb_ref = None
        for c in conds:
            fi, bb, asg = enc(m3, b[c].to(DEV), stats=True); F3[c].append(fi)
            if c == 'normal':
                bb_ref = bb
            st[c]['bb_cos'].append(F.cosine_similarity(bb.flatten(1), bb_ref.flatten(1), 1))
            st[c]['bb_mean'].append(bb.mean((1, 2, 3))); st[c]['bb_zero'].append((bb <= 0).float().mean((1, 2, 3)))
            st[c]['assign'].append(asg)
        XY.append(b['xy']); CS.append(b['cs']); CITY += list(b['city']); BLK.append(b['blk'])
        if bi == 0:  # sanity: re-implemented head == model.forward
            with torch.no_grad():
                p_ref, d_ref = m3(torch.cat([b['tiles'], b['normal'].unsqueeze(1)], 1)[:4].to(DEV))
            p_me, d_me, *_ = head(m3, F3['normal'][0][:4], NB3[0][:4])
            chk = float((p_ref - p_me).abs().max() + (d_ref - d_me).abs().max())
        print(f'batch {bi + 1}/{len(dl)}', end='\r', flush=True)
    cat = lambda L: torch.cat(L)
    F3 = {k: cat(v) for k, v in F3.items()}; NB3, NB2, F2sat = cat(NB3), cat(NB2), cat(F2sat)
    XY, CS, BLK = cat(XY).to(DEV), cat(CS).to(DEV), cat(BLK).numpy(); CITY = np.array(CITY)
    R['sanity_head_vs_forward_maxabs'] = chk
    fi, nb = F3['normal'], NB3

    # ---------------- D1: module interventions (cross-view model, normal light)
    pos, dr, attn, psg, ctx, ce = head(m3, fi, nb)
    base = summ(*errs(pos, dr, XY, CS))
    ctx_mean = ctx.mean(0, keepdim=True)
    # wrong neighbours: same city, different block
    perm = np.arange(len(fi))
    for c in np.unique(CITY):
        idx = np.where(CITY == c)[0]
        for _ in range(20):
            p = rng.permutation(idx)
            if (BLK[p] != BLK[idx]).all():
                break
        perm[idx] = p
    perm_t = torch.from_numpy(perm).to(DEV)
    perm_x = torch.from_numpy(rng.permutation(len(fi))).to(DEV)
    I = {}
    I['baseline'] = base
    pw, dw, *_ = head(m3, fi, nb[perm_t]); I['neighbors_wrong_block_same_city'] = summ(*errs(pw, dw, XY, CS))
    pz, dz, *_ = head(m3, fi, torch.zeros_like(nb)); I['neighbors_zero'] = summ(*errs(pz, dz, XY, CS))
    pf, df_, *_ = head(m3, fi[perm_x], nb); I['uav_wrong_neighbors_right'] = summ(*errs(pf, df_, XY, CS))
    pr, drr, *_ = head(m3, fi, nb[:, [3, 2, 1, 0]])  # 180deg layout flip: matching model must flip quadrant
    I['neighbors_slot_flip180'] = summ(*errs(pr, drr, XY, CS))
    I['neighbors_slot_flip180']['pred_follows_gt_pct'] = round(float(((pr * XY).sum(1) > 0).float().mean() * 100), 1)
    I['neighbors_slot_flip180']['pred_follows_flip_pct'] = round(float(((pr * -XY).sum(1) > 0).float().mean() * 100), 1)
    pc, dc, *_ = head(m3, fi, nb, ctx_mode='mean', ctx_mean=ctx_mean); I['CA_ctx_replaced_by_mean'] = summ(*errs(pc, dc, XY, CS))
    pp, dp, *_ = head(m3, fi, nb, psg_mode='zero'); I['PSG_zeroed'] = summ(*errs(pp, dp, XY, CS))
    I['delta_pred_wrong_vs_base_m'] = round(float((pw - pos).norm(dim=1).mean() * M_PER_UNIT), 2)
    R['D1_interventions_crossview'] = I

    # internals
    q = gt_quadrant(XY)
    ent = -(attn * attn.clamp_min(1e-9).log()).sum(1)
    psg_np, xy_np = psg.cpu().numpy(), XY.cpu().numpy()
    R['D1_internals'] = dict(
        feat_norm=round(float(nb.norm(dim=-1).mean()), 3),
        coord_emb_norm=[round(float(x), 3) for x in ce[0].norm(dim=-1)],
        coord_emb_pairwise_cos=round(float(F.cosine_similarity(ce[0].unsqueeze(1), ce[0].unsqueeze(0), -1).triu(1).sum() / 6), 3),
        CA_attn_entropy_mean=round(float(ent.mean()), 3), CA_attn_entropy_max_possible=round(float(np.log(4)), 3),
        CA_attn_max_weight_mean=round(float(attn.max(1).values.mean()), 3),
        CA_attn_argmax_eq_gt_quadrant_pct=round(float((attn.argmax(1) == q).float().mean() * 100), 1),
        CA_attn_argmax_hist=np.bincount(attn.argmax(1).cpu().numpy(), minlength=4).tolist(),
        CA_ctx_between_sample_std_over_mean_abs=round(float(ctx.std(0).mean() / ctx.abs().mean()), 4),
        PSG_abs_mean=round(float(np.abs(psg_np).mean()), 4), PSG_std=round(float(psg_np.std(0).mean()), 4),
        PSG_corr=dict(psg0_x=round(float(np.corrcoef(psg_np[:, 0], xy_np[:, 0])[0, 1]), 3),
                      psg0_y=round(float(np.corrcoef(psg_np[:, 0], xy_np[:, 1])[0, 1]), 3),
                      psg1_x=round(float(np.corrcoef(psg_np[:, 1], xy_np[:, 0])[0, 1]), 3),
                      psg1_y=round(float(np.corrcoef(psg_np[:, 1], xy_np[:, 1])[0, 1]), 3)),
        cos_uav_to_tiles_mean=round(float(F.cosine_similarity(fi.unsqueeze(1), nb, -1).mean()), 3),
        cos_uav_to_tiles_spread_mean=round(float(F.cosine_similarity(fi.unsqueeze(1), nb, -1).std(1).mean()), 4),
    )
    # first-layer weight share: how much of regressor input is driven by fi vs ctx vs psg
    W = m3.pos_regressor[0].weight.detach()
    R['D1_internals']['pos_reg_L1_weight_norm'] = dict(fi=round(float(W[:, :1024].norm()), 2),
                                                        ctx=round(float(W[:, 1024:2048].norm()), 2),
                                                        psg=round(float(W[:, 2048:].norm()), 3))

    # nav.py decoding uses block_size//2=256 px/unit, labels use UNI_PIXEL=128 px/unit
    R['D1_nav_decode_bug'] = dict(MLE_correct_scale=base['MLE'],
                                  MLE_as_decoded_in_nav=round(float((2 * pos - XY).norm(dim=1).mean() * M_PER_UNIT), 2))

    # ---------------- D2: GLUF retrieval (which tile contains the UAV centre?)
    def retr(f, nbx):
        s = F.cosine_similarity(f.unsqueeze(1), nbx, -1)
        return round(float((s.argmax(1) == q).float().mean() * 100), 1)
    R['D2_GLUF_retrieval_top1_pct'] = dict(chance=25.0, crossview_model_uav3d=retr(fi, nb),
                                            crossview_model_sat2d=retr(F3['sat2d'], nb),
                                            satview_model_sat2d=retr(F2sat, NB2))
    p2, d2, *_ = head(m2, F2sat, NB2)
    R['D2_satview_model_on_sat2d'] = summ(*errs(p2, d2, XY, CS))
    p3s, d3s, *_ = head(m3, F3['sat2d'], nb)
    R['D2_crossview_model_on_sat2d'] = summ(*errs(p3s, d3s, XY, CS))

    # ---------------- D3: low light
    L = {}
    for c in conds:
        pc_, dc_, at_, *_ = head(m3, F3[c], nb)
        s = summ(*errs(pc_, dc_, XY, CS))
        s['retr_acc'] = retr(F3[c], nb)
        s['desc_cos_to_normal'] = round(float(F.cosine_similarity(F3[c], fi, 1).mean()), 3)
        s['bb_map_cos_to_normal'] = round(float(cat(st[c]['bb_cos']).mean()), 3)
        s['bb_mean_act'] = round(float(cat(st[c]['bb_mean']).mean()), 3)
        s['bb_zero_frac'] = round(float(cat(st[c]['bb_zero']).mean()), 3)
        s['cluster_usage'] = [round(float(x), 3) for x in cat(st[c]['assign']).mean(0)]
        s['pred_shift_from_normal_m'] = round(float((pc_ - pos).norm(dim=1).mean() * M_PER_UNIT), 2)
        L[c] = s
    R['D3_lowlight_crossview'] = L

    # ---------------- D4: unseen cities (generated 2D samples; seen cities generated the same way as control)
    rng2 = np.random.default_rng(7)
    G = {}
    # validate generator against dataset files (seen city)
    r0 = df[df['target_path'].str.contains('citya')].iloc[0]
    rsi = cv2.cvtColor(cv2.imread(glob.glob(f"Bearing_UAV_90K/city_rsi/{SEEN['34bc']}*_city.jpg")[0]), cv2.COLOR_BGR2RGB)
    sx, sy = int(r0.block_x) * 256, int(r0.block_y) * 256
    t_gen = rsi[sy:sy + 256, sx:sx + 256]; t_file = imread_rgb(r0.p1_path)
    s_gen = crop_rot(rsi, sx + 256 + r0.x_norm * 128, sy + 256 + r0.y_norm * 128, r0.theta)
    s_file = imread_rgb(r0.target_path.replace('/uav_', '/sat_').replace('_v3d.jpg', '.jpg'))
    G['generator_check_mean_absdiff'] = dict(tile=round(float(np.abs(t_gen.astype(int) - t_file).mean()), 2),
                                             target=round(float(np.abs(s_gen.astype(int) - s_file).mean()), 2))
    for grp, D in [('seen', SEEN), ('unseen', UNSEEN)]:
        for name, pre in D.items():
            f = [p for p in glob.glob(f'Bearing_UAV_90K/city_rsi/{pre}*_city.jpg') if '6144' not in p][0]
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            S = gen_samples(img, a.n_gen, rng2)
            G[f'{grp}:{name}'] = dict(satview_model=eval_generated(m2, S), crossview_model=eval_generated(m3, S))
            print(name, G[f'{grp}:{name}'], flush=True)
    R['D4_generated_2d'] = G

    # ---------------- D5: latency (batch 1, 5 patches)
    x = torch.randn(1, 5, 3, 256, 256, device=DEV)
    def tm(fn, it=20):
        for _ in range(3): fn()
        if DEV == 'cuda': torch.cuda.synchronize()
        t = time.time()
        for _ in range(it): fn()
        if DEV == 'cuda': torch.cuda.synchronize()
        return round((time.time() - t) / it * 1000, 2)
    with torch.no_grad():
        xb = x[0]
        bbo = m3.backbone(xb)
        R['D5_latency_ms_gpu'] = dict(
            full_forward=tm(lambda: m3(x)),
            backbone_x5=tm(lambda: m3.backbone(xb)),
            csmg_x5=tm(lambda: m3.csmg(bbo)),
            jointnet_numpy_centroid_loop_x5=tm(lambda: m3.sgm(xb)) - tm(lambda: m3.csmg(m3.backbone(xb))),
            head=tm(lambda: head(m3, fi[:1], nb[:1])),
            uav_only_with_cached_tiles=tm(lambda: head(m3, enc(m3, xb[4:]), nb[:1])))
        R['params_M'] = dict(backbone=round(sum(p.numel() for p in m3.backbone.parameters()) / 1e6, 2),
                             csmg=round(sum(p.numel() for p in m3.csmg.parameters()) / 1e6, 2),
                             coord_enc=round(sum(p.numel() for p in m3.coord_encoder.parameters()) / 1e6, 2),
                             ca=round(sum(p.numel() for p in m3.neighbors_cross_attn.parameters()) / 1e6, 2),
                             heads=round(sum(p.numel() for p in list(m3.pos_regressor.parameters()) + list(m3.dir_regressor.parameters())) / 1e6, 2))

    json.dump(R, open(f'{a.out}/diag_results.json', 'w'), indent=1)
    print('\n' + json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
