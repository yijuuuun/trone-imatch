# CSMG 대체 모듈 학습. 사용: python3 train_swap.py <tag> <epochs>   (tag 예: csmg, grid, grid8, grid_s1, csmg_s2)
# 세 변형 모두 같은 데이터(4개 학습 도시 90K), 같은 손실(SmoothL1, 위치 0.8 + 방향 0.2), 같은 옵티마이저, 같은 시간 예산.
# VGG16은 얼림. 증강은 하지 않음(세 변형 공통).
import sys, time, json
from swap_lib import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader

variant, epochs = sys.argv[1], float(sys.argv[2])
if epochs != 4 and not re.search(r'_e\d+', variant):
    variant = f'{variant}_e{int(epochs)}'  # 4 epoch이 아니면 저장 이름에 epoch을 붙여 기존 모델을 덮어쓰지 않게 함 (예: grid_e20)
north = is_north(variant); torch.manual_seed(parse_tag(variant)[2])  # 시드: 초기화 + 데이터 순서
dark_train = is_dark_train(variant)  # 학습 UAV 영상의 50%를 어둡게: simple = L1~L3 단순 합성, phys = 물리 카메라 모델
from diag.diag_modules import darken
os.makedirs(OUTD, exist_ok=True)
LOG = f'{OUTD}/train_{variant}.txt'; open(LOG, 'w').close()
def log(s):
    print(s, flush=True); open(LOG, 'a').write(s + '\n')

df = pd.read_csv('Bearing_UAV_90K/c4m_254k_96bc_b15_s100/metadata/metadata.csv')
tcols = ['p1_path', 'p2_path', 'p3_path', 'p4_path']
tiles = sorted(set(df[tcols].values.ravel())); tid = {p: i for i, p in enumerate(tiles)}
log(f'device={DEV} variant={variant} north={north} dark_train={dark_train} samples={len(df)} unique_tiles={len(tiles)} epochs={epochs}')

m = build(variant)
# 타일은 종류가 적으므로 VGG 특징을 미리 계산해 GPU에 둔다 (VGG는 얼려 있으니 결과 동일)
with torch.no_grad(), torch.autocast(DT, dtype=torch.float16):
    TF = torch.cat([m.backbone(torch.stack([to_tensor(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in tiles[i:i+16]]).to(DEV)).half()
                    for i in range(0, len(tiles), 16)])
log(f'tile feature cache {tuple(TF.shape)} {TF.numel()*2/1e9:.2f} GB')


class DS(Dataset):
    def __init__(self, d): self.d = d.reset_index(drop=True)
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        u = cv2.cvtColor(cv2.imread(r.target_path), cv2.COLOR_BGR2RGB)
        if north: u = unrotate(u, float(r.theta))
        if dark_train and np.random.rand() < 0.5:
            rng = np.random.default_rng()
            u = phys_dark(u, rng) if dark_train == 'phys' else darken(u, ['L1', 'L2', 'L3'][np.random.randint(3)], rng)
        return (to_tensor(u), torch.tensor([tid[r[c]] for c in tcols]),
                torch.tensor([r.x_norm, r.y_norm], dtype=torch.float32), torch.tensor([r.x_cosa, r.y_sina], dtype=torch.float32))

NW = 2 if (DT == 'cuda' and os.name != 'nt') else 0  # Windows는 worker가 스크립트를 다시 실행하므로 0
perm = np.random.default_rng(42).permutation(len(df)); nva = int(0.05 * len(df))
tr = DataLoader(DS(df.iloc[perm[nva:]]), 64, shuffle=True, num_workers=NW, drop_last=True, persistent_workers=NW > 0)
va = DataLoader(DS(df.iloc[perm[:nva]]), 128, num_workers=NW)

params = [p for n, p in m.named_parameters() if not n.startswith('backbone.')]
log(f'학습 파라미터: 전체 {sum(p.numel() for p in params)/1e6:.3f} M, 교체 모듈 {sum(p.numel() for p in m.csmg.parameters())/1e6:.3f} M')
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
crit = nn.SmoothL1Loss(); wpos, wdir = 0.8, (0.0 if north else 0.2)   # 북쪽 정렬 시 방향은 학습 불가 -> 위치만
# MPS는 GradScaler를 지원하지 않는다(float64 불가). MPS에서는 얼린 VGG만 fp16, 학습되는 부분은 fp32로 계산한다.
AMP_ALL = DT == 'cuda'
scaler = torch.amp.GradScaler(DT, enabled=AMP_ALL)
# 같은 epoch 수의 코사인 스케줄
total, step, t0 = None, 0, time.time()


def forward(x, ti):
    with torch.no_grad(), torch.autocast(DT, dtype=torch.float16):
        fu = m.backbone(x)
    B = x.size(0)
    fi = desc(m, fu.float())
    nb = desc(m, TF[ti.flatten()].float()).view(B, 4, -1)
    return head_train(m, fi, nb)


@torch.no_grad()
def evaluate():
    m.eval(); E = []
    for x, ti, xy, cs in va:
        with torch.autocast(DT, dtype=torch.float16, enabled=AMP_ALL):
            pos, _ = forward(x.to(DEV), ti.to(DEV))
        E.append((pos.float() - xy.to(DEV)).norm(dim=1) * 32)
    m.train(); m.backbone.eval(); E = torch.cat(E)
    return float(E.mean()), float(E.median())


m.train(); m.backbone.eval(); ep = 0; done = False
while not done:
    ep += 1; run = 0.
    for x, ti, xy, cs in tr:
        x, ti, xy, cs = x.to(DEV, non_blocking=True), ti.to(DEV), xy.to(DEV), cs.to(DEV)
        with torch.autocast(DT, dtype=torch.float16, enabled=AMP_ALL):
            pos, dr = forward(x, ti)
            loss = wpos * crit(pos.float(), xy) + (wdir * crit(dr.float(), cs) if wdir else 0.)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1; run += float(loss.detach())
        if step == 1:
            total = int(epochs * len(tr)); log(f'총 {total} 스텝 ({epochs} epoch)')
        if step == 50:
            log(f'{(time.time()-t0)/50*1000:.0f} ms/step')
        if total:
            for g in opt.param_groups: g['lr'] = 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + np.cos(np.pi * min(step / total, 1)))
        if step % 500 == 0:
            log(f'step {step} ep {ep} loss {run/500:.4f} {(time.time()-t0)/60:.1f}min'); run = 0.
        if total and step >= total:
            done = True; break
    mle, med = evaluate()
    log(f'epoch {ep} 끝 (step {step}) val 위치오차 평균 {mle:.2f} m, 중앙값 {med:.2f} m')
torch.save({k: v for k, v in m.state_dict().items() if not k.startswith('backbone.')}, f'{OUTD}/{variant}.pth')
log(f'# done {(time.time()-t0)/60:.1f}min')
