# 모델이 예측한 위치가 블록 중앙에서 얼마나 떨어져 있나 (|pos|), 실제 정답과의 오차 비교. NY 200샘플.
import numpy as np, torch
from swap_lib import build, OUTD, DEV
from diag.diag_modules import UNSEEN, load_model, to_tensor, head, M_PER_UNIT
from diag.diag_nav import load_rsi
from diag.diag_gluf import gen_rgb
S = gen_rgb(load_rsi(UNSEEN['NY']), 200, np.random.default_rng(5))
gt = torch.tensor([s[2] for s in S], dtype=torch.float32, device=DEV)
ms = {'공개 가중치': load_model('Bearing_UAV/satellite_view')}
for t in ['csmg', 'grid', 'grid8', 'grid8_phys', 'grid1']:
    m = build(t); m.load_state_dict(torch.load(f'{OUTD}/{t}.pth', map_location=DEV), strict=False); m.eval(); ms[t] = m
print('정답의 중앙 거리 평균 = 항상 중앙 오차 %.1f m' % (gt.norm(dim=1).mean() * M_PER_UNIT))
with torch.no_grad():
    for k, m in ms.items():
        d = lambda imgs: torch.cat([m.csmg(m.backbone(torch.stack([to_tensor(x) for x in imgs[i:i+16]]).to(DEV)))[2] for i in range(0, len(imgs), 16)])
        dt = d([t for s in S for t in s[0]]).view(len(S), 4, -1); du = d([s[1] for s in S])
        p = torch.cat([head(m, du[i:i+64], dt[i:i+64])[0] for i in range(0, len(S), 64)])
        print(f'{k}: 예측 중앙 거리 평균 {p.norm(dim=1).mean()*M_PER_UNIT:.1f} m, 오차 평균 {(p-gt).norm(dim=1).mean()*M_PER_UNIT:.1f} m', flush=True)
