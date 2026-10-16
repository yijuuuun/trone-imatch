# 타일 맞히기 (학습 없음): 블록 2x2 타일 중 UAV 영상 중심이 있는 타일을 맞힌 비율 (우연 25%).
# 실험 5와 같은 방식(희윤님 diag_gluf.gen_rgb, 도시당 200샘플, rng 5)으로 Grid 모델까지 비교한다.
#   cos_raw   : UAV 영상 그대로 요약 벡터 코사인 유사도로 고름
#   cos_north : 나침반 방향으로 UAV 영상을 북쪽 위로 되돌린 뒤 고름 (+θ, −θ 둘 다 재서 규약 확인)
#   model     : Bearing 전체 모델(RCE→CA→PSG→헤드)이 예측한 위치의 사분면
# 사용: python tile_check.py csmg grid grid8 grid8_phys
import sys, json
import numpy as np, torch, torch.nn.functional as F
from swap_lib import build, unrotate, OUTD, DEV
from diag.diag_modules import SEEN, UNSEEN, load_model, to_tensor, head, gt_quadrant
from diag.diag_nav import load_rsi
from diag.diag_gluf import gen_rgb, dense_match

WEIGHTS = 'Bearing_UAV/satellite_view'
tags = sys.argv[1:]
models = {'공개 가중치': load_model(WEIGHTS)}
for t in tags:
    m = build(t); m.load_state_dict(torch.load(f'{OUTD}/{t}.pth', map_location=DEV), strict=False); m.eval(); models[t] = m
cities = {'Tokyo(학습)': SEEN['34bc'], 'NY': UNSEEN['NY'], 'Paris': UNSEEN['Paris'], 'London': UNSEEN['London']}


@torch.no_grad()
def desc(m, imgs):
    return torch.cat([m.csmg(m.backbone(torch.stack([to_tensor(x) for x in imgs[i:i + 16]]).to(DEV)))[2]
                      for i in range(0, len(imgs), 16)])


def acc(cos, q):
    return round(float((cos.argmax(1) == q).float().mean() * 100), 1)


out_path = f'{OUTD}/tile_check_{"_".join(tags)}.txt'
open(out_path, 'w').write('# 타일 맞히기 (%), 도시당 200샘플, rng 5. 우연 25%\n')
for city, pre in cities.items():
    S = gen_rgb(load_rsi(pre), 200, np.random.default_rng(5))
    gt = torch.tensor([s[2] for s in S], dtype=torch.float32, device=DEV); q = gt_quadrant(gt)
    tiles = [t for s in S for t in s[0]]
    uav = [s[1] for s in S]
    up = [unrotate(s[1], s[3]) for s in S]; um = [unrotate(s[1], -s[3]) for s in S]
    row = {}
    with torch.no_grad():
        for name, m in models.items():
            dt = desc(m, tiles).view(len(S), 4, -1)
            du, dp, dm = desc(m, uav), desc(m, up), desc(m, um)
            pos = torch.cat([head(m, du[i:i + 64], dt[i:i + 64])[0] for i in range(0, len(S), 64)])
            row[name] = dict(cos_raw=acc(F.cosine_similarity(du.unsqueeze(1), dt, -1), q),
                             cos_north_p=acc(F.cosine_similarity(dp.unsqueeze(1), dt, -1), q),
                             cos_north_m=acc(F.cosine_similarity(dm.unsqueeze(1), dt, -1), q),
                             model=acc(-(pos.unsqueeze(1) - torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]], device=DEV)).norm(dim=2), q))
        m0 = models['공개 가중치']
        D = np.array([dense_match(m0, s[0], s[1], [s[3]], True)[:2] for s in S])
        row['dense(나침반)'] = dict(model=round(float((((D[:, 0] > 0) + 2 * (D[:, 1] > 0)) == q.cpu().numpy()).mean() * 100), 1))
    line = f'{city} ' + json.dumps(row, ensure_ascii=False)
    print(line, flush=True); open(out_path, 'a').write(line + '\n')
open(out_path, 'a').write('# done\n')
