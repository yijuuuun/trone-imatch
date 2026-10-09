# CSMG 대체 평가: 닫힌 루프 항법 (NY 주간 / NY L2 / Tokyo 주간), 32궤적, 시드 2026. 사용: python3 eval_swap.py v1 v2 ...
import sys
from swap_lib import *
from diag.diag_modules import UNSEEN, SEEN, darken
from diag.diag_nav import load_rsi
from diag.diag_phase0 import make_traj, nav_run, wilson, mcnemar, ModelLoc

OUT = f'{OUTD}/eval.txt'
def log(s):
    print(s, flush=True); open(OUT, 'a').write(s + '\n')

class DarkWrap:
    def __init__(self, loc, lvl, seed=0): self.loc, self.lvl, self.rng = loc, lvl, np.random.default_rng(seed)
    def __call__(self, crops, headings, blocks):
        return self.loc([darken(c, self.lvl, self.rng) for c in crops], headings, blocks)

imgs = {'NY': load_rsi(UNSEEN['NY']), 'Tokyo': load_rsi(SEEN['34bc'])}
starts, goals = make_traj(32, 1000, 2026)
conds = [('NY', 'day'), ('NY', 'L2'), ('Tokyo', 'day')]

pub = load_model(WEIGHTS)
# 검증: SwapLoc(공개 가중치) == ModelLoc
a = SwapLoc(pub, imgs['NY'], False); b = ModelLoc(pub, imgs['NY'])
from diag.diag_phase0 import Cropper
cr = Cropper(imgs['NY']); cs = [cr(s[0], s[1], 30.) for s in starts[:8]]; bl = [(5, 5)] * 8
log(f'# sanity: SwapLoc vs ModelLoc 최대 차이 {np.abs(a(cs, [30.]*8, bl) - b(cs, [30.]*8, bl)).max():.2e}')

res = {}
def run(name, m, north):
    for city, cond in conds:
        loc = SwapLoc(m, imgs[city], north)
        L = loc if cond == 'day' else DarkWrap(loc, cond, 0)
        s, _, err = nav_run(L, imgs[city], starts, goals, decode_px=128)
        res[(name, city, cond)] = s
        extra = ''
        for ref in ['공개 가중치', 'csmg']:
            if ref != name and (ref, city, cond) in res:
                x, y, p = mcnemar(s, res[(ref, city, cond)]); extra += f'\tvs {ref} +{x}/-{y} p={p:.3g}'
        log(f'{name}\t{city} {cond}\t{s.sum()}/32 ({100*s.mean():.1f}%)\t{wilson(int(s.sum()), 32)}\t오차중앙 {np.median(err):.1f} m{extra}')

log('# 모델\t조건\t성공\tCI95\t위치오차 중앙값\tMcNemar')
run('공개 가중치', pub, False)
for v in sys.argv[1:]:
    m = build(v); m.load_state_dict(torch.load(f'{OUTD}/{v}.pth', map_location=DEV), strict=False); m.eval()
    run(v, m, is_north(v))
log('# done')
