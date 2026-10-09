# 궤적 수를 늘린 비교: NY 96궤적 + 다른 낯선 도시(Paris, London) 32궤적. 사용: python3 eval_more.py [--l2] <tag> ... (기본 csmg grid)
# --l2: UAV 영상만 L2로 어둡게 (노이즈 시드 0, 이전 L2 평가와 같은 방식)
import sys
from swap_lib import *
from diag.diag_modules import UNSEEN
from diag.diag_nav import load_rsi
from diag.diag_phase0 import make_traj, nav_run, wilson, mcnemar
from diag.diag_modules import darken
L2 = '--l2' in sys.argv; TAGS = [a for a in sys.argv[1:] if not a.startswith('--')] or ['csmg', 'grid']

class DarkWrap:
    def __init__(self, loc, lvl, seed=0): self.loc, self.lvl, self.rng = loc, lvl, np.random.default_rng(seed)
    def __call__(self, crops, headings, blocks):
        return self.loc([darken(c, self.lvl, self.rng) for c in crops], headings, blocks)

OUT = f'{OUTD}/eval_more_' + ('L2_' if L2 else '') + '_'.join(TAGS) + '.txt'
def log(s):
    print(s, flush=True); open(OUT, 'a').write(s + '\n')
models = {'공개 가중치': load_model(WEIGHTS)}
for v in TAGS:
    m = build(v); m.load_state_dict(torch.load(f'{OUTD}/{v}.pth', map_location=DEV), strict=False); m.eval(); models[v] = m
open(OUT, 'w').close()
log(f'# 모델\t도시(궤적)\t성공\tCI95\tMcNemar\t조건={"L2" if L2 else "주간"}')
for city, n in [('NY', 96), ('Paris', 32), ('London', 32)]:
    img = load_rsi(UNSEEN[city]); starts, goals = make_traj(n, 1000, 2026); res = {}
    for name, m in models.items():
        loc = SwapLoc(m, img, is_north(name) if name in TAGS else False)
        s, _, _ = nav_run(DarkWrap(loc, 'L2', 0) if L2 else loc, img, starts, goals, decode_px=128); res[name] = s
        extra = ''.join(f'\tvs {r} +{a}/-{b} p={p:.3g}' for r in res if r != name for a, b, p in [mcnemar(s, res[r])])
        log(f'{name}\t{city}({n})\t{s.sum()}/{n} ({100*s.mean():.1f}%)\t{wilson(int(s.sum()), n)}{extra}')
log('# done')
