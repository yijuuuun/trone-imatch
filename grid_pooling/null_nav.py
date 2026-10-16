# 기준선 확인: 위치 추정기가 항상 "블록 중앙"(pos=0)을 내놓을 때 항법 성공률. 영상을 전혀 보지 않는다.
import numpy as np
from swap_lib import OUTD
from diag.diag_modules import UNSEEN
from diag.diag_nav import load_rsi
from diag.diag_phase0 import make_traj, nav_run, wilson
import collections
null = lambda crops, headings, blocks: np.zeros((len(crops), 2))
for city, n in [('NY', 96), ('Paris', 32), ('London', 32)]:
    img = load_rsi(UNSEEN[city]); starts, goals = make_traj(n, 1000, 2026)
    s, reason, med = nav_run(null, img, starts, goals, decode_px=128)
    print(city, n, int(s.sum()), f'{100*s.mean():.1f}%', wilson(int(s.sum()), n), dict(collections.Counter(reason)), flush=True)
