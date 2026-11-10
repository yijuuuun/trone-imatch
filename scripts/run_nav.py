"""1 km 항법 실행: 도시마다 같은 궤적(make_traj, 시드 2026)으로 성공률을 잰다.

  python scripts/run_nav.py --sets 3d_ho37bc --frames 3d --matcher loma --angles 1 --capstone1 ../bearinguav --out outputs/nav
  python scripts/run_nav.py --sets 3d_ho38bc --frames 2d --matcher cv-sift --device cpu --n-traj 4 --out outputs/nav_check

--frames 3d: 가장 가까운 3D 렌더 (imatch/nav.py Nearest3D). 2d: 지도 크롭, 동작 확인용.
--oracle: 정답 위치로 블록을 고른다. 블록 선택을 고쳤을 때 도달할 수 있는 상한이다.
학습 도시(규칙을 정하는 곳): --data data/im_train_3d_500 --sets 3d_ho34bc,3d_ho36bc --n-traj 32 --seed 7
출력 (--out 폴더): <도시>_<frames>_<matcher><tag>_steps.csv (걸음별 기록), 표는 stdout.

--variants: 복구 단계를 쉼표로 여러 개 (docs/04 R0~R4). 한 프로세스에서 돌려 매칭 캐시를 같이 쓴다.
  r0 복구 없음 · r1 inlier < 15이면 모름 → 추측 항법 · r2g30 r1 + 추측 위치와 30 m 넘게 차이나면 모름
  r3g30 r2g30 + 모름이면 주변 5블록 재탐색 · r4g30n3 r3g30 + 3걸음 연속 모름이면 3×3 (imatch/nav.py Research)
  r4g30n5w20 r4g30n5 + 3×3에서만 inlier >= 20
  center 매칭 없이 항상 블록 중심 · dr 매칭 없이 추측 항법만
  출력 파일은 <도시>_<frames>_<matcher><tag>-<variant>_steps.csv (걸음별)과 _traj.csv (궤적별 성공, 종료 이유)
  r3·r4의 걸음 기록에는 재탐색 열(rec_*)이 붙고, sec는 재탐색 시간을 포함한다 (rec_sec가 그중 재탐색)
--odo vio: 실제 이동에 오차를 넣는다 (imatch/nav.py ODO). 없으면 추측 항법이 공짜로 맞는다.
"""
import re
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imatch import get_matcher  # noqa: E402
from imatch.nav import ODO, Crop2D, MatchLoc, Nearest3D, NoMatch, Research, make_traj, nav_run  # noqa: E402

CITY = {'3d_ho38bc': 'SF', '3d_ho37bc': 'Singapore', '3d_ho34bc': 'Tokyo', '3d_ho36bc': 'Taipei'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/im_t1_3d_1000'); ap.add_argument('--out', required=True)
    ap.add_argument('--sets', default='3d_ho38bc,3d_ho37bc'); ap.add_argument('--frames', choices=['2d', '3d'], default='3d')
    ap.add_argument('--capstone1', default='', help='--frames 3d일 때 필요'); ap.add_argument('--matcher', default='loma')
    ap.add_argument('--angles', type=int, default=1); ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-traj', type=int, default=96); ap.add_argument('--dist', type=float, default=1000)
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--oracle', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--variants', default='', help='예: r0,r1,r2g30,r3g30,r4g30n3,center,dr. 비우면 r0 한 번 (예전과 같음)')
    ap.add_argument('--odo', default='none', choices=list(ODO))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    matcher = get_matcher(a.matcher, a.device)
    angles = list(np.arange(0, 360, 360 / a.angles))
    print(f'입력 {a.frames}, 매처 {a.matcher} {a.angles}방향, 시드 {a.seed}', flush=True)
    print('| 도시 | 복구 | 이동 오차 | 블록 | 궤적 | 성공률 | 종료 이유 | 걸음 오차 중앙값 | 모름 걸음 | 걸음당 s | 새로 매칭 |')
    print('| --- ' * 11 + '|')
    for st in a.sets.split(','):
        city = cv2.cvtColor(cv2.imread(os.path.join(a.data, 'maps', f'{st}.jpg')), cv2.COLOR_BGR2RGB)
        frames = Nearest3D(a.capstone1, st) if a.frames == '3d' else Crop2D(city)
        starts, goals = make_traj(a.n_traj, a.dist, a.seed)
        match = MatchLoc(matcher, city, angles=angles)
        for v in (a.variants.split(',') if a.variants else ['']):
            m = re.fullmatch(r'(r0|r1|r([234])g(\d+)(?:n(\d+)(?:w(\d+))?)?|center|dr|)', v)
            if not m or (m.group(2) == '4') != bool(m.group(4)): sys.exit(f'모르는 variant: {v}')
            loc = NoMatch() if v in ('center', 'dr') else match
            kw = dict(reject_inl=15 if v.startswith(('r1', 'r2', 'r3', 'r4')) or v == 'dr' else 0, gate_m=float(m.group(3) or 0))
            rec = (Research(match, gate_m=kw['gate_m'], wide_after=int(m.group(4) or 0), wide_inl=int(m.group(5) or 0))
                   if m.group(2) in ('3', '4') else None)
            log, t0, n0 = [], time.time(), len(match.cache)
            ok, reason, med = nav_run(loc, frames, starts, goals, oracle_block=a.oracle, log=log,
                                      odo=ODO[a.odo], seed=a.seed, recover=rec, **kw)
            L = pd.DataFrame(log)
            if rec and rec.log:  # recover는 "모름" 걸음마다 그 순서대로 한 번 불린다
                R = pd.DataFrame(rec.log, index=L.index[L.unknown == 1])
                assert (R.traj == L.traj[R.index]).all()
                L = L.join(R.drop(columns='traj'))
                L['sec'] += L.rec_sec.fillna(0)
            name = f"{CITY[st]}_{a.frames}_{a.matcher}{a.tag}{'-' + v if v else ''}_steps.csv"
            L.to_csv(os.path.join(a.out, name), index=False)
            pd.DataFrame(dict(traj=range(len(ok)), success=ok.astype(int), reason=reason)).to_csv(
                os.path.join(a.out, name.replace('_steps.csv', '_traj.csv')), index=False)
            why = {x: int((reason == x).sum()) for x in np.unique(reason)}
            print(f"| {CITY[st]} | {v or 'r0'} | {a.odo} | {'정답' if a.oracle else '추정'} | {len(ok)} | {ok.mean()*100:.1f}% | {why} "
                  f"| {np.median(med):.2f} m | {L.unknown.mean()*100:.1f}% | {(time.time() - t0) / len(L):.2f} | {len(match.cache) - n0} |", flush=True)


if __name__ == '__main__':
    main()
