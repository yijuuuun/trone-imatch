#!/usr/bin/env python
"""UAV-VisLoc 실제 비행 재생 항법 (imatch/replay.py). 3D 1 km 항법(run_nav.py)과 따로 둔다.

  python scripts/run_visloc_nav.py --data data/visloc/seq_eval --out outputs/visloc_nav \
      --matcher loma --angles 8 --stop-inl 15 --variants r0,r1,r4g0n5w20 --odo vio --seed 2026

variants (run_nav.py와 같은 이름)
  r0           매칭 결과를 항상 믿는다 (기준선과 같은 조건)
  r1           inlier < 15이면 "모름" → 추측 위치 (v1 예방)
  r4g0n5w20    r1 + 5걸음 연속 "모름"이면 추측 위치 주변 3×3 재탐색, inlier >= 20 (v1 복구)
  r3g0         r1 + "모름"이면 바로 십자 5블록 재탐색
  dr           매칭 없이 추측 항법만 (이동량 오차만 쌓이는 하한)
같은 seed면 모든 variant와 기준선이 같은 구간·출발점·이동 오차를 쓴다.

출력 (variant마다)
  visloc_<매처><tag>-<variant>_steps.csv   걸음별 기록 (err_m, unknown, used, block_x, ...)
  visloc_<매처><tag>-<variant>_segs.csv    구간별 (마지막 오차, 걸음 10/20/32 m 비율)
표: 걸음 10/20/32 m 이내(출발 걸음 제외), 마지막 오차 중앙값, 마지막 40 m 이내(3D 항법 성공 기준 거리 2 × 20 m),
    모름 걸음, 정답이 고른 블록 밖인 걸음.

--max-dist 300이면 구간 앞 300 m까지만 (거리별 비교, 분담 7번).
기준선(B1 등)은 이 스크립트 대신 imatch.replay.run_segments에 자기 loc을 넘긴다 (replay.py 맨 위 설명).
"""
import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.matchers import get_matcher  # noqa: E402
from imatch.nav import MatchLoc, NoMatch, Research  # noqa: E402
from imatch.replay import run_segments, summarize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/visloc/seq_eval'); ap.add_argument('--out', required=True)
    ap.add_argument('--matcher', default='loma'); ap.add_argument('--angles', type=int, default=8)
    ap.add_argument('--stop-inl', type=int, default=0, help='각도 탐색 조기 종료 (0 = 끔, 15 = 모름 기준과 같게)')
    ap.add_argument('--variants', default='r0,r1,r4g0n5w20')
    ap.add_argument('--odo', default='vio', choices=['none', 'vio'])
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--sets', default='', help='구간 이름 쉼표 목록 (비우면 전부)')
    ap.add_argument('--max-dist', type=float, default=None, help='구간 시작부터 이 거리(m)까지만')
    ap.add_argument('--oracle', action='store_true', help='블록을 정답 위치로 고른다 (블록 고르기 오류 제외)')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--tag', default='')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    variants = a.variants.split(',')
    need_match = any(v not in ('dr', 'center') for v in variants)
    matcher = get_matcher(a.matcher, a.device) if need_match else None
    angles = list(np.arange(0, 360, 360 / a.angles))
    sets = a.sets.split(',') if a.sets else None
    print(f'재생 항법: {a.data}, 매처 {a.matcher} {a.angles}방향, 이동 오차 {a.odo}, 시드 {a.seed}'
          f"{f', 앞 {a.max_dist:g} m' if a.max_dist else ''}", flush=True)
    print('| variant | 구간 | 걸음 | 10 m | 20 m | 32 m | 걸음 오차 중앙값 | 마지막 오차 중앙값 | 마지막 40 m 이내 | 모름 | 블록 밖 | 걸음당 s |')
    print('| --- ' * 12 + '|')
    cache = {}  # 구간 지도별 MatchLoc. variant끼리 같은 (사진, 블록) 매칭을 다시 하지 않는다
    for v in variants:
        m = re.fullmatch(r'(r0|r1|r([234])g(\d+)(?:n(\d+)(?:w(\d+))?)?|center|dr)', v)
        if not m or (m.group(2) == '4') != bool(m.group(4)):
            sys.exit(f'모르는 variant: {v}')
        gate = float(m.group(3) or 0)

        def make_loc(city, v=v):
            if v in ('center', 'dr'):
                return NoMatch()
            key = hashlib.md5(city.tobytes()).hexdigest()
            if key not in cache:
                cache[key] = MatchLoc(matcher, city, angles=angles, stop_inl=a.stop_inl)
            return cache[key]

        make_rec = ((lambda loc, m=m, gate=gate: Research(loc, gate_m=gate, wide_after=int(m.group(4) or 0),
                                                          wide_inl=int(m.group(5) or 0)))
                    if m.group(2) in ('3', '4') else None)
        t0 = time.time()
        steps, segs = run_segments(make_loc, a.data, sets=sets, odo=a.odo, seed=a.seed,
                                   reject_inl=15 if v.startswith(('r1', 'r2', 'r3', 'r4')) or v == 'dr' else 0,
                                   gate_m=gate, make_recover=make_rec, oracle_block=a.oracle, max_dist=a.max_dist)
        name = f"visloc_{a.matcher}{a.tag}{'-oracle' if a.oracle else ''}-{v}"
        steps.to_csv(os.path.join(a.out, f'{name}_steps.csv'), index=False)
        segs.to_csv(os.path.join(a.out, f'{name}_segs.csv'), index=False)
        s = summarize(steps, segs)
        print(f"| {v} | {s['segments']} | {s['steps']} | {s['w10']}% | {s['w20']}% | {s['w32']}% | {s['med_err_m']} m "
              f"| {s['final_med_m']} m | {s['final_le40']}% | {s['unknown']}% | {s['outside_block']}% "
              f"| {(time.time() - t0) / max(len(steps), 1):.2f} |", flush=True)


if __name__ == '__main__':
    main()
