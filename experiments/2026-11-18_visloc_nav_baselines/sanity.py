"""어댑터 좌표 확인: 지도에서 정답 위치 중심 256 px를 잘라 '드론 사진'으로 넣는다 (계절·시점 차이 없음).
좌표 변환이 맞으면 오차가 블록 가운데만 찍는 것보다 훨씬 작아야 한다. rot = 크롭을 돌린 각도(반시계, 90° 단위).

  python experiments/2026-11-18_visloc_nav_baselines/sanity.py --models osfpi,corr38,bearing --n 40
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baselines as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/visloc/seq_eval'); ap.add_argument('--models', default='osfpi,corr38,bearing')
    ap.add_argument('--cap', default=os.path.expanduser('~/capstone/bearinguav'))
    ap.add_argument('--osfpi', default=os.path.expanduser('~/capstone/ext/OS-FPI'))
    ap.add_argument('--n', type=int, default=40); ap.add_argument('--rots', default='0,1')
    a = ap.parse_args()
    S = pd.read_csv(os.path.join(a.data, 'samples.csv'))
    S = S[(S.map_x > 400) & (S.map_x < 3700) & (S.map_y > 400) & (S.map_y < 3700)].sample(a.n, random_state=0)
    maps = {}
    for name in a.models.split(','):
        if name == 'osfpi':
            loc = B.OSFPILoc(a.osfpi)
        elif name == 'bearing':
            loc = B.BearingLoc(a.cap)
        else:
            loc = B.CorrLoc(a.cap, {'corr38': 'results/corr/p2bv4_3d_ho38bc_20261013_135730',
                                    'corr37': 'results/corr/p2bv4_3d_ho37bc_20261013_173659'}[name])
        for r in map(int, a.rots.split(',')):
            e, e0 = [], []
            for _, s in S.iterrows():
                if s.set not in maps:
                    maps[s.set] = cv2.cvtColor(cv2.imread(os.path.join(a.data, 'maps', f'{s.set}.jpg')), cv2.COLOR_BGR2RGB)
                city = maps[s.set]
                x, y = int(round(s.map_x)), int(round(s.map_y))
                crop = np.ascontiguousarray(np.rot90(city[y - 128:y + 128, x - 128:x + 128], r))
                bx, by = int(np.clip(round((x - 256) / 256), 0, 14)), int(np.clip(round((y - 256) / 256), 0, 14))
                p = loc.bind(city)([crop], np.array([0.]), [(bx, by)])[0]
                est = np.array([bx * 256 + 256, by * 256 + 256]) + p * 128
                e.append(np.linalg.norm(est - [x, y]) / 4)
                e0.append(np.linalg.norm(np.array([bx * 256 + 256, by * 256 + 256]) - [x, y]) / 4)
            e, e0 = np.array(e), np.array(e0)
            print(f'{name:8s} rot {90 * r:3d}°: 오차 중앙값 {np.median(e):6.1f} m, 10 m 이내 {(e <= 10).mean() * 100:5.1f}% '
                  f'| 블록 가운데 {np.median(e0):.1f} m', flush=True)


if __name__ == '__main__':
    main()
