"""OS-FPI 입력 설정을 개발용 single_calib(513장, 정답 블록)에서 고른다. 시험용 seq_eval은 보지 않는다.

드론 256 px(64 m) 크롭의 가운데 몇 px를 넣을지(256 / 192 / 128)만 바꾼다. 블록(512 px, 128 m)과 4방향 탐색은 그대로.
기준: 한 장 위치 오차 중앙값 (같으면 20 m 이내 비율).

  python experiments/2026-11-18_visloc_nav_baselines/calib_osfpi.py --data data/visloc/single_calib
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baselines as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/visloc/single_calib')
    ap.add_argument('--osfpi', default=os.path.expanduser('~/capstone/ext/OS-FPI'))
    ap.add_argument('--crops', default='256,192,128')
    a = ap.parse_args()
    S = pd.read_csv(os.path.join(a.data, 'samples.csv'))
    imgs = [(cv2.cvtColor(cv2.imread(os.path.join(a.data, s.set, f'k{s.k:03d}_uav.jpg')), cv2.COLOR_BGR2RGB),
             cv2.cvtColor(cv2.imread(os.path.join(a.data, s.set, f'k{s.k:03d}_block.jpg')), cv2.COLOR_BGR2RGB),
             np.array([s.x_norm, s.y_norm])) for s in S.itertuples()]
    e0 = np.array([np.linalg.norm(t) * 32 for _, _, t in imgs])
    print(f'{len(imgs)}장, 블록 가운데만: 중앙값 {np.median(e0):.1f} m, 10/20 m {(e0 <= 10).mean()*100:.1f} / {(e0 <= 20).mean()*100:.1f}%')
    for c in map(int, a.crops.split(',')):
        loc = B.OSFPILoc(a.osfpi, uav_crop=c)
        e = []
        for uav, blk, t in imgs:
            p = loc.bind(blk)([uav], np.array([0.]), [(0, 0)])[0]   # 지도 = 블록 한 장, 블록 (0, 0)
            e.append(np.linalg.norm(p - t) * 32)
        e = np.array(e)
        print(f'uav_crop {c:3d} px ({c / 4:.0f} m): 중앙값 {np.median(e):.1f} m, 10/20/32 m {(e <= 10).mean()*100:.1f} / '
              f'{(e <= 20).mean()*100:.1f} / {(e <= 32).mean()*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
