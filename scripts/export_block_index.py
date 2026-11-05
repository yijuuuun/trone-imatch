"""후보 블록 실험(⑥ 블록 선택)에 필요한 파일을 데이터 폴더에 추가한다. 한 번만 돌리면 된다.

  python scripts/export_block_index.py --capstone1 ../bearinguav

추가되는 파일 (--data 폴더, git에 안 올림)
  block_index.csv       set, k, block_x, block_y  (samples.csv의 row로 capstone1 메타데이터와 연결)
  maps/<set>.jpg        도시 지도 원본 (4096 px). 후보 블록은 여기서 자른다
"""
import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imatch.blocks import crop_block  # noqa: E402

META = 'Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'
MAP_PREFIX = {'3d_ho38bc': '37.755', '3d_ho37bc': '1.289'}  # city_rsi 파일 이름의 위도


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capstone1', required=True, help='capstone1 저장소 경로 (Bearing_UAV_90K가 있는 곳)')
    ap.add_argument('--data', default='data/im_t1_3d_1000')
    a = ap.parse_args()
    S = pd.read_csv(os.path.join(a.data, 'samples.csv'))
    M = pd.read_csv(os.path.join(a.capstone1, META), usecols=['block_x', 'block_y', 'x_norm', 'y_norm']).iloc[S.row]
    if not (np.allclose(M.x_norm, S.x_norm) and np.allclose(M.y_norm, S.y_norm)):
        sys.exit('samples.csv의 row가 메타데이터와 맞지 않는다')
    S['block_x'], S['block_y'] = M.block_x.to_numpy(), M.block_y.to_numpy()
    S[['set', 'k', 'block_x', 'block_y']].to_csv(os.path.join(a.data, 'block_index.csv'), index=False)

    os.makedirs(os.path.join(a.data, 'maps'), exist_ok=True)
    for st, g in S.groupby('set'):
        src = [p for p in glob.glob(os.path.join(a.capstone1, 'Bearing_UAV_90K', 'city_rsi', f'{MAP_PREFIX[st]}*_city.jpg'))
               if '6144' not in p][0]
        dst = os.path.join(a.data, 'maps', f'{st}.jpg')
        shutil.copyfile(src, dst)
        # 지도에서 자른 블록이 저장된 정답 블록과 같은지 앞 20개로 확인한다
        city = cv2.imread(dst)
        diff = [np.abs(crop_block(city, s.block_x, s.block_y).astype(float)
                       - cv2.imread(os.path.join(a.data, st, f'k{s.k:03d}_block.jpg'))).mean() for s in g.head(20).itertuples()]
        print(f'{st}: 샘플 {len(g)}, 지도 {city.shape[1]}x{city.shape[0]}, 정답 블록과 차이 평균 {np.mean(diff):.2f} (0~255)')


if __name__ == '__main__':
    main()
