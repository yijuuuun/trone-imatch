# capstone1 저장소 루트에서 실행해야 한다 (diag.harness를 쓴다). 보통은 data/README.md의 zip을 받으면 된다.
# Image-matching flow (IM_): export the first N samples of harness T1 (same rows, same order) as plain files,
# so off-the-shelf matchers can run on another machine (Kaggle) without this repo.
#   out/<view>_ho<city>/k###_block.jpg  2x2 tiles stitched (512 px, p1 p2 / p3 p4)
#   out/<view>_ho<city>/k###_uav.jpg    UAV view (256 px)
#   out/samples.csv                     set, k, row, x_norm, y_norm, theta
#   python <trone-imatch>/scripts/export_from_capstone1.py --n 100 --out <dir>
import os, sys, argparse
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.getcwd())
from diag.harness import heldout_rows, load_rows, SIZES

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--out', required=True)
ap.add_argument('--views', default='2d,3d')
ap.add_argument('--cities', default='38bc,37bc', help='시험 도시 38bc(SF), 37bc(Singapore). 학습 도시는 34bc, 36bc')
a = ap.parse_args()

recs = []
for view in a.views.split(','):
    for ho in a.cities.split(','):
        rows, df = heldout_rows(view, ho, SIZES['full']['t1'])  # same draw as harness T1; its prefix is unbiased
        rows = rows[:a.n]
        S = load_rows(df, rows)
        name = f'{view}_ho{ho}'; d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        for k, ((t, u, (xn, yn), th), r) in enumerate(zip(S, rows)):
            block = np.vstack([np.hstack([t[0], t[1]]), np.hstack([t[2], t[3]])])
            cv2.imwrite(os.path.join(d, f'k{k:03d}_block.jpg'), cv2.cvtColor(block, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
            cv2.imwrite(os.path.join(d, f'k{k:03d}_uav.jpg'), cv2.cvtColor(u, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
            recs.append(dict(set=name, k=k, row=int(r), x_norm=xn, y_norm=yn, theta=th,
                             uav_h=u.shape[0], uav_w=u.shape[1], block_h=block.shape[0], block_w=block.shape[1]))
        print(name, len(S), flush=True)
pd.DataFrame(recs).to_csv(os.path.join(a.out, 'samples.csv'), index=False)
