"""항법용 Kaggle 데이터셋 폴더를 만든다: 도시 지도(maps/) + 그 도시의 3D 렌더와 메타데이터.

  python scripts/kaggle/pack_nav.py --capstone1 ~/bearing/data --data data/im_train_3d_500 --sets 3d_ho34bc,3d_ho36bc --out outputs/nav_train_pack
  kaggle datasets create -p outputs/nav_train_pack      # 처음 한 번 (dataset-metadata.json은 직접 만든다)

출력 폴더: samples.csv(커널이 데이터 위치를 찾는 표시), nav.zip(maps/, Bearing_UAV_90K/의 필요한 부분만).
"""
import argparse
import os
import re
import shutil
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from imatch.nav import META_3D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capstone1', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--sets', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sets = a.sets.split(',')
    hos = [re.search(r'(\d+bc)', st).group(1) for st in sets]
    M = pd.read_csv(os.path.join(a.capstone1, META_3D))
    M = M[M.target_path.str.contains('|'.join(f'_{h}_' for h in hos))]
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.data, 'samples.csv'), a.out)
    with zipfile.ZipFile(os.path.join(a.out, 'nav.zip'), 'w', zipfile.ZIP_STORED) as z:
        for st in sets:
            z.write(os.path.join(a.data, 'maps', f'{st}.jpg'), f'maps/{st}.jpg')
        meta = M.to_csv(index=False)
        z.writestr(META_3D, meta)
        for p in M.target_path:
            z.write(os.path.join(a.capstone1, p), os.path.normpath(p))
    print(f'{a.out}: 렌더 {len(M)}장, {os.path.getsize(os.path.join(a.out, "nav.zip")) / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
