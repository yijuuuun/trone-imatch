"""샘플 데이터(약 340 MB)를 GitHub Release에서 받아 data/im_t1_3d_1000/ 에 푼다.
저장소가 비공개라 GitHub CLI(gh) 로그인이 필요하다:  gh auth login  (처음 한 번)

  python scripts/download_data.py
"""
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO, TAG, ASSET = 'yijuuuun/trone-grid-pooling', 'data-v1', 'im_t1_3d_1000.zip'
DATA = os.path.join(ROOT, 'data')
ZIP, OUT = os.path.join(DATA, ASSET), os.path.join(DATA, 'im_t1_3d_1000')


def main():
    if os.path.exists(os.path.join(OUT, 'samples.csv')):
        print('이미 있음:', OUT); return
    if not os.path.exists(ZIP):
        if not shutil.which('gh'):
            sys.exit(f'GitHub CLI(gh)가 필요하다 (https://cli.github.com). 또는 브라우저에서 '
                     f'https://github.com/{REPO}/releases/tag/{TAG} 의 {ASSET}을 받아 data/에 두고 다시 실행.')
        subprocess.run(['gh', 'release', 'download', TAG, '-R', REPO, '-p', ASSET, '-D', DATA], check=True)
    zipfile.ZipFile(ZIP).extractall(OUT)
    n = {d: len(os.listdir(os.path.join(OUT, d))) for d in sorted(os.listdir(OUT)) if os.path.isdir(os.path.join(OUT, d))}
    print('완료:', OUT, n)


if __name__ == '__main__':
    main()
