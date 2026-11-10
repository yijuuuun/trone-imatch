"""Kaggle GPU에서 돌릴 커널 폴더를 만든다. Kaggle 스크립트 커널은 파일 1개만 올라가므로
imatch/ 패키지와 scripts/run.py를 run.py 안에 base64 문자열로 넣는다.

  python scripts/kaggle/build_kernel.py --user <kaggle아이디> --dataset <kaggle아이디>/im-t1-3d-1000 \
      --slug loma-1dir --args "--matchers loma --angles 1 --sets 3d_ho38bc,3d_ho37bc"
  kaggle kernels push -p kaggle_kernels/loma-1dir --accelerator NvidiaTeslaT4
  kaggle kernels status <kaggle아이디>/loma-1dir
  kaggle kernels output <kaggle아이디>/loma-1dir -p outputs/loma-1dir --file-pattern 'out/.*'

--dataset: data/im_t1_3d_1000.zip을 올린 Kaggle 데이터셋 (data/README.md 참고).
--script run_nav: 1 km 항법. 데이터셋은 scripts/kaggle/pack_nav.py로 만든다.
"""
import argparse
import base64
import glob
import json
import os
import shlex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAUNCHER = r'''# 자동 생성 파일 (scripts/kaggle/build_kernel.py). 직접 고치지 말 것.
import base64, glob, os, shutil, subprocess, sys, zipfile
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vismatch'], check=True)

SRC = '/kaggle/working/src'
for rel, text in FILES.items():
    p = os.path.join(SRC, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(base64.b64decode(text))

# 데이터셋을 /kaggle/working/data 로 풀기 (zip이든 폴더든)
src = os.path.dirname(glob.glob('/kaggle/input/**/samples.csv', recursive=True)[0])
DATA = '/kaggle/working/data'; os.makedirs(DATA, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith('.zip'): zipfile.ZipFile(p).extractall(os.path.join(DATA, f[:-4]))
    elif os.path.isdir(p): shutil.copytree(p, os.path.join(DATA, f), dirs_exist_ok=True)
    else: shutil.copy(p, DATA)
for d in glob.glob(DATA + '/*_ho*'):
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner):
        for f in os.listdir(inner): shutil.move(os.path.join(inner, f), d)
# zip 안에 samples.csv와 도시 폴더가 함께 들어 있던 경우: 그 폴더를 데이터 폴더로 쓴다
ROOTS = [os.path.dirname(p) for p in glob.glob(DATA + '/**/samples.csv', recursive=True)
         if glob.glob(os.path.join(os.path.dirname(p), '*_ho*', 'k*_uav.jpg'))]
if ROOTS: DATA_DIR = ROOTS[0]
else: DATA_DIR = DATA
print('data', DATA_DIR, sorted(os.listdir(DATA_DIR)), flush=True)

sys.path.insert(0, SRC)
try:
    if SCRIPT == 'run_nav':
        # 항법: 지도(maps/)가 있는 폴더가 --data, Bearing_UAV_90K가 있는 폴더가 --capstone1
        DATA_DIR = os.path.dirname(glob.glob(DATA + '/**/maps', recursive=True)[0])
        CAP = os.path.dirname(glob.glob(DATA + '/**/Bearing_UAV_90K', recursive=True)[0])
        from scripts.run_nav import main
        sys.argv = ['run_nav', '--data', DATA_DIR, '--capstone1', CAP, '--out', '/kaggle/working/out'] + ARGS
        main()
    else:
        from scripts.run import main
        main(['--data', DATA_DIR, '--out', '/kaggle/working/out'] + ARGS)
finally:
    shutil.rmtree(DATA, ignore_errors=True)  # 출력 용량을 작게. 실패해도 지워야 로그를 받을 수 있다
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--slug', required=True); ap.add_argument('--args', default='--matchers loma --angles 8')
    ap.add_argument('--out', default=os.path.join(ROOT, 'kaggle_kernels'))
    ap.add_argument('--script', default='run', choices=['run', 'run_nav'], help='run_nav: 1 km 항법 (데이터셋에 maps/와 Bearing_UAV_90K 필요)')
    a = ap.parse_args()
    files = {os.path.relpath(p, ROOT): base64.b64encode(open(p, 'rb').read()).decode()
             for p in sorted(glob.glob(os.path.join(ROOT, 'imatch', '*.py'))) + [os.path.join(ROOT, 'scripts', f) for f in ('run.py', 'run_nav.py')]}
    files['scripts/__init__.py'] = ''
    d = os.path.join(a.out, a.slug); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8') as f:
        f.write(f'FILES = {files!r}\nARGS = {shlex.split(a.args)!r}\nSCRIPT = {a.script!r}\n' + LAUNCHER)
    json.dump({'id': f'{a.user}/{a.slug}', 'title': a.slug, 'code_file': 'run.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
               'dataset_sources': [a.dataset], 'competition_sources': [], 'kernel_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('만듦:', d)


if __name__ == '__main__':
    main()
