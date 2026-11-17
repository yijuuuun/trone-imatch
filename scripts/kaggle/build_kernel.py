"""Kaggle GPU에서 돌릴 커널 폴더를 만든다. Kaggle 스크립트 커널은 파일 1개만 올라가므로
imatch/ 패키지와 scripts/run.py를 run.py 안에 base64 문자열로 넣는다.

  python scripts/kaggle/build_kernel.py --user <kaggle아이디> --dataset <kaggle아이디>/im-t1-3d-1000 \
      --slug loma-1dir --args "--matchers loma --angles 1 --sets 3d_ho38bc,3d_ho37bc"
  kaggle kernels push -p kaggle_kernels/loma-1dir --accelerator NvidiaTeslaT4
  kaggle kernels status <kaggle아이디>/loma-1dir
  kaggle kernels output <kaggle아이디>/loma-1dir -p outputs/loma-1dir --file-pattern 'out/.*'

--dataset: data/im_t1_3d_1000.zip을 올린 Kaggle 데이터셋 (data/README.md 참고).
--script run_nav: 1 km 항법. 데이터셋은 scripts/kaggle/pack_nav.py로 만든다.
--script run_visloc_nav: UAV-VisLoc 재생 항법. --dataset heeyunju/visloc-seq-eval
--script run_nav_b1: Bearing-UAV 항법. --dataset <항법 데이터셋>,heeyunju/bearing-ckpt (쉼표로 둘).
    bearing-ckpt는 capstone1의 cvphr/, diag/diag_modules.py와 가중치 폴더(best_model.pth, training_configure.json)를
    cap/ 아래 capstone1과 같은 경로로 담는다. --args 안의 ';'로 실행을 여러 번 나눈다:
    --args "--ckpt Bearing_UAV/cross_view --sets 3d_ho38bc --dist 100 ... ; --ckpt ... --dist 300 ..."
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
if SCRIPT == 'run_nav_b1':  # 데이터 풀기 전에 가중치·블록 타일이 들어왔는지 확인 (없으면 바로 멈춰 GPU 시간을 아낌)
    n_tile = len(glob.glob('/kaggle/input/**/sat_254k_*/block_*.jpg', recursive=True))
    n_ckpt = len(glob.glob('/kaggle/input/**/best_model.pth', recursive=True))
    print('bearing-ckpt: 타일', n_tile, '가중치', n_ckpt, flush=True)
    if n_tile == 0 or n_ckpt == 0: raise SystemExit('bearing-ckpt에 블록 타일이나 가중치가 없다')
else:
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
for d in [d for d in glob.glob(DATA + '/*') if os.path.isdir(d)]:  # zip이 폴더째 들어 있던 경우 한 단계 올림
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner):
        for f in os.listdir(inner): shutil.move(os.path.join(inner, f), d)
# zip 안에 samples.csv와 도시 폴더가 함께 들어 있던 경우: 그 폴더를 데이터 폴더로 쓴다
ROOTS = [os.path.dirname(p) for p in glob.glob(DATA + '/**/samples.csv', recursive=True)
         if glob.glob(os.path.join(os.path.dirname(p), '*', 'k*_uav.jpg'))]  # 3D(*_ho*), UAV-VisLoc(visloc_*) 공통
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
    elif SCRIPT == 'run_visloc_nav':
        # UAV-VisLoc 재생 항법: maps/와 samples.csv가 있는 폴더(seq_eval)가 --data
        DATA_DIR = os.path.dirname(glob.glob(DATA + '/**/maps', recursive=True)[0])
        from scripts.run_visloc_nav import main
        main(['--data', DATA_DIR, '--out', '/kaggle/working/out'] + ARGS)
    elif SCRIPT == 'run_nav_b1':
        # Bearing-UAV 항법: Bearing_UAV_90K가 있는 폴더에 bearing-ckpt의 cap/(코드·가중치)를 합쳐 --capstone1로 쓴다
        CAP = os.path.dirname(glob.glob(DATA + '/**/Bearing_UAV_90K', recursive=True)[0])
        CK = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/cvphr/__init__.py', recursive=True)[0]))
        shutil.copytree(CK, CAP, dirs_exist_ok=True)
        # cvphr의 학습용 증강 모듈은 imgaug와 numpy 1 전용 코드를 불러와 Kaggle(numpy 2)에서 import가 실패한다.
        # 추론에는 쓰지 않으므로 빈 모듈로 대신한다 (로컬 bearing_env 결과와 걸음 오차 차이 최대 4e-5 m 확인)
        import types
        ut = types.ModuleType('cvphr.utils.utils_transform')
        def _unused(*a, **k): raise RuntimeError('학습용 증강은 추론에서 쓰지 않음')
        ut.transform_pipeline3 = ut.transform_pipeline1_gentle = ut.transform_pipeline_weather = _unused
        sys.modules['cvphr.utils.utils_transform'] = ut
        from scripts.run_nav_b1 import main
        runs = [[]]
        for t in ARGS:
            if t == ';': runs.append([])
            else: runs[-1].append(t)
        for r in runs:
            sys.argv = ['run_nav_b1', '--capstone1', CAP, '--out', '/kaggle/working/out'] + r
            main()
    else:
        from scripts.run import main
        main(['--data', DATA_DIR, '--out', '/kaggle/working/out'] + ARGS)
finally:
    shutil.rmtree(DATA, ignore_errors=True)  # 출력 용량을 작게. 실패해도 지워야 로그를 받을 수 있다
'''


def source_files():
    """imatch/*.py와 scripts/run.py, run_nav.py를 {상대 경로: base64}로. Colab 노트북(build_colab.py)도 쓴다."""
    files = {os.path.relpath(p, ROOT): base64.b64encode(open(p, 'rb').read()).decode()
             for p in sorted(glob.glob(os.path.join(ROOT, 'imatch', '*.py'))) + [os.path.join(ROOT, 'scripts', f) for f in ('run.py', 'run_nav.py', 'run_visloc_nav.py', 'run_nav_b1.py')]}
    files['scripts/__init__.py'] = ''
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--slug', required=True); ap.add_argument('--args', default='--matchers loma --angles 8')
    ap.add_argument('--out', default=os.path.join(ROOT, 'kaggle_kernels'))
    ap.add_argument('--script', default='run', choices=['run', 'run_nav', 'run_visloc_nav', 'run_nav_b1'],
                    help='run_nav: 1 km 항법 (데이터셋에 maps/와 Bearing_UAV_90K 필요). run_visloc_nav: UAV-VisLoc 재생 항법 (seq_eval)')
    a = ap.parse_args()
    files = source_files()
    d = os.path.join(a.out, a.slug); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8') as f:
        f.write(f'FILES = {files!r}\nARGS = {shlex.split(a.args)!r}\nSCRIPT = {a.script!r}\n' + LAUNCHER)
    json.dump({'id': f'{a.user}/{a.slug}', 'title': a.slug, 'code_file': 'run.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
               'dataset_sources': a.dataset.split(','), 'competition_sources': [], 'kernel_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('만듦:', d)


if __name__ == '__main__':
    main()
