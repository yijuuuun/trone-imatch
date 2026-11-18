"""Kaggle T4 커널(한 번에 모든 기존 모델)과 가중치 데이터셋 폴더를 만든다.

  python experiments/2026-11-18_visloc_nav_baselines/build_kernel.py --pack outputs/visloc_baselines_pack \
      --cap ~/capstone/bearinguav --osfpi ~/capstone/ext/OS-FPI --geoanchor ~/capstone/ext/GeoAnchor
  kaggle datasets create -p outputs/visloc_baselines_pack --dir-mode skip      # heeyunju/visloc-baselines
  kaggle kernels push -p kaggle_kernels/visloc-nav-baselines --accelerator NvidiaTeslaT4

데이터셋 3개: heeyunju/visloc-seq-eval (92구간), heeyunju/bearing-ckpt (cvphr, Bearing_UAV/cross_view),
heeyunju/visloc-baselines (이 스크립트가 만드는 폴더: CorrMatchNet p2bv4 두 개, OS-FPI 코드·net_040.pth, GeoAnchor src).
"""
import argparse
import base64
import glob
import json
import os
import shlex
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CORR = ['p2bv4_3d_ho38bc_20261013_135730', 'p2bv4_3d_ho37bc_20261013_173659']

LAUNCHER = r'''# 자동 생성 (experiments/2026-11-18_visloc_nav_baselines/build_kernel.py)
import base64, glob, importlib, os, shutil, subprocess, sys
if importlib.util.find_spec('kornia') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kornia'], check=True)
SRC = '/kaggle/working/src'
for rel, text in FILES.items():
    p = os.path.join(SRC, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(base64.b64decode(text))
def find(pat):
    r = glob.glob('/kaggle/input/**/' + pat, recursive=True)
    if not r: raise SystemExit('없음: ' + pat)
    return r[0]
DATA = os.path.dirname(os.path.dirname(find('maps/visloc_01_s00.jpg')))
CK = os.path.dirname(os.path.dirname(find('cvphr/__init__.py')))
PK = find('corr/' + CORR[0] + '/best_model.pth').rsplit('/corr/', 1)[0]
CAP = '/kaggle/working/cap'
for d in ('cvphr', 'diag', 'Bearing_UAV/cross_view'):
    shutil.copytree(os.path.join(CK, d), os.path.join(CAP, d), dirs_exist_ok=True)
for c in CORR:
    shutil.copytree(os.path.join(PK, 'corr', c), os.path.join(CAP, 'results/corr', c), dirs_exist_ok=True)
print('data', DATA, '| cap', sorted(os.listdir(CAP)), '| pack', sorted(os.listdir(PK)), flush=True)
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'experiments/2026-11-18_visloc_nav_baselines'))
os.chdir(SRC)
from run_baselines import main
main(['--data', DATA, '--out', '/kaggle/working/out', '--cap', CAP, '--osfpi', os.path.join(PK, 'osfpi'),
      '--geoanchor', os.path.join(PK, 'ga_pkg')] + ARGS)
'''


def pack(out, cap, osfpi, geoanchor):
    if os.path.exists(out):
        shutil.rmtree(out)
    for c in CORR:
        shutil.copytree(os.path.join(cap, 'results/corr', c), os.path.join(out, 'corr', c))
    shutil.copytree(os.path.join(osfpi, 'models'), os.path.join(out, 'osfpi', 'models'),
                    ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('opts.yaml', 'net_040.pth', 'README.md'):
        shutil.copy(os.path.join(osfpi, f), os.path.join(out, 'osfpi', f))
    g = os.path.join(out, 'ga_pkg', 'geoanchor'); os.makedirs(g)
    for f in ('__init__.py', 'matching.py', 'particle_filter.py', 'features.py'):
        shutil.copy(os.path.join(geoanchor, 'src', f), g)
    shutil.copy(os.path.join(geoanchor, 'LICENSE'), os.path.join(out, 'ga_pkg', 'LICENSE'))
    json.dump({'title': 'visloc-baselines', 'id': 'heeyunju/visloc-baselines', 'licenses': [{'name': 'other'}],
               'isPrivate': True}, open(os.path.join(out, 'dataset-metadata.json'), 'w'))
    print('데이터셋 폴더:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default=''); ap.add_argument('--cap', default=os.path.expanduser('~/capstone/bearinguav'))
    ap.add_argument('--osfpi', default=os.path.expanduser('~/capstone/ext/OS-FPI'))
    ap.add_argument('--geoanchor', default=os.path.expanduser('~/capstone/ext/GeoAnchor'))
    ap.add_argument('--slug', default='visloc-nav-baselines'); ap.add_argument('--user', default='heeyunju')
    ap.add_argument('--args', default='--models bearing,corr38,corr37,osfpi,geoanchor')
    a = ap.parse_args()
    if a.pack:
        pack(a.pack, a.cap, a.osfpi, a.geoanchor)
    files = {os.path.relpath(p, ROOT): base64.b64encode(open(p, 'rb').read()).decode()
             for p in sorted(glob.glob(os.path.join(ROOT, 'imatch', '*.py')) + glob.glob(os.path.join(HERE, '*.py')))}
    d = os.path.join(ROOT, 'kaggle_kernels', a.slug); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8') as f:
        f.write(f'FILES = {files!r}\nARGS = {shlex.split(a.args)!r}\nCORR = {CORR!r}\n' + LAUNCHER)
    json.dump({'id': f'{a.user}/{a.slug}', 'title': a.slug, 'code_file': 'run.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
               'dataset_sources': ['heeyunju/visloc-seq-eval', 'heeyunju/bearing-ckpt', 'heeyunju/visloc-baselines'],
               'competition_sources': [], 'kernel_sources': []}, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('커널:', d)


if __name__ == '__main__':
    main()
