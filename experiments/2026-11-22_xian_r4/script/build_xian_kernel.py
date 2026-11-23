"""XIAN 재생 항법(scripts/run_xian_nav.py)용 Kaggle 커널 폴더를 만든다. 팀 build_kernel.py는 고치지 않고 그대로 가져다 쓴다.

  python scripts/kaggle/build_xian_kernel.py --user jimin562 --dataset jimin562/xian-seq-test --slug xian-nav
  kaggle kernels push -p kaggle_kernels/xian-nav --accelerator NvidiaTeslaT4
"""
import argparse
import base64
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_kernel as bk  # noqa: E402

BRANCH = """    elif SCRIPT == 'run_xian_nav':
        DATA_DIR = os.path.dirname(glob.glob(DATA + '/**/maps', recursive=True)[0])
        from scripts.run_xian_nav import main
        main(['--data', DATA_DIR, '--out', '/kaggle/working/out'] + ARGS)
    elif SCRIPT == 'run_nav_b1':"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--slug', required=True); ap.add_argument('--args', default='')
    ap.add_argument('--out', default=os.path.join(bk.ROOT, 'kaggle_kernels'))
    a = ap.parse_args()
    files = bk.source_files()
    p = os.path.join(bk.ROOT, 'scripts', 'run_xian_nav.py')
    files[os.path.relpath(p, bk.ROOT)] = base64.b64encode(open(p, 'rb').read()).decode()
    files = {k.replace('\\', '/'): v for k, v in files.items()}   # Windows 경로(\)를 Kaggle(Linux)용으로
    marker = "    elif SCRIPT == 'run_nav_b1':"
    assert bk.LAUNCHER.count(marker) == 1, 'build_kernel.py LAUNCHER 형식이 바뀜'
    launcher = bk.LAUNCHER.replace(marker, BRANCH)
    d = os.path.join(a.out, a.slug); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8') as f:
        f.write(f'FILES = {files!r}\nARGS = {shlex.split(a.args)!r}\nSCRIPT = "run_xian_nav"\n' + launcher)
    json.dump({'id': f'{a.user}/{a.slug}', 'title': a.slug, 'code_file': 'run.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
               'dataset_sources': a.dataset.split(','), 'competition_sources': [], 'kernel_sources': []},
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('만듦:', d)


if __name__ == '__main__':
    main()
