"""Kaggle GPU 쿼터가 모자랄 때 Colab에서 항법(run_nav, run_visloc_nav)을 돌리는 노트북을 만든다.
저장소가 비공개라 clone하지 않고, build_kernel.py처럼 코드를 노트북 안에 base64로 넣는다.

  python scripts/kaggle/build_colab.py --slug nav-r4w-tk --dataset heeyunju/nav-train-tokyo \
      --args "--sets 3d_ho34bc --frames 3d --matcher loma --angles 8 --n-traj 32 --seed 7 --odo vio --tag=-vio --variants r2g30,r4g30n5w20"
  python scripts/kaggle/build_colab.py --slug visloc-nav-full --dataset gmldbs3104/visloc-seq-eval-full --script run_visloc_nav \
      --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20ac4 --odo vio --seed 2026"
  python scripts/kaggle/build_colab.py --slug visloc-nav-full-vio4 --dataset heeyunju/visloc-seq-eval-full --script run_visloc_nav \
      --cache-from visloc-nav-full --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20ac4 --odo vio4 --seed 2026"

Colab에서: 파일 > 노트북 업로드 → 런타임 유형 T4 GPU → 보안 비밀(열쇠 아이콘)에 KAGGLE_API_TOKEN
(또는 KAGGLE_USERNAME, KAGGLE_KEY) → 모두 실행. 출력은 Google Drive의 MyDrive/trone-nav/<slug>/.
끊겨도 다시 모두 실행하면 끝난 variant는 건너뛰고, 매칭 캐시(10분마다 Drive에 저장)를 이어 쓴다.
이어 쓴 매칭은 시간(sec)이 처음 잰 값이라, 걸음당 시간은 끊기지 않은 실행에서만 믿는다.
T4가 아니면 GPU 계산 차이로 Kaggle T4 결과와 조금 다를 수 있으니, 같은 노트북의 r2g30과만 비교한다.
"""
import argparse
import json
import os
import shlex

from build_kernel import ROOT, source_files

INTRO = """# {slug}: {script} ({variants})

`scripts/kaggle/build_colab.py`로 만든 노트북. 코드는 아래 셀에 들어 있어 저장소 접근이 필요 없다.

1. **런타임 > 런타임 유형 변경 > T4 GPU**
2. 왼쪽 **열쇠 아이콘(보안 비밀)**에 `KAGGLE_API_TOKEN` (또는 `KAGGLE_USERNAME`, `KAGGLE_KEY`)을 넣고 노트북 액세스를 켠다.
   데이터셋 `{dataset}`을 볼 수 있는 Kaggle 계정이어야 한다.
3. **런타임 > 모두 실행.** Drive 연결을 묻으면 허용한다.

결과: Google Drive `MyDrive/trone-nav/{slug}/` 의 `*_steps.csv`, `*{done}`.
끊기면 다시 **모두 실행**: 끝난 variant는 건너뛰고, 매칭 캐시(10분마다 저장)를 이어 쓴다.
"""

CONFIG = """SCRIPT = {script!r}
DATASET = {dataset!r}
ARGS = {args!r}
OUT = '/content/drive/MyDrive/trone-nav/{slug}'
CACHE_FROM = {cache_from!r}  # 처음 돌릴 때 이 매칭 캐시로 시작 (같은 데이터, 같은 매처·각도의 이전 실행)"""

GPU = """import subprocess
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', gpu or '없음')
assert gpu, '런타임 > 런타임 유형 변경에서 T4 GPU를 고르세요'
if 'T4' not in gpu:
    print('주의: T4가 아니다. Kaggle T4 결과와 나란히 놓지 말고 이 노트북 안의 variant끼리만 비교할 것')"""

SETUP = """import glob, os, subprocess, sys, zipfile
from google.colab import drive, userdata
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

def secret(k):
    try:
        return userdata.get(k)
    except Exception:
        return None

if secret('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = secret('KAGGLE_API_TOKEN')
elif secret('KAGGLE_USERNAME') and secret('KAGGLE_KEY'):
    os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = secret('KAGGLE_USERNAME'), secret('KAGGLE_KEY')
else:
    raise SystemExit('보안 비밀에 KAGGLE_API_TOKEN (또는 KAGGLE_USERNAME + KAGGLE_KEY)을 넣고 노트북 액세스를 켜세요')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'vismatch', 'kaggle'], check=True)

# 데이터셋을 /content/input에 받고, 안의 zip도 푼다 (이미 받았으면 건너뜀)
IN = '/content/input'
if not glob.glob(IN + '/**/maps', recursive=True):
    subprocess.run(['kaggle', 'datasets', 'download', DATASET, '-p', IN, '--unzip'], check=True)
    for z in glob.glob(IN + '/**/*.zip', recursive=True):
        zipfile.ZipFile(z).extractall(z[:-4]); os.remove(z)
for d in [d for d in glob.glob(IN + '/**/*', recursive=True) if os.path.isdir(d)]:  # zip이 폴더째 들어 있던 경우 한 단계 올림
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner):
        for f in os.listdir(inner): os.rename(os.path.join(inner, f), os.path.join(d, f))
        os.rmdir(inner)
DATA_DIR = os.path.dirname(glob.glob(IN + '/**/maps', recursive=True)[0])
CAP = os.path.dirname(glob.glob(IN + '/**/Bearing_UAV_90K', recursive=True)[0]) if SCRIPT == 'run_nav' else None
print('data', DATA_DIR, 'capstone1', CAP)"""

FILES = "FILES = {files!r}"

RUN = """import base64, os, pickle, sys, threading, time
SRC = '/content/src'
for rel, text in FILES.items():
    p = os.path.join(SRC, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(base64.b64decode(text))
sys.path.insert(0, SRC)

# 끝난 variant(Drive에 run_nav는 _traj.csv, run_visloc_nav는 _segs.csv가 있음)는 건너뛴다
i = ARGS.index('--variants')
DONE = '_traj.csv' if SCRIPT == 'run_nav' else '_segs.csv'
todo = [v for v in ARGS[i + 1].split(',') if not glob.glob(f'{OUT}/*-{v}{DONE}')]
print('남은 variant:', todo or '없음')

# 매칭 캐시를 Drive에 10분마다 저장하고, 다시 돌리면 불러온다 (끊긴 뒤 같은 매칭을 다시 하지 않게).
# 지도마다 MatchLoc가 하나씩 생기므로(run_visloc_nav는 구간 지도마다) 지도 이름별로 나눠 저장한다.
import imatch.nav as nav
CACHE, LIVE = OUT + '/match_cache.pkl', {}
SAVED = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
if not SAVED and CACHE_FROM:
    # 캐시 키는 (사진, 블록, 각도)뿐이라 이동 오차·variant가 달라도 같은 매칭 답이다. 시간(sec)도 처음 잰 값.
    assert os.path.exists(CACHE_FROM), f'{CACHE_FROM}이 없다. 이전 실행의 match_cache.pkl을 그 자리에 두거나 CACHE_FROM = None'
    SAVED = pickle.load(open(CACHE_FROM, 'rb')); print('이전 실행 매칭 캐시로 시작:', CACHE_FROM)
if SAVED and not all(isinstance(v, dict) for v in SAVED.values()):
    SAVED = {'': SAVED}   # 예전 형식(지도 하나)
_init = nav.MatchLoc.__init__

def map_name(city):
    key = getattr(city, 'key', None)
    return os.path.basename(key) if key else ''

def save_cache():
    allc = dict(SAVED)
    for name, m in list(LIVE.items()):
        allc[name] = dict(m.cache)
    pickle.dump(allc, open(CACHE + '.tmp', 'wb')); os.replace(CACHE + '.tmp', CACHE)

def init(self, matcher, city, *a, **k):
    _init(self, matcher, city, *a, **k)
    name = map_name(city)
    if name in SAVED:
        self.cache.update(SAVED[name]); print('이어 쓰는 매칭', name, len(self.cache), flush=True)
    LIVE[name] = self

def every_10_min():
    while True:
        time.sleep(600); save_cache()

nav.MatchLoc.__init__ = init
if todo:
    threading.Thread(target=every_10_min, daemon=True).start()
    rest = ARGS[:i + 1] + [','.join(todo)] + ARGS[i + 2:]
    if SCRIPT == 'run_nav':
        from scripts.run_nav import main
        sys.argv = ['run_nav', '--data', DATA_DIR, '--capstone1', CAP, '--out', OUT] + rest
        main()
    else:
        from scripts.run_visloc_nav import main
        main(['--data', DATA_DIR, '--out', OUT] + rest)
    save_cache()
print('끝. 결과:', sorted(os.listdir(OUT)))"""


def cell(kind, text):
    c = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--args', required=True, help='run_nav / run_visloc_nav 인자. --variants가 있어야 한다 (끝난 것 건너뛰기에 쓴다)')
    ap.add_argument('--script', default='run_nav', choices=['run_nav', 'run_visloc_nav'],
                    help='run_visloc_nav: UAV-VisLoc 재생 항법 (데이터셋에 maps/와 samples.csv)')
    ap.add_argument('--cache-from', default=None,
                    help='이전 Colab 실행의 slug. 그 Drive 폴더의 match_cache.pkl로 시작한다 (데이터·매처·각도가 같을 때만)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'colab'))
    a = ap.parse_args()
    cache_from = f'/content/drive/MyDrive/trone-nav/{a.cache_from}/match_cache.pkl' if a.cache_from else None
    args = shlex.split(a.args)
    assert '--variants' in args, '--variants가 필요하다'
    variants = args[args.index('--variants') + 1]
    files = cell('code', FILES.format(files=source_files()))
    files['metadata'] = {'cellView': 'form', 'jupyter': {'source_hidden': True}}
    nb = dict(nbformat=4, nbformat_minor=0,
              metadata={'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []},
                        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
              cells=[cell('markdown', INTRO.format(slug=a.slug, variants=variants, dataset=a.dataset, script=a.script,
                                                   done='_traj.csv' if a.script == 'run_nav' else '_segs.csv')),
                     cell('code', CONFIG.format(script=a.script, dataset=a.dataset, args=args, slug=a.slug, cache_from=cache_from)),
                     cell('code', GPU), cell('code', SETUP), files, cell('code', RUN)])
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f'{a.slug}.ipynb')
    json.dump(nb, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('만듦:', p)


if __name__ == '__main__':
    main()
