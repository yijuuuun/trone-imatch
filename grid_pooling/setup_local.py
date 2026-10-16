# 어느 컴퓨터에서나(Windows / Linux / Mac / Colab) 실험 환경을 만드는 설치 스크립트.
# 사용:  python setup_local.py            (기본 설치 위치: 홈 폴더의 bearing)
#        python setup_local.py D:\bearing  (위치 지정)
# 하는 일: 원 저장소 코드, 모델 가중치(131 MB), 데이터셋(6.3 GB)을 내려받아 풀고, 이 폴더의 실험 코드를 넣는다.
# 필요: Python 3.9 이상, 디스크 여유 약 15 GB (설치 후 약 7 GB).
import os, sys, shutil, zipfile, urllib.request

BASE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), 'bearing'))
CODE = os.path.dirname(os.path.abspath(__file__))
REPO, OUT = os.path.join(BASE, 'repo'), os.path.join(BASE, 'out')
URLS = {
    'repo': 'https://codeload.github.com/liukejia121/bearinguav/zip/refs/heads/main',
    'weights': 'https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip',
    'data': 'https://huggingface.co/datasets/HaoyZhou/bearinguav/resolve/main/Bearing_UAV_90K.zip',
}


def download(url, path):
    """끊겨도 이어받는다 (다시 실행하면 이어서 받음)."""
    for _ in range(20):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get('Content-Length', 0))
                if have and r.status != 206:      # 서버가 이어받기를 거부하면 처음부터
                    have, total = 0, int(r.headers.get('Content-Length', 0))
                with open(path, 'ab' if have else 'wb') as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b); have += len(b)
                        print(f'\r  {os.path.basename(path)}: {have/1e9:.2f} / {total/1e9:.2f} GB', end='', flush=True)
            print(); return
        except urllib.error.HTTPError as e:
            if e.code == 416:                     # 이미 다 받음
                print(); return
            print(f'\n  다시 시도 ({e})')
        except Exception as e:
            print(f'\n  다시 시도 ({e})')
    sys.exit('다운로드 실패. 인터넷 연결을 확인하고 다시 실행하세요 (이어받기 됨).')


def unzip(path, dest):
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


os.makedirs(BASE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
print(f'설치 위치: {BASE}')

if not os.path.isdir(os.path.join(REPO, 'cvphr')):
    print('[1/4] 원 저장소 코드'); z = os.path.join(BASE, 'repo.zip'); download(URLS['repo'], z)
    unzip(z, BASE); shutil.move(os.path.join(BASE, 'bearinguav-main'), REPO); os.remove(z)

if not os.path.exists(os.path.join(REPO, 'Bearing_UAV', 'satellite_view', 'best_model.pth')):
    print('[2/4] 모델 가중치'); z = os.path.join(BASE, 'weights.zip'); download(URLS['weights'], z)
    unzip(z, REPO); os.remove(z)

if not os.path.isdir(os.path.join(REPO, 'Bearing_UAV_90K', 'city_rsi')):
    print('[3/4] 데이터셋 6.3 GB (가장 오래 걸림)'); z = os.path.join(BASE, 'data.zip'); download(URLS['data'], z)
    print('  압축 해제 중...'); unzip(z, REPO); os.remove(z)

print('[4/4] 실험 코드')
os.makedirs(os.path.join(REPO, 'diag'), exist_ok=True)
for f in os.listdir(os.path.join(CODE, 'diag')):
    if f.endswith('.py'):
        shutil.copy(os.path.join(CODE, 'diag', f), os.path.join(REPO, 'diag', f))
open(os.path.join(REPO, 'diag', '__init__.py'), 'a').close()
for f in ['swap_lib.py', 'train_swap.py', 'eval_swap.py', 'eval_more.py', 'run.py']:
    shutil.copy(os.path.join(CODE, f), os.path.join(REPO, f))
# 이준이 학습한 기존 모델(비교 대상)을 out 폴더에 넣는다
md = os.path.join(CODE, 'models')
if os.path.isdir(md):
    for f in os.listdir(md):
        if f.endswith('.pth') and not os.path.exists(os.path.join(OUT, f)):
            shutil.copy(os.path.join(md, f), os.path.join(OUT, f)); print(f'  기존 모델 복사: {f}')
with open(os.path.join(BASE, 'bearing_paths.txt'), 'w') as f:
    f.write(f'{REPO}\n{OUT}\n')
shutil.copy(os.path.join(BASE, 'bearing_paths.txt'), os.path.join(CODE, 'bearing_paths.txt'))
print(f'\n설치 완료. 결과는 {OUT} 에 저장됩니다.')
print(f'다음:  python "{os.path.join(CODE, "run.py")}" check')
