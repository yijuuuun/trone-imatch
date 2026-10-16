# 실험 실행기. 어느 컴퓨터에서나 같은 명령을 쓴다.
#   python run.py check                    설치와 GPU 확인 + 속도 측정 (약 5분)
#   python run.py train <tag> <epoch>      학습 (예: python run.py train csmg_s1 4)
#   python run.py eval [--l2] <tag> ...    항법 평가 (선택)
# 결과(.pth, train_<tag>.txt)는 설치 위치의 out 폴더에 저장된다. 다른 폴더에 저장하려면 --out <폴더>.
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
paths = os.path.join(HERE, 'bearing_paths.txt')
if not os.path.exists(paths):
    sys.exit('먼저 setup_local.py를 실행하세요.')
REPO, OUT = open(paths).read().split('\n')[:2]
args = sys.argv[1:]
if '--out' in args:
    i = args.index('--out'); OUT = os.path.abspath(args[i + 1]); del args[i:i + 2]
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, BEARING_ROOT=REPO, BEARING_OUT=OUT, PYTHONUNBUFFERED='1')
cmd = args[0] if args else 'help'

if cmd == 'check':
    code = r'''
import torch, time
from swap_lib import *
print('장치:', DEV, '| torch', torch.__version__)
m = build('csmg'); m.train(); m.backbone.eval()
x = torch.randn(64, 3, 256, 256, device=DEV); tf = torch.randn(256, 512, 32, 32, device=DEV)
def sync():
    if DEV == 'cuda': torch.cuda.synchronize()
    elif DEV == 'mps': torch.mps.synchronize()
for k in range(3):
    sync(); t = time.time()
    with torch.no_grad(): fu = m.backbone(x)
    fi = desc(m, fu); nb = desc(m, tf).view(64, 4, -1); p, d = head_train(m, fi, nb); (p.sum() + d.sum()).backward(); m.zero_grad()
    sync(); dt = time.time() - t
print(f'CSMG 한 스텝 약 {dt:.1f}초 -> 4 epoch 약 {dt*5340/3600:.1f}시간 (Grid는 이보다 훨씬 빠름)')
print('CHECK_OK')
'''
    sys.exit(subprocess.call([sys.executable, '-c', code], cwd=REPO, env=env))
if cmd == 'train':
    sys.exit(subprocess.call([sys.executable, 'train_swap.py'] + args[1:], cwd=REPO, env=env))
if cmd == 'eval':
    sys.exit(subprocess.call([sys.executable, 'eval_more.py'] + args[1:], cwd=REPO, env=env))
print(__doc__ if __doc__ else open(__file__).read().split('import')[0])
