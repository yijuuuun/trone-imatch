# 실험 5: CSMG가 위치 정보를 잃나? (학습 없음, 희윤의 diag_gluf.eval_set 그대로 사용)
# 같은 frozen VGG conv4_3 지도 위에서 CSMG(학습) / CSMG 내부 GAP / VGG GAP·GeM(학습 없음) / dense 비교.
# retr_top1 = UAV 중심이 블록 2x2 타일 중 어디 있는지 맞춘 비율(우연 25%). full_model MLE = Bearing 전체 위치오차.
import os, sys, json
from unittest.mock import MagicMock
import numpy as np
for k, v in {'complex_': np.complex128, 'float_': np.float64, 'unicode_': np.str_, 'string_': np.bytes_}.items():
    if not hasattr(np, k): setattr(np, k, v)
sys.modules['imgaug'] = MagicMock(); sys.modules['imgaug.augmenters'] = MagicMock()
os.chdir('/content/bearinguav'); sys.path.insert(0, '/content/bearinguav')
from diag.diag_modules import SEEN, UNSEEN, load_model
from diag.diag_nav import load_rsi
from diag.diag_gluf import eval_set, gen_rgb, clean

OUT = '/content/drive/MyDrive/bearinguav_cache/csmg_check.txt'
m2 = load_model('/content/bearinguav/Bearing_UAV/satellite_view')
rng = np.random.default_rng(5); N = 200
sets = {'Tokyo(학습) 주간': (SEEN['34bc'], None), 'NY(낯선) 주간': (UNSEEN['NY'], None), 'NY(낯선) L2': (UNSEEN['NY'], 'L2')}
open(OUT, 'w').close()
for name, (pre, dark) in sets.items():
    o, _ = eval_set(m2, gen_rgb(load_rsi(pre), N, rng, dark=dark))
    line = f'{name} (n={N}) ' + json.dumps(clean(o), ensure_ascii=False)
    print(line, flush=True); open(OUT, 'a').write(line + '\n')
open(OUT, 'a').write('# done\n')
