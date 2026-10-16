# CSMG 대체 실험 공통 코드 (이준). Bearing의 VGG16(얼림) + RCE/CA/PSG/헤드 구조는 그대로, CSMG 자리만 교체.
import os, sys
from unittest.mock import MagicMock
import numpy as np
for k, v in {'complex_': np.complex128, 'float_': np.float64, 'unicode_': np.str_, 'string_': np.bytes_}.items():
    if not hasattr(np, k): setattr(np, k, v)
sys.modules['imgaug'] = MagicMock(); sys.modules['imgaug.augmenters'] = MagicMock()
ROOT = os.environ.get('BEARING_ROOT', '/content/bearinguav')
OUTD = os.environ.get('BEARING_OUT', '/content/drive/MyDrive/bearinguav_cache/swap')
os.chdir(ROOT); sys.path.insert(0, ROOT)

import re, cv2, torch, torch.nn as nn, torch.nn.functional as F
from diag.diag_modules import DEV, to_tensor, load_model
from cvphr.sceneGraphEncodingNet.nets import CSMG_soft
from cvphr.models.posaglreg.models import load_config_and_model

WEIGHTS = f'{ROOT}/Bearing_UAV/satellite_view'
DT = 'cuda' if DEV == 'cuda' else ('mps' if DEV == 'mps' else 'cpu')  # autocast 장치


class GridPool(nn.Module):
    """CSMG 대체: 1x1 conv(512->64) + 4x4 격자 평균 풀링 -> 64*16 = 1024 (CSMG 출력과 같은 크기).
    클러스터로 뭉뚱그리지 않고 칸별로 따로 요약해 '어디에 무엇이 있는지'를 남긴다."""
    def __init__(self, cin=512, c=64, g=4, hidden=0):
        super().__init__()
        if hidden:  # 교란 변수 분리용: 1x1 conv를 두 층(512->hidden->c)으로 해서 파라미터 수만 바꾼다
            self.proj = nn.Sequential(nn.Conv2d(cin, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                                      nn.Conv2d(hidden, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        else:
            self.proj = nn.Sequential(nn.Conv2d(cin, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.g = g
    def forward(self, x):
        y = F.adaptive_avg_pool2d(self.proj(x), self.g)        # [B,64,4,4]
        y = F.normalize(y, dim=1)                               # 칸마다 정규화
        return None, None, F.normalize(y.flatten(1), dim=1), None  # CSMG와 같은 반환 형식


def unrotate(rgb, a):
    """나침반 방향으로 UAV 영상을 북쪽이 위가 되게 되돌림 (diag_gluf.rot_tpl과 같은 각도 규약, 자르지 않음)."""
    R = cv2.getRotationMatrix2D((128, 128), a, 1)
    return cv2.warpAffine(rgb, R, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def parse_tag(tag):
    """'csmg', 'grid', 'grid_north', 'grid8', 'grid_s1', 'grid_dark', 'grid_dark_s1' ... -> (base, 격자 크기 g, 시드).
    gridN은 N×N 격자 (출력 1024 유지: 채널 = 1024 / N²). _dark는 저조도 사본 학습. _sK는 학습 시드 K (기본 0)."""
    tag = re.sub(r'_h\d+', '', tag)  # _hN: GridPool 은닉 채널 (hidden_of 참고)
    tag = re.sub(r'_e\d+', '', tag)  # _eN: 학습 epoch 표시 (저장 이름 구분용, 모델 구조와 무관)
    seed = 0
    if '_s' in tag and tag.rsplit('_s', 1)[1].isdigit():
        tag, seed = tag.rsplit('_s', 1)[0], int(tag.rsplit('_s', 1)[1])
    base = re.sub(r'_(dark|phys)$', '', tag)
    g = int(base[4:]) if base.startswith('grid') and base[4:].isdigit() else 4
    return base, g, seed


def hidden_of(tag):
    """'_hN'이 있으면 GridPool의 1x1 conv를 512->N->c 두 층으로 만든다 (파라미터 수만 조절). 없으면 0."""
    h = re.search(r'_h(\d+)', tag)
    return int(h.group(1)) if h else 0


def is_dark_train(tag):
    """저조도 학습 방식: '_dark' = 단순 합성(darken), '_phys' = 희윤님 물리 카메라 모델(capture). 없으면 None."""
    return 'phys' if '_phys' in tag else ('simple' if '_dark' in tag else None)


LIGHT_MIN, LIGHT_MAX, STRATEGIES = 1 / 400, 1 / 8, ['iso', 'exposure', 'auto']  # 희윤님 lowlight_train과 같은 분포


def phys_dark(rgb, rng, light=None, strategy=None):
    """희윤님 diag_lowlight.capture()로 저조도 영상을 만든다. light/strategy를 안 주면 학습용 분포에서 뽑는다."""
    from diag.diag_lowlight import capture
    if light is None:
        light = float(np.exp(rng.uniform(np.log(LIGHT_MIN), np.log(LIGHT_MAX))))
    if strategy is None:
        strategy = STRATEGIES[int(rng.integers(3))]
    return capture(rgb, light, strategy, rng)


def is_north(tag):
    return parse_tag(tag)[0] == 'grid_north'


def build(tag, init='fresh'):
    """tag: parse_tag 참고. init='fresh': VGG만 공개 가중치, 나머지 전부 새로 초기화 (모든 변형 같은 조건)."""
    base, g, seed = parse_tag(tag)
    m = load_model(WEIGHTS)
    if init == 'fresh':
        cls, kw = load_config_and_model(WEIGHTS)
        torch.manual_seed(seed); fresh = cls(**kw).to(DEV)
        sd = {k: v for k, v in fresh.state_dict().items() if not k.startswith('backbone.')}
        m.load_state_dict(sd, strict=False)
    if base.startswith('grid'):
        assert 1024 % (g * g) == 0, g
        torch.manual_seed(seed); m.csmg = GridPool(c=1024 // (g * g), g=g, hidden=hidden_of(tag)).to(DEV)
    for p in m.backbone.parameters(): p.requires_grad = False
    m.backbone.eval()
    return m


def head_train(m, fi, nb):
    """diag_modules.head와 같은 계산 (grad 허용). RCE -> CA -> PSG -> 회귀."""
    B = fi.size(0)
    kc = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], device=fi.device)
    ce = m.coord_encoder(kc).unsqueeze(0).expand(B, -1, -1)
    psg = m.sim_pos_prior(fi, nb)
    nbk = nb + ce if m.add_patch_coord else nb
    ctx = m.neighbors_cross_attn(fi, nbk)
    comb = torch.cat([fi, ctx], 1)
    return m.pos_regressor(torch.cat([comb, psg], 1)), m.dir_regressor(comb)


def desc(m, feat):
    return m.csmg(feat)[2]


class SwapLoc:
    """항법용 위치추정기 (ModelLoc과 같은 인터페이스). north=True면 나침반으로 UAV를 북쪽 정렬."""
    def __init__(self, m, img, north):
        self.m, self.img, self.north, self.c = m, img, north, {}
    @torch.no_grad()
    def tile(self, k):
        if k not in self.c:
            t = self.img[k[1]*256:(k[1]+1)*256, k[0]*256:(k[0]+1)*256]
            self.c[k] = desc(self.m, self.m.backbone(to_tensor(t).unsqueeze(0).to(DEV)))[0]
        return self.c[k]
    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        if self.north: crops = [unrotate(c, h) for c, h in zip(crops, headings)]
        x = torch.stack([to_tensor(c) for c in crops]).to(DEV)
        fi = torch.cat([desc(self.m, self.m.backbone(x[k:k+64])) for k in range(0, len(x), 64)])
        nb = torch.stack([torch.stack([self.tile((bx+dx, by+dy)) for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]])
                          for bx, by in blocks])
        return head_train(self.m, fi, nb)[0].cpu().numpy()
