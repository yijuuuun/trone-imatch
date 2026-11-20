"""파이프라인 ②③: 특징 추출과 매칭. 모든 매처는 (드론 영상, 블록) → magsac() 결과를 돌려준다.

새 매처를 실험하려면 __call__(im0, im1) 을 가진 클래스를 만들고 get_matcher()에 이름을 추가하면 된다.
"""
import os

import cv2
import numpy as np

from .geometry import magsac

# vismatch의 get_matcher에 없는 LoMa 변형은 직접 만든다 ('loma' = LoMa-B는 get_matcher로)
LOMA_ARCH = {'loma-b128': 'LoMa-B128', 'loma-l': 'LoMa-L', 'loma-g': 'LoMa-G'}

class VisMatch:
    """vismatch 라이브러리의 공개 매처 (loma, romav2, sift-lightglue 등). GPU 권장.

    LoMa 변형: 'loma'(= LoMa-B, 주력), 'loma-b128'(경량), 'loma-l', 'loma-g', 'loma-r'.
    이름 끝에 '-cache'를 붙이면 지도 블록 특징을 캐시한다 (예: 'loma-cache').
    """

    def __init__(self, name, device='cuda'):
        import torch
        cache = name.endswith('-cache')   # 'loma-cache': 지도 블록 특징 캐시 (결과는 'loma'와 같고 시간만 줄어든다)
        name = name[:-len('-cache')] if cache else name
        self.name, self.torch = name + ('-cache' if cache else ''), torch
        if name in LOMA_ARCH:
            from vismatch.im_models import loma
            self.m = loma.LoMaMatcher(device, 2048, arch=LOMA_ARCH[name])
        else:
            from vismatch import get_matcher
            self.m = get_matcher(name, device=device)
        if name.startswith('loma') and os.environ.get('LOMA_FP32'):
            # GTX 1650은 fp16에서 키포인트가 전부 NaN이 되어 오류 없이 짝 0개가 나온다. LOMA_FP32=1로 fp32를 쓴다.
            # T4 결과(fp16)와 조금 다를 수 있으니 보고용 숫자는 T4 기본 설정으로 낸다
            from vismatch.utils import force_float32
            force_float32(self.m.matcher)
        if cache:
            if not name.startswith('loma'):
                raise ValueError('-cache는 LoMa만 지원')
            self._cache_block_features(max_items=int(os.environ.get('LOMA_CACHE_ITEMS', 512)))
        if name == 'romav2':
            # 기본 'precise' 설정은 T4(bf16 없음)에서 메모리 부족으로 멈췄다. 우리 영상은 256 / 512 px라
            # 'base' 설정(640 px, 단방향, fp32)을 쓴다.
            from vismatch.im_models import romav2 as rv2
            rv2._disable_autocast()
            self.m.romav2_model.apply_setting(os.environ.get('ROMAV2_SETTING', 'base'))
            self.name = f"romav2-{os.environ.get('ROMAV2_SETTING', 'base')}"

    def _cache_block_features(self, max_items=512, min_side=400):
        """지도 블록의 LoMa 키포인트·디스크립터를 한 번만 계산한다 (교수님 피드백: 지도 특징 캐시).

        같은 블록은 방향(최대 4개)마다, 항법에서는 걸음·재탐색마다 다시 들어온다. 드론 사진만 돌리므로 블록 쪽
        detect_and_describe 결과는 매번 같다(DaD 키포인트는 top-k라 결정적). 그래서 블록(한 변 >= min_side px)만
        내용 해시로 캐시하고, 드론 사진(256 px)은 그대로 계산한다. 결과는 캐시 없을 때와 같다.
        """
        import hashlib
        from collections import OrderedDict
        lm = self.m.matcher
        orig, store = lm.detect_and_describe, OrderedDict()
        self.cache_hits = self.cache_misses = 0

        def cached(image, num_keypoints=None):
            if isinstance(image, str) or min(image.shape[-2:]) < min_side:
                return orig(image, num_keypoints)
            key = (hashlib.md5(image.detach().cpu().numpy().tobytes()).hexdigest(), tuple(image.shape), num_keypoints)
            if key in store:
                store.move_to_end(key); self.cache_hits += 1
                return store[key]
            self.cache_misses += 1
            out = store[key] = orig(image, num_keypoints)
            if len(store) > max_items:
                store.popitem(last=False)
            return out

        lm.detect_and_describe = cached

    def __call__(self, im0, im1):
        t = lambda im: self.torch.from_numpy(im).permute(2, 0, 1).float() / 255.
        r = self.m(t(im0), t(im1))
        p0, p1 = r['matched_kpts0'], r['matched_kpts1']
        # vismatch는 inlier 점만 주고 마스크는 안 주므로 같은 점으로 MAGSAC++를 다시 돌린다
        return magsac(np.float32(p0).reshape(-1, 2), np.float32(p1).reshape(-1, 2))


class CvSift:
    """GPU 없이 도는 기준선: OpenCV SIFT + Lowe 비율 검사 + MAGSAC++. 설치·동작 확인용."""
    name = 'cv-sift'

    def __init__(self):
        self.sift = cv2.SIFT_create(4096); self.bf = cv2.BFMatcher()

    def __call__(self, im0, im1):
        k0, d0 = self.sift.detectAndCompute(cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY), None)
        k1, d1 = self.sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), None)
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), None, np.zeros(0, bool)
        mm = [m for m, n in self.bf.knnMatch(d0, d1, k=2) if m.distance < 0.8 * n.distance]
        p0 = np.float32([k0[m.queryIdx].pt for m in mm]).reshape(-1, 2)
        p1 = np.float32([k1[m.trainIdx].pt for m in mm]).reshape(-1, 2)
        return magsac(p0, p1)


class VggMNN:
    """비교 실험 'VGG16 백본은 그대로 두면?' (docs/02_실험기록.md 7장). 원 Bearing-UAV 체크포인트의 고정
    VGG16 conv4_3(B2가 상관을 계산하는 그 특징)을 점 단위 기술자로 쓴다.

    특징 맵의 칸(stride 8, 중심 8 i + 4)이 점이다. 블록 평균을 뺀 코사인 유사도로 상호 최근접 짝을 짓고,
    이후는 다른 매처와 같은 MAGSAC++ / 호모그래피를 쓴다. 회전으로 생긴 검은 모서리의 칸은 뺀다.
    capstone1 저장소와 체크포인트가 필요하다:
      CAPSTONE1_DIR=<capstone1 경로> VGG_CKPT=Bearing_UAV/cross_view python scripts/run.py --matchers vgg-mnn --device cpu
    결과: Singapore 44%(B2 69%, 100샘플) → 원 모델 특징으로는 점 단위 매칭이 안 된다.
    """
    name = 'vgg-mnn'

    def __init__(self):
        import sys
        import torch
        root = os.environ['CAPSTONE1_DIR']
        sys.path.insert(0, root)
        from diag.diag_modules import DEV, load_model, to_tensor
        self.torch, self.tt, self.dev = torch, to_tensor, DEV
        self.bb = load_model(os.path.join(root, os.environ.get('VGG_CKPT', 'Bearing_UAV/cross_view'))).backbone

    def _feat(self, im):
        with self.torch.no_grad():
            f = self.bb(self.tt(im).unsqueeze(0).to(self.dev))[0]          # [C, h, w]
        C, h, w = f.shape; st = im.shape[0] / h
        yy, xx = np.mgrid[0:h, 0:w]
        pts = np.stack([xx.ravel() * st + st / 2, yy.ravel() * st + st / 2], 1).astype(np.float32)
        return f.reshape(C, -1).T, pts

    def __call__(self, im0, im1):
        F = self.torch.nn.functional
        d0, p0 = self._feat(im0); d1, p1 = self._feat(im1)
        valid = im0[np.clip(p0[:, 1].astype(int), 0, im0.shape[0] - 1), np.clip(p0[:, 0].astype(int), 0, im0.shape[1] - 1)].sum(1) > 0
        d0, p0 = d0[self.torch.from_numpy(valid)], p0[valid]
        mu = d1.mean(0, keepdim=True)
        d0, d1 = F.normalize(d0 - mu, dim=1), F.normalize(d1 - mu, dim=1)
        S = d0 @ d1.T
        j = S.argmax(1); i = S.argmax(0)
        mutual = (i[j] == self.torch.arange(len(j), device=j.device)).cpu().numpy()
        j = j.cpu().numpy()
        return magsac(p0[mutual], p1[j[mutual]])


def get_matcher(name, device='cuda'):
    if name == 'cv-sift':
        return CvSift()
    if name == 'vgg-mnn':
        return VggMNN()
    return VisMatch(name, device)
