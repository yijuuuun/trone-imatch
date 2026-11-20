"""UAV-VisLoc 재생 항법(imatch/replay.py)에 꽂는 기존 모델 어댑터.

모든 모델이 같은 구간·출발 위치(GPS)·이동 오차(vio, seed 2026)·블록 고르기 규칙(추측 위치를 담은 512 px 블록)을 쓴다.
loc(crops, headings, blocks) → (n, 2) 블록 중심 기준 위치 (1 = 128 px, x 오른쪽, y 아래). loc.last에 걸음 시간(sec)을 남긴다.

- BearingLoc   원 Bearing-UAV 공개 가중치 (#42와 같은 입력: 블록을 256 px 타일 4장 + 드론 256 px). 방향을 스스로 추정.
- OSFPILoc     OS-FPI (Chen et al. 2024, github.com/cjhing/OS-FPI, net_040.pth). 드론 96 px, 블록 384 px.
               회전을 다루지 않는 모델이라 4방향(0/90/180/270°)으로 돌려 넣고 heatmap 최고점이 가장 높은 방향을 쓴다.
- CorrLoc      CorrMatchNet (capstone1, 학습 도시 3D로 학습). 드론을 north_up으로 돌려 넣는 모델이라 나침반 대신
               4방향을 넣고 신뢰도(conf)가 가장 높은 방향을 쓴다.
- run_geoanchor GeoAnchor(github.com/YusufGUNEL/GeoAnchor)의 융합 방식을 우리 조건으로: 그 저장소의 LoFTR 매처·
               ParticleFilter·DINOv2 임베더와 step() 규칙(게이트, 주변 타일, 잃었을 때 전역 재탐색, 불일치 시 재초기화)을
               그대로 쓰고, 입력만 우리 것(64 m 드론 크롭, 같은 블록, 모의 이동량)이다. 나침반이 없으므로 4방향 탐색,
               온라인 yaw·축척 보정과 흐림 게이트는 끈다(아래 run_geoanchor 설명).
"""
import os
import sys
import time
import types

import cv2
import numpy as np
import pandas as pd
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def block_img(city, bx, by):
    """추측 위치 블록 = 지도 px (bx*256, by*256)에서 512 px 정사각형. 중심은 (bx*256+256, by*256+256)."""
    return city[by * 256:by * 256 + 512, bx * 256:bx * 256 + 512]


def _norm(rgb):
    x = (rgb.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def _sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def stub_cvphr_transform():
    """cvphr 학습용 증강 모듈(imgaug, numpy 1 전용)은 추론에서 쓰지 않는다. build_kernel.py run_nav_b1과 같은 처리."""
    ut = types.ModuleType('cvphr.utils.utils_transform')

    def _unused(*a, **k):
        raise RuntimeError('학습용 증강은 추론에서 쓰지 않음')
    ut.transform_pipeline3 = ut.transform_pipeline1_gentle = ut.transform_pipeline_weather = _unused
    sys.modules['cvphr.utils.utils_transform'] = ut


# ------------------------------------------------------------------------------------------------ Bearing-UAV
class BearingLoc:
    """#42 bearing_adapter.py와 같은 입력·전처리. 블록 타일 순서 TL, TR, BL, BR + 드론."""

    def __init__(self, cap, weights='Bearing_UAV/cross_view', device='cuda'):
        sys.path.insert(0, cap)
        stub_cvphr_transform()
        from cvphr.models.posaglreg.models import load_config_and_model
        d = os.path.join(cap, weights)
        cls, kw = load_config_and_model(d)
        self.m = cls(**kw)
        st = torch.load(os.path.join(d, 'best_model.pth'), map_location='cpu', weights_only=False)
        self.m.load_state_dict(st['model_state_dict'], strict=True)
        self.m = self.m.float().to(device).eval()
        self.dev, self.city, self.last = device, None, []

    def bind(self, city):
        self.city = city
        return self

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            t0 = time.time()
            b = block_img(self.city, bx, by)
            patches = [b[y:y + 256, x:x + 256] for x, y in [(0, 0), (256, 0), (0, 256), (256, 256)]] + [crop]
            x = torch.stack([_norm(p) for p in patches])[None].to(self.dev)
            pos, _ = self.m(x)
            _sync(self.dev)
            out.append(pos[0].float().cpu().numpy())
            self.last.append(dict(sec=time.time() - t0))
        return np.array(out)


# ------------------------------------------------------------------------------------------------ OS-FPI
def _stub_osfpi_deps():
    """OS-FPI 백본·neck이 import하는 mmdet/mmcv/resnest. 추론 경로에서 쓰는 것은 auto_fp16 데코레이터뿐이고
    fp16_enabled가 없으면 mmcv에서도 아무 일도 하지 않는다. 사전학습 백본 불러오기(load_checkpoint)는 우리가 쓰지 않는다
    (net_040.pth 전체를 load_state_dict로 넣는다). resnest는 CNN 백본용이라 os_pcpvt_small에서 쓰지 않는다."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules.setdefault(name, m)
        return m

    class _Reg:
        def register_module(self, *a, **k):
            return a[0] if a and callable(a[0]) else (lambda c: c)

    def ident(*a, **k):
        return lambda f: f

    def _unused(*a, **k):
        raise RuntimeError('OS-FPI 추론에서 쓰지 않음')
    mod('mmdet'); mod('mmdet.models'); mod('mmdet.models.builder', BACKBONES=_Reg())
    mod('mmdet.utils', get_root_logger=lambda *a, **k: None)
    mod('mmcv'); mod('mmcv.runner', load_checkpoint=_unused, auto_fp16=ident, force_fp32=ident)
    mod('resnest'); mod('resnest.torch', resnest50=_unused, resnest101=_unused)


class OSFPILoc:
    """tool/model_test_server.py의 디코딩 그대로: sigmoid → hanning(centerR) → argmax + loc 오프셋."""

    def __init__(self, repo, ckpt='net_040.pth', device='cuda', angles=4, uav_crop=256):
        """uav_crop: 드론 256 px 크롭의 가운데 몇 px를 넣을지. OS-FPI는 저고도(80~100 m) UL14로 학습돼 드론/위성 폭
        비율에 민감하다. 개발용 single_calib에서 고른다 (calib_osfpi.py)."""
        self.uav_crop = int(uav_crop)
        _stub_osfpi_deps()
        sys.path.insert(0, repo)
        import yaml
        from models.model import make_model
        cfg = yaml.safe_load(open(os.path.join(repo, 'opts.yaml')))
        opt = types.SimpleNamespace(**cfg)
        self.m = make_model(opt)
        sd = torch.load(os.path.join(repo, ckpt), map_location='cpu', weights_only=False)
        self.m.load_state_dict({k.split('module.')[-1]: v for k, v in sd.items()})
        self.m = self.m.to(device).eval()
        self.uav_hw, self.sat_hw, self.R = tuple(opt.UAVhw), tuple(opt.Satellitehw), int(opt.centerR)
        h = np.outer(np.hanning(self.R + 2), np.hanning(self.R + 2))
        self.kernel = (h / h.sum())[1:-1, 1:-1]
        self.dev, self.k_rot, self.city, self.last = device, list(range(angles)) if angles == 4 else [0], None, []

    def bind(self, city):
        self.city = city
        return self

    def _resize(self, rgb, hw):  # torchvision Resize(interpolation=3 = bicubic) + ToTensor (정규화 없음)
        from PIL import Image
        im = Image.fromarray(rgb).resize((hw[1], hw[0]), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.)

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            t0 = time.time()
            sat = self._resize(np.ascontiguousarray(block_img(self.city, bx, by)), self.sat_hw)
            o = (crop.shape[0] - self.uav_crop) // 2
            cc = crop[o:o + self.uav_crop, o:o + self.uav_crop]
            z = torch.stack([self._resize(np.ascontiguousarray(np.rot90(cc, r)), self.uav_hw) for r in self.k_rot])
            x = sat[None].expand(len(self.k_rot), -1, -1, -1)
            cls, loc = self.m(z.to(self.dev), x.to(self.dev))
            cls = torch.sigmoid(cls).float().cpu().numpy()[:, 0]
            loc = loc.float().cpu().numpy()
            best = None
            for i in range(len(self.k_rot)):
                mp = cv2.filter2D(cls[i], -1, self.kernel)
                r, c = divmod(int(np.argmax(mp)), mp.shape[-1])
                rc = (np.array([r, c], float) + loc[i][:, r, c]) * self.sat_hw[0] / loc.shape[-1]
                if best is None or mp[r, c] > best[0]:
                    best = (float(mp[r, c]), rc, self.k_rot[i])
            _sync(self.dev)
            row, col = best[1] * 512. / self.sat_hw[0]
            out.append([(col - 256.) / 128., (row - 256.) / 128.])
            self.last.append(dict(sec=time.time() - t0, conf=best[0], rot=90 * best[2]))
        return np.array(out)


# ------------------------------------------------------------------------------------------------ CorrMatchNet
class CorrLoc:
    """scripts/making_loc.py CorrNavLoc과 같은 전처리. 방향(나침반) 대신 4방향 중 conf 최고."""

    def __init__(self, cap, ckpt_dir, device='cuda', angles=4):
        sys.path.insert(0, cap)
        stub_cvphr_transform()
        from cvphr.models.corrmatch import CorrMatchNet, north_up
        self.m = CorrMatchNet.load(os.path.join(cap, ckpt_dir), device=device)
        self.north_up = north_up
        self.angles = [0., 90., 180., 270.] if angles == 4 else [0.]
        self.dev, self.city, self.cache, self.last = device, None, {}, []

    def bind(self, city):
        self.city, self.cache = city, {}
        return self

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            t0 = time.time()
            if (bx, by) not in self.cache:
                b = block_img(self.city, bx, by)
                tiles = [b[y:y + 256, x:x + 256] for x, y in [(0, 0), (256, 0), (0, 256), (256, 256)]]
                self.cache[(bx, by)] = self.m.tile_features(torch.stack([_norm(t) for t in tiles]).to(self.dev))
            tf = self.cache[(bx, by)][None].expand(len(self.angles), -1, -1, -1, -1)
            u = torch.stack([_norm(self.north_up(crop, a)) for a in self.angles]).to(self.dev)
            pos, conf = self.m.predict(u, tf)
            i = int(torch.argmax(conf))
            _sync(self.dev)
            out.append(pos[i].float().cpu().numpy())
            self.last.append(dict(sec=time.time() - t0, conf=float(conf[i]), rot=self.angles[i]))
        return np.array(out)


# ------------------------------------------------------------------------------------------------ R1: 모델 점수로 "모름"
class Gated:
    """기존 모델의 점수(key)를 replay의 n_inl 자리에 넣는다. run_segments(reject_inl=T)면 점수 < T일 때 "모름" →
    추측 위치. v1의 R1(LoMa inlier < 15)과 같은 처리이고, T는 개발용 single_calib에서 고른다(calib_gate.py, gate.json)."""

    def __init__(self, loc, key):
        self.loc, self.key, self.last = loc, key, []

    def bind(self, city):
        self.loc.bind(city)
        return self

    def __call__(self, crops, headings, blocks):
        out = self.loc(crops, headings, blocks)
        self.last = [dict(d, n_inl=d[self.key]) for d in self.loc.last]
        return out


# ------------------------------------------------------------------------------------------------ 검색형 (MEAN, Sample4Geo, Game4Loc)
def _purge(prefix):
    """MEAN과 Sample4Geo 저장소가 같은 패키지 이름(sample4geo)을 써서, 다른 쪽을 불러오기 전에 지운다."""
    for k in [k for k in sys.modules if k == prefix or k.startswith(prefix + '.')]:
        del sys.modules[k]


class RetrievalLoc:
    """검색형 모델을 항법에 꽂는다: 블록 안 후보 위치마다 그 점을 중심으로 512 px(128 m) 위성 타일을 잘라
    드론 사진과 코사인 유사도가 가장 높은 타일의 중심을 위치로 쓴다 (모델은 그대로, 위치 = 고른 타일 중심).

    후보: 블록 중심 ±128 px를 64 px(16 m) 간격 5×5로 보고, 가장 높은 점 주변 ±32 px(8 m 간격 3×3)로 한 번 더 좁힌다.
    드론 사진은 4방향(0/90/180/270°) 중 유사도 최고. 불러오기·전처리는 재홍 #33(results/kaggle_2026-11-14 source/
    retrieval_worker.py)과 같다: 공식 val 변환(384 px INTER_LINEAR_EXACT + Normalize), fp32, MEAN은 출력[-2](전역 특징).
    Game4Loc은 공식 eval_gta.py와 같다 (DesModel, img_size 384, 같은 가중치로 드론·위성).
    """
    STEP1, STEP2 = 64, 32

    def __init__(self, name, repo, ckpt, device='cuda', batch=32):
        self.name, self.dev, self.batch = name, device, batch
        _purge('sample4geo'); _purge('game4loc')
        sys.path.insert(0, repo)
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        for key in ('state_dict', 'model_state_dict', 'model'):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if all(k.startswith('module.') for k in state):
            state = {k[7:]: v for k, v in state.items()}
        if name == 'sample4geo':
            # MEAN 코드는 import될 때 timm의 convnext 등록을 자기 구현으로 바꾼다. 그 뒤에 만들면 다른 구조가 된다.
            if getattr(RetrievalLoc, '_mean_loaded', False):
                raise RuntimeError('Sample4Geo는 MEAN보다 먼저 불러와야 한다 (--models 순서)')
            from sample4geo.model import TimmModel
            m = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, img_size=384)
            cfg = m.get_config()
        elif name == 'mean':
            RetrievalLoc._mean_loaded = True
            import importlib
            from sample4geo.hand_convnext.model import make_model
            mod = importlib.import_module('sample4geo.hand_convnext.ConvNext.make_model')
            orig = mod.create_model
            mod.create_model = lambda *a, **k: orig(*a, **{**k, 'pretrained': False})
            try:
                m = make_model(types.SimpleNamespace(views=2, nclasses=701, block=2, triplet_loss=.3, resnet=False))
            finally:
                mod.create_model = orig
            exp = m.state_dict()   # 재홍 #33과 같은 키 이름 변환 (model_1.MSE.* → model_1.DEG.*, 다섯 개)
            for s in ('FC11.weight', 'FC12.weight', 'FC13.weight', 'FC1.weight', 'FC1.bias'):
                if 'model_1.MSE.' + s in state and 'model_1.DEG.' + s in exp:
                    state['model_1.DEG.' + s] = state.pop('model_1.MSE.' + s)
            cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        elif name == 'game4loc':
            if 'thop' not in sys.modules:   # model.py가 import만 함 (FLOPs 측정용)
                sys.modules['thop'] = types.ModuleType('thop'); sys.modules['thop'].profile = None
            from game4loc.models.model import DesModel
            m = DesModel('vit_base_patch16_rope_reg1_gap_256.sbb_in1k', pretrained=False, img_size=384, share_weights=True)
            cfg = m.get_config()
        else:
            raise ValueError(name)
        m.load_state_dict(state, strict=True)
        self.m = m.float().to(device).eval()
        self.mean = np.array(cfg['mean'], np.float32); self.std = np.array(cfg['std'], np.float32)
        self.city, self.cache, self.last = None, {}, []

    def bind(self, city):
        self.city = cv2.copyMakeBorder(city, 512, 512, 512, 512, cv2.BORDER_REFLECT)   # 가장자리 타일용
        self.cache = {}
        return self

    @torch.no_grad()
    def embed(self, imgs):
        out = []
        for i in range(0, len(imgs), self.batch):
            x = np.stack([(cv2.resize(im, (384, 384), interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32) / 255.
                           - self.mean) / self.std for im in imgs[i:i + self.batch]])
            x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.dev)
            with torch.autocast('cuda', enabled=False):
                y = self.m(x)
            if self.name == 'mean':
                y = y[-2]
            out.append(torch.nn.functional.normalize(y.float(), dim=-1))
        return torch.cat(out)

    def tiles(self, pts):
        """지도 px 점들 → 각 점 중심 512 px 타일 임베딩 (구간 안 캐시). 새로 계산한 시간을 돌려준다."""
        new = [p for p in pts if p not in self.cache]
        t0 = time.time()
        if new:
            e = self.embed([self.city[y + 256:y + 768, x + 256:x + 768] for x, y in new])
            for p, v in zip(new, e):
                self.cache[p] = v
        _sync(self.dev)
        return torch.stack([self.cache[p] for p in pts]), time.time() - t0

    def __call__(self, crops, headings, blocks):
        out, self.last = [], []
        for crop, (bx, by) in zip(crops, blocks):
            t0 = time.time()
            q = self.embed([np.ascontiguousarray(np.rot90(crop, r)) for r in range(4)])   # [4, d]
            cx, cy = bx * 256 + 256, by * 256 + 256
            p1 = [(cx + dx, cy + dy) for dy in range(-128, 129, self.STEP1) for dx in range(-128, 129, self.STEP1)]
            t1, s1 = self.tiles(p1)
            sim1 = (q @ t1.T).max(0).values
            bx1, by1 = p1[int(sim1.argmax())]
            p2 = [(bx1 + dx, by1 + dy) for dy in (-self.STEP2, 0, self.STEP2) for dx in (-self.STEP2, 0, self.STEP2)]
            t2, s2 = self.tiles(p2)
            sim2 = (q @ t2.T).max(0).values
            i = int(sim2.argmax())
            _sync(self.dev)
            sec = time.time() - t0
            out.append([(p2[i][0] - cx) / 128., (p2[i][1] - cy) / 128.])
            self.last.append(dict(sec=sec, sec_query=sec - s1 - s2, sim=float(sim2[i])))
        return np.array(out)


# ------------------------------------------------------------------------------------------------ GeoAnchor 방식
def run_geoanchor(ga_dir, data, odo='vio', seed=2026, device='cuda', sets=None, verbose=True):
    """GeoAnchor SequentialLocalizer.step()의 규칙을 우리 재생 항법 조건으로 옮긴 것.

    그대로 쓰는 것 (ga_dir의 geoanchor 패키지 = 그 저장소 src/): LoFTRMatcher(outdoor, 640 px), ParticleFilter·PFConfig 기본값,
    sigma/weight_from_inliers, Dinov2Embedder(vit_small dinov2, both). 규칙 수치도 기본값: local_min_inliers 40,
    gate = clip(3 × spread, 120, 400) m, quality_target 150, 주변 타일 400 m 안 3개, 잃었을 때(is_lost) 재탐색 쿨다운 4,
    전역 후보 top-40 → 200 m 떨어진 8개, early_exit 200, 불일치 40 m가 5걸음이면 측정으로 재초기화.
    바꾼 것 (우리 조건이라서)
      - 입력: 드론 사진 = seq_eval 256 px(64 m) 크롭, 위성 = 추측 위치를 담은 512 px(128 m) 블록 (원본: 300 m 프레임,
        추측 위치 중심 400 m 크롭). 주변 타일·전역 후보도 같은 블록 격자(64 m 간격)에서 고른다.
      - 이동량: 우리 모의 VIO (다른 방법과 같은 난수). 원본은 프레임 간 SIFT 시각 오도메트리 + IMU yaw.
      - 나침반 없음: 드론 크롭을 4방향(0/90/180/270°)으로 돌려 매칭해 inlier 최대(150이면 멈춤). 원본은 IMU yaw로 북쪽 정렬.
        그래서 온라인 yaw·축척 보정(나침반·고도계 오차 학습)은 끈다. 흐림 게이트·블러 맞춤도 끈다(우리 크롭엔 흐린 사진 없음).
      - 자세(pitch/roll) 보정 없음 (다른 방법도 쓰지 않음).
      - RANSAC 임계값: 원본 4 px @ 0.625 m/px = 2.5 m → 블록 px(0.25 m/px)로 10 px.
    출력: imatch.replay.run_segments와 같은 열의 (걸음 표, 구간 표).
    """
    sys.path.insert(0, ga_dir)
    from geoanchor.features import Dinov2Embedder
    from geoanchor.matching import LoFTRMatcher
    from geoanchor.particle_filter import Candidate, ParticleFilter, PFConfig, sigma_from_inliers, weight_from_inliers
    from imatch.blocks import N_BLOCK
    from imatch.nav import ODO, PX
    from imatch.replay import _rot, load_segments

    odo = ODO[odo] if isinstance(odo, str) else odo
    matcher = LoFTRMatcher(size=640, device=device)
    emb = Dinov2Embedder(device=device)
    MIN_INL, QUAL, EARLY, THR = 40, 150, 200, 10.0

    def match_block(gray_uav, bgray, bx, by):
        """4방향 중 inlier 최대. 반환 (map px 위치 또는 None, inlier)."""
        best = (None, 0)
        for r in range(4):
            q = np.ascontiguousarray(np.rot90(gray_uav, r))
            pa, pb, _ = matcher.match(q, bgray)
            if len(pa) < MIN_INL:
                continue
            H, mask = cv2.findHomography(pa.astype(np.float32), pb.astype(np.float32), cv2.USAC_MAGSAC, THR,
                                         maxIters=10_000, confidence=0.999)
            if H is None or mask is None:
                continue
            n = int(mask.sum())
            if n < MIN_INL or n <= best[1]:
                continue
            p = H @ np.array([128., 128., 1.])
            if abs(p[2]) < 1e-9:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if not (-256 < u < 768 and -256 < v < 768):   # 원본: 크롭 밖 반 폭까지
                continue
            best = (np.array([bx * 256 + u, by * 256 + v]), n)
            if n >= QUAL:
                break
        return best

    segs = load_segments(data, sets)
    steps, rows = [], []
    for i, (st, g) in enumerate(segs.items()):
        city = cv2.cvtColor(cv2.imread(os.path.join(data, 'maps', f'{st}.jpg')), cv2.COLOR_BGR2RGB)
        gcity = cv2.cvtColor(city, cv2.COLOR_RGB2GRAY)
        grid = [(bx, by) for by in range(N_BLOCK) for bx in range(N_BLOCK)]
        cen = np.array([[bx * 256 + 256, by * 256 + 256] for bx, by in grid], float)
        tile_emb = emb.embed(np.stack([np.ascontiguousarray(block_img(city, bx, by)) for bx, by in grid]))
        real = g[['map_x', 'map_y']].to_numpy(float)
        rng = np.random.default_rng([seed + 1, i])          # replay_one과 같은 난수 → 같은 이동 오차
        n = len(g)
        bias = rng.normal(0, odo['bias_deg']) if odo else 0.0
        dh = rng.normal(0, odo['head_deg'], n) if odo else np.zeros(n)
        ds = rng.normal(0, odo['scale'], n) if odo else np.zeros(n)
        pf = ParticleFilter(PFConfig(), rng=np.random.default_rng(seed))
        pf.initialize(real[0, 0] / PX, real[0, 1] / PX, sigma_m=3.0)   # 출발 위치는 GPS (다른 방법과 같음)
        since_reloc, disagree = 999, 0
        log = [dict(set=st, traj=i, k=int(g.k[0]), dist_m=float(g.dist_m[0]), real_x=real[0, 0], real_y=real[0, 1],
                    guess_x=real[0, 0], guess_y=real[0, 1], est_x=real[0, 0], est_y=real[0, 1], err_m=0.0,
                    raw_err_m=0.0, guess_err_m=0.0, outside_block=0, unknown=0, used='start', sec=0.0)]
        for k in range(1, n):
            d = _rot(real[k] - real[k - 1], bias + dh[k]) * (1 + ds[k])
            pf.predict(d[0] / PX, d[1] / PX)
            pm = np.array(pf.estimate())
            guess = pm * PX
            gate = float(np.clip(3.0 * pf.spread(), 120.0, 400.0))
            crop = cv2.cvtColor(cv2.imread(os.path.join(data, st, f'k{int(g.k[k]):03d}_uav.jpg')), cv2.COLOR_BGR2GRAY)
            t0 = time.time()
            bx = int(np.clip(np.round((guess[0] - 256) / 256), 0, N_BLOCK - 1))
            by = int(np.clip(np.round((guess[1] - 256) / 256), 0, N_BLOCK - 1))

            def cand(pos, inl):
                return Candidate(north=float(pos[0] / PX), east=float(pos[1] / PX),
                                 weight=weight_from_inliers(inl), sigma=sigma_from_inliers(inl))
            cands, inl, meas, mode = [], 0, None, ''
            c_pos, c_inl = match_block(crop, block_img(gcity, bx, by), bx, by)
            c = cand(c_pos, c_inl) if c_pos is not None else None
            if c is not None:
                meas = c_pos
                if np.hypot(c.north - pm[0], c.east - pm[1]) <= gate:
                    cands.append(c); inl = c_inl
            if not cands or inl < QUAL:   # 주변 타일 (400 m 안, 가까운 순 3개, 이미 본 블록 제외)
                dd = np.linalg.norm(cen / PX - pm, axis=1)
                near = [j for j in np.argsort(dd) if dd[j] <= 400 and grid[j] != (bx, by)][:3]
                for j in near:
                    p2, n2 = match_block(crop, block_img(gcity, *grid[j]), *grid[j])
                    if p2 is not None:
                        c2 = cand(p2, n2)
                        if np.hypot(c2.north - pm[0], c2.east - pm[1]) <= gate:
                            cands.append(c2)
                            if n2 > inl:
                                inl, meas = n2, p2
                            if inl >= QUAL:
                                break
            if not cands and pf.is_lost() and since_reloc >= 4:   # 전역 재탐색 (DINOv2 → LoFTR)
                since_reloc = 0
                rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                qe = emb.embed(np.stack([np.ascontiguousarray(np.rot90(rgb, r)) for r in range(4)]))
                sims = (qe @ tile_emb.T).max(0)
                keep = []
                for j in np.argsort(-sims)[:40]:
                    if all(np.linalg.norm(cen[j] - cen[q]) / PX >= 200 for q in keep):
                        keep.append(j)
                        if len(keep) >= 8:
                            break
                best = (None, 0)
                for j in keep:
                    p3, n3 = match_block(crop, block_img(gcity, *grid[j]), *grid[j])
                    if p3 is not None and n3 > best[1]:
                        best = (p3, n3)
                    if best[1] >= EARLY:
                        break
                if best[0] is not None:
                    cands.append(cand(*best))
                    pf.inject(cands)
                    inl, meas, mode = best[1], best[0], 'reloc'
            else:
                since_reloc += 1
            if c is not None:   # 불일치: 측정이 계속 믿음과 40 m 넘게 다르면 측정으로 재초기화
                disagree = disagree + 1 if np.hypot(c.north - pm[0], c.east - pm[1]) > pf.cfg.disagree_m else 0
                if disagree >= pf.cfg.disagree_patience:
                    pf.initialize(c.north, c.east, sigma_m=max(10.0, c.sigma))
                    disagree = 0
                    cands, mode = [c], 'jump'
            measured = pf.update(cands)
            sec = time.time() - t0
            est = np.array(pf.estimate()) * PX
            raw = meas if meas is not None else est
            log.append(dict(set=st, traj=i, k=int(g.k[k]), dist_m=float(g.dist_m[k]), real_x=real[k, 0], real_y=real[k, 1],
                            guess_x=guess[0], guess_y=guess[1], est_x=est[0], est_y=est[1], block_x=bx, block_y=by,
                            err_m=float(np.linalg.norm(est - real[k]) / PX), raw_err_m=float(np.linalg.norm(raw - real[k]) / PX),
                            guess_err_m=float(np.linalg.norm(guess - real[k]) / PX),
                            outside_block=int(np.abs(real[k] - (np.array([bx, by]) * 256 + 256)).max() > 128),
                            unknown=int(not measured), used=mode or ('match' if measured else 'guess'), sec=sec,
                            n_inl=inl, spread_m=pf.spread()))
        steps += log
        e = np.array([s['err_m'] for s in log])
        rows.append(dict(set=st, traj=i, n_steps=len(log), length_m=float(g.dist_m.iloc[-1]), final_err_m=float(e[-1]),
                         max_err_m=float(e.max()), med_err_m=float(np.median(e)), w10=float((e <= 10).mean()),
                         w20=float((e <= 20).mean()), w32=float((e <= 32).mean()),
                         unknown=float(np.mean([s['unknown'] for s in log]))))
        if verbose and (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(segs)} 구간', flush=True)
    return pd.DataFrame(steps), pd.DataFrame(rows)
