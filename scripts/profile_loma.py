"""LoMa-B 단계별 연산량(GFLOPs)·시간·메모리를 잰다 (docs/06_경량화_제안.md 1장).

방향 1개 = 드론 영상 특징 + 지도 블록 특징 + 매칭 트랜스포머 + MAGSAC++. 드론 영상은 45° 돌려 넣는다.
FLOPs는 하드웨어와 무관하므로 다른 GPU(젯슨 등)로 환산할 때 쓴다. 시간은 돌린 GPU 기준이다.

  python scripts/profile_loma.py --data data/im_t1_3d_1000 --n 5
  LOMA_FP32=1 python scripts/profile_loma.py ...   # GTX 1650 (fp16에서 NaN)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from imatch.geometry import magsac, turn  # noqa: E402
from imatch.matchers import get_matcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/im_t1_3d_1000'); ap.add_argument('--set', default='3d_ho37bc')
    ap.add_argument('--n', type=int, default=5); ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    from vismatch.im_models.loma import filter_matches, to_pixel_coords

    M = get_matcher('loma', 'cuda'); L = M.m.matcher
    rd = lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    prep = lambda im: M.m.preprocess(torch.from_numpy(im).permute(2, 0, 1).float() / 255.)[0]
    sync = torch.cuda.synchronize

    def tm(fn):
        fn(); sync(); t0 = time.time()
        for _ in range(a.reps):
            out = fn()
        sync(); return out, (time.time() - t0) / a.reps

    def gflops(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / 1e9

    rows = []
    with torch.inference_mode():
        for k in range(a.n):
            d = os.path.join(a.data, a.set)
            u, b = prep(turn(rd(f'{d}/k{k:03d}_uav.jpg'), 45.0)), prep(rd(f'{d}/k{k:03d}_block.jpg'))
            torch.cuda.reset_peak_memory_stats()
            (k0, d0, _, _), tu = tm(lambda: L.detect_and_describe(u, 2048))
            (k1, d1, _, _), tb = tm(lambda: L.detect_and_describe(b, 2048))
            sc, tmat = tm(lambda: L(k0, k1, d0, d1)['scores'])
            m0, _, _, _ = filter_matches(sc, L.cfg.filter_threshold); v = m0[0] > -1
            p0 = to_pixel_coords(k0[0][torch.where(v)[0]], *u.shape[-2:]).cpu().numpy()
            p1 = to_pixel_coords(k1[0][m0[0][v]], *b.shape[-2:]).cpu().numpy()
            t0 = time.time(); inl = magsac(np.float32(p0), np.float32(p1))[3]; tg = time.time() - t0
            rows.append([gflops(lambda: L.detect_and_describe(u, 2048)), gflops(lambda: L.detect_and_describe(b, 2048)),
                         gflops(lambda: L(k0, k1, d0, d1)), tu, tb, tmat, tg, int(v.sum()), int(inl.sum()),
                         torch.cuda.max_memory_allocated() / 2**20])
            print(k, ' '.join(f'{x:.3g}' for x in rows[-1]), flush=True)
    r = np.array(rows).mean(0)
    print(f"\nGPU {torch.cuda.get_device_name()}, 특징 dtype {d0.dtype}, 기술자 {d0.shape[-1]}차원, "
          f"파라미터 {sum(p.numel() for p in L.parameters()) / 1e6:.1f} M")
    print(f"GFLOPs | 드론 특징 {r[0]:.0f} | 지도 블록 특징 {r[1]:.0f} | 매칭 트랜스포머 {r[2]:.0f}")
    print(f"초     | 드론 특징 {r[3]:.3f} | 지도 블록 특징 {r[4]:.3f} | 매칭 트랜스포머 {r[5]:.3f} | MAGSAC++ {r[6]:.4f}")
    print(f"짝 {r[7]:.0f}, inlier {r[8]:.0f}, 최대 GPU 메모리 {r[9]:.0f} MiB")


if __name__ == '__main__':
    main()
