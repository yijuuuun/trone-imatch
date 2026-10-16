"""파이프라인 ①~⑤를 한 샘플에 적용한다.

① 회전 탐색: 드론 영상을 angles의 각 각도로 돌린다 (angles=[0]이면 탐색 없음, 원 Bearing-UAV와 같은 조건)
②③ 매칭: matcher(돌린 드론 영상, 블록)
④ 검증: MAGSAC++ inlier
⑤ 위치·heading: 호모그래피 → pose_from_homography
→ inlier가 가장 많은 각도의 결과를 고른다.
"""
import time

import cv2
import numpy as np

from .geometry import pose_from_homography, rot, turn


def localize(matcher, uav, block, gt, scale=1.0, angles=(0.0,), px_tol=8.0):
    """각도별 기록 목록과 고른 결과(best, 없으면 None)를 돌려준다.

    gt = (x_norm, y_norm, theta): 위치 추정에는 쓰지 않고, 대응점이 맞는지(good_*) 채점하는 데만 쓴다.
    """
    hu, wu = uav.shape[:2]; cu = np.array([wu / 2, hu / 2])
    U = cv2.resize(uav, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else uav
    B = cv2.resize(block, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else block
    xn, yn, th = gt; C = np.array([256 + xn * 128, 256 + yn * 128]); Rg = rot(th)
    recs = []
    for a in angles:
        t0 = time.time()
        p0, p1, H, inl = matcher(turn(U, a), B)
        dt = time.time() - t0
        p0 = np.asarray(p0, np.float64) / scale; p1 = np.asarray(p1, np.float64) / scale
        # 원래 드론 영상 좌표로 되돌림: p_turned = cu + R(a) (p - cu)  ->  p = cu + R(a)^T (p_turned - cu)
        po = (p0 - cu) @ rot(a) + cu if len(p0) else p0
        good = np.linalg.norm(C + (po - cu) @ Rg.T - p1, axis=1) < px_tol if len(p0) else np.zeros(0, bool)
        r = dict(angle=a, n_match=int(len(p0)), n_inl=int(inl.sum()), sec=dt,
                 good_match=float(good.mean()) if len(good) else np.nan,
                 good_inl=float(good[inl].mean()) if inl.sum() else np.nan)
        if H is not None and inl.sum() >= 4:
            r.update(pose_from_homography(H, scale, a, cu))
        recs.append(r)
    ok = [r for r in recs if 'px' in r]
    best = max(ok, key=lambda r: r['n_inl']) if ok else None
    return recs, best
