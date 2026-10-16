"""파이프라인 ④⑤: 대응점 검증(MAGSAC++)과 호모그래피 → 위치·heading.

좌표 규칙 (Bearing-UAV와 같음)
- 드론 영상 256 px, 지도 블록 512 px (2x2 타일). 블록 중심이 (0, 0), 1단위 = 128 px = 32 m.
- 정답 변환: block_px = C + R(theta) (uav_px - 128),  R(t) = [[cos t, sin t], [-sin t, cos t]]
"""
import cv2
import numpy as np

BLOCK_CENTER = 256.0
PX_PER_UNIT = 128.0
M_PER_UNIT = 32.0

RANSAC_PX = 3.0  # MAGSAC++ 임계값(px). scripts/run.py --ransac-px 로 바꾼다


def set_ransac_px(px):
    global RANSAC_PX
    RANSAC_PX = float(px)


def rot(a):
    t = np.deg2rad(a); c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def turn(img, a):
    """영상을 a도 돌린다 (반시계, cv2 규칙). 빈 모서리는 검은색."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def wrap(d):
    """각도 차이의 절댓값 (0~180도)."""
    return np.abs((np.asarray(d) + 180) % 360 - 180)


def magsac(p0, p1):
    """대응점 → (p0, p1, 호모그래피 H, inlier 마스크). 점이 4개 미만이면 H = None."""
    if len(p0) < 4:
        return p0, p1, None, np.zeros(len(p0), bool)
    H, mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=2000, confidence=0.995)
    return p0, p1, H, (mask[:, 0].astype(bool) if mask is not None else np.zeros(len(p0), bool))


def pose_from_homography(H, scale, angle, uav_center):
    """돌린 드론 영상 → 블록 호모그래피에서 위치(x, y: 블록 단위)와 heading(도), 축척을 구한다.

    위치: 드론 영상 중심을 블록으로 옮긴 점. heading: 중심에서의 국소 선형 변환 L의 회전 성분.
    회전 탐색으로 영상을 angle만큼 돌렸으므로 L에 R(angle)을 곱해 원래 영상 기준으로 되돌린다.
    """
    S = np.diag([scale, scale, 1.])
    Hb = np.linalg.inv(S) @ H @ S                                   # 축척 보정 (base px 기준)
    h = Hb @ np.array([uav_center[0], uav_center[1], 1.]); c = h[:2] / h[2]  # 회전해도 중심은 그대로
    L = (Hb[:2, :2] - np.outer(c, Hb[2, :2])) / h[2] @ rot(angle)   # 중심에서의 야코비안
    return dict(px=float((c[0] - BLOCK_CENTER) / PX_PER_UNIT), py=float((c[1] - BLOCK_CENTER) / PX_PER_UNIT),
                head=float(np.degrees(np.arctan2(L[0, 1], L[0, 0]))), scale_est=float(np.sqrt(abs(np.linalg.det(L)))))
