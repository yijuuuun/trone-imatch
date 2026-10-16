"""팀 공통 평가 규칙 (2026-10-15 확정). 모든 표의 숫자는 이 규칙으로 계산한다.

- 추정은 2x2 블록 안으로 clip 한다. 추정이 없으면 블록 중심 (0, 0)을 답으로 낸다.
  (B2·CorrMatchNet은 원래 블록 안만 답할 수 있으므로 같은 조건을 맞춘 것)
- 실패 = 추정 없음 또는 블록 밖으로 0.1단위(3.2 m) 넘게 나감.
- 성공 = 위치 오차 10 m 이내.
- 비교는 같은 샘플끼리 짝지어 McNemar 정확 검정 (나만 성공 / 상대만 성공).
"""
import numpy as np
from scipy.stats import binomtest

from .geometry import M_PER_UNIT


def errors(pred_x, pred_y, gt_x, gt_y):
    """(위치 오차 m, 실패 여부) 배열."""
    px, py = np.asarray(pred_x, float), np.asarray(pred_y, float)
    found = np.isfinite(px) & np.isfinite(py)
    px, py = np.nan_to_num(px), np.nan_to_num(py)
    fail = ~found | (np.abs(px) > 1.1) | (np.abs(py) > 1.1)
    e = np.hypot(np.clip(px, -1, 1) - np.asarray(gt_x), np.clip(py, -1, 1) - np.asarray(gt_y)) * M_PER_UNIT
    return e, fail


def mcnemar(ok_mine, ok_other):
    """'나만 성공 / 상대만 성공, p' 문자열."""
    a, b = int((ok_mine & ~ok_other).sum()), int((~ok_mine & ok_other).sum())
    return f"{a} / {b}, p = {binomtest(a, a + b).pvalue:.2g}" if a + b else "0 / 0"
