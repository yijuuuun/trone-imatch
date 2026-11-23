"""팀 공통 평가 규칙 (2026-10-15 확정). 모든 표의 숫자는 이 규칙으로 계산한다.

- 추정은 2x2 블록 안으로 clip 한다. 추정이 없으면 블록 중심 (0, 0)을 답으로 낸다.
  (B2·CorrMatchNet은 원래 블록 안만 답할 수 있으므로 같은 조건을 맞춘 것)
- 실패 = 추정 없음 또는 블록 밖으로 0.1단위(3.2 m) 넘게 나감.
- 성공 = 위치 오차 10 m 이내.
- 비교는 같은 샘플끼리 짝지어 McNemar 정확 검정 (나만 성공 / 상대만 성공).
"""
import numpy as np
from scipy.stats import binomtest

from .blocks import UNITS_PER_BLOCK
from .geometry import M_PER_UNIT


def errors(pred_x, pred_y, gt_x, gt_y, lim=1.0):
    """(위치 오차 m, 실패 여부) 배열. lim = clip 범위 (블록 단위)."""
    px, py = np.asarray(pred_x, float), np.asarray(pred_y, float)
    found = np.isfinite(px) & np.isfinite(py)
    px, py = np.nan_to_num(px), np.nan_to_num(py)
    fail = ~found | (np.abs(px) > lim + 0.1) | (np.abs(py) > lim + 0.1)
    e = np.hypot(np.clip(px, -lim, lim) - np.asarray(gt_x), np.clip(py, -lim, lim) - np.asarray(gt_y)) * M_PER_UNIT
    return e, fail


def errors_chosen_block(pred_x, pred_y, dx, dy, gt_x, gt_y):
    """후보 블록 결과용 errors. pred는 고른 블록(정답 블록 + (dx, dy)) 기준이고, clip은 그 블록의 영상 범위(±2)로 한다.

    errors의 기본 ±1은 정답 블록일 때만 맞다. 정답 블록 좌표로 옮겨서 ±1로 clip하면 이웃 블록을 고른 추정이 정답
    쪽으로 끌려와 오차가 작게 나오고, 고른 블록 기준 ±1로 clip하면 이웃 블록의 맞는 추정(예: -1.45)이 틀어진다.
    이 함수의 숫자는 errors로 잰 표와 바로 비교하면 안 된다. 같은 후보 결과끼리만 비교한다.
    """
    dx, dy = np.nan_to_num(np.asarray(dx, float)), np.nan_to_num(np.asarray(dy, float))
    return errors(pred_x, pred_y, np.asarray(gt_x) - UNITS_PER_BLOCK * dx, np.asarray(gt_y) - UNITS_PER_BLOCK * dy, lim=2.0)


def mcnemar(ok_mine, ok_other):
    """'나만 성공 / 상대만 성공, p' 문자열."""
    a, b = int((ok_mine & ~ok_other).sum()), int((~ok_mine & ok_other).sum())
    return f"{a} / {b}, p = {binomtest(a, a + b).pvalue:.2g}" if a + b else "0 / 0"
