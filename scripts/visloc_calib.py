#!/usr/bin/env python
"""UAV-VisLoc 보정값 정하기 + 비행별 결과 표.

export_visloc.py로 만든 single_calib을 run.py로 돌린 결과에서, 비행마다
  - heading 보정 head_off (정답 heading과 추정 heading 차이의 중앙값)
  - 위치 보정 along_m, right_m (추정 - 정답을 드론 진행 방향 기준으로 본 중앙값)
  - 축척 확인 scale_med (추정 축척의 중앙값. 1에서 멀면 gsd가 틀림. --apply-scale이면 gsd에 곱한다)
을 구해 calib.csv로 쓴다. 지금 samples.csv에 적힌 보정값 위에 더한다(여러 번 돌려도 됨).
보정은 inlier >= --min-inl이고 오차가 --max-err m 안인 샘플만 쓴다 (틀린 매칭 제외).

  python scripts/visloc_calib.py outputs/visloc_calib/loma_s1.csv --data data/visloc/single_calib --out data/visloc_calib.csv
  python scripts/visloc_calib.py outputs/visloc_eval/loma_s1.csv --data data/visloc_v2/single_eval      # 표만

표: 비행별 found, 모름(inlier < 15), 10/20/32 m 이내(팀 규칙 imatch/metrics.errors), heading 오차.
같은 숫자를 보정 전 GPS 정답(x_norm_raw, y_norm_raw)으로도 낸다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.geometry import M_PER_UNIT, wrap  # noqa: E402
from imatch.metrics import errors  # noqa: E402


def signed(d):
    return (np.asarray(d, float) + 180) % 360 - 180


def report(R, label):
    print(f'\n{label}')
    print('| 비행 | n | found | 모름 | 10 m | 20 m | 32 m | 답한 것 10 m | heading 중앙값 오차 |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for fl, g in list(R.groupby('flight')) + [('전체', R)]:
        e, fail = errors(g.pred_x, g.pred_y, g.gx, g.gy)
        ok = ~fail
        unk = g.n_inl.fillna(0).to_numpy() < 15
        ans = ok & ~unk
        he = wrap(g.pred_theta - g.theta)
        pct = lambda m: f'{np.mean(m) * 100:.1f}%'
        print(f'| {fl} | {len(g)} | {pct(g.pred_x.notna())} | {pct(unk)} | {pct(ok & (e <= 10))} | {pct(ok & (e <= 20))} '
              f'| {pct(ok & (e <= 32))} | {pct((e <= 10)[ans]) if ans.any() else "-"} | {np.nanmedian(he):.1f}° |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred', help='run.py 출력 CSV')
    ap.add_argument('--data', required=True, help='export_visloc.py의 single_calib 또는 single_eval 폴더')
    ap.add_argument('--out', default='', help='보정값 CSV. 비우면 표만 낸다')
    ap.add_argument('--min-inl', type=float, default=15)
    ap.add_argument('--max-err', type=float, default=60, help='보정에 쓸 샘플의 최대 오차 (m, 보정 전 기준)')
    ap.add_argument('--apply-scale', action='store_true', help='gsd에 추정 축척 중앙값을 곱한다')
    a = ap.parse_args()

    S = pd.read_csv(os.path.join(a.data, 'samples.csv'), dtype={'flight': str})
    P = pd.read_csv(a.pred)
    if 'pred_x' not in P:
        sys.exit('run.py 결과에 pred_x가 없음')
    P = P.drop(columns=[c for c in ['gt_x', 'gt_y', 'gt_theta'] if c in P])
    R = S.merge(P, on=['set', 'k'], how='left')
    R['flight'] = R.flight.str.zfill(2)

    R['gx'], R['gy'] = R.x_norm, R.y_norm
    report(R, '지금 정답(x_norm, y_norm) 기준')
    if (R.x_norm != R.x_norm_raw).any() or (R.y_norm != R.y_norm_raw).any():
        Rr = R.copy(); Rr['gx'], Rr['gy'] = R.x_norm_raw, R.y_norm_raw
        report(Rr, '보정 전 GPS 정답(x_norm_raw, y_norm_raw) 기준')

    if not a.out:
        return
    if not (S.split == 'calib').all():
        print('\n경고: --data가 single_calib이 아님. 평가용 샘플로 보정값을 정하면 안 된다.')
    out = []
    for fl, g in R.groupby('flight'):
        dE = (g.pred_x - g.x_norm) * M_PER_UNIT
        dN = -(g.pred_y - g.y_norm) * M_PER_UNIT
        use = (g.n_inl.fillna(0) >= a.min_inl) & (np.hypot(dE, dN) <= a.max_err)
        g, dE, dN = g[use], dE[use], dN[use]
        head_d = signed(g.pred_theta - g.theta)
        d_med = float(np.median(head_d)) if len(g) else np.nan
        # 진행 방향은 보정된 heading으로 (pred_theta - theta = d → 참 heading_cw = heading_cw - d)
        h = np.radians(g.heading_cw - d_med)
        along = dE * np.sin(h) + dN * np.cos(h)
        right = dE * np.cos(h) - dN * np.sin(h)
        scale = float(np.median(g.scale_est)) if len(g) and 'scale_est' in g else np.nan
        base = g.iloc[0] if len(g) else R[R.flight == fl].iloc[0]
        gsd = float(base.gsd) * (scale if a.apply_scale and np.isfinite(scale) else 1.0)
        out.append(dict(
            flight=fl, gsd=round(gsd, 5),
            head_off=round(float(base.head_off) - d_med, 2) if len(g) else float(base.head_off),
            along_m=round(float(base.along_m) + float(np.median(along)), 2) if len(g) else float(base.along_m),
            right_m=round(float(base.right_m) + float(np.median(right)), 2) if len(g) else float(base.right_m),
            n_used=int(len(g)), n_total=int((R.flight == fl).sum()),
            head_d_med=round(d_med, 2), head_mad=round(float(np.median(np.abs(head_d - d_med))), 2) if len(g) else np.nan,
            along_med=round(float(np.median(along)), 2) if len(g) else np.nan,
            along_mad=round(float(np.median(np.abs(along - np.median(along)))), 2) if len(g) else np.nan,
            right_med=round(float(np.median(right)), 2) if len(g) else np.nan,
            right_mad=round(float(np.median(np.abs(right - np.median(right)))), 2) if len(g) else np.nan,
            scale_med=round(scale, 4)))
    C = pd.DataFrame(out)
    C.to_csv(a.out, index=False)
    print('\n보정값 (head_off, along_m, right_m는 지금 값 + 이번에 잰 중앙값)')
    print(C.to_string(index=False))
    few = C[C.n_used < 15]
    if len(few):
        print(f'\n경고: 보정 샘플이 15개 미만인 비행 {", ".join(few.flight)}. 값이 불안정하다 (--calib-frac를 늘리거나 0으로 둘 것)')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
