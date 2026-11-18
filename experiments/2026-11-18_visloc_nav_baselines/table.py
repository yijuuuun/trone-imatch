"""기존 모델(이 실험)과 우리 방법(#45)을 seq_eval 92구간 한 표로. GPU 불필요.

  python experiments/2026-11-18_visloc_nav_baselines/table.py   # 표 출력 + table.csv

지표: 마지막 20 m 이내(Bearing-UAV 논문 항법 SR@20), 마지막 40 m 이내(3D 항법 도착 기준), 마지막 오차 중앙값,
걸음 오차 중앙값(출발 제외), 걸음당 s(매칭·추론 시간, Kaggle T4). v1 대비 = 같은 구간에서 v1만 / 상대만 성공, McNemar(정확).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, '..', '2026-11-16_visloc_nav_v1')
CACHE = os.path.join(HERE, '..', '2026-11-18_map_cache')
ROWS = [  # (이름, 폴더, 파일 앞부분, 설명)
    ('추측 항법만', V1, 'visloc_loma-dr', '매칭 없음'),
    ('Bearing-UAV (공개 가중치)', HERE, 'visloc_bearing-r0', 'R0. #42 재실행(T4), 걸음 오차 #42와 같음'),
    ('CorrMatchNet (SF 뺀 학습)', HERE, 'visloc_corr38-r0', 'R0, 4방향 중 conf 최고'),
    ('CorrMatchNet (싱가포르 뺀 학습)', HERE, 'visloc_corr37-r0', 'R0, 4방향 중 conf 최고'),
    ('OS-FPI', HERE, 'visloc_osfpi-r0', 'R0, 4방향 중 heatmap 최고, 드론 가운데 192 px'),
    # GeoAnchor 방식(visloc_geoanchor-r0_*)은 표에서 뺀다: 입력·이동량·방향·보정을 우리가 바꿔 원본 성능이 아니고,
    # 매칭 통과 기준(inlier 40)이 64 m 크롭에 맞지 않아 걸음의 3.9%만 측정됐다 (README 4장).
    ('Sample4Geo (검색형)', HERE, 'visloc_sample4geo-r0', 'R0, 블록 안 타일 검색, 위치 = 타일 중심'),
    ('MEAN (검색형)', HERE, 'visloc_mean-r0', 'R0, 블록 안 타일 검색, 위치 = 타일 중심'),
    ('Game4Loc (검색형)', HERE, 'visloc_game4loc-r0', 'R0, 블록 안 타일 검색, 위치 = 타일 중심'),
    ('LoMa R0', V1, 'visloc_loma-r0', '매칭을 항상 믿음'),
    ('R1', V1, 'visloc_loma-r1', 'inlier < 15면 모름'),
    ('v1 (R1 + R4)', V1, 'visloc_loma-r4g0n5w20', '확정 v1'),
]
SEC_CACHE = {'R1': 'visloc_loma-cache-r1', 'v1 (R1 + R4)': 'visloc_loma-cache-r4g0n5w20'}


def mcnemar(a, b):
    n1, n2 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n1, n1 + n2, 0.5).pvalue if n1 + n2 else 1.0
    return n1, n2, p


def main():
    data = {}
    for name, d, f, _ in ROWS:
        sp, st = os.path.join(d, f + '_segs.csv'), os.path.join(d, f + '_steps.csv')
        if os.path.exists(sp):
            data[name] = (pd.read_csv(sp).sort_values('set').reset_index(drop=True), pd.read_csv(st))
    ref = data['v1 (R1 + R4)'][0]
    out = []
    print('| 방법 | 마지막 20 m 이내 | 마지막 40 m 이내 | 마지막 오차 중앙값 | 걸음 오차 중앙값 | 걸음당 s | v1 대비 20 m (v1만 / 상대만) | v1 대비 40 m |')
    print('| --- ' * 8 + '|')
    for name, d, f, note in ROWS:
        if name not in data:
            print(f'| {name} | (결과 없음) |'); continue
        segs, steps = data[name]
        assert (segs.set.values == ref.set.values).all(), name
        fe = segs.final_err_m.values
        s = steps[steps.used != 'start'] if 'used' in steps else steps
        sec = float(s.sec.mean()) if 'sec' in s and s.sec.notna().any() else 0.0
        cs = ''
        if name in SEC_CACHE and os.path.exists(os.path.join(CACHE, SEC_CACHE[name] + '_steps.csv')):
            c = pd.read_csv(os.path.join(CACHE, SEC_CACHE[name] + '_steps.csv'))
            cs = f" (캐시 {c[c.used != 'start'].sec.mean():.2f})"
        r = dict(method=name, note=note, final_le20=int((fe <= 20).sum()), final_le40=int((fe <= 40).sum()),
                 final_med_m=round(float(np.median(fe)), 1), step_med_m=round(float(s.err_m.median()), 1), sec=round(sec, 3))
        cmp = {}
        for t in (20, 40):
            if name == 'v1 (R1 + R4)':
                cmp[t] = '—'
            else:
                n1, n2, p = mcnemar(ref.final_err_m.values <= t, fe <= t)
                cmp[t] = f'{n1} / {n2}, p = {p:.2g}'
                r[f'vs_v1_le{t}'] = cmp[t]
        out.append(r)
        n = len(fe)
        print(f"| {name} | {r['final_le20']} ({r['final_le20'] / n * 100:.1f}%) | {r['final_le40']} ({r['final_le40'] / n * 100:.1f}%) "
              f"| {r['final_med_m']} m | {r['step_med_m']} m | {sec:.2f}{cs} | {cmp[20]} | {cmp[40]} |")
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'table.csv'), index=False)


if __name__ == '__main__':
    main()
