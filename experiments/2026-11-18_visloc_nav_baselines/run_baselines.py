#!/usr/bin/env python
"""UAV-VisLoc seq_eval 92구간에서 기존 모델 항법 (R0: 모든 답을 믿음). baselines.py의 어댑터를 imatch.replay에 꽂는다.

  python experiments/2026-11-18_visloc_nav_baselines/run_baselines.py --data data/visloc/seq_eval --out <폴더> \
      --models bearing,osfpi,corr38,corr37,geoanchor --cap ~/capstone/bearinguav \
      --osfpi ~/capstone/ext/OS-FPI --geoanchor <geoanchor 패키지가 든 폴더>

models
  bearing    원 Bearing-UAV 공개 가중치 (--cap 아래 Bearing_UAV/cross_view). #42 재현 확인과 T4 시간용
  osfpi      OS-FPI net_040.pth, 4방향
  corr38     CorrMatchNet p2bv4 (SF 뺀 학습), 4방향      corr37   같은 구조 (싱가포르 뺀 학습)
  geoanchor  GeoAnchor 방식 (LoFTR + 파티클 필터), 우리 조건
  sample4geo, mean, game4loc   검색형, 블록 안 타일 검색 (baselines.RetrievalLoc)
  <model>-r1 모델 점수 < T면 "모름" → 추측 위치 (v1의 R1과 같은 처리). T는 calib_gate.py가 개발용에서 고른 gate.json
출력: visloc_<model>-r0_{steps,segs}.csv (R1이면 -r1). 같은 seed면 run_visloc_nav.py(LoMa, dr)와 같은 구간·출발점·이동 오차.
표: 마지막 20 m 이내(Bearing-UAV 논문 SR@20) / 40 m 이내, 마지막 오차 중앙값, 걸음 오차 중앙값(출발 제외), 걸음당 s.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from imatch.replay import run_segments  # noqa: E402


# 검색형: (저장소 폴더, 가중치 파일). 가중치 SHA-256 앞자리: MEAN bc0f0798 (#33과 같음), Sample4Geo 04f4c899 (#33과 같음,
# 공식 pretrained.zip의 university/), Game4Loc 12ec61d1 (HF Yux1ang/gta_uav_pretrained_models, GTA-UAV cross-area 학습)
RET = {'mean': ('MEAN', 'MEAN_weights_e2_0.9355.pth'),
       'sample4geo': ('Sample4Geo', 's4g_u1652_weights_e1_0.9515.pth'),
       'game4loc': ('GTA-UAV/Game4Loc', 'vit_base_eva_gta_cross_area.pth')}


def table_row(name, steps, segs):
    s = steps[steps.used != 'start']
    f = segs.final_err_m
    return (f"| {name} | {int((f <= 20).sum())} ({(f <= 20).mean() * 100:.1f}%) | {int((f <= 40).sum())} "
            f"({(f <= 40).mean() * 100:.1f}%) | {f.median():.1f} m | {s.err_m.median():.1f} m | {s.sec.mean():.3f} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/visloc/seq_eval'); ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='bearing,osfpi,corr38,corr37,geoanchor')
    ap.add_argument('--cap', default=os.path.expanduser('~/capstone/bearinguav'))
    ap.add_argument('--osfpi', default=os.path.expanduser('~/capstone/ext/OS-FPI'))
    ap.add_argument('--geoanchor', default='', help='geoanchor/ 패키지(GeoAnchor 저장소 src/)가 든 폴더')
    ap.add_argument('--ret', default=os.path.expanduser('~/capstone/ext'),
                    help='검색형: MEAN/, Sample4Geo/, GTA-UAV/Game4Loc/ 저장소와 weights/가 든 폴더')
    ap.add_argument('--odo', default='vio'); ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--sets', default='', help='구간 이름 쉼표 목록 (비우면 92구간 전부)')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sets = a.sets.split(',') if a.sets else None
    import torch
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, 데이터 {a.data}, '
          f'이동 오차 {a.odo}, seed {a.seed}', flush=True)
    rows = []
    for name in a.models.split(','):
        t0 = time.time()
        print(f'== {name}', flush=True)
        base, r1 = (name[:-3], True) if name.endswith('-r1') else (name, False)
        if r1:   # R1: 개발용에서 고른 점수 기준 (calib_gate.py → gate.json). 출력 visloc_<model>-r1_*
            import json
            g = json.load(open(os.path.join(HERE, 'gate.json')))[base]
            print(f'   점수 {g["score"]} < {g["T"]}면 모름 (개발용 거절 {g["reject_pct"]}%)', flush=True)
        if name == 'geoanchor':
            from baselines import run_geoanchor
            steps, segs = run_geoanchor(a.geoanchor, a.data, odo=a.odo, seed=a.seed, device=a.device, sets=sets)
        else:
            import baselines as B
            if base == 'bearing':
                loc = B.BearingLoc(a.cap, device=a.device)
            elif base == 'osfpi':
                # 드론 크롭 가운데 192 px: 개발용 single_calib에서 고름 (calib_osfpi.py, 중앙값 52.2 m vs 256 px 56.0 m)
                loc = B.OSFPILoc(a.osfpi, device=a.device, uav_crop=192)
            elif base in ('corr38', 'corr37'):
                d = {'corr38': 'results/corr/p2bv4_3d_ho38bc_20261013_135730',
                     'corr37': 'results/corr/p2bv4_3d_ho37bc_20261013_173659'}[base]
                loc = B.CorrLoc(a.cap, d, device=a.device)
            elif base in RET:
                repo, w = RET[base]
                loc = B.RetrievalLoc(base, os.path.join(a.ret, repo), os.path.join(a.ret, 'weights', w), device=a.device)
            else:
                sys.exit(f'모르는 모델: {name}')
            T = 0.
            if r1 and g['T'] is not None:
                loc, T = B.Gated(loc, g['score']), float(g['T'])
                assert T > 0, 'replay는 reject_inl > 0일 때만 거른다'
            steps, segs = run_segments(loc.bind, a.data, sets=sets, odo=a.odo, seed=a.seed, reject_inl=T)
        tag = f'{base}-r1' if r1 else f'{name}-r0'
        steps.to_csv(os.path.join(a.out, f'visloc_{tag}_steps.csv'), index=False)
        segs.to_csv(os.path.join(a.out, f'visloc_{tag}_segs.csv'), index=False)
        rows.append(table_row(name, steps, segs))
        print(f'   {len(segs)}구간, {time.time() - t0:.0f} s', flush=True)
        print(rows[-1], flush=True)
    print('\n| 모델 | 마지막 20 m 이내 | 마지막 40 m 이내 | 마지막 오차 중앙값 | 걸음 오차 중앙값 | 걸음당 s |')
    print('| --- ' * 6 + '|')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
