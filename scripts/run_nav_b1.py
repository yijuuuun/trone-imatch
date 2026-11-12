"""원 모델(B1, Bearing-UAV 누수 없이 재학습)로 1 km 항법: run_nav.py와 같은 nav_run·궤적·이동 오차.

  python scripts/run_nav_b1.py --capstone1 ~/capstone/bearinguav \
      --ckpt results/c4ma/phr5_d96100_3d_ho38bc_b16_l0.5_e100_gNone_20261004_225155 \
      --sets 3d_ho34bc,3d_ho36bc --n-traj 32 --seed 7 --odo vio --tag=-ho38 --out outputs/nav_b1

capstone1(bearinguav) 저장소의 diag/, cvphr/와 Bearing_UAV_90K가 필요하다 (Python 3.9, torch 2.1 환경 bearing_env).
--ckpt는 capstone1 기준 경로. 위치 추정은 scripts/making_loc.py(강지민)의 B1Loc과 같다:
  드론 영상 = 가장 가까운 3D 렌더 그대로(Nearest3D), 타일 = 메타데이터의 블록 타일 p1~p4, heading 입력 없음.
  B1은 "모름"을 내지 않으므로 R0(복구 없음)만 있다.
출력: <도시>_3d_b1<tag>-r0_{steps,traj}.csv (run_nav.py와 같은 열, scripts/nav_metrics.py로 비교).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from imatch.nav import META_3D, ODO, Nearest3D, make_traj, nav_run  # noqa: E402

CITY = {'3d_ho38bc': 'SF', '3d_ho37bc': 'Singapore', '3d_ho34bc': 'Tokyo', '3d_ho36bc': 'Taipei'}


class B1Loc:
    """B1 forward. 위치는 블록 중심 기준 블록 단위 (1 = 128 px). heading 출력은 self.last의 b1_head(도)에 남긴다."""
    def __init__(self, m, meta, code, base, enc, head, to_tensor, dev):
        self.m, self.base, self.enc, self.head, self.tt, self.dev = m, base, enc, head, to_tensor, dev
        d = meta[meta.p1_path.astype(str).str.contains(code, regex=False)]
        self.rows = d.drop_duplicates(['block_x', 'block_y']).set_index(['block_x', 'block_y'])
        self.feat, self.last = {}, []

    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.feat:
            import cv2
            r = self.rows.loc[(bx, by)]
            tiles = [cv2.cvtColor(cv2.imread(os.path.join(self.base, r[c])), cv2.COLOR_BGR2RGB)
                     for c in ('p1_path', 'p2_path', 'p3_path', 'p4_path')]  # TL, TR, BL, BR
            self.feat[(bx, by)] = self.enc(self.m, torch.stack([self.tt(t) for t in tiles]).to(self.dev))
        return self.feat[(bx, by)]

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        t0 = time.time()
        nb = torch.stack([self.block(int(bx), int(by)) for bx, by in blocks])
        fi = self.enc(self.m, torch.stack([self.tt(c) for c in crops]).to(self.dev))
        pos, dr = self.head(self.m, fi, nb)[:2]
        if self.dev != 'cpu':
            torch.cuda.synchronize()
        sec = (time.time() - t0) / len(crops)
        hd = torch.rad2deg(torch.atan2(dr[:, 1], dr[:, 0])).cpu().numpy()
        self.last = [dict(sec=sec, b1_head=float(h)) for h in hd]
        return pos.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capstone1', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--sets', default='3d_ho34bc,3d_ho36bc')
    ap.add_argument('--n-traj', type=int, default=32); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dist', type=float, default=1000); ap.add_argument('--odo', default='vio', choices=list(ODO))
    ap.add_argument('--oracle', action='store_true'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    cap = os.path.abspath(os.path.expanduser(a.capstone1))
    os.chdir(cap); sys.path.insert(0, cap)  # diag 모듈이 cwd 기준 상대 경로를 쓴다
    from diag.diag_modules import DEV, enc, head, load_model, to_tensor
    m = load_model(a.ckpt)
    meta = pd.read_csv(os.path.join(cap, META_3D))
    print(f'B1 {a.ckpt}, 이동 오차 {a.odo}, 시드 {a.seed}, {a.n_traj}궤적', flush=True)
    print('| 도시 | 블록 | 궤적 | 성공률 | 종료 이유 | 걸음 | 걸음 10 m 이내 | 걸음 32 m 넘음 | 걸음당 s |')
    print('| --- ' * 9 + '|')
    for st in a.sets.split(','):
        code = st.split('_ho')[1]
        loc = B1Loc(m, meta, code, cap, enc, head, to_tensor, DEV)
        frames = Nearest3D(cap, st)
        starts, goals = make_traj(a.n_traj, a.dist, a.seed)
        log = []
        ok, reason, _ = nav_run(loc, frames, starts, goals, oracle_block=a.oracle, log=log, odo=ODO[a.odo], seed=a.seed)
        L = pd.DataFrame(log)
        name = f"{CITY[st]}_3d_b1{a.tag}{'-oracle' if a.oracle else ''}-r0_steps.csv"
        L.to_csv(os.path.join(out, name), index=False)
        pd.DataFrame(dict(traj=range(len(ok)), success=ok.astype(int), reason=reason)).to_csv(
            os.path.join(out, name.replace('_steps.csv', '_traj.csv')), index=False)
        why = {x: int((reason == x).sum()) for x in np.unique(reason)}
        print(f"| {CITY[st]} | {'정답' if a.oracle else '추정'} | {len(ok)} | {ok.mean()*100:.1f}% ({ok.sum()}) | {why} | {len(L)} "
              f"| {(L.err_m < 10).mean()*100:.1f}% | {(L.err_m > 32).mean()*100:.1f}% | {L.sec.mean():.3f} |", flush=True)


if __name__ == '__main__':
    main()
