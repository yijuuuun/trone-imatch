"""기존 모델에 붙일 "모름" 기준(R1)을 개발용 single_calib(513장)에서만 정한다. 시험용 seq_eval은 보지 않는다.

v1의 R1은 LoMa inlier < 15면 "모름"이다(15는 학습 도시에서 정함, VisLoc에서 고르지 않음). 기존 모델은 inlier가 없으므로
각자 내는 점수를 쓴다: CorrMatchNet conf, OS-FPI heatmap 최고값, 검색형(Sample4Geo, MEAN, Game4Loc) 코사인 유사도.
Bearing-UAV는 점수를 내지 않아 붙일 수 없다.

고르는 규칙 (결과를 보기 전에 정함, #48 single_gate와 같은 채점):
  - 한 장마다 모델 답과 점수를 구한다. 점수 < T면 "모름" → 블록 가운데를 답한다.
  - T 후보는 개발용 점수의 분위수(거절 비율 0, 5, ..., 95%)다.
  - 20 m 이내 비율(Bearing-UAV 논문 항법 기준)이 가장 높은 T를 고른다. 같으면 거절 비율이 낮은 쪽.

검색형은 128 m 타일이 블록 밖까지 나가므로 블록 한 장만으로는 항법과 같은 입력이 안 된다. 그래서 원 UAV-VisLoc
GeoTIFF에서 블록 둘레 ±128 m를 export_visloc.py와 같은 방식으로 다시 그리고, 가운데가 저장된 k###_block.jpg와
같은지 확인한다 (--root).

  python experiments/2026-11-18_visloc_nav_baselines/calib_gate.py --root /mnt/c/Users/heeyu/Desktop/dataset \
      --models corr38,corr37,osfpi,sample4geo,game4loc,mean        # → calib_gate_<model>.csv, gate.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baselines as B  # noqa: E402

CTX = 512        # 블록 둘레 여유 (px). 검색형 타일이 닿는 범위: 블록 가운데 ±(128 + 32 + 256) px = 블록 밖 160 px
SCORE = {'corr38': 'conf', 'corr37': 'conf', 'osfpi': 'conf', 'sample4geo': 'sim', 'game4loc': 'sim', 'mean': 'sim'}
QS = np.arange(0, 100, 5)


def build_contexts(data, root, out):
    """single_calib 각 장의 블록 둘레 지도 (1536 px, 블록은 (512, 512)에서 시작 = 블록 번호 (2, 2))."""
    import export_visloc as E
    S = pd.read_csv(os.path.join(data, 'samples.csv'))
    os.makedirs(out, exist_ok=True)
    rows = []
    for st, g in S.groupby('set', sort=False):
        flight = st.split('_')[1]
        todo = [r for r in g.itertuples() if not os.path.exists(os.path.join(out, f'{st}_k{r.k:03d}.png'))]
        if not todo:
            continue
        fdir = Path(root) / flight
        df, _, _ = E.load_frames(fdir, flight, 10, 5, 15)        # export_visloc 기본값 (height_tol, tilt_max, turn_deg)
        mos = E.Mosaic(fdir, float(df.lat.mean()))
        for r in todo:
            X, Y = mos.xy(r.lat, r.lon)                          # export_visloc.gt_positions와 같은 보정 (지금 데이터는 0)
            h = np.radians((r.Phi1 + r.head_off) % 360)
            X = X + (r.along_m * np.sin(h) + r.right_m * np.cos(h)) * E.PX_PER_M
            Y = Y - (r.along_m * np.cos(h) - r.right_m * np.sin(h)) * E.PX_PER_M
            cx, cy = X - r.x_norm * 128, Y - r.y_norm * 128      # 블록 가운데 (평면 px)
            bx, by = int(round((cx - 256) / 256)), int(round((cy - 256) / 256))
            ctx = mos.render(bx * 256 - CTX, by * 256 - CTX, 512 + 2 * CTX, 512 + 2 * CTX)
            blk = cv2.cvtColor(cv2.imread(os.path.join(data, st, f'k{r.k:03d}_block.jpg')), cv2.COLOR_BGR2RGB)
            full = ctx is not None
            if not full:                                         # 지도 가장자리: 블록만 반사로 채움 (RetrievalLoc.bind와 같음)
                ctx = cv2.copyMakeBorder(blk, CTX, CTX, CTX, CTX, cv2.BORDER_REFLECT)
            diff = float(np.abs(ctx[CTX:CTX + 512, CTX:CTX + 512].astype(np.int16) - blk).mean())
            cv2.imwrite(os.path.join(out, f'{st}_k{r.k:03d}.png'), cv2.cvtColor(ctx, cv2.COLOR_RGB2BGR))
            rows.append(dict(set=st, k=r.k, full=int(full), block_diff=round(diff, 2)))
        print(f'  {st}: {len(todo)}장', flush=True)
    if rows:
        R = pd.DataFrame(rows)
        p = os.path.join(out, 'contexts.csv')
        R = pd.concat([pd.read_csv(p), R]) if os.path.exists(p) else R
        R.to_csv(p, index=False)
    R = pd.read_csv(os.path.join(out, 'contexts.csv'))
    print(f'둘레 지도 {len(R)}장: 원 지도로 그림 {R.full.sum()}, 가장자리라 반사 {len(R) - R.full.sum()} | '
          f'가운데와 저장된 블록의 평균 차이 중앙값 {R[R.full == 1].block_diff.median():.2f}, 최대 '
          f'{R[R.full == 1].block_diff.max():.2f} (0~255, JPEG 손실 정도면 맞음)', flush=True)
    return R


def make_loc(name, a):
    if name == 'osfpi':
        return B.OSFPILoc(a.osfpi, uav_crop=192)
    if name in ('corr38', 'corr37'):
        return B.CorrLoc(a.cap, {'corr38': 'results/corr/p2bv4_3d_ho38bc_20261013_135730',
                                 'corr37': 'results/corr/p2bv4_3d_ho37bc_20261013_173659'}[name])
    from run_baselines import RET
    return B.RetrievalLoc(name, os.path.join(a.ret, RET[name][0]), os.path.join(a.ret, 'weights', RET[name][1]))


def choose(D):
    """점수 < T면 블록 가운데. 20 m 이내 비율 최고, 같으면 거절 비율 낮은 쪽."""
    best, rows = None, []
    for q in QS:
        T = float(np.percentile(D.score, q)) if q else -np.inf
        e = np.where(D.score >= T, D.err_m, D.err0_m)
        r = dict(reject_pct=int(q), T=T, rejected=float((D.score < T).mean() * 100),
                 le10=float((e <= 10).mean() * 100), le20=float((e <= 20).mean() * 100), med=float(np.median(e)))
        rows.append(r)
        if best is None or r['le20'] > best['le20'] + 1e-9:
            best = r
    return best, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.expanduser('~/capstone/trone-imatch/data/visloc/single_calib'))
    ap.add_argument('--root', default='/mnt/c/Users/heeyu/Desktop/dataset', help='원 UAV-VisLoc 폴더 (01/ ... 11/)')
    ap.add_argument('--ctx', default=os.path.join(ROOT, 'outputs/visloc_calib_ctx'))
    ap.add_argument('--models', default='corr38,corr37,osfpi,sample4geo,game4loc,mean')
    ap.add_argument('--cap', default=os.path.expanduser('~/capstone/bearinguav'))
    ap.add_argument('--osfpi', default=os.path.expanduser('~/capstone/ext/OS-FPI'))
    ap.add_argument('--ret', default=os.path.expanduser('~/capstone/ext'))
    a = ap.parse_args()
    S = pd.read_csv(os.path.join(a.data, 'samples.csv'))
    build_contexts(a.data, a.root, a.ctx)
    gp = os.path.join(HERE, 'gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else {}
    for name in a.models.split(','):
        out = os.path.join(HERE, f'calib_gate_{name}.csv')
        if not os.path.exists(out):
            loc, t0, rows = make_loc(name, a), time.time(), []
            for r in S.itertuples():
                ctx = cv2.cvtColor(cv2.imread(os.path.join(a.ctx, f'{r.set}_k{r.k:03d}.png')), cv2.COLOR_BGR2RGB)
                uav = cv2.cvtColor(cv2.imread(os.path.join(a.data, r.set, f'k{r.k:03d}_uav.jpg')), cv2.COLOR_BGR2RGB)
                p = loc.bind(ctx)([uav], np.array([0.]), [(2, 2)])[0]
                t = np.array([r.x_norm, r.y_norm])
                rows.append(dict(set=r.set, k=r.k, est_x=p[0], est_y=p[1], err_m=float(np.linalg.norm(p - t) * 32),
                                 err0_m=float(np.linalg.norm(t) * 32), score=loc.last[0][SCORE[name]]))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f'{name}: {len(rows)}장 {time.time() - t0:.0f} s', flush=True)
            del loc
        D = pd.read_csv(out)
        best, tab = choose(D)
        tab.to_csv(os.path.join(HERE, f'calib_gate_{name}_sweep.csv'), index=False)
        r0 = tab.iloc[0]
        gate[name] = dict(score=SCORE[name], T=best['T'] if best['reject_pct'] else None, reject_pct=best['reject_pct'], le20=round(best['le20'], 1),
                          le20_always=round(r0.le20, 1), le20_centre=round(float((D.err0_m <= 20).mean() * 100), 1),
                          med_always=round(r0.med, 1))
        print(f"{name}: 항상 답함 20 m {r0.le20:.1f}% (중앙값 {r0.med:.1f} m) → 점수 < {best['T']:.4g}면 모름 "
              f"(거절 {best['rejected']:.0f}%): 20 m {best['le20']:.1f}% | 블록 가운데만 {gate[name]['le20_centre']}%",
              flush=True)
        json.dump(gate, open(gp, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
