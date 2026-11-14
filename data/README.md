# 데이터

git에는 올리지 않는다(약 340 MB). `data/im_t1_3d_1000/`에 아래 구조로 둔다.

```
data/im_t1_3d_1000/
├── samples.csv          set, k, row, x_norm, y_norm, theta, ...  (정답: 블록 단위 위치, heading 도)
├── 3d_ho38bc/           SF:        k000_uav.jpg (256 px), k000_block.jpg (512 px, 2x2 타일) ... k999
└── 3d_ho37bc/           Singapore: 같은 형식
```

## 받는 법
```bash
gh auth login                     # 처음 한 번 (저장소가 비공개라 필요)
python scripts/download_data.py   # Release data-v1에서 받아 data/im_t1_3d_1000/ 에 푼다
```
브라우저로 받으려면 저장소의 Releases → `data-v1` → `im_t1_3d_1000.zip`을 받아 `data/`에 두고 위 스크립트를 실행한다.

Kaggle에서 돌리려면 이 zip을 자기 Kaggle 데이터셋으로 올리고 `build_kernel.py --dataset`에 그 이름을 넣는다.

## 후보 블록 실험용 파일 (`run.py --block-radius`, `run_nav.py`)
capstone1 저장소의 Bearing_UAV-90K가 있으면 한 번만 만들면 된다.
```bash
python scripts/export_block_index.py --capstone1 ../bearinguav
```
`data/im_t1_3d_1000/`에 `block_index.csv`(set, k, block_x, block_y)와 `maps/3d_ho38bc.jpg`, `maps/3d_ho37bc.jpg`(도시 지도, 장당 약 4 MB)가 생긴다. Kaggle에는 zip 하나가 아니라 이 폴더째 `kaggle datasets create -p data/im_t1_3d_1000 --dir-mode zip`으로 올린다. 커널이 압축 안 된 `samples.csv`를 찾기 때문이다.

## 처음부터 다시 만들기 (보통은 필요 없음)
capstone1 저장소와 Bearing-UAV-90K 원본이 있어야 한다.
```bash
cd capstone1 && python <이 저장소>/scripts/export_from_capstone1.py --n 1000 --out <폴더>
```
행 번호(row)는 capstone1 `diag/harness.py`의 T1 순서와 같다. 그래서 results/의 기준선과 샘플이 1:1로 맞는다.

## UAV-VisLoc (외부 데이터, `scripts/export_visloc.py`)

[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) 전체(비행 01~11, 약 23 GB)를 받아 `--root`로 준다. 07은 자세 열이 없고 위성 지도가 띠라 뺀다.

**받기**: Kaggle 비공개 데이터셋 `heeyunju/visloc-single-eval`, `heeyunju/visloc-seq-eval` (공동작업자로 추가받아야 보임). 직접 만들려면:

```bash
python scripts/export_visloc.py --root <UAV-VisLoc> --out data/visloc        # 약 4분, 880 MB
python scripts/visloc_calib.py outputs/<결과>.csv --data data/visloc/single_eval   # run.py 결과의 비행별 표 (10/20/32 m, 모름, heading)
```

정답은 **GPS 그대로**다(위치 보정 없음). 좌표 어긋남을 확인한 결과와 보정을 넣지 않은 이유는 [experiments/2026-11-13_visloc_data/README.md](../experiments/2026-11-13_visloc_data/README.md). 보정을 넣으려면 `visloc_calib.py --out`으로 만든 CSV를 `export_visloc.py --calib`에 준다.

| 폴더 | 내용 | 쓰는 곳 |
| --- | --- | --- |
| `single_calib/` | 비행마다 앞쪽 20% 줄, 3장마다 1장 | 좌표 어긋남 확인에만 씀. 평가에 쓰지 않음 |
| `single_eval/` | 나머지 줄, 3장마다 1장. set = `visloc_<비행>` | 한 장 비교 (원 Bearing-UAV, LoMa, 다른 모델, 8/4방향) |
| `seq_eval/` | 평가 줄을 지도 한 장(4096 px = 1,024 m)에 들어가는 구간으로 자름. 사진은 순서대로 전부. `maps/`, `block_index.csv` 포함 | 항법 비교 (R1, R1 + R4, 원 Bearing-UAV, 다른 모델), 거리별 비교, 프레임 간 매칭 |

- 블록은 고정 격자에서 자른다 (GPS 중심이 아님). 정답이 블록 안 여기저기에 있다.
- `x_norm`, `y_norm`은 정답(지금은 GPS 그대로라 `x_norm_raw`, `y_norm_raw`와 같다). `tilted`, `turning`은 어려운 사진 표시(빼지 않음).
- 10 m 기준은 GPS 어긋남(약 10~16 m)에 민감하므로 20 m, 32 m도 같이 보고한다.
- `seq_eval`의 `dist_m`은 구간 시작부터 누적 거리. 사진 간격이 60~130 m라 시뮬레이터(15 m)와 다르다.
- `seq_eval` 항법은 `run_nav.py`(3D 닫힌 루프)가 아니라 재생 항법으로 돌린다. 실제 사진은 드론이 지나간 곳에서만 있어서, 찍힌 순서대로 사진을 주고 "직전 추정 + 이동량(VIO 오차)"으로 다음 블록을 고른다. GPS는 채점과 이동량 시뮬레이션에만 쓴다.

```bash
# 우리 방법 (R0, R1, R1 + R4, 추측 항법만)
python scripts/run_visloc_nav.py --data data/visloc/seq_eval --out outputs/visloc_nav --matcher loma --angles 8 \
    --stop-inl 15 --variants r0,r1,r4g0n5w20,dr --odo vio --seed 2026
# Kaggle
python scripts/kaggle/build_kernel.py --user <아이디> --dataset heeyunju/visloc-seq-eval --slug <이름> \
    --script run_visloc_nav --args "--matcher loma --angles 8 --stop-inl 15 --variants r0,r1,r4g0n5w20,dr"
```

기준선(B1, CorrMatchNet 등)은 3D 항법에 꽂았던 loc을 그대로 넘긴다. 모델은 고치지 않는다.

```python
from imatch.replay import run_segments, summarize
# B1Loc: 3D 항법 때 b1_loc_sg를 만든 클래스 (지도 city만 구간마다 바뀜). reject_inl=0 = R0와 같은 조건
steps, segs = run_segments(lambda city: B1Loc(model, city), 'data/visloc/seq_eval', odo='vio', seed=2026)
print(summarize(steps, segs))
```

같은 `--seed`면 모든 방법이 같은 구간·출발점·이동 오차·블록 고르기 규칙을 쓴다. 지표는 걸음별 10/20/32 m, 마지막 오차, 모름 비율("도착 성공"은 경로가 정해져 있어 쓸 수 없다). `--max-dist 300`이면 구간 앞 300 m까지만 쓴다.
- 한계: GPS·촬영 시각 정렬은 검증하지 않았다. 드론 축척은 카메라 보정값이 아니라 연속 사진으로 잰 값이다. 드론 사진의 가운데 64 m만 쓴다.
