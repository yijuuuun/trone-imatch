# Trone: 신뢰도 기반 GPS-Free 드론 위치추정 및 항법

Trone 팀 졸업프로젝트 · 2026-11-27

GPS 없이 **드론 영상과 위성 지도만으로** 위치를 찾고, 그 위치로 목적지까지 날아간다. 학습 없이 공개 이미지 매처(LoMa)로 대응점을 찾고 기하 계산으로 위치·heading을 구한다.

연구의 핵심은 매처가 아니라 **매칭을 언제 믿고 언제 버릴지**다. 틀린 매칭을 그대로 쓰면 다음 지도 블록을 잘못 골라 오차가 연쇄적으로 쌓인다.

**최종 v1 = LoMa 4방향 매칭 + R1(신뢰도 기반 매칭 거부) + R4(경량 이웃 블록 복구)**

| 구성 | 하는 일 |
| --- | --- |
| 4방향 매칭 | 0°부터 90°씩 돌려 매칭하고, inlier ≥ 15가 나오면 나머지 방향은 건너뛴다 |
| **R1** | inlier < 15면 "모름"으로 보고 매칭 결과를 버린다. 대신 이동 센서로 추측한 위치(추측 항법)를 쓴다 |
| R4 | "모름"이 이어지면 이웃 블록을 다시 찾는다. 처음엔 십자 4칸(마지막 성공 방향 먼저), 5·10·15번째 연속 실패에서 3×3(inlier ≥ 20) |

> 이전 방향인 Grid Pooling(CSMG 교체)은 태그 [`grid-pooling-final`](https://github.com/yijuuuun/trone-imatch/tree/grid-pooling-final/grid_pooling)에 있다. 방향 전환 계기는 [docs/00_교수님_피드백.md](docs/00_교수님_피드백.md).

## 주요 결과

모든 비교는 같은 구간끼리 짝지은 McNemar 검정(양측 정확 이항)이다. 항법은 기록된 비행 사진을 순서대로 넣는 재생 항법이고, 이동 오차는 모의값이다. 실제 기체에서는 돌리지 않았다.

### 연속 항법: UAV-VisLoc 평가용 92구간 (약 1 km, 마지막 오차 기준)

| 방법 | 20 m 이내 | 40 m 이내 | 마지막 오차 중앙값 | 걸음당 시간 (T4) |
| --- | --- | --- | --- | --- |
| 추측 항법만 | 25 (27.2%) | 56 (60.9%) | 33.1 m | 0 |
| Bearing-UAV | 19 (20.7%) | 41 (44.6%) | 62.5 m | 0.05~0.09초 |
| 기존 모델 5종 (CorrMatchNet, OS-FPI, Sample4Geo, MEAN, Game4Loc) | 4~13 | 10~28 | 60.5~137.7 m | 0.14~1.54초 |
| LoMa R0 (항상 믿음) | 25 (27.2%) | 41 (44.6%) | 63.1 m | 0.79초 |
| R1 | 44 (47.8%) | 69 (75.0%) | 20.8 m | 0.54초 |
| **최종 v1 (R1 + R4)** | **45 (48.9%)** | **71 (77.2%)** | **20.3 m** | **1.48초** |

- **R1 vs R0 (40 m): 28 vs 0, p = 7e-09.** 같은 LoMa에서 "모름" 규칙 하나로 41 → 69구간.
- v1 vs Bearing-UAV (40 m): 36 vs 6, p = 3e-06. v1 vs 추측 항법 (40 m): 23 vs 8, p = 0.01.
- 기존 모델 5종에 같은 R1을 붙여도 LoMa R1에 못 미친다(`experiments/2026-11-18_visloc_nav_baselines`).

### 처음 보는 실제 비행 데이터 (설정을 모두 정한 뒤 처음 실행, 추가 학습 없음)

| 데이터 | 구간 | 추측 항법 | R1 | 최종 v1 | R1 vs 추측 항법 (40 m) |
| --- | --- | --- | --- | --- | --- |
| AerialVL | 85 | 28 (32.9%) | 71 (83.5%) | 72 (84.7%) | 49 vs 6, p = 2e-09 |
| VPAIR | 118 | 20 (16.9%) | 101 (85.6%) | 102 (86.4%) | 81 vs 0, p < 1e-20 |

### R4: 언제 도움이 되나

| 조건 | R1 → v1 (40 m 도착) | 짝 비교 |
| --- | --- | --- |
| 보통 이동 오차 (VisLoc·AerialVL·VPAIR·XIAN·도쿄 3D) | 차이 −1 ~ +2구간 | 모두 유의하지 않음 |
| 이동 오차 4배, VisLoc 개발용 77구간 × 새 seed 4개 | 137 → 155 / 308 | 22 vs 4, p = 0.0005 |
| 이동 오차 4배, XIAN-VisLoc 217구간 × seed 4개 | 575 → 633 / 868 | 64 vs 6, p = 2e-13 |
| 수십 km 장거리 (VisLoc 481 km), 이동 오차 1·2·4배 | 이득 확인 안 됨 | p ≥ 0.074 |

R4는 평소 성능을 거의 바꾸지 않고, **이동 센서 오차가 큰 약 1~2 km 구간**에서 도착을 유의하게 늘리는 보조 복구 장치다. 대신 걸음당 시간이 약 2배다.

### 장거리: R1 오차는 쌓이지 않는다

VisLoc 시험용 줄 11개(481 km)를 출발 때만 GPS를 주고 끊지 않고 날았다. 50 km 지점에서 40 m 이내는 R1 5/7, 추측 항법 0/7이고, R1 오차 중앙값은 22 m다. 길을 잃어도 매칭되는 곳을 지나면 다시 맞춰진다(`experiments/2026-11-20_visloc_nav_full`).

### 속도

지도 특징 캐시, 4방향 + 조기 종료, R4 경량화(방향 기억, 3×3 쉬는 시간)로 결과를 거의 그대로 두고 걸음당 3.57초 → 1.48초(T4). Jetson 등 기체 장비에서는 재지 않았다.

## 파이프라인

```
드론 영상 ─ ① 회전 탐색 ─ ② 특징 추출·매칭 ─ ③ 검증 ─ ④ 위치·heading ─ ⑤ R1 판정 ─ ⑥ 다음 블록
               (4방향)         (LoMa)          (MAGSAC++)   (호모그래피)     inlier < 15면   추정 위치 또는
지도 블록 ──────────────────┘                                              "모름" → R4     추측 위치로 선택
```

| 단계 | 코드 | 하는 일 |
| --- | --- | --- |
| ① 회전 탐색 | `imatch/pipeline.py` | 드론 영상을 돌려 각각 매칭, 조기 종료 |
| ② 매칭 | `imatch/matchers.py` | LoMa(`loma`, 지도 특징 캐시 `loma-cache`), RoMa v2, SIFT-LightGlue, OpenCV SIFT |
| ③ 검증 | `imatch/geometry.py` `magsac` | MAGSAC++로 틀린 짝 제거 |
| ④ 위치·heading | `imatch/geometry.py` `pose_from_homography` | 드론 중심을 지도로 옮겨 위치, 회전 성분으로 heading |
| ⑤⑥ 항법 | `imatch/nav.py`, `imatch/replay.py` | R0·R1·R4, 블록 선택, 추측 항법, 이동 오차 모의(`vio`, `vio2`, `vio4`) |
| 지도 블록 | `imatch/blocks.py` | 지도 격자, 이웃 블록, 큰 지도 타일 읽기 |
| 평가 | `imatch/metrics.py` | 실패 정의, McNemar 검정 |

## 폴더 구조

```
trone-imatch/
├── imatch/                 파이프라인·항법 코드 (위 표)
├── scripts/
│   ├── run.py              단일 영상 매칭 → 샘플별 CSV
│   ├── evaluate.py         짝지어 비교 → 표
│   ├── run_visloc_nav.py   실제 비행 재생 항법 (UAV-VisLoc, AerialVL, VPAIR 형식)
│   ├── run_nav.py          Bearing-UAV 3D 1 km 항법
│   ├── export_visloc.py    UAV-VisLoc → 항법 구간·지도 (--full: 긴 구간)
│   ├── export_aerialvl.py  AerialVL → 항법 구간 (2.5배 축척 규칙)
│   ├── export_vpair.py     VPAIR → 항법 구간
│   └── kaggle/             Kaggle·Colab GPU 커널 만들기
├── experiments/            실험마다 README(미리 정한 기준, 명령어, 결과)와 CSV
├── data/                   데이터 (git에 안 올림, data/README.md)
└── docs/                   교수님 피드백, 계획, 실험 기록, 주간 정리
```

## 시작하기

```bash
pip install -r requirements.txt

# 1) 설치 확인 (CPU, 몇 초)
python scripts/download_data.py      # 샘플 데이터 (GitHub Release, gh auth login 필요)
python scripts/run.py --data data/im_t1_3d_1000 --out outputs/check --matchers cv-sift --n 5 --device cpu

# 2) 최종 v1 항법 (GPU): 추측 항법, R1, 최종 v1을 같은 구간에서
python scripts/run_visloc_nav.py --data data/visloc/seq_eval --out outputs/v1 \
    --matcher loma-cache --angles 4 --stop-inl 15 --variants dr,r1,r4g0n5w20ac4

# 이동 오차 4배, 다른 seed
python scripts/run_visloc_nav.py ... --odo vio4 --seed 2027

# 3) Kaggle T4에서 돌리기
python scripts/kaggle/build_kernel.py --user <아이디> --dataset <아이디>/<데이터> --slug <이름> --args "..."
kaggle kernels push -p kaggle_kernels/<이름> --accelerator NvidiaTeslaT4
```

GTX 1650에서는 LoMa가 fp16에서 짝 0개를 돌려준다. `LOMA_FP32=1`을 붙이면 돌아간다. 보고용 숫자는 모두 Kaggle T4 기본 설정이다.

## 데이터와 평가 조건

| 데이터 | 용도 | 분할 |
| --- | --- | --- |
| Bearing-UAV-90K (3D 렌더) | 매처·회전·"모름" 기준 고르기, 1 km 항법 | 학습 도시 = 개발용, SF·Singapore = 평가용 |
| UAV-VisLoc | 주 평가 (단일 1,738장, 항법 92구간) | 비행 줄마다 앞 20% 개발용, 나머지 평가용 |
| AerialVL, VPAIR | 처음 보는 데이터 일반화 | 전부 평가용 |
| XIAN-VisLoc | R4 확인 실험 (비행 12개, 217구간) | 사전 등록 후 실행 |

- 도착 성공 = 마지막 오차 40 m 이내(20 m도 함께 보고). 단일 영상은 10·15·20·32 m.
- 결과를 보고 정한 규칙은 그 결과를 본 데이터로 판정하지 않는다. 예외 두 가지(매처를 평가용 도시 k < 100으로 고름, R4 채택을 평가용 결과를 본 뒤 정함)는 각 실험 README에 적었다.
- 실험별 데이터·분할·seed·스크립트는 각 `experiments/<날짜>_<이름>/README.md`에 있다.
