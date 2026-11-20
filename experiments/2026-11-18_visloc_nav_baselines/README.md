# UAV-VisLoc 항법: 기존 모델 비교 (주희윤, 2026-11-18)

교수님 피드백 6("동작하는 다른 방법과 비교")과 8("같은 데이터")에 대한 실험이다. 시험용 `seq_eval` 92구간에서 기존 오픈소스 모델을 우리 v1과 같은 재생 항법 조건으로 돌렸다.

**결론 한 줄**: 모델을 바꾸지 않고 꽂은 기존 모델 여섯 개(위치 회귀형 Bearing-UAV, CorrMatchNet, OS-FPI와 검색형 Sample4Geo, MEAN, Game4Loc)는 모두 v1보다 유의하게 적게 도착하고, 추측 항법보다도 낮다. 마지막 20 m 이내(Bearing-UAV 논문 기준)는 v1 45 / 92, 기존 모델 4~19 / 92다. 기존 모델에도 v1과 같은 "모름" 처리(R1)를 각자의 점수로 붙이면 추측 항법 수준(8~25 / 92)까지는 오르지만, 여전히 LoMa R1(44)과 v1(45)보다 유의하게 낮다(20 m p ≤ 0.005). 차이는 매처의 점수가 맞는 답을 가려내는지에서 나온다(2.1절). GeoAnchor는 조건을 맞추지 못해 비교에서 뺐다(4장).

## 1. 방법

- 데이터: `heeyunju/visloc-seq-eval` 92구간, 1,020걸음.
- 공통 조건 (`imatch/replay.py`, [#45](../2026-11-16_visloc_nav_v1/README.md)와 같음)
  - 출발 위치는 GPS로 준다.
  - 이동 오차는 `vio`, seed 2026이다(치우침 3°, 걸음 방향 2°, 축척 5%).
  - 블록은 추측 위치를 담은 512 px(128 m) 블록이다.
  - 모델이 할 일은 "블록 안 위치 추정" 한 칸이고, 이 칸만 모델마다 바꿨다.
- 기존 모델은 신뢰도를 쓰지 않는 R0(답을 항상 믿음)로 돌렸다. 나침반이 없으므로, 방향을 모르는 모델에는 우리와 같은 4방향 탐색을 붙였다(모델에 유리한 쪽).

| 모델 | 출처 | 학습 데이터 | 우리 조건에 맞춘 것 |
| --- | --- | --- | --- |
| Bearing-UAV | 공개 `cross_view` 가중치 | Bearing-UAV-90K | 없음 ([#42](../2026-11-16_bearing_visloc_colab/README.md)와 같은 입력). 방향을 스스로 추정 |
| CorrMatchNet | capstone1 p2bv4 (SF 뺀 학습 / 싱가포르 뺀 학습) | Bearing-UAV-90K 3D 렌더 | 4방향 중 신뢰도(conf) 최고 |
| OS-FPI | [cjhing/OS-FPI](https://github.com/cjhing/OS-FPI) `net_040.pth` | UL14 (저고도 80~100 m) | 4방향 중 heatmap 최고. 드론 크롭 가운데 192 px (개발용 `single_calib`에서 고름, `calib_osfpi.py`) |
| Sample4Geo (검색형) | [Skyy93/Sample4Geo](https://github.com/Skyy93/Sample4Geo) University-1652 가중치 (SHA-256 `04f4c899`, #33과 같음) | University-1652 | 아래 타일 검색 |
| MEAN (검색형) | [ISChenawei/MEAN](https://github.com/ISChenawei/MEAN) `weights_e2_0.9355` (`bc0f0798`, #33과 같음) | University-1652 | 아래 타일 검색 |
| Game4Loc (검색형) | [Yux1angJi/GTA-UAV](https://github.com/Yux1angJi/GTA-UAV) cross-area 가중치 (`12ec61d1`, HF `Yux1ang/gta_uav_pretrained_models`) | GTA-UAV (게임 렌더, VisLoc 미포함) | 아래 타일 검색 |

**검색형 모델을 넣은 방법** (`baselines.py`의 `RetrievalLoc`)
- 검색형은 "두 사진이 같은 곳인가"만 답하고 블록 안 좌표를 내지 않는다. 그래서 블록 안 후보 위치마다 위성 타일을 잘라 비교하고, 가장 비슷한 타일의 중심을 위치로 썼다.
- 후보: 블록 가운데 ±32 m를 16 m 간격 5×5로 보고, 가장 비슷한 점 둘레를 8 m 간격 3×3으로 한 번 더 본다.
- 타일은 후보 점 중심 128 m(512 px)이다. 이 크기는 결과를 보기 전에 세 모델 공통 규칙으로 정했고 바꾸지 않았다.
- 방향은 드론 사진을 4방향으로 돌려 가장 높은 유사도를 쓴다.
- 전처리는 각 저장소의 공식 검증 변환(384 px, 정규화)이고 fp32다. 모델과 가중치는 바꾸지 않았다.

**기존 모델 + R1 (v1과 같은 "모름" 처리)** (`calib_gate.py`, `baselines.py`의 `Gated`)
- v1의 R1은 LoMa inlier < 15면 "모름"이라 하고 추측 위치를 쓴다. 기존 모델은 inlier가 없으므로 각자 내는 점수를 쓴다: CorrMatchNet conf, OS-FPI heatmap 최고값, 검색형 코사인 유사도. Bearing-UAV는 점수를 내지 않아 붙일 수 없다.
- 기준 T는 개발용 `single_calib`(513장)에서만 골랐고, 규칙은 결과를 보기 전에 정했다.
  - 한 장마다 점수 < T면 블록 가운데를 답한다(#48 single_gate와 같은 채점).
  - T 후보는 개발용 점수의 분위수(거절 0, 5, …, 95%)다. 20 m 이내 비율이 가장 높은 T를 고른다.
  - v1의 15는 학습 도시(3D)에서 정했고 VisLoc에서 고르지 않았다. 그래서 이 규칙은 기존 모델에 유리한 쪽이다.
- 검색형은 128 m 타일이 블록 밖까지 나간다. 그래서 개발용 사진마다 원 UAV-VisLoc GeoTIFF에서 블록 둘레 ±128 m를 `export_visloc.py`와 같은 방식으로 다시 그렸다. 가운데 512 px가 저장된 블록과 같았다(평균 차이 0.94 / 255, JPEG 손실 수준). 지도 가장자리 21장은 반사로 채웠다.
- 시험용 92구간은 로컬 GTX 1650에서 돌렸다. 걸음마다 모델 계산은 R0와 같으므로, 표의 걸음당 s는 T4 R0 값이다.

**연결 확인**
- Bearing-UAV: 1,112걸음 모두 #42와 걸음 오차 차이 최대 6e-5 m로 같다.
- 지도에서 정답 위치 조각을 잘라 드론 사진으로 넣으면(`sanity.py`, 40장) 다음과 같다.
  - CorrMatchNet: 오차 중앙값 1.3 m
  - OS-FPI: 11.6 m
  - Bearing-UAV: 13.8 m
  - Sample4Geo: 15.9 m, MEAN: 20.2 m
  - Game4Loc: 32.7 m
  - 블록 가운데만 찍으면: 24.6 m
- 즉 어댑터의 좌표 변환은 맞다. 검색형은 타일 단위로만 구분하므로 같은 영상에서도 오차가 크다. Game4Loc은 블록 가운데보다 나쁜데, 게임 데이터 학습 때의 드론-타일 크기 관계가 128 m 타일과 다를 수 있다(4장).

```bash
python experiments/2026-11-18_visloc_nav_baselines/calib_osfpi.py                      # OS-FPI 입력 (개발용)
python experiments/2026-11-18_visloc_nav_baselines/build_kernel.py --pack outputs/visloc_baselines_pack
kaggle datasets create -p outputs/visloc_baselines_pack --dir-mode zip                   # heeyunju/visloc-baselines
kaggle kernels push -p kaggle_kernels/visloc-nav-baselines --accelerator NvidiaTeslaT4  # 2.45 h (GeoAnchor 포함)
# 검색형: 저장소와 가중치는 ~/capstone/ext/{MEAN,Sample4Geo,GTA-UAV}, ~/capstone/ext/weights/
python experiments/2026-11-18_visloc_nav_baselines/build_kernel.py --ret-pack outputs/visloc_retrieval_pack \
    --slug visloc-nav-retrieval --args "--models sample4geo,game4loc,mean"   # 순서 고정: MEAN이 timm convnext 등록을 덮어씀
kaggle datasets create -p outputs/visloc_retrieval_pack --dir-mode zip                  # heeyunju/visloc-retrieval
kaggle kernels push -p kaggle_kernels/visloc-nav-retrieval --accelerator NvidiaTeslaT4  # 1.0 h
# R1판: 개발용에서 기준 고르기 (원 UAV-VisLoc 폴더 필요, GTX 1650 약 1 h) → gate.json, 그다음 92구간 (약 2 h)
python experiments/2026-11-18_visloc_nav_baselines/calib_gate.py --root <UAV-VisLoc 폴더>
python experiments/2026-11-18_visloc_nav_baselines/run_baselines.py --data data/visloc/seq_eval --out outputs/visloc_nav_r1 \
    --models corr38-r1,corr37-r1,osfpi-r1,sample4geo-r1,game4loc-r1,mean-r1
python experiments/2026-11-18_visloc_nav_baselines/table.py                              # 아래 표, table.csv
```

## 2. 결과 (Kaggle T4, 92구간)

| 방법 | 마지막 20 m 이내 | 마지막 40 m 이내 | 마지막 오차 중앙값 | 걸음 오차 중앙값 | 걸음당 s | v1 대비 20 m (v1만 / 상대만) | v1 대비 40 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 추측 항법만 | 25 (27.2%) | 56 (60.9%) | 33.1 m | 17.3 m | 0 | 32 / 12, p = 0.004 | 23 / 7, p = 0.005 |
| Bearing-UAV | 19 (20.7%) | 41 (44.6%) | 62.5 m | 43.0 m | 0.09 | 35 / 9, p = 1e-04 | 36 / 5, p = 8e-07 |
| CorrMatchNet (SF 뺀 학습) | 9 (9.8%) | 27 (29.3%) | 70.8 m | 57.4 m | 0.16 | 37 / 1, p = 3e-10 | 50 / 5, p = 2e-10 |
| CorrMatchNet (싱가포르 뺀 학습) | 4 (4.3%) | 25 (27.2%) | 94.9 m | 61.2 m | 0.16 | 42 / 1, p = 1e-11 | 53 / 6, p = 2e-10 |
| OS-FPI | 4 (4.3%) | 10 (10.9%) | 137.7 m | 103.6 m | 0.14 | 43 / 2, p = 6e-11 | 65 / 3, p = 4e-16 |
| Sample4Geo (검색형) | 11 (12.0%) | 28 (30.4%) | 81.5 m | 60.4 m | 1.23 (질의 0.04) | 40 / 6, p = 3e-07 | 49 / 5, p = 4e-10 |
| MEAN (검색형) | 13 (14.1%) | 26 (28.3%) | 88.0 m | 65.4 m | 0.54 (질의 0.03) | 36 / 4, p = 2e-07 | 51 / 5, p = 1e-10 |
| Game4Loc (검색형) | 4 (4.3%) | 26 (28.3%) | 60.5 m | 52.2 m | 1.54 (질의 0.03) | 45 / 4, p = 8e-10 | 52 / 6, p = 3e-10 |
| LoMa R0 (#45) | 25 (27.2%) | 41 (44.6%) | 63.1 m | 42.5 m | 0.79 | 20 / 0, p = 2e-06 | 31 / 0, p = 9e-10 |
| R1 (#45) | 44 (47.8%) | 69 (75.0%) | 20.8 m | 20.0 m | 0.74 (캐시 0.54) | 2 / 1, p = 1 | 3 / 0, p = 0.25 |
| **v1 (#45)** | **45 (48.9%)** | **72 (78.3%)** | **20.3 m** | 20.1 m | 3.57 (캐시 2.32) | — | — |

- Bearing-UAV 걸음당 시간은 #42(Colab T4) 0.05 s와 다르다. 이번은 Kaggle T4이고, 블록 타일 4장 인코딩을 포함한다.
- 캐시는 [#49](../2026-11-18_map_cache/README.md)의 지도 특징 캐시 시간이다(결과는 같다).
- 검색형의 걸음당 시간에는 후보 타일 34장 인코딩이 들어 있다. 괄호 안 "질의"는 드론 사진 4방향 인코딩만이다. 지도 타일을 미리 인코딩해 두면 걸음당 시간은 질의 쪽에 가까워진다.

### 2.1 기존 모델 + R1

개발용 `single_calib`에서 고른 기준 (513장, 20 m 이내 비율):

| 모델 | 점수 | 항상 답함 | 고른 기준으로 거름 | 거절 비율 |
| --- | --- | --- | --- | --- |
| CorrMatchNet (SF 뺀 학습) | conf < 0.884 | 24.8% | 29.4% | 70% |
| CorrMatchNet (싱가포르 뺀 학습) | conf < 0.923 | 24.4% | 28.3% | 80% |
| OS-FPI | heatmap < 0.924 | 12.1% | 28.5% | 85% |
| Sample4Geo | 유사도 < 0.871 | 19.9% | 27.3% | 95% |
| Game4Loc | 유사도 < 0.741 | 18.5% | 28.1% | 95% |
| MEAN | 유사도 < 0.881 | 19.3% | 29.2% | 95% |
| (블록 가운데만) | | | 28.8% | 100% |
| (참고: LoMa inlier < 15, #48) | | 45.6% | 46.8% | 48.5% |

시험용 `seq_eval` 92구간:

| 방법 | 마지막 20 m 이내 | 마지막 40 m 이내 | 마지막 오차 중앙값 | 걸음 오차 중앙값 | 모름 비율 | 걸음당 s | LoMa R1 대비 20 m (R1만 / 상대만) | v1 대비 40 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 추측 항법만 | 25 | 56 | 33.1 m | 17.3 m | 100% | 0 | 31 / 12, p = 0.005 | 23 / 7, p = 0.005 |
| CorrMatchNet (SF 뺀 학습) + R1 | 8 | 37 | 49.5 m | 36.4 m | 69% | 0.16 | 40 / 4, p = 2e-08 | 40 / 5, p = 8e-08 |
| CorrMatchNet (싱가포르 뺀 학습) + R1 | 11 | 44 | 42.9 m | 29.7 m | 77% | 0.16 | 38 / 5, p = 3e-07 | 31 / 3, p = 8e-07 |
| OS-FPI + R1 | 14 | 35 | 48.9 m | 30.9 m | 81% | 0.14 | 37 / 7, p = 5e-06 | 45 / 8, p = 2e-07 |
| Sample4Geo + R1 | 23 | 55 | 34.0 m | 18.8 m | 97% | 1.23 | 30 / 9, p = 0.001 | 24 / 7, p = 0.003 |
| MEAN + R1 | 25 | 48 | 39.2 m | 20.2 m | 92% | 0.54 | 30 / 11, p = 0.004 | 29 / 5, p = 4e-05 |
| Game4Loc + R1 | 21 | 47 | 39.5 m | 19.7 m | 95% | 1.54 | 32 / 9, p = 4e-04 | 30 / 5, p = 2e-05 |
| **LoMa R1 (#45)** | **44** | 69 | 20.8 m | 20.0 m | 49% | 0.74 (캐시 0.54) | — | 3 / 0, p = 0.25 |
| **v1 (#45)** | **45** | **72** | **20.3 m** | 20.1 m | 49% | 3.57 (캐시 2.32) | 1 / 2, p = 1 | — |

받아들인 답(모름이 아닌 걸음)의 질:

| 방법 | 받아들인 걸음 | 받아들인 답 오차 중앙값 | 추측 위치보다 나은 비율 |
| --- | --- | --- | --- |
| LoMa R1 | 523 | 20.0 m | 47% |
| CorrMatchNet + R1 (SF / 싱가포르) | 316 / 230 | 49.2 / 46.5 m | 31 / 34% |
| OS-FPI + R1 | 192 | 46.3 m | 30% |
| Sample4Geo / Game4Loc / MEAN + R1 | 28 / 51 / 85 | 31.7 / 37.1 / 60.9 m | 36 / 31 / 28% |

## 3. 해석

- **v1이 여섯 기존 모델보다 유의하게 많이 도착한다.** 20 m 기준(p ≤ 1e-4)과 40 m 기준(p ≤ 8e-7) 모두 그렇다.
- **여섯 기존 모델은 실제 사진에서 추측 항법보다 나쁘다.**
  - 이 모델들은 항상 답을 믿는다. 그래서 틀린 위치로 다음 블록을 고르고, 걸음의 58~79%에서 정답이 고른 블록 밖에 있다(검색형 67~72%).
  - CorrMatchNet은 3D 렌더로만 학습했고, OS-FPI는 저고도 UL14로 학습했다. 둘 다 UAV-VisLoc 실제 사진에 일반화되지 않는다.
  - OS-FPI는 정답 블록을 줘도 한 장 오차 중앙값이 52 m로, 블록 가운데만 찍는 것(26 m)보다 나쁘다(개발용 513장).
- **검색형 세 모델도 같은 자리다(마지막 40 m 이내 26~28 / 92).** University-1652(저고도 합성·실사)와 GTA-UAV(게임)로 학습해 VisLoc 고고도 실사와 조건이 다르고, 검색은 타일 단위라 블록 안 위치 구분이 거칠다. Game4Loc은 마지막 오차 중앙값(60.5 m)과 걸음 오차 중앙값(52.2 m)은 셋 중 가장 낮지만, 마지막 20 m 이내는 4로 가장 적다.
- **LoMa R0도 41로 Bearing-UAV와 같다.** "모름" 처리 없이는 우리 매처도 추측 항법보다 낮다.
- **같은 "모름" 처리를 붙여도 기존 모델은 v1에 못 미친다(2.1절).**
  - R1은 기존 모델을 추측 항법 쪽으로 끌어올린다. 마지막 40 m 이내가 Sample4Geo 28 → 55, CorrMatchNet(싱가포르 뺀 학습) 25 → 44가 된다.
  - 그래도 LoMa R1보다 20 m 기준 유의하게 낮고(p ≤ 0.005), 추측 항법(25 / 56)보다 유의하게 높은 것은 없다. CorrMatchNet 두 개와 OS-FPI는 R1을 붙여도 20 m 기준 추측 항법보다 유의하게 낮다(SF 뺀 학습 20 / 3, p = 0.0005; 싱가포르 뺀 학습 20 / 6, p = 0.009; OS-FPI 16 / 5, p = 0.03, 추측 항법만 / 상대만). 검색형 셋은 20 m에서 추측 항법과 차이가 없다(p ≥ 0.39). 40 m에서는 Game4Loc이 추측 항법보다 낮다(12 / 3, p = 0.035).
  - 원인은 점수의 질이다. 개발용에서 기존 모델의 점수로 거르면 블록 가운데만 찍는 것(28.8%)과 비슷한 27~29%까지만 오르고, 고른 기준이 70~95%를 거절한다. 시험용에서 받아들인 답도 오차 중앙값 32~61 m로 추측 위치보다 나은 것이 28~36%다. LoMa의 inlier는 받아들인 답이 20.0 m, 47%가 추측보다 낫다.
  - 그래서 v1의 이점은 "모름" 처리 자체가 아니라, **inlier 수가 맞는 답을 가려내는 신뢰도로 작동한다는 것**이다. 기존 모델의 conf·heatmap·유사도는 VisLoc 실제 사진에서 그 역할을 하지 못한다.
- **걸음당 시간은 위치 회귀형이 0.09~0.16 s로 v1(캐시 2.32 s)보다 15~25배 빠르다.** 검색형은 타일 인코딩 포함 0.54~1.54 s, 질의만 0.03~0.04 s다. 정확도-시간 트레이드오프 표(피드백 9)에 함께 넣을 수 있다.

## 4. 한계

- **GeoAnchor는 비교에서 뺐다.**
  - [YusufGUNEL/GeoAnchor](https://github.com/YusufGUNEL/GeoAnchor)(MIT)의 LoFTR·파티클 필터·DINOv2·규칙 코드를 가져와 우리 조건으로 돌렸다(`baselines.py`의 `run_geoanchor`, 결과 `visloc_geoanchor-r0_*`는 기록으로만 남김).
  - 원본은 300 m 원본 사진, IMU 나침반, 정사영 지도를 쓴다. 우리는 입력(64 m 크롭), 이동량(모의값), 방향(4방향 탐색), 보정(yaw·축척·흐림 끔)을 바꿨다.
  - 그 결과 원본의 매칭 통과 기준(inlier 40)을 걸음의 3.9%만 넘어, 사실상 추측 항법이 됐다(마지막 40 m 이내 60 vs 56, p = 0.13).
  - 원인이 우리가 바꾼 조건이라 GeoAnchor의 성능으로 볼 수 없다. 통과 기준을 다시 정하려 해도, 연속 비행 개발용 데이터가 없어 정할 수 없었다.
  - 원본 논문 숫자(비행별 오차 중앙값 8~25 m)는 300 m 사진, IMU 나침반, 원본 해상도 지도 조건이라 이 표와 비교할 수 없다.
- 기존 모델 모두 파인튜닝하지 않았다(우리 LoMa도 같다).
- 4방향 탐색은 CorrMatchNet, OS-FPI, 검색형 세 모델에 붙인 것이고, 모델 자체는 바꾸지 않았다. OS-FPI 입력(가운데 192 px)만 개발용에서 골랐다.
- 검색형의 "타일 중심 = 위치"와 타일 크기 128 m, 후보 간격(16 m → 8 m)은 우리가 정한 연결 방식이다. 검색 모델 논문은 블록 안 위치를 내는 방법을 정하지 않으므로, 이 결과는 "검색 모델을 이 항법에 넣었을 때"의 성능이다. 타일 크기는 개발용에서 고르지 않았고(세 모델 공통 고정값), 특히 Game4Loc은 지도 조각 확인에서도 블록 가운데보다 나빠 이 크기가 맞지 않을 수 있다.
- R1판의 기준은 한 장 채점(모름이면 블록 가운데)으로 골랐다. 항법에서는 모름일 때 추측 위치를 쓰고 이것이 블록 가운데보다 정확하므로, 항법에 맞는 기준은 더 높을 수 있다. 연속 비행 개발용 데이터가 없어 항법으로 고르지 못했다. 검색형 세 모델은 후보 범위의 끝(거절 95%)이 골라졌다. 범위를 넓혀 100%가 되면 추측 항법과 같아진다.
- R1판에는 v1의 R4(복구)를 붙이지 않았다. 그래서 짝 비교 대상은 LoMa R1이다(v1과 차이 1 / 2, p = 1).
- MEAN은 import할 때 timm의 convnext 등록을 덮어쓴다. 그래서 Sample4Geo를 먼저 만들고, 순서가 어긋나면 멈추게 했다.
- 한 번의 실행, 한 가지 이동 오차 seed이고, GPS 라벨 어긋남(10~16 m)은 보정하지 않았다.

## 5. 파일

| 파일 | 내용 |
| --- | --- |
| `baselines.py` | 어댑터 (BearingLoc, CorrLoc, OSFPILoc, RetrievalLoc, R1용 Gated; run_geoanchor는 기록용) |
| `calib_gate.py`, `gate.json`, `calib_gate_<model>.csv`, `calib_gate_<model>_sweep.csv` | R1 기준 고르기 (개발용 한 장 결과, 기준별 채점) |
| `visloc_{corr38,corr37,osfpi,sample4geo,game4loc,mean}-r1_{steps,segs}.csv`, `local_r1.log` | R1판 결과 (로컬 GTX 1650) |
| `run_baselines.py` | 92구간 실행, 표 출력 |
| `sanity.py`, `calib_osfpi.py` | 좌표 확인(지도 조각), OS-FPI 입력 선택(개발용) |
| `build_kernel.py` | Kaggle 커널·데이터셋(`heeyunju/visloc-baselines`, `heeyunju/visloc-retrieval`) 만들기 |
| `table.py`, `table.csv` | 위 표 (기존 #45 결과와 합침, McNemar) |
| `visloc_{bearing,corr38,corr37,osfpi,sample4geo,mean,game4loc}-r0_{steps,segs}.csv` | 결과 (검색형 steps에는 `sec_query`, `sim` 열이 더 있음) |
| `visloc_geoanchor-r0_{steps,segs}.csv` | 비교에서 뺀 GeoAnchor 방식 실행 기록 (4장) |
| `kaggle_t4.log`, `kaggle_t4_retrieval.log` | Kaggle 실행 로그 |
