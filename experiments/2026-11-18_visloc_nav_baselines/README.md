# UAV-VisLoc 항법: 기존 모델 비교 (주희윤, 2026-11-18)

교수님 피드백 6("동작하는 다른 방법과 비교")과 8("같은 데이터")에 대한 실험이다. 시험용 `seq_eval` 92구간에서 기존 오픈소스 모델을 우리 v1과 같은 재생 항법 조건으로 돌렸다.

**결론 한 줄**: 모델을 바꾸지 않고 꽂은 기존 모델 세 개(Bearing-UAV, CorrMatchNet, OS-FPI)는 모두 v1보다 유의하게 적게 도착하고, 추측 항법보다도 낮다. 마지막 20 m 이내(Bearing-UAV 논문 기준)는 v1 45 / 92, 기존 모델 4~19 / 92다. GeoAnchor는 조건을 맞추지 못해 비교에서 뺐다(4장).

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

**연결 확인**
- Bearing-UAV: 1,112걸음 모두 #42와 걸음 오차 차이 최대 6e-5 m로 같다.
- 지도에서 정답 위치 조각을 잘라 드론 사진으로 넣으면(`sanity.py`, 40장) 다음과 같다.
  - CorrMatchNet: 오차 중앙값 1.3 m
  - OS-FPI: 11.6 m
  - Bearing-UAV: 13.8 m
  - 블록 가운데만 찍으면: 24.6 m
- 즉 어댑터의 좌표 변환은 맞다.

```bash
python experiments/2026-11-18_visloc_nav_baselines/calib_osfpi.py                      # OS-FPI 입력 (개발용)
python experiments/2026-11-18_visloc_nav_baselines/build_kernel.py --pack outputs/visloc_baselines_pack
kaggle datasets create -p outputs/visloc_baselines_pack --dir-mode zip                   # heeyunju/visloc-baselines
kaggle kernels push -p kaggle_kernels/visloc-nav-baselines --accelerator NvidiaTeslaT4  # 2.45 h (GeoAnchor 포함)
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
| LoMa R0 (#45) | 25 (27.2%) | 41 (44.6%) | 63.1 m | 42.5 m | 0.79 | 20 / 0, p = 2e-06 | 31 / 0, p = 9e-10 |
| R1 (#45) | 44 (47.8%) | 69 (75.0%) | 20.8 m | 20.0 m | 0.74 (캐시 0.54) | 2 / 1, p = 1 | 3 / 0, p = 0.25 |
| **v1 (#45)** | **45 (48.9%)** | **72 (78.3%)** | **20.3 m** | 20.1 m | 3.57 (캐시 2.32) | — | — |

- Bearing-UAV 걸음당 시간은 #42(Colab T4) 0.05 s와 다르다. 이번은 Kaggle T4이고, 블록 타일 4장 인코딩을 포함한다.
- 캐시는 [#49](../2026-11-18_map_cache/README.md)의 지도 특징 캐시 시간이다(결과는 같다).

## 3. 해석

- **v1이 세 기존 모델보다 유의하게 많이 도착한다.** 20 m 기준(p ≤ 1e-4)과 40 m 기준(p ≤ 8e-7) 모두 그렇다.
- **세 기존 모델은 실제 사진에서 추측 항법보다 나쁘다.**
  - 이 모델들은 항상 답을 믿는다. 그래서 틀린 위치로 다음 블록을 고르고, 걸음의 58~79%에서 정답이 고른 블록 밖에 있다.
  - CorrMatchNet은 3D 렌더로만 학습했고, OS-FPI는 저고도 UL14로 학습했다. 둘 다 UAV-VisLoc 실제 사진에 일반화되지 않는다.
  - OS-FPI는 정답 블록을 줘도 한 장 오차 중앙값이 52 m로, 블록 가운데만 찍는 것(26 m)보다 나쁘다(개발용 513장).
- **LoMa R0도 41로 Bearing-UAV와 같다.** 차이를 만드는 것은 매처가 아니라 R1의 "모름" 처리다(#45와 같은 결론).
- **걸음당 시간은 기존 모델이 0.09~0.16 s로 v1(캐시 2.32 s)보다 15~25배 빠르다.** 정확도-시간 트레이드오프 표(피드백 9)에 함께 넣을 수 있다.

## 4. 한계

- **GeoAnchor는 비교에서 뺐다.**
  - [YusufGUNEL/GeoAnchor](https://github.com/YusufGUNEL/GeoAnchor)(MIT)의 LoFTR·파티클 필터·DINOv2·규칙 코드를 가져와 우리 조건으로 돌렸다(`baselines.py`의 `run_geoanchor`, 결과 `visloc_geoanchor-r0_*`는 기록으로만 남김).
  - 원본은 300 m 원본 사진, IMU 나침반, 정사영 지도를 쓴다. 우리는 입력(64 m 크롭), 이동량(모의값), 방향(4방향 탐색), 보정(yaw·축척·흐림 끔)을 바꿨다.
  - 그 결과 원본의 매칭 통과 기준(inlier 40)을 걸음의 3.9%만 넘어, 사실상 추측 항법이 됐다(마지막 40 m 이내 60 vs 56, p = 0.13).
  - 원인이 우리가 바꾼 조건이라 GeoAnchor의 성능으로 볼 수 없다. 통과 기준을 다시 정하려 해도, 연속 비행 개발용 데이터가 없어 정할 수 없었다.
  - 원본 논문 숫자(비행별 오차 중앙값 8~25 m)는 300 m 사진, IMU 나침반, 원본 해상도 지도 조건이라 이 표와 비교할 수 없다.
- 기존 모델 모두 파인튜닝하지 않았다(우리 LoMa도 같다).
- 4방향 탐색은 CorrMatchNet과 OS-FPI에 붙인 것이고, 모델 자체는 바꾸지 않았다. OS-FPI 입력(가운데 192 px)만 개발용에서 골랐다.
- MEAN·Sample4Geo(검색 모델)는 블록 안 위치를 내지 않아 이 틀에 넣지 않았다.
- 한 번의 실행, 한 가지 이동 오차 seed이고, GPS 라벨 어긋남(10~16 m)은 보정하지 않았다.

## 5. 파일

| 파일 | 내용 |
| --- | --- |
| `baselines.py` | 어댑터 (BearingLoc, CorrLoc, OSFPILoc; run_geoanchor는 기록용) |
| `run_baselines.py` | 92구간 실행, 표 출력 |
| `sanity.py`, `calib_osfpi.py` | 좌표 확인(지도 조각), OS-FPI 입력 선택(개발용) |
| `build_kernel.py` | Kaggle 커널·데이터셋(`heeyunju/visloc-baselines`) 만들기 |
| `table.py`, `table.csv` | 위 표 (기존 #45 결과와 합침, McNemar) |
| `visloc_{bearing,corr38,corr37,osfpi}-r0_{steps,segs}.csv` | 결과 |
| `visloc_geoanchor-r0_{steps,segs}.csv` | 비교에서 뺀 GeoAnchor 방식 실행 기록 (4장) |
| `kaggle_t4.log` | Kaggle 실행 로그 |
