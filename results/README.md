# 결과 (2026-10-16 기준)

모든 숫자는 `python scripts/evaluate.py <csv>`로 다시 만들 수 있다. 3D, 처음 보는 도시.

## 파일

| 파일 | 내용 |
| --- | --- |
| `baselines_3d_k0-999.csv` | 기준선의 샘플별 오차: B1 원모델(누수 없이 재학습), B2(VGG 상관), CorrMatchNet |
| `loma_8dir.csv` | LoMa-B, 8방향, 두 도시 각 1,000샘플 |
| `loma_8dir_per_angle_k100-999.csv` | 위 실행의 각도별 기록 (k = 100~999). `subset_angles.py`로 방향 수 줄인 결과를 만든다 |
| `romav2_8dir.csv` | RoMa v2('base' 설정), 8방향, 두 도시 각 1,000샘플 |
| `ensemble_loma_romav2.csv` | 두 매처 중 inlier 비율이 높은 쪽 선택 (`scripts/ensemble.py`). k = 100~999로만 평가 |
| `stage1_smoke_k0-99/` | 1단계 매처 고르기 (k = 0~99, 두 도시 각 100샘플): OpenCV SIFT, SIFT-LightGlue, LoMa, RoMa v2. `_scale2` = 영상 2배 확대 |
| `vgg_backbone_k0-99/` | 원 모델 VGG16 특징 + 새 뒷단 (`--matchers vgg-mnn`), MAGSAC++ 3 px / 8 px |

## 주요 결과

| 방법 | SF 10 m 이내 / 평균 오차 | Singapore 10 m 이내 / 평균 오차 | 1장당 |
| --- | --- | --- | --- |
| B1 원 모델 | 15.8% / 20.63 m | 15.2% / 21.63 m | 0.07 s (GTX 1650) |
| B2 | 88.8% / 7.36 m | 62.6% / 16.96 m | — |
| CorrMatchNet | 88.5% / 6.12 m | 63.0% / 12.99 m | — |
| **LoMa 8방향** | **96.6% / 3.94 m** | **76.1% / 10.10 m** | 2.3 s (T4) |
| RoMa v2 8방향 | 96.5% / 3.94 m | 72.5% / 12.16 m | 19.9 s (T4) |

LoMa 짝지은 검정 (LoMa만 성공 / 상대만 성공):
- vs B1: SF 812 / 4, Singapore 639 / 30 (p < 1e-149)
- vs B2: SF 81 / 3, Singapore 178 / 43 (p < 1e-19)
- vs RoMa v2: SF 3 / 2 (차이 없음), Singapore 95 / 59 (p = 0.005) → 주력은 LoMa

## 1단계: 매처 고르기 (k = 0~99)

| 매처 | SF 10 m 이내 | Singapore 10 m 이내 | 1장당 |
| --- | --- | --- | --- |
| OpenCV SIFT | 89% | 19% | — |
| SIFT-LightGlue | 92% | 41% | 5.5 s |
| RoMa v2 | 95% | 78% | 19.4 s |
| **LoMa** | **95%** | **82%** | **2.4 s** |
| VGG16(원 모델 특징) + 새 뒷단, 8 px | 84% | 44% | 9.3 s (CPU) |

같은 뒷단이어도 특징에 따라 Singapore 19~82%로 갈린다 → 문제는 특징 매칭이다. 원 모델 VGG16 특징은 맞는 짝이 12개 수준이라 B2(69%)보다도 나쁘다.

## 방향 수 (LoMa, k = 100~999, 900샘플)

| 방향 | SF | Singapore | 1장당 |
| --- | --- | --- | --- |
| 8 | 96.8% | 75.4% | 2.3 s |
| 4 | 96.8% | 75.9% | 1.16 s |
| 2 | 96.7% | 74.8% | 0.58 s |
| 1 (탐색 없음) | 96.0% | 72.9% | 0.29 s |

1방향도 heading이 호모그래피에서 나온다(중앙값 0.2° / 1.6°). 원 Bearing-UAV도 방향 정보 없이 1번 계산하므로 같은 조건이다.

## 모델 크기

| 모델 | 파라미터 | 가중치 |
| --- | --- | --- |
| 원 Bearing-UAV | 17.1 M | 69 MB |
| LoMa-B (주력) | 341.6 M (DINOv2 ViT-L 304.4 M 포함) + DaD 6.5 M | 758 MB + 26 MB |
| LoMa-B128 (경량, 미실험) | 31.8 M + DaD 6.5 M | 127 MB + 26 MB |
| RoMa v2 | 425.5 M | 1,096 MB |

## 남은 약점
Singapore(3D 새 도시): 실패 13%, 32 m 초과 8%, inlier 중앙값 57개(SF 294개). LoMa와 RoMa v2 중 하나라도 맞힌 비율은 82%지만, 단순 inlier 비교 조합은 76%로 효과가 없다.
