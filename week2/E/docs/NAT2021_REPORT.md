# 실제 NAT2021 야간 영상에서의 Full LoMa-B 매칭

작업일: 2026-11-10 KST. **실제 영상 40쌍 × 원본·Gamma·CLAHE = 120건의 GPU 추론을 완료했습니다.** T4R 복원과 카메라 파라미터 감사는 진행하지 않았습니다.

## 간단한 결론

현재 Full LoMa-B는 이번 **5프레임 간격·높은 겹침의 실제 야간 영상**에서 원본 기준 평균 1,355개 대응점과 654개 MAGSAC inlier를 생성했습니다. 40쌍 모두 유효 Homography와 inlier≥15 기준을 통과했습니다. Gamma·CLAHE는 밝기를 높였지만 평균 inlier와 비율은 감소해, 이번 조건에서는 기본 보정으로 채택하지 않습니다.

이는 같은 시퀀스 프레임 간의 기하 일관성 결과입니다. 정답 대응점 precision, 위성 매칭 성능, 위치 정확도는 아닙니다. 위성 지도·위치 GT가 없어 **Success@10m는 계산하지 않았습니다.**

## NAT2021 원본·보정별 실측표

| 보정 | 평균 Matching | 평균 Inlier | Inlier 비율¹ | 유효 H | 평균 / 중앙 / P95 지연 |
|---|---:|---:|---:|---:|---|
| none | 1355.0 | 654.2 | 47.2% | 100% | 1367.4 / 1372.8 / 1388.8ms |
| gamma | 1251.5 | 498.8 | 38.7% | 100% | 1404.7 / 1404.3 / 1411.2ms |
| clahe | 1264.0 | 482.9 | 37.2% | 100% | 1428.2 / 1427.2 / 1435.5ms |

¹ 각 쌍의 inlier/matching 비율을 평균했습니다. 전체 inlier 합/전체 matching 합인 pooled 비율은 별도 CSV에 있으며, 원본은 48.3%입니다. 두 평균 방식은 다릅니다.

모든 방법의 기하 통과(inlier≥15 및 유효 H)는 40/40입니다. 유효 H는 finite·rank3·inlier≥4인 수치 기준입니다. 이 기준의 통과는 정답 대응점이나 항법 성공을 보장하지 않습니다. Peak allocated VRAM은 세 방법 모두 약 4,476MiB였습니다.

원본 대비 Gamma는 평균 inlier 155.5개 감소(39쌍 감소·1쌍 증가), CLAHE는 171.4개 감소(40쌍 모두 감소)했습니다. 시퀀스 단위 paired bootstrap의 평균 inlier 차이 95% 구간은 Gamma [-194.6,-117.3], CLAHE [-199.8,-143.9]개입니다. 이 구간은 관측 코호트의 기하 일관성 차이이며 정답 매칭 정확도 차이가 아닙니다.

## 밝기가 낮으면 얼마나 나빠졌는가?

| 원본 밝기 층 | 쌍 | 평균 Y′ /255 | 평균 Inlier | Inlier 비율¹ | 기하 통과 |
|---|---:|---:|---:|---:|---:|
| dark | 14 | 10.75 | 632.0 | 46.8% | 100% |
| middle | 13 | 22.81 | 749.5 | 53.2% | 100% |
| bright | 13 | 46.19 | 583.0 | 41.8% | 100% |

원본 쌍 평균 밝기의 3분위 경계는 17.418·31.133/255입니다. 보정 비교에도 원본 기준의 동일 그룹을 적용했습니다. 어두운 층의 평균 inlier는 밝은 층보다 오히려 높았고 중간 층이 가장 높았습니다. 시퀀스 평균 밝기–inlier의 Spearman 상관은 -0.131였습니다. **이 표본에서는 밝기가 낮을수록 일관되게 성능이 저하되는 추세를 확인하지 못했습니다.**

장면의 텍스처·움직임·반복 무늬·blur와 밝기가 함께 달라집니다. 따라서 이 결과로 조도 단독 효과를 추정하거나 실제 야간 전반에서 밝기가 중요하지 않다고 결론 내리지 않습니다. 주간 대비 저하율도 이 실험에서는 측정하지 않았습니다.

## 실제 영상 예시 5개와 사례

`visual_examples_manifest.csv`에 서로 다른 밝기의 5시퀀스를 기록했습니다. 각 시퀀스의 원본 쌍과 원본/Gamma/CLAHE 매칭 그림이 `visual_examples/`에 있습니다.

| 예시 쌍 | 원본 평균 밝기 /255 | Matching / Inlier |
|---|---:|---:|
| N09006_first | 6.15 | 1195 /471 |
| N04004_first | 18.36 | 1513 /988 |
| N03001_first | 24.04 | 1518 /998 |
| N04023_first | 39.09 | 1396 /703 |
| N04032_first | 44.50 | 1307 /642 |

- 강한 기하 일관성 사례: **N02037_first**, matching 1,532개/inlier 1,064개(69.5%).
- 최약 원본 사례: **N02018_middle**, matching 785개/inlier 82개(10.4%). Gamma 72개, CLAHE 57개로 더 줄었습니다. 반복되는 도로 구조와 blur가 보이지만 원인을 분리 검증한 것은 아닙니다.
- 정한 기하 기준의 실패는 **120건 중 0건**입니다. 실패를 만들어내지 않고 최약 사례를 별도로 제시했습니다. GT가 없으므로 거짓 Homography/오매칭의 실제 비율까지 검증한 것은 아닙니다.

녹색선은 MAGSAC inlier, 빨간선은 outlier의 시각 표본입니다. 최대60/20개를 고정 순서로 표시했고 전체 대응점은 각 NPZ에 있습니다. 표시용 resize는 실제 모델 입력과 별개입니다. 원본 영상 예시는 로컬 연구용이며 외부 업로드하지 않았습니다.

## 고정한 평가 조건과 겹침 한계

- 기존 20시퀀스에서 첫·중간 anchor 각 1쌍, partner는 모두 +5프레임. 공식 30fps 기준 약 0.167초입니다. 밝기나 모델 출력으로 표본을 제외하지 않았습니다.
- 전체 1280×720 화면, 단일 방향, FP32·TF32 off, cache 없음. 입력을 1274×714로 맞추고 원래 좌표로 복원했습니다. 같은 시퀀스이므로 8방향 지도 탐색을 하지 않았습니다.
- 공식 Full LoMa-B pretrained strict load, keypoints2048, threshold0.1, MAGSAC3px, maxIters10000, confidence0.999. RGB·좌표·score finite 및 실제 FP32 Conv/Linear 경로를 확인했습니다.
- Gamma x^0.6, LAB-L CLAHE clip2/grid8을 **양쪽 프레임 모두**에 적용했습니다. 합성 저조도나 추가 모델은 없습니다.
- 시간은 decoded RGB부터 보정·resize·양쪽 특징 추출·matching/filter·MAGSAC까지입니다. 파일 decode·모델 로딩·저장·겹침 진단은 제외했습니다. 조건별 warm-up10회와 CUDA 동기화를 적용했습니다. 고정 순서 1회 측정이어서 순서/열 상태 영향은 남습니다.

시간 간격과 화면 범위는 고정했지만 실제 물리적 겹침을 GT 없이 완전히 동일하게 강제할 수는 없습니다. 원본의 H 기반 겹침 proxy는 평균 **0.985**, 최소 **0.852**였고 40쌍 모두 0.8 이상입니다. 이것은 모델 추정에서 나온 사후 값이지 독립 overlap GT가 아닙니다. 이 값으로 실패나 어려운 표본을 거르지 않았습니다. 높은 겹침과 짧은 시간 간격이므로 긴 간격·큰 시점 변화로 일반화할 수 없습니다.

## 기존 Bearing-UAV 합성 결과 — 별도 태스크

아래는 기존 20쌍 실측의 사본입니다. 위 NAT 지표와 같은 정확도로 해석하지 않습니다.

| 기존 Bearing-UAV 조건 | 보정 | 평균 Matching | 평균 Inlier | 독립 위치 Success@10m |
|---|---|---:|---:|---:|
| daylight | none | 202.2 | 59.4 | 65% |
| mild | none | 201.2 | 62.6 | 70% |
| severe | none | 174.3 | 61.4 | 60% |
| mild | gamma | 192.1 | 62.4 | 70% |
| mild | clahe | 191.7 | 58.4 | 70% |
| severe | gamma | 169.5 | 56.2 | 55% |
| severe | clahe | 173.6 | 54.9 | 60% |
| severe | gamma_clahe | 170.0 | 53.8 | 60% |

**Bearing-UAV:** UAV–Satellite Oracle T1, FP16, 8방향, UAV256/지도512px, UAV에만 보정. 위치 GT가 있어 독립 Success@10m를 계산했습니다.

**이번 NAT2021:** 실제 야간의 같은 시퀀스 프레임 매칭, FP32, 1방향, native720p, 양쪽 보정. 위치 GT가 없어 Success@10m가 없습니다.

태스크·모달리티·입력 크기·정밀도·방향 수가 달라 Matching 수나 latency도 직접적인 우열 비교에 쓰지 않습니다. 두 실험 모두 이번 고정 Gamma/CLAHE가 도움이 되지 않은 관측은 있으나, 실제 야간 위성 위치 정확도나 모든 보정 방법의 일반 성능을 뜻하지는 않습니다.

## 파일과 재현

- `samples.csv`: 120행 샘플별 실측. `matching_summary.csv`: 세 방법 요약.
- `brightness_groups.csv`, `paired_samples.csv`, `paired_summary.csv`, `sequence_summary.csv`: 밝기층과 짝지은 분석.
- `PAIR_MANIFEST.json`: 고정40쌍의 원본 경로·CRC/SHA. 기존40프레임 재사용, +5프레임40개만 공개 원본에서 추가 확보했습니다.
- `none/`, `gamma/`, `clahe/`: 원시 JSON·대응점 NPZ. `cases.csv`, `geometry_failure_cases.csv`: 사례 기록.
- `visual_examples/`: 밝기별 예시5개, 최강·최약 사례. `environment.json`, `protocol.json`, `model.json`, `dtype_audit.json`: 환경·측정 경계·모델 증거.

```bash
cd ${WEEK2_WORKSPACE}
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -u standalone/run_nat_matching.py
.venv-gpu/bin/python standalone/report_nat_matching.py
```

완료된 GPU 샘플은 건너뛰며 source/protocol 변경 시 재개를 거부합니다. 보고서 생성기는 기존 산출물 덮어쓰기를 거부하므로 재분석에는 새 경로가 필요합니다. 초기 생성 보고서는 `REPORT.generated.md`에도 보존했습니다.

기존260건·원본 파일·가중치·코드는 보존했습니다. T4R 복원, 새 카메라 모델, 재학습, Git Commit/Push/PR, 팀의 잠긴 위치 시험은 수행하지 않았습니다. 원본 영상의 이용 조건은 이전 감사 기록을 유지하며 코드 라이선스를 원본 영상의 확정 라이선스로 전용하지 않았습니다.
