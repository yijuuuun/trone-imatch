# LoMa-B128 Lightweight Evaluation & External Dataset Validation

## 1. 프로젝트 개요

본 실험은 Bearing-UAV 환경에서 LoMa-B128 경량 매칭 모델의 정확도와 추론 속도를 평가하고, 외부 UAV 데이터셋에서도 평가 파이프라인을 실행할 수 있는지 확인하는 것을 목표로 한다.

실험은 다음 네 가지로 구성된다.

| 실험 | 내용 |
|---|---|
| E1 | LoMa-B128 1방향 매칭 평가 |
| E2 | LoMa-B128 8방향 매칭 평가 |
| E5 | 원본 Bearing-UAV 모델의 T4 추론 속도 측정 |
| C5 | UAV-VisLoc 외부 데이터셋 변환 및 실행 검증 |

**실험 환경**
- GPU: NVIDIA Tesla T4
- LoMa 모델: LoMa-B128
- 입력 데이터: Bearing-UAV 및 UAV-VisLoc
- 평가 대상: 위치 오차, 방향 오차, 매칭 성공률, 추론 시간

---

## 2. 실험 구성

### E1. LoMa-B128 1방향 평가

- 매칭 방향: 1방향
- Scale: 1
- Angle: 1
- 데이터: Bearing-UAV 3D SF, 3D Singapore
- 샘플 수: 총 2,000개

| Dataset | N | Found (%) | MLE (m) | MedLE (m) | LSR10 (%) | Sec/sample |
|---|---:|---:|---:|---:|---:|---:|
| 3D Singapore | 1,000 | 98.6 | 19.86 | 6.30 | 64.9 | 0.222 |
| 3D SF | 1,000 | 99.4 | 4.22 | 3.02 | 95.7 | 0.217 |

### E2. LoMa-B128 8방향 평가

- 매칭 방향: 8방향
- Scale: 1
- Angle: 8
- 데이터: Bearing-UAV 3D SF, 3D Singapore
- 샘플 수: 총 2,000개

| Dataset | N | Found (%) | MLE (m) | MedLE (m) | LSR10 (%) | Sec/sample |
|---|---:|---:|---:|---:|---:|---:|
| 3D Singapore | 1,000 | 100.0 | 16.24 | 5.85 | 69.8 | 1.725 |
| 3D SF | 1,000 | 99.9 | 4.78 | 3.05 | 96.1 | 1.729 |

### E1–E2 비교

| Dataset | LSR10 변화 | MLE 변화 | 실행시간 변화 |
|---|---:|---:|---:|
| 3D Singapore | 64.9% → 69.8% (+4.9%p) | 19.86 → 16.24 m | 0.222 → 1.725 s |
| 3D SF | 95.7% → 96.1% (+0.4%p) | 4.22 → 4.78 m | 0.217 → 1.729 s |

8방향 탐색은 Singapore에서 위치 정확도를 개선했으며, SF에서는 LSR10이 소폭 개선되었다. 다만 1방향 대비 실행시간은 약 7.8~8.0배 증가했다.

---

### E5. 원본 Bearing-UAV 추론 속도 측정

NVIDIA Tesla T4 환경에서 원본 Bearing-UAV 모델의 추론 비용을 측정했다.

| 항목 | 측정 결과 |
|---|---:|
| Full model forward | 79.9 ms/sample(0.0799 s/sample) |
| Precomputed tile features | 16.5 ms/sample(0.0165 s/sample) |
| 모델 파라미터 | 17.13M |
| Backbone 파라미터 | 약 9.5M |
| 가중치 크기 | 약 69 MB |

**측정 조건**
- Batch size: 1
- Precision: FP32
- Warm-up: 3회
- 측정 반복: 20회 평균
- 이미지 읽기 및 정규화 시간 제외

Full model forward와 사전 계산된 tile feature를 사용하는 경우는 측정 범위가 다르므로 직접적인 동일 조건 비교로 해석하지 않는다.

---

## 3. C5. 외부 데이터셋 검증

### 데이터셋

- Dataset: UAV-VisLoc
- Sequence: 03
- 원본 데이터: 768개 샘플
- 평가 샘플: 기존 20개 + 추가 20개
- 총 평가 샘플: 40개
- 기존 20개와 추가 20개는 서로 중복되지 않도록 구성했다.

*UAV-VisLoc 데이터셋이 16.4G로 커서 2.4G인 Example 데이터셋으로 진행함*

### 데이터 변환

UAV-VisLoc 원본 데이터를 LoMa 평가 파이프라인의 `data/` 형식으로 변환했다.

| 항목 | 설정 |
|---|---|
| UAV 이미지 | 중앙 512×512 crop |
| UAV 입력 크기 | 256×256 resize |
| Satellite 이미지 | GPS 중심 512×512 crop |
| Heading 부호 | +1 |
| Heading offset | 0° |
| 좌표 및 방향 기준 | 잠정 설정 |

### C5 평가 결과

| 구분 | N | Found (%) | MLE (m) | MedLE (m) | LSR10 (%) | MHE (°) |
|---|---:|---:|---:|---:|---:|---:|
| 기존 20개 | 20 | 100.0 | 19.84 | 15.14 | 25.0 | 104.11 |
| 추가 20개 | 20 | 85.0 | 64.11 | 15.17 | 40.0 | 101.44 |

**주의사항**

C5의 GPS 좌표와 이미지 중심의 대응 관계, 카메라 내부 파라미터, FOV, 지상 촬영 범위 및 heading convention은 완전히 검증되지 않았다.

따라서 위 위치·방향 오차는 정식 벤치마크 정확도로 해석하지 않는다. `Found`는 위치 추정 결과가 생성되었음을 의미하며, 실제 정답 위치에 성공적으로 도달했다는 의미가 아니다.

C5의 핵심 성과는 UAV-VisLoc 외부 데이터 40개를 변환하고 LoMa-B128 평가 파이프라인에서 실행한 것이다.

---

## 4. 실행 방법

### 4.1 외부 데이터 변환

원본 UAV-VisLoc 데이터셋 경로와 출력 경로를 지정한다.

```bash
python scripts/export_external.py \
  --root "<UAV-VisLoc 원본 데이터 경로>" \
  --out "<변환 데이터 출력 경로>" \
  --n 20 \
  --uav-crop-px 512 \
  --block-px 512 \
  --theta-sign 1 \
  --theta-offset 0 \
  --provisional
```

추가 20개를 생성할 때는 기존 20개와 겹치지 않도록 샘플 인덱스를 지정해야 한다. 현재 추가 샘플 실행은 별도의 샘플 선택 설정을 적용한 실행본으로 진행했다.

### 4.2 LoMa-B128 평가

`run.py`는 `samples.csv`가 포함된 데이터 디렉터리를 입력으로 사용한다.

```bash
python run.py \
  --data "<변환 데이터 경로>" \
  --out "<결과 출력 경로>" \
  --matchers loma \
  --scales 1 \
  --angles 1 \
  --n 0 \
  --device cuda
```

> **실행 환경 참고:** 실험 당시 Kaggle 실행 환경에서는 LoMa-B128 전용 matcher adapter와 가중치 로딩 설정을 적용했다. 따라서 위 명령은 해당 adapter 및 가중치 설정이 준비된 환경에서 실행해야 한다. 일반적인 `run.py`만으로 별도 설정 없이 재현된다고 가정하지 않는다.

---

## 5. 결과 파일

실험 결과는 다음 디렉터리에 정리한다.

```text
experiments/
└── 2026-11-06_light/
    ├── E1_LoMaB128_1dir_results/
    │   ├── loma-b128_E1_lomaB128_1dir_s1.csv
    │   ├── loma-b128_E1_lomaB128_1dir_s1_angles.csv
    │   └── summary.csv
    │
    ├── E2_LoMaB128_8dir_results/
    │   ├── loma-b128_E2_lomaB128_8dir_s1.csv
    │   ├── loma-b128_E2_lomaB128_8dir_s1_angles.csv
    │   └── summary.csv
    │
    ├── E5_BearingUAV_timing_result/
    │    ├── E5_T6_run1_3d_ho37bc_quick.json
    │    ├── E5_T6_run2_3d_ho37bc_quick.json
    │    ├── E5_T6_run3_3d_ho37bc_quick.json
    │    └── summary_quick.xlsx
    │
    └── C5_ready/
        ├── C5_UAVVisLoc_LoMaB128_results/
        │   ├── loma-b128_uavvisloc20_s1.csv
        │   ├── loma-b128_uavvisloc20_s1_angles.csv
        │   └── summary.csv
        │
        └── C5_UAVVisLoc_extra20_LoMaB128/
            ├── loma-b128_extra20_s1.csv
            ├── loma-b128_extra20_s1_angles.csv
            └── summary.csv
 ```
---
## 6. 종합 결론

LoMa-B128의 8방향 탐색은 일부 데이터셋에서 위치 정확도를 개선했지만 약 8배의 실행시간 증가가 발생했다. 또한 원본 Bearing-UAV의 T4 추론 비용을 측정하고 UAV-VisLoc 외부 데이터 40개를 변환·실행하여 외부 데이터 평가 파이프라인의 실행 가능성을 확인했다.

단, C5의 좌표 및 heading 기준은 잠정 설정이므로 정량적인 외부 데이터 정확도 검증은 향후 기하 보정 이후 진행해야 한다.

---

## 7. 재현 및 해석 시 유의사항

- 본 실험은 기존 학습 가중치를 사용하며 재학습을 수행하지 않았다.
- E1과 E2는 동일한 2,000개 샘플을 사용했다.
- E5의 시간 측정 범위는 E1·E2의 샘플당 실행시간과 다르다.
- C5의 `Found`와 위치 오차를 정식 벤치마크 성공률로 해석하지 않는다.
- C5의 좌표 및 방향 기준 검증 후 정확도 비교를 진행한다.
