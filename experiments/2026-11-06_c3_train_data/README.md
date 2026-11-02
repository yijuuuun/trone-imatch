# C3 감지기 학습 데이터 (김이준, 2026-11-02)

## 무엇을

2주차 C3(실패 감지기)를 학습할 데이터를 **학습 도시**에서 만들었다. 시험 도시(SF, Singapore)로 학습하면 같은 문제로 공부하고 시험 보는 셈이라 쓰지 않는다.

1. 학습 도시 도쿄(34bc), 타이베이(36bc)의 3D 샘플을 500개씩 뽑았다 (capstone1 harness와 같은 방식, `--cities` 옵션 추가).
2. LoMa 16방향으로 돌렸다 (Kaggle T4, 도시마다 약 45분).
3. 라벨 점검(`scripts/check_labels.py`)으로 정답 라벨이 의심되는 샘플을 뺐다.
4. 후보(샘플 × 각도) 1개 = 1행의 특징 표로 바꿨다 (`scripts/c3_features.py`) → `results/c3_train.csv`.

```bash
# 샘플 뽑기 (capstone1 루트에서, 원본 데이터 필요)
python <trone-imatch>/scripts/export_from_capstone1.py --n 500 --views 3d --cities 34bc,36bc --out <trone-imatch>/data/im_train_3d_500
# Kaggle: 데이터셋 yijuniverse/im-train-3d-500 (비공개)
python scripts/kaggle/build_kernel.py --user yijuniverse --dataset yijuniverse/im-train-3d-500 --slug c3-train-tk \
    --args "--matchers loma --angles 16 --sets 3d_ho34bc --tag=-16dir"         # 타이베이는 c3-train-tp, 3d_ho36bc
# 라벨 점검과 특징 표
python scripts/check_labels.py experiments/2026-11-06_c3_train_data/loma_16dir_all.csv --data data/im_train_3d_500 --out results/label_check_train.csv
python scripts/c3_features.py experiments/2026-11-06_c3_train_data/loma_16dir_{tk,tp}_angles.csv \
    --samples data/im_train_3d_500/samples.csv --labels results/label_check_train.csv --out results/c3_train.csv
```
(`loma_16dir_all.csv`는 `loma_16dir_tk.csv`와 `loma_16dir_tp.csv`를 이어 붙인 것)

## 결과

| 도시 | 샘플 (라벨 의심 제외) | 라벨 의심 | inlier 1등 각도 10 m 이내 | 맞는 후보 비율 | 맞는 각도가 하나라도 있는 비율 | 1장당 (16방향) |
| --- | --- | --- | --- | --- | --- | --- |
| 도쿄 | 498 | 2 | 77.2% | 66.2% | 89.2% | 5.2 s |
| 타이베이 | 460 | 40 | 96.4% | 98.0% | 99.3% | 4.9 s |

- **도쿄는 Singapore와 비슷하게 어렵고, 타이베이는 SF와 비슷하게 쉽다.** 두 도시 모두 3D 렌더지만, 도쿄는 드론 영상과 지도의 계절이 다르다(드론은 갈색 겨울, 지도는 초록 여름). 타이베이는 드론 영상과 지도의 색이 거의 같다(추정 위치에서 NCC 중앙값 0.98). 예시 `taipei_tokyo_examples.jpg` (위 2줄 타이베이, 아래 2줄 도쿄).
- 타이베이도 SF처럼 라벨이 의심되는 샘플이 8%(40건) 있다. 감지기가 틀린 라벨을 배우지 않도록 뺐다.
- 후보 15,328개(맞음 81%, 틀림 19%). 틀린 후보는 대부분 도쿄에서 나온다.

### 특징별로 맞는 후보와 틀린 후보 (중앙값)

| 특징 | 맞는 후보 | 틀린 후보 |
| --- | --- | --- |
| n_inl (inlier 수) | 294 | 6 |
| inl_ratio (inlier / 짝) | 0.71 | 0.17 |
| inl_rel (같은 샘플 1등 대비) | 0.84 | 0.60 |
| **agree_pos** (다른 각도 추정과 위치가 10 m 안인 비율) | **1.00** | **0.07** |
| agree_head (heading이 20° 안인 비율) | 1.00 | 0.13 |
| scale_est (추정 축척, 정상은 약 1) | 1.00 | 0.77 |

## 2주차에 할 것

- 이 표로 작은 분류기(로지스틱 회귀 또는 gradient boosting)를 학습한다. 학습 도시 안에서는 도시 단위로 나눠 검증한다(도쿄로 학습 → 타이베이 확인, 반대도).
- 평가는 시험 도시(Singapore, SF)의 기존 16방향 결과(`experiments/2026-11-06_rotation/`)에서 한 번만 한다: 감지기 점수 1등 각도 vs inlier 1등 각도, 짝지은 검정.
- 어려운 쪽(도쿄)이 Singapore와 비슷하므로 도쿄 비중이 낮지 않은지 본다. 필요하면 도쿄만으로 학습한 것과 비교한다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `loma_16dir_{tk,tp}.csv` | 샘플별 결과 (inlier 1등 각도) |
| `loma_16dir_{tk,tp}_angles.csv` | 각도별 기록 (특징 표의 원본) |
| `taipei_tokyo_examples.jpg` | 두 도시의 드론 영상과 지도 예시 |
| `../../results/c3_train.csv` | C3 학습 표 (후보 1개 = 1행, 특징 9개 + 라벨 `ok`) |
| `../../results/label_check_train.csv` | 학습 도시 라벨 점검 결과 |
| `../../data/im_train_3d_500/` | 샘플 이미지 (git에 안 올림, 180 MB) |
