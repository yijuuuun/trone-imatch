# Trone: 이미지 매칭 기반 재설계 (Bearing-UAV)

Trone 팀 졸업프로젝트 · 작성 이준 · 2026-10-16

> **방향 전환 (2026-10-15 교수님 피드백 이후):** 이제 이 저장소의 주 방향은 **이미지 매칭 기반 재설계**다. 문제는 결국 "특징 매칭이 안 되는 것"이므로, CSMG 교체(Grid Pooling)가 아니라 검증된 이미지 매칭 파이프라인으로 위치·heading을 구한다. 계획 전체는 [docs/01_재설계계획.md](docs/01_재설계계획.md), 계기는 [docs/00_교수님_피드백.md](docs/00_교수님_피드백.md).
>
> 이전 방향인 Grid Pooling(CSMG 교체) 코드·모델·결과·문서는 main에서 뺐다. 태그 [`grid-pooling-final`](https://github.com/yijuuuun/trone-grid-pooling/tree/grid-pooling-final/grid_pooling)에 그대로 있다 (`git checkout grid-pooling-final`).

드론 영상 1장과 지도 블록(2×2 타일)을 받아 **위치와 heading**을 구한다. 학습 없이, 공개 이미지 매처(LoMa)로 대응점을 찾고 기하 계산으로 답한다.

## 한눈에

| 3D, 처음 보는 도시, 1,000샘플 | SF 10 m 이내 | Singapore 10 m 이내 | 사진 1장당 |
| --- | --- | --- | --- |
| 원 Bearing-UAV (B1, 누수 없이 재학습) | 15.8% | 15.2% | 0.07 s (GTX 1650) |
| B2 (VGG 상관) | 88.8% | 62.6% | — |
| CorrMatchNet (이전 최종안) | 88.5% | 63.0% | — |
| **이 파이프라인: LoMa, 8방향** | **96.6%** | **76.1%** | 2.3 s (T4) |
| 이 파이프라인: LoMa, 1방향 (900샘플) | 96.0% | 72.9% | 0.29 s (T4) |

모두 같은 샘플끼리 짝지은 검정에서 기준선보다 유의하게 좋다(p < 1e-12). 자세한 표는 [results/README.md](results/README.md).

## 파이프라인

```
드론 영상 ─ ① 회전 탐색 ─ ②③ 특징 추출·매칭 ─ ④ 검증 ─ ⑤ 위치·heading
                (0~8방향)      (LoMa 등)        (MAGSAC++)   (호모그래피)
지도 블록 ──────────────────┘                    inlier가 가장 많은 방향의 결과를 쓴다
```

| 단계 | 코드 | 하는 일 |
| --- | --- | --- |
| ① 회전 탐색 | `imatch/pipeline.py` | 드론 영상을 여러 각도로 돌려 각각 매칭 (`--angles 1`이면 탐색 없음) |
| ②③ 매칭 | `imatch/matchers.py` | LoMa, RoMa v2, SIFT-LightGlue ([vismatch](https://pypi.org/project/vismatch/)), OpenCV SIFT |
| ④ 검증 | `imatch/geometry.py` `magsac` | MAGSAC++로 틀린 짝 제거 |
| ⑤ 위치·heading | `imatch/geometry.py` `pose_from_homography` | 드론 중심을 지도로 옮겨 위치, 중심의 회전 성분으로 heading |
| 평가 규칙 | `imatch/metrics.py` | 블록 안으로 clip, 실패 정의, McNemar 검정 |

## 폴더 구조

```
trone-grid-pooling/
├── imatch/                 파이프라인 코드 (위 표)
├── scripts/
│   ├── run.py              매칭 실행 → 샘플별 CSV
│   ├── evaluate.py         기준선·LoMa와 짝지어 비교 → 표
│   ├── subset_angles.py    다시 안 돌리고 방향 수만 줄인 결과 만들기
│   ├── kaggle/build_kernel.py   Kaggle GPU용 커널 만들기
│   └── export_from_capstone1.py 원 데이터셋에서 샘플 뽑기 (capstone1 저장소 필요, 보통은 안 씀)
├── data/                   샘플 데이터 (git에 안 올림, data/README.md)
├── results/                지금까지의 결과 CSV와 요약
├── experiments/README.md   팀원이 돌릴 실험 목록
└── docs/                   교수님 피드백, 계획서, 실험 기록, 기여 추천안 (archive/: 이전 정리 문서)
```

## 시작하기

```bash
pip install -r requirements.txt
# 데이터: data/README.md 대로 data/im_t1_3d_1000/ 에 풀기 (zip은 이준에게 받기)

# 1) 설치 확인 (CPU, 몇 초): OpenCV SIFT로 5샘플
python scripts/run.py --data data/im_t1_3d_1000 --out outputs/check --matchers cv-sift --n 5 --device cpu
python scripts/evaluate.py outputs/check/cv-sift_s1.csv

# 2) 주력 LoMa (GPU 필요. 로컬 GPU 또는 Colab / Kaggle)
python scripts/run.py --data data/im_t1_3d_1000 --out outputs/loma_1dir --matchers loma --angles 1
python scripts/evaluate.py outputs/loma_1dir/loma_s1.csv

# 3) Kaggle에서 돌리기
python scripts/kaggle/build_kernel.py --user <아이디> --dataset <아이디>/im-t1-3d-1000 \
    --slug loma-1dir --args "--matchers loma --angles 1"
kaggle kernels push -p kaggle_kernels/loma-1dir --accelerator NvidiaTeslaT4
```

GPU에서 LoMa는 방향 1개당 약 0.29 s, 1,000샘플 × 2도시 × 8방향이 약 1.3시간(T4)이다.

## 데이터와 평가 조건

- Bearing-UAV-90K의 T1 테스트, **3D 렌더 영상**, 학습에 없는 두 도시: SF(`3d_ho38bc`), Singapore(`3d_ho37bc`). 도시마다 1,000샘플(k = 0~999).
- k = 0~99는 1단계(매처 고르기)에 썼다. 그 결과를 보고 정한 규칙(예: 조합 방식)은 **k = 100~999로만** 평가한다(`--kmin 100`).
- 성공 = 10 m 이내. 1단위 = 32 m. 규칙은 `imatch/metrics.py`.
