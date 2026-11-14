# Kaggle 비교 결과와 이전 iMatch 평가 기록 — 2026-11-14

> **2026-11-15 후속 계산:** 원본 samples.csv 좌표를 확보해 [MEAN·Sample4Geo 미터 정확도](meter_accuracy_2026-11-15/README.md)를 계산했다. 아래 N/A는 원래 실행 당시 기록이며, 파생 결과는 별도 파일에 보관한다.

이번 실행은 **Bearing-UAV·MEAN·Sample4Geo 3개 모델 완료**, iMatch는 `not_completed`다.
이전 iMatch 표와 파인튜닝 기록은 아래 별도 표로 보존하며, 이번 실행의 원본 comparison.csv에 합쳐 쓰지 않았다.

## 1. 이번 Kaggle 실행

출처: [원본 comparison.csv](run_20261114T034717155134Z/comparison.csv), [환경](run_20261114T034717155134Z/environment.json), [데이터 감사](run_20261114T034717155134Z/data_audit.json).
Tesla T4, PyTorch 2.11.0+cu130, FP32 검사, batch=1, warmup=5, 표본당 반복 3회.
각 완료 모델은 동일 ID 1,738개, 예측 5,214행을 포함한다.
정확도는 첫 반복만, 시간은 세 반복 전체로 계산한다.

| 모델 | 상태 | <10m (%) | <20m (%) | <32m (%) | 평균 ms | p50 ms | p95 ms | FPS | 전체 파라미터 M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bearing-UAV official cross_view | 완료 | 10.4143 | 37.5719 | 74.5109 | 91.971 | 92.587 | 94.962 | 10.873 | 17.1346 |
| iMatch | 이번 RUN 미완료 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MEAN | 완료 | N/A | N/A | N/A | 15.498 | 15.457 | 16.688 | 64.525 | 36.4988 |
| Sample4Geo | 완료 | N/A | N/A | N/A | 38.379 | 38.462 | 40.269 | 26.056 | 87.5665 |

**정확도 순위와 네 모델의 공정한 속도 순위를 이 표로 확정할 수 없다.**

- 모든 모델의 `position_ranking_valid=False`. 이번 `all_labels_at_center=False`이고 중앙 예측 <10m 기준선은 7.8251%다. 모든 정답이 중앙인 데이터라고 설명하면 안 된다. 진단 모드이며 데이터 생성/학습·시험 중복 검증이 완료되지 않았다.
- MEAN/Sample4Geo는 전체 1,738개 갤러리의 특징을 미리 계산한 검색이다. 공통 지리 좌표와 독립 relevance label이 없어 미터 정확도 및 Recall@1은 N/A다. `valid_pct=100`은 위치 정확도 100%를 뜻하지 않는다.
- Bearing-UAV는 주어진 UAV–지도 블록 쌍 처리 시간이다. 검색 모델은 사전 계산한 GPU 갤러리에서 query 전처리·H2D·모델·정규화·Top-1 검색 시간을 측정한다. 각 범위에서 디코딩은 제외한다.
- 갤러리 사전 계산 wall time: MEAN **32.509초**, Sample4Geo **70.668초**. 각 모델의 `gallery_timing.csv`에 별도 저장돼 있으며 query FPS에는 포함되지 않는다.

## 2. 이전 1,738장 iMatch 표 — 별도 실행 참고

출처: [기존 노트북에 저장된 text/plain 출력](previous_imatch_table.txt).
기존 원시 예측/manifest를 이번 ZIP과 대조하지 못했으므로 동일 표본 수만으로 데이터·실행 조건의 완전 일치를 단정하지 않는다.

| 이전 모델 | <10m (%) | <20m (%) | <32m (%) | p50 ms | p95 ms | FPS | 전체 파라미터 M |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bearing-UAV official cross_view | 10.414269 | 37.571922 | 74.510932 | 47.863363 | 50.601915 | 20.863050 | 17.134600 |
| iMatch LoMa-B / 8dir / stop=15 | 8.975834 | 29.459148 | 51.093211 | 4505.185072 | 6019.785883 | 0.287985 | 341.523567 |

이전 표도 `position_ranking_valid=False`, FP32, batch=1, repeats=3이다.
이전 Bearing-UAV p50 47.863ms와 이번 92.587ms가 다르므로 이전 iMatch 시간을 이번 세 모델과 같은 실행 조건의 수치로 간주하지 않는다.
현재 저장소 최상단의 3D 도시별 1,000장 표와도 데이터/평가 규칙이 다르다.

## 3. iMatch LoMa-B 파인튜닝 — 독립 128장 시험 기록

파인튜닝 대상은 **Bearing-UAV 원 모델이 아니라 Bearing 데이터로 미세조정한 LoMa-B**다.
전체 파라미터 341,523,567개와 학습 대상 1,316,609개를 구분한다.
사용자가 제공한 학습 설명과 다음 고정 커밋의 비교 CSV·체크포인트 기록을 참조했다.

[독립 도시 test comparison.csv](https://github.com/imjahk18-a11y/loma-lightweight-research/blob/9bc0b0bb53119c70b876149b51a0f8b764a1dd6f/bearing_uav_finetune/finetune/bearing_loma/artifacts/test_gpu_128_v1/comparison.csv)
· [체크포인트/validation 기록](https://github.com/imjahk18-a11y/loma-lightweight-research/blob/9bc0b0bb53119c70b876149b51a0f8b764a1dd6f/bearing_uav_finetune/CHECKPOINT_REFERENCES.json)

| 128장 독립 test (SF 64 + Singapore 64) | 공개 LoMa-B | LoMa-B FT step100 |
|---|---:|---:|
| <10m 성공률 (%) | 79.6875 | 79.6875 |
| raw 위치 평균 오차 (m) | 9.4245 | 9.0303 |
| raw 위치 중앙 오차 (m) | 3.7566 | 3.4995 |
| Heading 평균 오차 (도) | 9.1504 | 13.3035 |
| p50 (ms) | 201.3229 | 204.7053 |
| p95 (ms) | 1604.5413 | 1620.3757 |
| 전체 파라미터 M | 341.523567 | 341.523567 |

이 결과의 79.6875%를 이전 1,738장 iMatch 8.975834% 대신 넣지 않는다.
파인튜닝 모델의 이번 1,738장 재평가 결과는 없다. 원본 Bearing-UAV 행도 수정하지 않는다.
사용자 제공 설명상 확대 학습의 최종 채택은 공개 LoMa-B이며, 확인한 체크포인트 기록에서도 세 FT 후보는 모두 `eligible=false`, `selected=null`이다.
확대 학습 validation의 step002765 성공률 57.421875%는 test 결과가 아니다.
GPU 재실행 및 서버의 실제 체크포인트 존재·해시 검증은 수행하지 않았다.

## 4. 결과 무결성과 보관 범위

[검증 결과](verification.json)에서 세 모델의 ID, 반복 수, 중복 (sample_id, repeat), 평균/p50/p95/FPS 및 Bearing 정확도를 원시 예측으로 재계산해 summary 및 comparison과 대조했다.

`source_hashes.json`의 `retrieval_worker.py` 해시는 초기 설치 시점 값이고 ZIP의 최종 파일과 다르다.
ZIP에는 MEAN 키 호환 패치 전 백업과 패치 후 소스가 모두 있다. 원래 기록을 덮어쓰지 않고 불일치와 실제 해시를 verification.json에 명시했다.
이 해시 목록만으로 최종 worker의 실행 시점 동일성을 보장할 수는 없다.
원본 comparison의 iMatch reason에 깨진 한글이 있어 원본은 그대로 보존하고 이 문서에서 상태를 설명했다.

[archive_manifest.json](archive_manifest.json)에 원본 ZIP SHA256과 보존/제외 파일별 해시를 기록했다.
CSV·JSON·로그·평가 소스는 원본 바이트를 보존했다. 재생성 가능한 `gallery_features.pt`, Python bytecode, 완료 CSV와 중복인 partial journal은 Git에서 제외했다.
모델 가중치와 원본 이미지는 이 결과 ZIP에 없으며 Git에 추가하지 않았다.

## 5. 재실행

[최신 독립 Kaggle 노트북](../../benchmarks/kaggle_four_models_fresh.ipynb)을 사용한다.
공통 셀 1~4 이후 각 모델의 준비·3장 확인 셀과 전체 평가 셀을 실행한다.
이 노트북은 오류 수정이 반영된 재실행용 버전이며, 이번 실행의 실제 소스 증거는 RUN 내부 `source/`다.

검증 재실행: 저장소 루트에서 `python scripts/verify_kaggle_results.py`. Python 표준 라이브러리만 필요하다.
