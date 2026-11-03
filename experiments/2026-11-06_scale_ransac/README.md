# 김재홍 · E3·E4 방법 B 실험 (2026-11-06 회의)

**방법 B: 동일하게 재구성한 Singapore 1,000개 입력에서 Baseline을 새로 실행하고, 그 Baseline만을 짝지은 비교 기준으로 사용한다.** 과거 72.9%는 별도 실험의 참고값이다. 기존 prepared 이미지와 픽셀 단위 동일성을 주장하지 않는다.

**완료:** RTX A6000 GPU 0에서 smoke25건·탐색500건·최종4,500건을 실제 실행하고 평가했다. 탐색에서 동결한 Scale1.5는 최종 채택 기준을 충족하지 못해 **Scale1·MAGSAC3px를 유지**한다. 결과와 한계는 아래 실측 표를 따른다.

## 데이터 재구성과 검증

원본은 `data/Bearing_UAV_90K.zip`(6,344,501,973 bytes, 273,653 ZIP entries)이다. CSV 메타데이터의 90,000행 중 Singapore cityc/37bc는 22,500행이다. 준비된 `im_t1_3d_1000.zip`, `samples.csv`, `3d_ho37bc/`는 ZIP 안에 없었다.

`results/raw_capstone1/out_imatch/samples_1000.csv`에 보존된 Singapore k=0~999와 global row를 그대로 사용했다. 그 row로 ZIP의 `c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv`를 조회했으며 1,000개 x_norm/y_norm/theta 모두 과거 LoMa GT와 차이 0이었다. B1/B2/CorrMatchNet의 원본 row도 모두 일치했고 stage1의 100개 prefix도 검증했다. 새 샘플 추출이나 GT 생성·정렬 변경은 없었다.

UAV 1,000장과 공유 위성 타일 892장, 총 **1,892개 원본 이미지(58,307,058 bytes)**만 ZIP에서 읽었다. 90K 전체를 압축 해제하지 않았다. 재구성 폴더는 `data/im_t1_3d_1000_reconstructed/`이며 1,000 UAV/위성 블록 쌍과 다음 기록을 포함한다.

- `samples.csv`: 원래 k,row,GT와 크기를 보존한 평가 manifest.
- `source_mapping.csv`: 각 k에 대응하는 원본 row, UAV와 p1/p2/p3/p4의 ZIP 경로.
- `export_provenance.json`: 원본 ZIP/CSV·원본 및 생성 이미지 SHA256, 전처리·한계.

원본 UAV와 타일은 256×256 RGB다. 기존 export 코드의 `[[p1,p2],[p3,p4]]` 순서로 512×512 블록을 만들고, OpenCV RGB decode → JPEG 품질 97 저장을 적용했다. **재구성 단계의 resize/crop/interpolation/augmentation은 없다.** 입력 1,892개 모두 현재 OpenCV/Pillow의 RGB decode 결과가 같았다. 출력 2,000장의 디코딩·크기·이름·GT를 전수 확인했다. 이는 과거 prepared 이미지와의 픽셀 동일성 검증을 의미하지 않는다.

`diag/harness.py`와 원래 prepared 이미지/해시는 접근 가능한 경로와 Git 기록에서 찾지 못했다. 실험을 수행한 원본 작업 사본에는 원격이 없고 로컬 source snapshot만 있었다. 방법 B는 이 한계를 유지한 채 **새 프로토콜 내부의 데이터 동일성**을 `method_b_dataset.json`으로 고정한다. 모든 실행 전에 manifest·2,000개 이미지 해시를 확인하고 달라지면 중단한다. 초기 export 기록의 `preprocessing_verified=false`와 `eligible_for_final_evaluation=false`는 당시의 과거 픽셀 재현 진단에 해당한다. 이를 덮어쓰지 않고 방법 B의 새 Baseline 평가 근거를 별도 seal로 고정했다.

좌표계는 원래 그대로다: 블록 중심=(0,0), 1단위=128px=32m, UAV 중심=(128,128), 위성 블록 중심=(256,256). T1의 정답 블록을 제공하며 블록 선택 알고리즘을 추가하지 않는다.

검증 근거: [ZIP 검사](validation/bearing90k_zip_audit.json), [export 검사](validation/bearing90k_export_audit.json), [이미지/GT 검사](validation/reconstructed_assets.json), [고정한 방법 B 데이터](method_b_dataset.json).

## 모델·환경·실험 조건

실행 Python은 `/home/gpu_03/trone-imatch-e3-e4/.venv-e3-e4/bin/python`이다. torch 2.8.0+cu126, torchvision 0.23.0+cu126, OpenCV 4.12.0, vismatch 1.4.0, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3을 사용한다. 실제 CUDA runtime은 12.6이고 시스템 Toolkit 12.2.128은 변경하지 않았다. GPU는 0번 RTX A6000(물리 VRAM 49,140MiB)이다. batch=1, 조건별 순차 실행, CPU torch threads=4, OpenCV threads=1이다.

공식 pretrained `loma_B.pt`, `dad.pth`, `dinov2_vitl14_pretrain.pth`는 `.cache/e3-e4/torch/hub/checkpoints/`에 받았고, 실제 CUDA 모델 로드·추론에 사용했다. [URL 및 소스](official_weights.json), [다운로드와 SHA256](validation/official_weights_download.json). 모델 구조·가중치·키포인트 수를 바꾸거나 학습하지 않았다. 다른 매처의 대규모 의존성을 줄이기 위해 vismatch는 LoMa 전용 최소 의존성 구성으로 설치했으며 전체 vismatch 매처 지원을 주장하지 않는다.

| 조건 | 입력 배율 | UAV / 위성 입력(px) | 원본 지도 기준 MAGSAC 임계값 | 변경 변수 |
| --- | --- | --- | --- | --- |
| baseline | 1 | 256 / 512 | 3 px | 새 1방향 Baseline |
| e3_scale1p5 | 1.5 | 384 / 768 | 3 px | Scale |
| e3_scale2 | 2 | 512 / 1024 | 3 px | Scale |
| e4_magsac3 | 1 | 256 / 512 | 3 px | 동일 설정 재현 검사 |
| e4_magsac8 | 1 | 256 / 512 | 8 px | threshold |

LoMa 1방향(0°), 기존 위치 추정·평가 규칙을 유지한다. E4 3px는 원래 기본값과 같으므로 별도 성능 개선 설정으로 해석하지 않는다.

E3에서는 기존 `localize`의 cubic resize를 유지한다. 대응점을 원본 좌표로 나눈 후 동일한 MAGSAC 임계값으로 Hbase를 추정하고, `Hscaled=S Hbase S⁻¹`을 전달한다. 기존 pose 함수의 `S⁻¹ Hscaled S`로 동일한 블록 좌표계를 복원한다. 따라서 scale1.5/2에서 원본 3px는 resize 좌표의 4.5/6px와 동등하다. vismatch 내부 14배수 resize의 좌표 복원과 기존 단순 배율 좌표 관례를 유지한다.

실제 호출은 `cv2.findHomography(..., cv2.USAC_MAGSAC, RANSAC_PX, maxIters=2000, confidence=0.995)`이다. 3/8px가 실제 위치 인자로 전달되는 테스트와 합성 homography/scale 복원 검사를 통과했다. wrapper 내부 MAGSAC H는 사용하지 않고 raw matched_kpts로 기존 파이프라인의 MAGSAC을 수행한다.

## 구간 분리·사전 채택 기준

- smoke: k=0~4, 조건마다 동일한 5개, 25개 측정 추론+5개 warm-up.
- 탐색: k=0~99, 조건마다 100개.
- 최종: k=100~999, 조건마다 900개. 탐색 뒤 freeze 파일 없이는 실행 불가.
- 전체 1,000개 표는 탐색과 최종을 합친 기술통계이며, 최종 평가와 구분한다.

`conditions.json`의 사전 규칙: E4 3px 중복을 제외하고 탐색 성공 수가 가장 높은 조건을 후보로 선택한다. 동률이면 평균 오차, 평균 시간 순으로 정한다. Baseline보다 성공 수가 늘지 않으면 Baseline을 유지한다. 최종에서는 동결한 후보가 성공률을 높이고 McNemar의 Holm4 보정 p<0.05이면 채택 근거로 삼으며, 아니면 Baseline 유지/채택 보류로 결론 낸다. 시간·메모리도 회의에 함께 제시한다. 네 후보 비교는 사전 선언됐으며 최종 결과로 후보를 바꾸거나 재탐색하지 않는다.

## 평가·실패 태그·통계

기존 `imatch.metrics.errors`와 `scripts/evaluate.py`를 재사용한다. 위치는 [-1,1] clip, 추정 없음은 블록 중심으로 평가하며 성공은 기존 코드와 같은 **error<10m**다. 정확히 10m는 실패다. `pipeline_failure`(추정 없음 또는 |pred|>1.1)와 오차≥10m 실패 수를 구분한다.

새 Baseline의 error≥10m인 샘플만 태그를 재계산한다. 우선순위는 추정 없음/블록 밖 > heading오차>45° > inlier<30 > 기타다. 회전 태그는 heading오차 기반 진단값이며 인과 원인 확정이 아니다. `baseline_failure_tags.csv`와 tag counts를 구간별 저장한다. 과거 failure_tags.csv의 146건은 `historical:` 참고 부분집합일 뿐 새 실패 집합으로 사용하지 않는다.

`analyze.py`는 새 Baseline과 동일 k를 짝지어 성공률, 오차·inlier 평균/중앙값, 복구/악화, 시간, peak allocated/reserved를 출력한다. 성공은 McNemar exact, 전체 네 후보에는 Holm 보정, 연속량은 sample-paired bootstrap 10,000회 평균 차이 95% CI와 Wilcoxon을 쓴다. 태그/연속량 p는 탐색적이다. 모든 샘플의 차이와 성공 전환은 CSV에 보관한다. 원래 evaluate 스크립트의 과거 연구 방법 비교는 역사적 참고 로그로만 해석한다.

시간은 warm-up을 제외하고 CUDA synchronize를 앞뒤로 적용한 resize+매칭+기하 계산이다. 이미지 I/O·모델 로드·seed 설정·결과 저장은 제외한다. peak allocated/reserved는 이미지별 reset 후 기록하며 reserved는 이전 조건의 allocator cache를 포함할 수 있다. `gpu.csv`는 2초 간격 GPU 전체 사용량으로 process peak와 다르다. 시간은 한 번씩 측정했으므로 정밀한 반복 속도 benchmark는 아니다.

## 실제 실행 명령

아래 경로를 사용했다. 모든 출력은 새 폴더에 저장하며 원본/기존 결과를 덮어쓰지 않는다. 기본 Python 및 다른 사용자의 프로세스도 변경하지 않는다.

```bash
cd /home/gpu_03/trone-imatch-e3-e4
export P="$PWD/.venv-e3-e4/bin/python"
export E=experiments/2026-11-06_scale_ransac
export D="$PWD/data/im_t1_3d_1000_reconstructed"
export W="$PWD/.cache/e3-e4/torch"
export R="$PWD/$E/runs/method_b_20261102_r01"
# 원본 검증/선별 export의 실제 명령(기존 출력 경로가 있으면 재실행 불가):
# "$P" "$E/reconstruct_singapore.py" --zip data/Bearing_UAV_90K.zip #   --report "$E/validation/bearing90k_export_audit.json" --export "$D" --diagnostic-only
# 실제 실행한 고정 명령(기존 seal이 있으면 덮어쓰지 않음):
# "$P" "$E/seal_method_b.py" --data "$D"
"$P" "$E/preflight.py" --data "$D" --gpu 0 --torch-home "$W"
"$P" "$E/test_protocol.py"
"$P" "$E/launch_tmux.py" --session jaehong-method-b-smoke --   --data "$D" --out "$R/smoke" --split smoke --gpu 0 --torch-home "$W"
"$P" "$E/analyze.py" --runs "$R/smoke" --split smoke --out "$R/smoke_analysis"
"$P" "$E/resource_plan.py" --smoke-run "$R/smoke" --out "$R/resource_estimate.json"
"$P" "$E/launch_tmux.py" --session jaehong-method-b-tune --   --data "$D" --out "$R/tune" --split tune --gpu 0 --torch-home "$W"
```

탐색 이후의 실제 후보명/근거는 freeze.json에 기록한다.

```bash
"$P" "$E/evaluate_existing.py" --run "$R/tune" --split tune --out "$R/tune/team_evaluate.md"
"$P" "$E/analyze.py" --runs "$R/tune" --split tune --out "$R/tune_analysis"
# 실제 탐색에서 선택·동결한 후보(최종 결과로 교체하지 않음):
export CANDIDATE=e3_scale1p5
export REASON="Tune-only:79/100 vs baseline77; recovery4/loss2; p=0.6875. Confirm positive final gain with Holm4 p<0.05, else retain baseline."
"$P" "$E/freeze.py" --tune-run "$R/tune" --tune-analysis "$R/tune_analysis"   --candidate "$CANDIDATE" --reason "$REASON" --out "$R/freeze.json"
"$P" "$E/launch_tmux.py" --session jaehong-method-b-final --   --data "$D" --out "$R/final" --split final --freeze "$R/freeze.json" --gpu 0 --torch-home "$W"
"$P" "$E/evaluate_existing.py" --run "$R/final" --split final --out "$R/final/team_evaluate.md"
"$P" "$E/analyze.py" --runs "$R/final" --split final --out "$R/final_analysis"
"$P" "$E/analyze.py" --runs "$R/tune" "$R/final" --split all --out "$R/all_analysis"
```

동일 경로 재실행은 덮어쓰지 않고 실패한다. 새 run을 만들 때는 새 이름을 사용한다. tmux attach는 `tmux attach -t jaehong-method-b-final`, 분리는 Ctrl-b,d다. 각 split의 console.log, 조건별 inference.log/gpu.csv/completed.json을 확인한다. 중단된 partial.csv/FAILED.json은 완료 결과가 아니다. OOM이면 batch는 이미 1이므로 배정 GPU의 여유를 확인하고 같은 설정으로 새 프로세스/run을 사용한다. 임의로 배율·키포인트·정밀도를 줄여 결과에 섞지 않는다.

## 측정 결과와 11/6 결론

탐색·최종·전체 결과와 새 실패 원인 표를 실제 `comparison.csv` 및 `REPORT.md`에서 작성했다. 최종 조건을 한 번 평가한 뒤 동결 기준으로 Baseline 유지 결론을 내렸다.

## 재현성과 한계

1. 과거 72.9%는 k100~999의 과거 1방향 결과이며 현재 Baseline과 동일 조건이라고 주장하지 않는다.
2. 원래 load_rows/이미지 해시가 없어 과거 prepared 픽셀 동일성은 미검증이다. 본 방법 B의 동일 입력/GT/전처리/가중치/설정 해시만 검증했다.
3. 단일 seed, 단일 도시, 정답 블록 T1 조건이다. 시계열 상관·실제 블록 검색·항법·외부 도시 일반화는 평가하지 않았다.
4. 새 1방향 실패 태그는 진단 휴리스틱이며 과거 8방향 146건과 혼동하지 않는다. subgroup·연속량 검정의 다중 비교 한계도 밝힌다.
5. 코드·설정·데이터·가중치·의존성/GPU 정체성이 동결 이후 달라지면 최종 실행을 중단한다. 새 결과를 보고 코드/설정을 조정해 재평가하지 않는다.
6. 원본 export를 추가 대조하려면 김이준 팀원에게 `diag/harness.py`의 load_rows/heldout_rows/SIZES와 그 함수가 import하는 dataset/image loader, 원래 samples_1000.csv 및 prepared 이미지 SHA256 또는 원래 im_t1_3d_1000.zip을 요청한다. 다른 모델 가중치나 회전/블록 선택 코드는 필요하지 않다.

## 완료한 탐색 결과 (k=0~99, 조건마다 100개)

| 조건 | 성공률 | 평균 오차(m) | 오차 중앙값(m) | inlier 중앙값 | 평균 시간(s) | 복구/악화 | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 새 Baseline | 77% | 9.825 | 4.117 | 43.5 | 0.1765 | — | — |
| Scale 1.5 | 79% | 9.349 | 3.897 | 50.5 | 0.2912 | 4/2 | 0.6875 |
| Scale 2 | 78% | 11.265 | 4.659 | 35.5 | 0.4523 | 6/5 | 1.0 |
| MAGSAC 3px | 77% | 9.825 | 4.117 | 43.5 | 0.1752 | 0/0 | 1.0 |
| MAGSAC 8px | 78% | 9.570 | 4.269 | 98.5 | 0.1762 | 1/0 | 1.0 |

 탐색의 새 Baseline 실패 23건은 블록 밖11/짝 부족9/회전2/기타1이다. 과거 146건을 재사용한 집합이 아니다. Baseline과 MAGSAC3의 기하/inlier는 동일했다. Scale1.5를 탐색 후보로 선택했으나 유의한 개선이라는 뜻은 아니다. 채택 여부는 동결한 기준으로 최종 평가를 한 번만 확인한다. [동결 기록](results/method_b_20261102_r01/freeze.json), [탐색 상세 보고서](results/method_b_20261102_r01/tune_analysis/REPORT.md).

## 최종 실측 결과 — k=100~999, 각 조건 900개

| 조건 | 성공률 | 평균 / 중앙 오차(m) | inlier 평균 / 중앙값 | ≥10m 실패 | 복구 / 악화 | 평균 시간(s) | McNemar p / Holm4 p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 새 Baseline | 74.22% | 10.629 / 5.545 | 62.75 / 42.0 | 232 | 0 / 0 | 0.1770 | — |
| Scale 1.5 | 71.89% | 11.373 / 5.681 | 68.24 / 44.0 | 253 | 39 / 60 | 0.2935 | 0.0438753 / 0.131626 |
| Scale 2.0 | 69.56% | 12.782 / 5.973 | 61.06 / 34.5 | 274 | 43 / 85 | 0.4540 | 0.000258527 / 0.00103411 |
| MAGSAC 3px | 74.22% | 10.629 / 5.545 | 62.75 / 42.0 | 232 | 0 / 0 | 0.1769 | 1 / 1 |
| MAGSAC 8px | 74.67% | 10.686 / 5.674 | 119.79 / 92.0 | 228 | 13 / 9 | 0.1764 | 0.523467 / 1 |

## 전체 실측 기술통계 — k=0~999, 각 조건 1,000개

탐색 100개를 포함한다. 이 표의 검정값으로 최종 채택을 다시 결정하지 않는다.

| 조건 | 성공률 | 평균 / 중앙 오차(m) | inlier 평균 / 중앙값 | ≥10m 실패 | 복구 / 악화 | 평균 시간(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 새 Baseline | 74.50% | 10.549 / 5.407 | 65.07 / 42.0 | 255 | 0 / 0 | 0.1770 |
| Scale 1.5 | 72.60% | 11.171 / 5.465 | 70.52 / 45.0 | 274 | 43 / 62 | 0.2932 |
| Scale 2.0 | 70.40% | 12.630 / 5.878 | 63.50 / 35.0 | 296 | 49 / 90 | 0.4538 |
| MAGSAC 3px | 74.50% | 10.549 / 5.407 | 65.07 / 42.0 | 255 | 0 / 0 | 0.1768 |
| MAGSAC 8px | 75.00% | 10.575 / 5.577 | 122.02 / 92.0 | 250 | 14 / 9 | 0.1764 |

## 짝지은 연속량·시간·메모리 — 최종 900개

차이는 조건−새 Baseline이다. 오차 평균 차이의 95% CI는 10,000회 sample-paired bootstrap이다. 연속량 p는 탐색적이다.

| 조건 | 오차 평균 차이 [95% CI](m) | 오차 / inlier Wilcoxon p | 평균 시간 / Baseline | allocated / reserved peak(MiB) |
| --- | --- | --- | --- | --- |
| 새 Baseline | 0.000 [0.000, 0.000] | 1 / 1 | 1.00× | 2802 / 4354 |
| Scale 1.5 | 0.744 [-0.021, 1.513] | 0.76838 / 4.917e-16 | 1.66× | 5278 / 8414 |
| Scale 2.0 | 2.152 [1.246, 3.070] | 0.00034469 / 0.0018661 | 2.56× | 8706 / 15568 |
| MAGSAC 3px | 0.000 [0.000, 0.000] | 1 / 1 | 1.00× | 2802 / 15568 |
| MAGSAC 8px | 0.057 [-0.151, 0.280] | 0.39359 / 2.319e-143 | 1.00× | 2802 / 15568 |

E4 3px의 reserved peak 증가는 앞선 Scale2 allocator cache 유지 때문이다. allocated peak는 Baseline과 동일하며 threshold 자체의 메모리 증가로 해석하지 않는다.

## 새 Baseline 실패 원인과 복구

| 범위 | 블록 밖 | 짝 부족 | 회전 태그 | 기타 | 합계 |
| --- | --- | --- | --- | --- | --- |
| k0~99 | 11 | 9 | 2 | 1 | 23 |
| k100~999 | 95 | 66 | 25 | 46 | 232 |
| k0~999 | 106 | 75 | 27 | 47 | 255 |

최종 새 Baseline의 블록 밖+짝 부족은 **161건**이며 전체에서는 **181건**이다. 과거 146건과 별개의 집합이다. 아래는 최종 161건 부분집합의 결과이며 전체 성공률이 아니다.

| 조건 | 블록 밖 복구/95 | 짝 부족 복구/66 | 합계 복구/161 | 해당 부분집합 오차 평균 차이(m) | 해당 부분집합 inlier 평균 차이 |
| --- | --- | --- | --- | --- | --- |
| 새 Baseline | 0 | 0 | 0 | 0.000 | 0.000 |
| Scale 1.5 | 12 | 18 | 30 | -2.511 | 0.901 |
| Scale 2.0 | 14 | 19 | 33 | -1.297 | -0.062 |
| MAGSAC 3px | 0 | 0 | 0 | 0.000 | 0.000 |
| MAGSAC 8px | 2 | 7 | 9 | -0.246 | 7.925 |

Scale1.5는 이 161건 중 30건을 복구했지만 전체 최종 평가에서는 복구39건보다 기존 성공의 악화60건이 많았다. 실패 부분집합만 보고 전체 개선으로 해석하면 안 된다. 짝 부족 66건만 보면 Scale1.5의 평균 오차 변화는 +2.945m이므로 복구 수 증가가 모든 실패의 오차 개선을 뜻하지 않는다.

## 11/6 회의 결론

**기존 설정인 Scale 1.0·MAGSAC 3px를 유지한다.** 탐색에서 고른 Scale1.5는 최종 성공률이 74.22%→71.89%(−2.33%p), 복구39/악화60, Holm p=0.132로 채택 기준을 충족하지 못했다. Scale2는 69.56%(−4.67%p), Holm p=0.00103으로 유의하게 악화됐다. MAGSAC8은 inlier 중앙값을 42→92로 늘렸지만 성공률은 74.67%(+0.44%p), Holm p=1.0으로 유의한 개선이 없었다.

Scale1.5의 inlier 평균 증가는 관측됐지만 위치 성공률을 개선하지 못했다. 입력 확대나 임계값 완화로 inlier가 늘어나는 것과 실제 위치 정확도 향상은 구분해야 한다. 동결한 후보를 최종 결과를 보고 교체하지 않았으며 추가 튜닝·재평가는 수행하지 않았다.

과거 72.9%는 참고값이다. 이 프로토콜의 새 Baseline은 최종 900개 74.22%, 전체 1,000개 74.50%다. 이는 과거 수치의 개선/재현을 입증하는 비교가 아니다.

## 실험 완료·보존 기록

GPU 0 RTX A6000에서 smoke25건과 본실험5,000건을 실제 실행했다(별도 warm-up 제외). smoke·탐색·최종 모두 정상 완료됐다. Baseline/MAGSAC3의 기하·inlier는 탐색/최종/전체에서 동일했다. 최종 조건 실행 wall은 1,168.41초(19분28초, 모델 초기화 제외)다. 테스트14개와 실제 OpenCV MAGSAC 검사가 모두 통과했고 skip은 없다.

[보존한 결과 전체](results/method_b_20261102_r01/), [최종 상세 보고서](results/method_b_20261102_r01/final_analysis/REPORT.md), [전체 1,000개 보고서](results/method_b_20261102_r01/all_analysis/REPORT.md), [동결 기록](results/method_b_20261102_r01/freeze.json), [채택 판단](results/method_b_20261102_r01/decision.json). 샘플별 비교·성공 전환·새 실패 태그·과거 태그 참고 분석·입력 manifest와 해시는 같은 보존 폴더에 있다. 원본 ZIP/기존 결과를 덮어쓰지 않았으며 remote push/PR은 수행하지 않았다.

검증 완료 기록: [method_b_final_verification.json](validation/method_b_final_verification.json). 원본 보존 폴더의 122개 파일 해시를 전수 확인했다. 제출 폴더의 `ARTIFACT_SHA256.json`은 선별한 93개 파일을 검증하며 `SOURCE_ARTIFACT_SHA256.json`은 생략 항목을 포함한 원본 122개 해시 목록이다. 제출본은 중복 로그·서버 환경 덤프·외부 패키지 소스 사본을 제외한 선별본이며 [제출 범위](SUBMISSION.md)를 따른다. CSV의 표준 CRLF를 보존하도록 Git attributes를 지정했으며, 최종 `git diff --check`도 통과했다. GPU 작업 프로세스는 정상 종료됐고 조회 시 15MiB·사용률0% 상태로 반환됐다. 기본 Python은 여전히 CPU용 torch2.0.1로 유지된다.

## 팀 저장소 통합 검증 (2026-11-03)

팀 `main`의 `3a77ff40c313961e39c8fd88cff490e5a40ca3ec`에서 만든 `experiment/gpu03-e3-e4` 브랜치에 선별 통합했다. 독립 로컬 snapshot 이력은 병합하지 않았다. 팀 공통 코드와 대조한 뒤 지연 import 변경만 반영했고, 실측 당시의 Python·설정·의존성 파일 25개와 전체 source fingerprint가 모두 일치함을 확인했다.

통합 사본에서 테스트14개(skip0), 기존 CPU SIFT 실행기 실제5샘플, 기존 evaluate·CLI·Python compile 검사를 통과했다. 보존된 최종 CSV를 재분석한 비교표·샘플별 차이·새 실패 태그/건수 CSV는 원본과 바이트 단위로 같다. 이 과정에서 GPU 추론은 반복하지 않았다. 팀 일정·타인의 실험·기존 연구 CSV는 변경하지 않았다. [팀 통합 검증](validation/team_integration.json), [테스트 출력](validation/team_protocol_tests.txt), [기존 평가 출력](validation/team_evaluate.txt).
