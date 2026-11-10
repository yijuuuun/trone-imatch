# [E] 저조도 — 독립 GPU Pilot 완료

| Condition | Enhancement | Independent Success@10m | Rejection | Large error recall | Mean ms |
|---|---|---:|---:|---|---:|
| daylight | none | 65% (13/20) | 20% | 1/2 (50.0%) | 832.7 |
| mild | none | 70% (14/20) | 20% | 1/2 (50.0%) | 859.0 |
| severe | none | 60% (12/20) | 25% | 1/3 (33.3%) | 920.5 |
| mild | gamma | 70% (14/20) | 20% | 2/3 (66.7%) | 876.0 |
| mild | clahe | 70% (14/20) | 20% | 4/5 (80.0%) | 892.5 |
| severe | gamma | 55% (11/20) | 30% | 4/6 (66.7%) | 842.4 |
| severe | clahe | 60% (12/20) | 30% | 2/3 (66.7%) | 853.7 |
| severe | gamma_clahe | 60% (12/20) | 30% | 3/4 (75.0%) | 811.0 |

전 표는 독립 Oracle T1 개발20쌍(k0..19)의 결과이며 팀 공식 정확도가 아닙니다. 원본 EPSG:4326 지도/UAV JSON과 GT 좌표 관계를 개발100개에서 검증했습니다. 위치 오차는 공식 원본 메타데이터의 0.25m/px에 따른 평면 거리입니다. WGS84 예측 좌표와 별도 haversine 오차도 샘플 CSV에 기록했습니다. 팀 imatch 좌표·heading·평가·run_nav 통합은 미완료입니다.

주 Success@10m는 error<10m AND 거절 아님, 전체20개 분모입니다. raw_success_10m는 거절 적용 전이며 별도 열로 보존했습니다. 거절은 inlier<15 또는 H 무효, 큰 오류는 거절 전 위치 오차>32m입니다. 큰 오류 검출 재현율=거절된 큰 오류/전체 큰 오류이며 조건별 분모가 다릅니다. no-pose는 성공 실패로 분모에 남기고 수치 오차는 공란으로 두도록 했습니다(이번260건은 모두 수치적으로 유효 H). 위치 clipping/중심 fallback은 없습니다.

모든 조건에서 Full LoMa-B, keypoints2048, threshold0.1, MAGSAC3px, batch1, cache 없음입니다. Master weights는 FP32, FP16은 AMP 연산입니다. 기본8방향 [0,45,...315],4방향 [0,90,180,270], GT로 후보를 선택하지 않았습니다. 동일한 최대-inlier/동률 match/최초 angle 선택 규칙을 유지했습니다. 원본 이미지·가중치는 변경하지 않았습니다.

Headline 시간은 decoded RGB→UAV 합성/보정→회전/전송/resize→양쪽 특징 추출→matching/filter/transfer→MAGSAC/H/블록 중심·후보선택입니다. CUDA 동기화와 조건별 single-heading warm-up10회를 적용했습니다. 파일 decode, 모델 로딩, 기록, GT 채점 및 WGS84 변환은 제외했습니다. finite guard와 heading별 sync 비용은 포함됩니다. 따라서 팀 navigation end-to-end라고 부르지 않습니다.

latency_breakdown.csv는 별도 k0..4의 동기화 wall profile입니다. feature_wall_ms는 detector+descriptor 양쪽 영상, transformer_wall_ms는 model.forward만, magsac_wall_ms는 cv2.findHomography만입니다. 그밖의 필터/전송·검사/후보선택은 차액에 포함됩니다. 별도 profile_e2e_ms와 headline20개 평균은 측정 pass와 표본이 달라 합산·대체하면 안 됩니다. Headline은 조건별 고정순서1회 측정이며 온도/clock/순서 영향에 대한 균형 반복은 미실시했습니다. 작은 해상도 가속차의 일반화는 보류합니다. Peak는 allocated와 reserved를 구분하며 모델 parameters/원본 weight 파일 크기는 줄지 않았습니다.

T4R/NAT2021: 공유 로컬 아카이브와 개인 GitHub 검색에서 설정을 확보하지 못했습니다. 공식 카메라 모델 재현 결과가 아닙니다. standalone/lowlight.py의 기존 임시 설정을 고정했습니다: 선형광 노출 mild0.45/severe0.15, Poisson photon scale1000/500, Gaussian read sigma0.008/0.015. Seed20261110+k, Gamma x^0.6, LAB-L CLAHE clip2/grid8. UAV만 변환하며 위성 원본은 유지합니다. 각 조건은 동일 seed·같은 원본을 사용합니다.

Daylight는 F1 실측20건을 공유하며 별도 GPU 평가로 중복 계수하지 않았습니다. Gamma+CLAHE는 severe 단독 보정 두 방법이 주간 성공률을 회복하지 못한 뒤 추가한 탐색 조건입니다. 무보정 대비 일관된 위치 회복이 없어 기본 보정을 채택하지 않습니다. 큰 오류 recall 증가만으로 개선을 주장하지 않습니다. 분모가 큰 오류3→6 또는3→4처럼 커질 수 있습니다. Denoising/Zero-DCE/학습은 미실시했습니다.

실행 명령

```bash
cd ${WEEK2_WORKSPACE}
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -u standalone/pilot_matrix.py \
  --out runs/pilot20_v1 --limit 20
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python -u standalone/extra_gamma_clahe.py \
  --out runs/pilot20_gamma_clahe_v1 --limit 20 --only E_severe_gamma_clahe
.venv-gpu/bin/python standalone/analyze_pilot.py \
  --run runs/pilot20_v1 --out runs/pilot20_analysis_v1 --limit 20
.venv-gpu/bin/python standalone/publish_pilot.py
```
기존 run은 프로토콜·소스 hash가 일치하는 완료 샘플을 재실행하지 않습니다. 새 GPU 측정에는 새 --out을 사용하세요. 분석·publish 출력은 기존 파일 덮어쓰기를 거부하므로 재분석 시 새 output을 사용하거나 새 버전으로 관리해야 합니다. GPU는 샌드박스 밖 실행이 필요합니다. 소스 수정 후 기존 run에 이어쓰기는 거부합니다.

환경: environment.json, 설정: config.json, 원시 대응점: ../../runs/pilot20_v1 및 pilot20_gamma_clahe_v1. 이전 scaffold는 history/pre_pilot20_v1에 보존했습니다.
