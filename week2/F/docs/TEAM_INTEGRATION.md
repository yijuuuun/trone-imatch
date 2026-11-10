# 팀 imatch 통합 준비서 — 설계만 작성, 팀 파일 미수정

대상 회의: 2026-11-11(수). 팀 소스가 없으므로 아래 경로는 **역할별 제안 경로**이며, 실제 존재·CLI 구조·시그니처를 확인한 목록이 아닙니다. 현재 실행 가능한 독립 코드는 ../standalone/lowlight.py, loma_backend.py, smoke_gpu.py입니다. 팀 저장소에 복사·패치·실행하지 않았습니다.

| 역할 | 제안 파일 / 실제 확인할 파일 | 준비할 변경 |
|---|---|---|
| 저조도 처리 | 신규 imatch/lowlight.py | 기존 standalone/lowlight.py의 RGB uint8 함수 이식. 개발 합성/보정 분리, 기본값 픽셀 불변, Satellite 우회 |
| LoMa 초기화·정밀도 | 기존 matcher wrapper 위치 확인; 필요 시 신규 imatch/loma_adapter.py | FP32 master weights, AMP 내부/외부 dtype 및 DINOv2 명시 cast 일치. FP32 TF32 상태 기록. strict loading 유지 |
| 방향 후보 옵션 | 기존 heading loop의 소유 파일 확인 | 명시적4방향일 때 [0,90,180,270],8방향일 때 [0,45,...315]. 회전/resize/후보선택·동률 처리 자체는 변경하지 않음 |
| CLI 옵션 전달 | 실제 run_nav.py 또는 해당 entrypoint 경로 확인 | CLI→config→matcher/전처리로 인자 전달. 실제 팀 기본값 상속, 옵션 미사용 시 원래 결과 유지 |
| 기하·좌표·heading | 기존 imatch.geometry / metrics의 실제 파일 확인 | 기존 변환·평가 함수를 호출. 독립 Scorer의 Singapore128px 정규화/좌표식을 팀 전체에 하드코딩하지 않음 |
| 평가 wrapper | experiments/2026-11-13_light2/eval_team_dev100.py 제안 | native 팀 파이프라인 호출 후 동일100쌍·세 조건 비교, 결과 새 경로 저장 |
| 통합 검증 | 저장소 기존 tests 또는 별도 experiments/.../integration_checks.py 제안 | 기본값 parity, dtype,좌표·heading,15/32 경계,분모,GT 비입력 확인 |

imatch/nav.py는 팀원의 R1~R4 변경 가능성이 있으므로 사본/현재 변경 상태를 확인하기 전 수정하지 않습니다. 이 문서의 함수·파일명은 팀 코드 발견 후 매핑해야 합니다.

## CLI 계약 제안 — 아직 구현/실행 확인하지 않음

| 옵션 | 제안 계약 | 기본값·제약 |
|---|---|---|
| --amp | 지정 시 FP16 AMP | 미지정 시 기존 팀 정밀도 상속. 기존 --amp 옵션과 충돌 여부 우선 확인 |
| --num-headings {4,8} | 명시한 고정 후보 목록으로 제한 | 미지정은 None으로 두고 팀 기존 후보 상속. GT heading으로 선택 금지 |
| --lowlight {none,mild,severe} | 개발 입력에만 임시 저조도 합성 | none. 실제 야간 영상에 재합성하지 않음. T4R/NAT2021 공식 재현이라고 표시 금지 |
| --enhance {none,gamma,clahe,gamma_clahe} | UAV RGB에만 보정 | none. 이번 결과에서 기본 보정 미채택, 옵션 존재와 채택을 구분 |
| --lowlight-seed | sample ID 기반 재현 가능한 seed | 개발용20261110+k. GT 수치/heading을 seed나 선택에 쓰지 않음 |

옵션 전달 순서: UAV decode → 선택적 합성 → 선택적 보정 → **기존 팀** 회전 후보/resize/추론 → 기존 MAGSAC/좌표/heading → 기존 선택·failure 규칙. Satellite 입력은 기존 경로 유지. 이미 팀 코드에 다른 처리 순서가 있으면 순서를 임의 변경하지 말고 차이를 먼저 기록합니다.

LoMa의 module-level device/amp_dtype와 각 module의 amp/amp_dtype가 함께 쓰입니다. 외부 autocast만 켜서는 FP16 설정이 충분하지 않을 수 있습니다. 여러 정밀도를 한 프로세스에서 동시 사용하는 구조라면 global dtype 충돌을 검토하고, 검증용 정밀도별 프로세스 분리를 우선합니다. CPU MAGSAC/기하는 FP16로 내리지 않습니다. 이번에는 모델 구조·feature cache·입력 해상도 축소를 팀 기본값에 추가하지 않습니다.

## 팀 코드 수령 후 대조할 항목

1. 팀 원본/작업 사본 및 현재 변경 상태 확인. gpu_03 경로 우회,원본 덮어쓰기,Git 변경은 하지 않음.
2. 옵션 미사용 기존 baseline과 연결 후 baseline의 대응점/좌표/inlier/거절/선택 angle parity 확인. 테스트 실행은 이번 문서 작성에서 수행하지 않음.
3. 현재 팀 회전 canvas·border·resize·COLMAP -0.5px 처리·H 방향·원래 좌표 복원·지도 mosaic 순서 확인. 독립 구현과 차이가 있으면 팀 코드를 기준으로 새 run에서 비교.
4. 팀 위치 및 heading 변환 함수를 그대로 재사용. GT 위치/heading은 채점 모듈에만 전달. Oracle T1 지도 조건과 실제 map search를 분리.
5. 선택 inlier14는 거절,15는 거절 아님. 큰 오류는32m 초과. 독립 Success는error<10m이며 팀 경계가 다르면 그대로 명시. 거절을 분모에서 빼지 않음. no-pose/clip/fallback 규칙을 숨기거나 임의 교체하지 않음.
6. FP32/FP16 실제 Conv/Linear dtype,FP32 master weights,NaN/Inf,strict checkpoint hash 확인. 방향 후보를 줄여도 선택 점수·동률 규칙은 동일하게 유지.
7. 동일개발100개에서 F0/F2/F3만 비교. 샘플별 판정·짝지은 전환·scene-aware 통계·시간/VRAM·단계별 profile 기록. 기존20개를 포함하므로 미노출 검증이라고 부르지 않음.
8. 팀 공식 결과는 새 디렉터리에 저장하고 기존 독립 Pilot과 혼합하지 않음. k100..999 및 잠긴 지역/궤적은 팀 평가 일정에 따름.

이번 산출물은 통합 사양과 검증 계획입니다. 팀 CLI 실행 가능,통합 완료,팀 공식 위치 정확도라고 주장하지 않습니다. T4R/NAT2021 파라미터가 도착하면 출처·단위·전처리 위치부터 확인하며, 임시 합성을 공식 모델로 이름만 바꾸지 않습니다.
