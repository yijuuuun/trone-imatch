# 팀원 실험 목록

누가 무엇을 언제까지 하는지는 [../docs/04_팀분담_일정.md](../docs/04_팀분담_일정.md).

규칙
- 한 번에 하나만 바꾼다. 같은 샘플에서 `scripts/evaluate.py`로 짝지어 비교한다.
- 새 규칙·설정을 결과를 보고 정했다면 `--kmin 100`으로만 평가한다 (k = 0~99는 이미 봤다).
- 결과는 `experiments/<날짜>_<이름>/` 폴더에 README(무엇을, 명령어, evaluate 표, 결론 한 줄)와 CSV를 남긴다. 브랜치에서 작업하고 PR로 올린다.

## 바로 할 수 있는 것

| # | 실험 | 명령어 (GPU) | 질문 |
| --- | --- | --- | --- |
| E1 | LoMa-B128 (경량, 파라미터 1/10) | `run.py --matchers loma-b128 --angles 1` | 정확도가 얼마나 떨어지나? 드론 탑재 후보가 되나? |
| E2 | LoMa-B128, 8방향 | `run.py --matchers loma-b128 --angles 8` | 경량 모델은 회전 탐색이 필요한가? |
| E3 | 확대 배율 | `run.py --matchers loma --angles 1 --scales 1.5` | 영상을 키우면 Singapore가 나아지나? |
| E4 | MAGSAC++ 임계값 | `run.py --matchers loma --angles 1 --ransac-px 8 --tag=-r8` | 3 px vs 8 px |
| E5 | 원 Bearing-UAV 속도를 T4에서 재기 | capstone1에서 | 같은 GPU 조건으로 속도 비교 |

## 우리 기여 후보 (docs/03_기여_추천안.md)

| # | 실험 | 기준 |
| --- | --- | --- |
| C3 | 후보 선택기: 여러 후보(방향 × 매처 × 배율) 중 맞는 것을 고르는 작은 모델. 입력은 inlier 수·비율, 호모그래피 축척·찌그러짐, 대응점 분포 | Singapore 76% → 82%(두 매처 중 하나라도 맞힌 비율)에 근접 |
| C2 | LoMa 매칭부를 Bearing 학습 도시로 미세조정 (DINOv2 고정) | 1방향 Singapore 72.9% → 76% 이상 |
| C4 | 연속 프레임 항법: 이전 위치로 블록·방향 후보를 좁히고 실패 시 복구 | 3D 1 km 항법 성공률 |

## 새 매처 추가하기
`imatch/matchers.py`에 `__call__(uav_rgb, block_rgb)`이 `magsac(p0, p1)` 결과를 돌려주는 클래스를 만들고 `get_matcher()`에 이름을 넣는다. 나머지(회전 탐색, 위치 계산, 평가)는 그대로 쓴다.

## 김재홍 E3·E4 완료 (방법 B, 2026-11-03)

[Singapore Scale·MAGSAC++ 평가](2026-11-06_scale_ransac/README.md): k=100~999, 조건당 900개, 위치 오차 <10m. 새 Baseline 74.22%, Scale1.5 71.89%, Scale2 69.56%, MAGSAC3 74.22%, MAGSAC8 74.67%. Scale1·MAGSAC3 유지. 재구성 입력의 새 Baseline에 대한 비교이며 과거 72.9%와 직접 비교하지 않는다.


## 2026-11-16 Bearing-UAV Colab 시퀀스 항법

[분석과 원시 결과](2026-11-16_bearing_visloc_colab/README.md): 92구간·1,020걸음, 마지막 40 m 이내 44.6%, 걸음 오차 중앙값 43.0 m. 동일 이동 오차의 DR은 60.9%. 기존 LoMa 결과와의 시작점 초기화 차이를 함께 기록했다.
