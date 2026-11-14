# MEAN · Sample4Geo 미터 정확도 사후 계산

2026-11-15 제공된 원본 samples.csv의 GPS·정규화 좌표를 이용해 **저장된 검색 결과만으로** 계산했다. 모델 재실행이나 검색 후보 변경은 하지 않았다. 원본 RUN의 N/A 값은 당시 좌표 미제공 상태의 기록으로 보존하고, 계산 결과를 이 디렉터리에 별도 저장한다.

## 결과

예측 위치는 **검색된 갤러리 블록의 중심**이다. UAV GPS와의 WGS84 타원체 최단거리(GeographicLib 2.1)를 사용하고, 정확도는 첫 반복(repeat=0) 1,738장에 대해 `error < threshold`로 계산한다.

| 모델 | <10m 성공/전체 | <10m (%) | <20m 성공/전체 | <20m (%) | <32m 성공/전체 | <32m (%) |
|---|---:|---:|---:|---:|---:|---:|
| MEAN | 6 / 1738 | **0.3452** | 49 / 1738 | **2.8193** | 132 / 1738 | **7.5949** |
| Sample4Geo | 15 / 1738 | **0.8631** | 55 / 1738 | **3.1646** | 122 / 1738 | **7.0196** |

[요약 CSV](retrieval_meter_summary.csv) · [MEAN 샘플별 오차](MEAN_predictions.csv) · [Sample4Geo 샘플별 오차](Sample4Geo_predictions.csv)

MEAN의 평균/중앙 오차는 약 370.39km / 18.95km, Sample4Geo는 약 441.61km / 146.01km다. 전체 비행을 섞은 고정 갤러리에서 다른 지역을 검색한 예측을 그대로 포함했다. query 정답 비행으로 후보를 제한하지 않았다. 동일 비행 검색 비율은 각각 36.59%, 34.70%다.

`paired_block_exact_match_pct`는 원본 UAV에 연결된 블록의 정확한 이미지 해시와 검색 해시가 같은 비율(8.8608%, 8.2278%)이다. 겹치는 이웃 블록도 위치상 정답이 될 수 있어 독립 relevance label로 평가한 Recall@1이라고 부르지 않는다.

## 좌표 복원 근거

- [원본 samples.csv](samples.csv)의 1,738개 `(set,k)`가 RUN manifest와 정확히 일치하며 `x_norm,y_norm`도 전부 일치한다. 이미지 실물은 받지 않았으므로 입력 이미지 자체의 추가 해시 검증을 했다는 의미는 아니다.
- [export_visloc.py](../../../scripts/export_visloc.py)의 위성 지도는 북쪽 위 0.25m/px, 128px/정규화 단위, 256px 격자 간격이다. 따라서 1단위=32m, 블록 중심 격자 간격=64m다. `gsd`는 UAV 원본 사진의 크롭 축척이므로 지도 거리 환산에 사용하지 않는다.
- 원본 CSV에 `lat,lon,x_norm_raw,y_norm_raw,along_m,right_m`가 있다. 모든 위치 보정은 0이며 raw/corrected 정규화 좌표는 동일해 query GT는 저장된 GPS다.
- exporter는 비행 전체 평균 위도에서 m/degree를 계산하지만 평균 위도 자체를 samples에 저장하지 않는다. 비행별 GPS 차이와 raw 정규화 오프셋 차이에 64m 격자 제약을 적용해 동/북 m/degree를 최소제곱으로 복원했다. 표본 평균 위도 축척으로 격자 정수를 초기화하고, 복원 후 격자 정수 일치·축척 변화 0.1% 미만·잔차 1mm 미만을 검사한다. 동축에서 역산한 기준 위도로 계산한 북축 축척도 대조했다.
- 10개 비행에서 최대 격자 잔차는 **0.0000293m**다. 이는 저장된 좌표와 격자 모델의 수치 일관성이며, 원본 GPS의 실제 측량 정확도를 의미하지 않는다.
- 각 블록 중심은 `center_lon = lon - 32*x_norm_raw/mx`, `center_lat = lat + 32*y_norm_raw/my`다. y가 남쪽으로 증가하므로 위도 식은 **더하기**다.
- 갤러리 ID는 manifest의 블록 SHA256과 gallery.csv의 이미지 SHA256으로 연결한다. 복원 좌표는 검색 결과와 무관하게 전체 갤러리에 먼저 부여한다.

[좌표 테이블](coordinates.csv) · [비행별 복원 감사](flight_coordinate_audit.csv) · [검증 및 입력 해시](verification.json)

동일 비행 검색에 대해 exporter 평면 거리와 WGS84 거리의 10/20/32m 성공 판정 차이는 모두 0건이다. 전체 예측 중 임계값과의 최소 거리는 약 0.051m로, 복원 격자 잔차보다 충분히 크다.

## 해석 범위

`position_ranking_valid=False`는 유지한다. 좌표를 확보한 것은 기존 데이터·후보 생성·학습/시험 중복 감사를 자동으로 완료한 것이 아니다.

Bearing-UAV/iMatch는 주어진 정답 지도 블록 안에서 연속 위치를 추정하고, MEAN/Sample4Geo는 모든 비행의 갤러리에서 블록을 찾아 그 중심을 답한다. 동일한 미터 임계값이라도 과제·입력 정보가 다르므로 이 수치만으로 공정한 모델 순위를 주장하지 않는다. 같은 블록을 골라도 블록 중심과 UAV의 거리가 10m를 넘으면 10m 실패다.

## 재현

```bash
python -m pip install geographiclib==2.1
python scripts/score_retrieval_meters.py
python scripts/test_score_retrieval_meters.py
python scripts/verify_kaggle_results.py
```

필요 입력은 이 디렉터리의 원본 samples.csv와 기존 RUN의 manifest, gallery, 모델별 predictions.csv다. 스크립트는 원본 RUN을 변경하지 않고 이 디렉터리의 파생 파일만 다시 생성한다.
