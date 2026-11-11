# SYNTHESIS_COMPARISON — 기존 설정 변경 없음

## 표본과 측정 정의

NAT2021-test 180시퀀스를 ID로 정렬해 등간격 인덱스 20개를 고르고, 각 첫·중간·마지막 프레임을 사용했습니다. `NAT2021_SAMPLE_MANIFEST.json`에 원본 member, SHA, CRC, 선택 규칙이 있습니다. 비교 합성은 기존 개발 k=0~19에 원래 함수와 seed=20261110+k를 적용한 Daylight/Mild/Severe 60건입니다. CPU 영상 통계만 계산했으며 LoMa를 재실행하지 않았습니다.

두 자료는 장소·시점·FOV가 다른 **unpaired 표본**입니다. 아래 값은 프레임별 통계의 등가중 평균입니다. median 열은 프레임별 픽셀 중앙값을 평균한 것으로, 전체 픽셀을 pooling한 median이 아닙니다.

Y′=0.2126R′+0.7152G′+0.0722B′는 정규화된 sRGB code의 밝기 proxy입니다. 선형 Y는 각 채널에 piecewise EOTF를 적용한 뒤 가중합합니다. ICC 없는 RGB를 sRGB로 가정했으며 실제 RAW radiance가 아닙니다.

## 밝기·색·endpoint 비교

| 표본 / 기존조건 | n | Mean Y′×255 | Mean frame median Y′×255 | Mean linear Y | Dark Y′<0.05 | Any RGB channel=0 | Any channel=255 | Mean HSV S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NAT2021_sample60 | 60 | 26.09 | 17.67 | 0.02540 | 40.89% | 0.568% | 0.273% | 0.187 |
| dev_daylight20 | 20 | 131.81 | 132.41 | 0.28042 | 0.06% | 0.040% | 0.080% | 0.192 |
| dev_mild20 | 20 | 90.14 | 90.89 | 0.12624 | 1.68% | 3.768% | 0.000% | 0.272 |
| dev_severe20 | 20 | 50.87 | 52.15 | 0.04297 | 9.61% | 24.992% | 0.000% | 0.516 |

Severe 평균은 본 NAT 표본의 약 1.95배, Mild는 3.46배입니다. 어두운 픽셀 비율도 NAT 40.89%에 비해 Mild 1.68%, Severe 9.61%로 낮습니다. 합성의 흰 endpoint 비율은 0%이며, NAT 표본에는 광원과 highlight가 관찰됩니다. Severe의 높은 색포화도·any-channel=0 비율은 채널별 noise/clipping이 색 분포를 바꾼 정황입니다. 같은 원인이 실제 센서에 존재한다고 결론 내리지는 않습니다.

개별 프레임 평균은 NAT 3.99~71.65/255, Mild 63.47~110.71, Severe 34.72~63.51입니다. 일부 밝은 NAT 장면과 범위가 겹쳐도 표본 전체 분포가 유사하다는 뜻은 아닙니다. ID 기반 편의 표본이며 전체 모집단 검정이나 센서 파라미터 피팅은 하지 않았습니다.

## 대비·noise 통계

CSV에는 평균·중앙값·밝기 분위수, RGB 채널 분포, HSV saturation, global RMS 대비, P95−P05, 16px tile 대비와 어두운 영역의 HH 고주파 proxy가 있습니다. 64bin 히스토그램은 `PIXEL_HISTOGRAMS_64BINS.json`에 있습니다. 색포화도와 clipping은 별개입니다. dark code<0.05/linear<0.01은 본 감사의 기술 지표이며 공식 카메라 파라미터가 아닙니다.

NAT의 dark-HH MAD proxy는 여러 영상에서 0으로 퇴화합니다. 이는 센서 noise가 0이라는 뜻이 아닙니다. JPEG 양자화·압축, ISP, 평탄 영역과 texture가 섞인 지표입니다. RMS/P95/zero-residual 비율은 `NAT2021_NOISE_PROXY_SUPPLEMENT.csv`에 추가했습니다. RAW·고정 장면 반복·ISO·black-level 없이 실제 Poisson/read 분산을 분리 추정하지 않았고, 이 proxy로 σ를 피팅하지 않았습니다.

NAT 1280×720과 UAV 256×256의 native 16px 대비·고주파에는 해상도와 장면 차이도 있습니다. 픽셀 규모가 같다고 같은 센서·공간 구조를 비교하는 것은 아닙니다.

## LAI와 장면별 편차

[논문 §4.3](https://arxiv.org/abs/2203.10541)의 LAI는 객체 중심 local intensity의 시퀀스 평균<20입니다. 전체 Y′ 평균이나 CSV의 `global_code_lt20of255_ratio_NOT_LAI`는 LAI가 아닙니다. 정확한 ROI·변환 코드가 없어 local LAI를 재현하거나 표본을 새로 라벨링하지 않았습니다.

원본별 공식 도로/캠퍼스 라벨은 확보하지 못했습니다. 아래는 첫 프레임 연락판의 시각적 구조로 부여한 **비공식 편의 그룹**입니다. dark_path는 잠정 라벨, road/urban은 의미가 겹치며 캠퍼스 정체는 확인하지 않았습니다. 첫 프레임 기준 라벨을 세 표본에 기술적으로 적용한 것으로 공식 장면 분포나 장면 유형 간 검정이 아닙니다.

| 수동 시각 그룹 | 시퀀스 / 프레임 | 그룹 평균 밝기 /255 | 시퀀스 평균 범위 |
|---|---|---:|---|
| road_urban | 13 / 39 | 29.05 | 12.45~45.45 |
| dark_path_uncertain | 4 / 12 | 8.96 | 4.83~14.21 |
| open_paved_or_sports_area | 2 / 6 | 42.70 | 35.92~49.49 |
| distant_urban_skyline | 1 / 3 | 22.87 | 22.87~22.87 |

시퀀스별 편차는 `NAT2021_SEQUENCE_STATISTICS.csv`, 라벨 근거는 `SCENE_VISUAL_LABELS.csv`에 있습니다. 이렇게 다른 장면들을 하나의 평균 밝기로 모두 보정하는 것은 적절한 센서 calibration을 보장하지 않습니다.

## 예시와 판정

- `visual_examples/NAT2021_first_frames_local_only.png`: 20시퀀스 첫 프레임. 표시용 resize는 통계에 사용하지 않았습니다.
- `visual_examples/existing_synthesis_k000_k005_k019.png`: 기존 조건·동일 seed의 비교. 원본 파일은 그대로입니다.

현재 합성이 본 NAT 표본의 분포와 유사하다는 근거는 없습니다. 그러나 이 비교만으로 정확한 물리 카메라 값을 정할 수도 없습니다. **통계는 측정했지만 합성을 보정하지 않았으므로 B: Data-calibrated 모델은 아직 없습니다.**
