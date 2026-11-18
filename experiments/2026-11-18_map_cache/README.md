# 지도 특징 캐시: 결과는 그대로, 시간 1.4~1.5배 단축

교수님 피드백("지도 특징 캐시")과 경량화. 지도 블록의 LoMa 키포인트·디스크립터를 한 번만 계산해 두고, 드론 사진 쪽만 매번 계산한다.

**결론 한 줄**: 결과는 캐시 없을 때와 한 장도 다르지 않고(위치 차이 0), 걸음당 시간이 사진 한 장 0.80 → 0.56초(1.43배), 항법 R1 0.74 → 0.54초(1.36배), v1(R1 + R4) 3.57 → 2.32초(1.54배)로 줄었다.

## 1. 방법

- LoMa는 두 영상 각각의 `detect_and_describe`(키포인트 + 디스크립터)를 구한 뒤 짝을 맞춘다. 지도 블록은 그대로 두고 드론 사진만 돌리므로, 블록 쪽 결과는 방향(최대 4개)마다, 항법에서는 걸음·재탐색마다 똑같이 반복 계산되고 있었다.
- 블록(한 변 ≥ 400 px)의 `detect_and_describe` 결과를 내용 해시로 캐시한다(최근 512개). 드론 사진(256 px)은 매번 계산한다. DaD 키포인트는 점수 상위 k개(top-k)라 결정적이어서 결과가 바뀌지 않는다.
- 코드: `imatch/matchers.py`의 `VisMatch._cache_block_features`. 매처 이름에 `-cache`를 붙이면 켜진다(`loma-cache`).
- 실제 드론에서는 비행 전에 지도 전체 블록의 특징을 미리 계산해 두는 것과 같다(이 실험은 처음 만난 블록에서 계산하므로 그 계산 시간이 포함돼 있다).
- FP16: LoMa는 T4(compute capability 7.5)에서 이미 fp16 autocast로 돈다(`loma/device.py`). 기존 숫자가 모두 fp16 결과라 따로 바꿀 것이 없었다.

```
# 사진 한 장 (개발용, 같은 커널에서 캐시 없음 / 있음)
python scripts/kaggle/build_kernel.py --user yijuniverse --dataset heeyunju/visloc-single-calib --slug calib-map-cache --args "--matchers loma,loma-cache --angles 4 --stop-inl 15"
# 항법 (시험용, #45와 비교)
python scripts/kaggle/build_kernel.py --user yijuniverse --dataset heeyunju/visloc-seq-eval --slug visloc-nav-map-cache --script run_visloc_nav --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20 --odo vio --seed 2026"
```

`scripts/run.py`에 `--stop-inl`을 넣었다(`run1113.py`와 같은 옵션).

## 2. 결과 (Kaggle T4)

### 사진 한 장: 개발용 `single_calib` 513장, 4방향 + 멈추기 15

| | 10 / 20 / 32 m | 한 장 시간 평균 | 각도당 시간 |
| --- | --- | --- | --- |
| 캐시 없음 | 15.6 / 45.6 / 68.2% | 0.80초 | 0.31초 |
| **캐시** | 15.6 / 45.6 / 68.2% | **0.56초** | 0.22초 |

- 513장 모두 inlier 수·고른 각도·위치가 같다(차이 0). #40(지민)과도 같다.
- 평균 시도 각도 2.59개. 블록 계산이 각도마다 → 한 번으로 줄었다.

### 항법: 시험용 `seq_eval` 92구간 (#45와 비교)

| | 마지막 40 m 이내 | 걸음당 시간 (#45 → 캐시) | 배 |
| --- | --- | --- | --- |
| R1 | 69 → 69 | 0.74 → **0.54초** | 1.36 |
| v1 (R1 + R4) | 72 → 72 | 3.57 → **2.32초** | 1.54 |
| └ 3×3 재탐색 걸음 | | 9.9 → 6.1초 | 1.6 |
| └ 나머지 걸음 | | 2.38 → 1.62초 | 1.5 |

- 1,020걸음 모두 추정 위치가 #45와 같다(차이 0). 시험용 결과는 바뀌지 않으므로 다시 고를 것이 없다.
- 재탐색에서 같은 블록이 여러 번 나와서 v1 쪽 효과가 더 크다.

## 3. 남은 시간은 어디서 드나

캐시 뒤 각도당 0.22초는 드론 사진의 `detect_and_describe`와 짝 맞추기다. 더 줄이려면:

- 방향마다 드론 사진을 다시 돌리지 않는 방법(방향 수 자체, 이미 4방향 + 멈추기)
- 키포인트 수 2048 → 1024, 더 작은 모델(LoMa-B128): 결과가 바뀌므로 개발용에서 다시 정해야 한다
- R4 3×3 횟수 줄이기: v1 설정이 바뀌므로 개발용에서 다시 정해야 한다
- `torch.compile`(LoMa 설정 `compile`): 결과는 같고 리눅스 GPU에서 빨라질 수 있다. 시도하지 않았다

## 4. 파일

| 파일 | 내용 |
| --- | --- |
| `calib_loma_s1.csv`, `calib_loma-cache_s1.csv` | 사진 한 장, 캐시 없음 / 있음 |
| `visloc_loma-cache-{r1,r4g0n5w20}_{segs,steps}.csv` | 항법, 캐시 |
