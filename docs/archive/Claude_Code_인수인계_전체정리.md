# Claude Code 인수인계: GPS-free 드론 위치추정 (Bearing-UAV) 프로젝트

작성일: 2026-10-08 (목) 17:10 KST
대상: 이 프로젝트를 이어서 도울 다음 Claude Code 세션
사용자: 김이준(이준), 동국대 컴퓨터AI학부 3학년. 팀 프로젝트 "Trone"(4명).

> 이 문서만 읽고 바로 이어서 작업할 수 있도록 정리했다. 확인된 것, 가설, 내가 실수한 것을 구분해서 적었다. 숫자는 사용자가 Colab에서 실행하고 붙여준 출력에서 가져온 것이다.

---

## 0. 먼저 알아둘 것

### 0-1. 작업 방식 (중요)
- **(10/8 갱신) Claude Code가 SSH로 Colab에 직접 연결된다.** Mac의 `~/.ssh/config`에 `Host colab`(cloudflared 터널)을 등록해 두었다. `ssh colab "..."`로 실행하고, `scp`로 파일을 올리며, 긴 작업은 `nohup`으로 돌린다. Colab 런타임이 재시작되면 터널 주소가 바뀌므로 사용자가 재연결 셀 하나를 실행하고 새 주소를 알려준다(sshd는 포트 2222에서 `-D`로 실행해야 한다. 22번 데몬 방식은 Colab에서 바로 죽는다). 사용자는 여러 셀을 복붙하는 방식을 힘들어한다. 예전 방식(Claude가 셀을 쓰고 사용자가 출력을 붙여줌)은 연결이 안 될 때만 쓴다.
- Claude의 클라우드 작업 공간에서 직접 돌리는 것은 어렵다(90K 데이터셋과 가중치를 받아야 하는데, 외부 접근이 허용 목록으로 제한됨. 시도해 보진 않았다).
- 사용자는 **쉬운 말로 설명**해 주기를 원한다. 항상 **문제 → 원인 → 해결** 순서로 말하면 이해가 빠르다.
- 사용자가 선호하는 문체는 담백하고 자연스러운 한국어다. 지어낸 내용이나 부정확한 프로젝트 설명은 잡아내서 알려주기를 원한다.
- **기여를 분리해서 적는다.** 팀원 희윤의 코드와 분석을 이준의 것으로 쓰지 않는다.

### 0-2. 일정과 팀 합의
- 중간발표 **10/23**, 기말발표 **12/11**. 오늘은 10/8.
- **10/7 회의에서 정해진 것**: Bearing-UAV의 문제는 CSMG라는 모듈이므로, **각자 "CSMG를 무엇으로 바꿀지"와 "그게 효과가 있는지"를 실험해 오기**. 이제 팀원들과 **각자 별개로** 진행한다(희윤과도 별개).
- "대체"의 방식은 각자 마음대로 해도 된다(CSMG만 끼워 넣어도 되고, 뒷단까지 바꿔도 됨).

### 0-3. 팀원
| 이름 | 방향 |
|---|---|
| 주희윤 | 저조도 아이디어 → 모듈별 진단 → dense matching(B2) baseline, 항법 시뮬레이터 코드, CSMG 교체 모델 학습 |
| 김재홍 | PiLoT + 3D Gaussian 맵 압축 (GPU 비용으로 제외됨) |
| 강지민 | PiDiNet + 속도 제약 + 2-Stage Reset (결과표는 예상치, 실험 아님) |
| 이준 | 장거리 쪽(블록 이탈 감지·재탐색), 지금은 이준만의 CSMG 대체안 |

그 외 팀원 문서 하나(작성자 미확인)가 GLUF 모듈 진단을 했다(5절).

---

## 1. 연구 개요

### 1-1. 기반 논문
**Bearing-UAV** (CVPR 2026, Liu et al.). GPS 없이 드론 영상과 위성지도를 매칭해 위치와 방향을 추정한다. 시뮬레이션만 사용한다. 야간·열화상은 범위 밖으로 정했다("원래대로" 선택).

### 1-2. 교수님 피드백 (10/1)
- 드론 탑재가 목적이므로 **가벼워야** 한다.
- 해법보다 **숨은 문제(hidden problem)**를 먼저 찾아야 한다.
- **baseline은 Bearing-UAV**여야 한다.
- 장거리에서 VO는 부적합하다.
- 모듈 단위로 교체하는 접근이 바람직하다.
- PiLoT, PiDiNet은 무겁다. 단순 재학습은 피한다.

### 1-3. 모델 구조
- 클래스: `PARCASGM_v5a` (`cvphr/models/posaglreg/models.py`), 모델 인자: `backbone_name=vgg16, feature_dim=256, num_clusters=4, freeze_backbone=True, add_patch_coord=True`.
- 흐름: UAV 영상 + 위성 타일 2×2(= 한 "블록") → **GLUF**(VGG16 conv4_3 + **CSMG**) → RCE(좌표 인코더) → CA(NeighborsCrossAttention) → PSG(SimilarityPositionPrior) → 위치·방향 회귀 헤드.
- **CSMG**(`cvphr/sceneGraphEncodingNet/nets.py`, `CSMG_soft` 사용)는 conv + Non-local + 클러스터 집계로, 특징 지도를 벡터 하나로 요약한다. 파라미터 1.31 M.
- 로드된 모델의 최상위 모듈: `backbone, csmg, coord_encoder, neighbors_cross_attn, pos_regressor, dir_regressor, sgm, sim_pos_prior`.

### 1-4. 좌표와 단위
- 라벨 단위 1 = 128 px = 32 m, GSD 0.25 m/px(`PX = 4` px/m). UAV 패치가 64 m를 덮는다.
- 도시 지도는 약 1 km²(15×15 블록, 256 px 타일).
- 이웃 블록은 타일 한 장(= 2 단위) 간격이다. 상하좌우는 타일 2장, 대각선은 1장 공유.
- 이웃 블록의 UAV 샘플 라벨은 현재 블록 기준으로 x' = x_norm + 2·dx, y' = y_norm + 2·dy.
- 타일 슬롯 축은 블록 축과 바뀌어 있다(PSG 좌표는 (y, x)).

### 1-5. 항법 시뮬레이션이란
- 15 m씩 전진한다. 한 스텝마다 (1) 드론이 지금 위치에서 아래를 찍고(위성 지도를 잘라 만든 크롭), (2) **직전 추정 위치 기준**으로 블록을 고르고, (3) 위치를 추정해 목적지 방향을 정하고 전진한다. 목적지 20 m 안이면 도착, 실제 위치가 40 m 안이면 성공.
- **닫힌 루프**라서 블록을 한 번 잘못 고르면 추정이 틀리고 다음 블록도 틀려서 복구되지 않는다. 이것이 장거리 실패의 핵심 구조다.
- 정답 블록(oracle)을 주면 95~100%다.

---

## 2. 환경 (Colab)

### 2-1. 복구 순서 (런타임이 초기화되면)
1. 런타임을 **GPU**로 설정 (Python 3.13, NumPy 2).
2. **복구셀**: Drive 마운트(`/content/drive/MyDrive/bearinguav_cache`), 저장소 clone(`github.com/liukejia121/bearinguav`, `.git` 확인 후), 필요한 패키지만 pip(`requirements.txt`는 3.13에서 실패하니 쓰지 않음), NumPy 호환 패치(`np.complex_`, `np.float_`, `np.unicode_`, `np.string_`) + `imgaug` MagicMock, 데이터셋과 가중치를 Drive 캐시 또는 HF에서 받음(HF repo `HaoyZhou/bearinguav`: `Bearing_UAV_90K.zip`(dataset), `Bearing_UAV.zip`(model)), `/content/bearinguav/Bearing_UAV_90K` 심볼릭 링크, Drive 캐시의 `diag/*.py`를 `/content/bearinguav/diag/`에 복사.
3. **로드셀** (아래 경로는 현재 실제 구조):

```python
import os, sys
os.chdir('/content/bearinguav'); sys.path.insert(0, '/content/bearinguav')
from diag.diag_modules import DEV, SEEN, UNSEEN, to_tensor, load_model, enc, head
from diag.diag_nav import load_rsi, TileCache
from diag.diag_phase0 import make_traj, Cropper, ModelLoc, nav_run, wilson, mcnemar, PX, DenseLoc
from diag.diag_gluf import rot_tpl
m2 = load_model('/content/bearinguav/Bearing_UAV/satellite_view')   # 중첩 폴더가 아님
```

- 가중치 위치: `/content/bearinguav/Bearing_UAV/{cross_view,satellite_view}/best_model.pth`. (예전에는 `Bearing_UAV/Bearing_UAV/...` 구조였는데 복구셀을 바꾸면서 한 단계 얕아졌다. 헷갈리면 `find /content/bearinguav -name best_model.pth`.)
- `diag/` 4개 파일(`diag_modules.py`, `diag_nav.py`, `diag_gluf.py`, `diag_phase0.py`)은 **희윤의 코드를 그대로 복사**한 것이다. 내가 수정하지 않았다. 사용자 Drive의 `bearinguav_cache/diag/`에 올려 두었다.

### 2-2. 주요 함수 (희윤의 `diag_phase0.py`)
- `make_traj(n_traj, dist_m, seed, n_block=15)` → `starts, goals` (픽셀 좌표). 평가 시드는 **2026**.
- `nav_run(loc, img, starts, goals, decode_px=128, oracle_block=False, step_m=15., arrive_m=20., n_block=15)` → `(success, reason, 스텝별 위치오차 중앙값 리스트)`.
- `loc(crops, headings, blocks)` → 라벨 단위 (x, y) 배열. `ModelLoc`(기존 모델), `DenseLoc`(B2, 학습 없는 VGG conv4_3 dense matching).
- `wilson(k, n)`, `mcnemar(a, b)` → `(a만 성공, b만 성공, p)`, 정확검정.
- `load_rsi(UNSEEN['NY'])`로 지도를 읽는다. 낯선 도시는 NY, Paris, London, Birmingham, Berlin. 학습한 도시는 Tokyo(`SEEN['34bc']`), SF(`SEEN['38bc']`).
- 어두움: `diag_modules.darken(rgb, lvl, rng, noise=True)`, `lvl`은 `DARK`의 키 `'L1','L2','L3'`. 감마 + 노이즈 + 8bit 양자화인 **단순 합성**이다.

---

## 3. 지금까지 한 일과 결과

모든 항법은 **2D 위성 크롭, `satellite_view` 모델, 나침반으로 진행 방향을 아는 조건**이다. 낯선 도시는 **NY 하나**, 궤적은 16~64개라 신뢰구간이 넓다.

### 3-1. 문제 진단 (확인됨)
- 낯선 도시(NY) 1 km 항법이 실패하는 건 위치를 못 읽어서가 아니라 **블록 선택을 한 번 틀리면 복구가 안 되어서**다. 정답 블록을 주면 NY **95.3%**(64궤적), Tokyo 100%.
- 블록이 틀리는 이유는 희윤의 진단으로 **CSMG가 특징 지도를 벡터 하나로 요약하며 위치 정보를 잃기 때문**이라고 본다(이준이 직접 확인한 것은 아님).

### 3-2. 모델 내부 값으로 confidence 만들기 → 폐기
| 신호 | 위치오차와의 상관 (n=200) |
|---|---|
| PSG softmax 분포 | -0.12 |
| CA attention 분포 | +0.05 |
| 둘의 평균 | +0.04 |

모두 약했다. 중간에 temperature scaling이 모델 예측 자체를 오염시켜 상관 +0.78이 나온 적이 있는데, 가짜 신호였다(confidence용 softmax를 분리해 다시 쟀다).

### 3-3. 블록 안/밖 신호와 이웃 재탐색
후보 9블록 중 진짜 블록을 1등으로 고른 비율(우연 11.1%). 분류기는 모델 내부 값 10개 + 로지스틱 회귀.

| 환경 | 분류기 | 최대 cos 단독 |
|---|---|---|
| 3D, Singapore | 66.3% | 42.7% |
| 2D, Tokyo (분류기 학습 제외) | 75.0% | 43.3% |
| 2D, NY (낯선 도시) | 48.3% | 35.3% |

항법(2D, 1 km, 같은 궤적 비교, 분류기는 Taipei·Singapore·SF로 학습):

| 도시 | 궤적 | 기본 | 정답 블록 | 재탐색 | 같은 궤적 비교 |
|---|---|---|---|---|---|
| Tokyo | 32 | 31.2% | 100% | **90.6%** | +21/-2, p=0.0001 |
| NY | 64 | 35.9% | 95.3% | 29.7% | +12/-16, p=0.57 |

**학습한 도시에서만 효과**가 있었다. (분류기 홀드아웃 AUROC 0.83.)

### 3-4. dense matching (희윤의 B2) 위에서 돌린 것
- B2는 **CSMG 교체 완성본이 아니라 baseline**이다(희윤 정정). 교체 모델의 실제 성능은 학습 후 판정한다.
- 희윤 측정: 5개 낯선 도시 1 km(480궤적) B1(누수 없이 재학습) 19~33% → B2 99.4%. **2D 상한선**이며 3D 렌더에서는 10 m 이내 비율이 SF 88.8%, Singapore 62.6%.
- 이준이 NY에서 돌린 것:

| 시험 | 기존 모델(공개 가중치) | dense |
|---|---|---|
| 1 km, 32궤적 | 21.9% (7/32) | **100%** (+25/-0) |
| 약 3 km(7구간×430 m), 16궤적 | 0% (0/16) | **87.5%** (14/16, CI 64.0~96.5) |
| dense + 매 스텝 이웃 8블록 재탐색 | — | 68.8% (오히려 나빠짐, +0/-10, p=0.002) |

- 3 km는 **같은 1 km² 지도 안을 돌아다닌 경로**다.
- 어두운 조건(UAV 크롭에만, 노이즈 시드 0):

| 조건 | 기존 모델 | dense | 같은 궤적 비교 |
|---|---|---|---|
| 주간 | 21.9% | 100% | +25/-0 |
| L2 | 50.0% | 40.6% | +5/-8, p=0.58 |
| L3 | 43.8% | 21.9% | +3/-10, p=0.09 |

### 3-5. 어두운 조건에서 dense는 어디서 틀리나 (NY, L2, 32궤적)
| | 주간 | L2 |
|---|---|---|
| 틀린 스텝(오차 10 m 초과) | 2.2% | 76.3% |
| 틀린 스텝 중 블록 밖 | 22/46 | 1626/1746 (93%) |
| 블록 안에서 틀린 비율 | 1.2% | 19% |
| 피크 점수(틀림/맞음) | 0.706/0.654 | 0.215/0.401 |
| 피크로 틀린 스텝 가리기 AUROC | — | **0.874** (블록 안 스텝만 0.887) |

- 처음 블록을 벗어나기 직전 스텝의 피크 중앙값 0.261은 블록 안에서 맞은 스텝(0.409)보다 이미 낮다.
- 스텝 단위 통계라 길을 잃은 뒤의 스텝이 많이 섞여 있다.
- **피크가 낮을 때만 재탐색**(임계값 0.269): L2에서 dense만 40.6% → 28.1%(+7/-11, p=0.48), 재탐색 비율 65%. 주간은 100%→100%. **감지는 되지만 복구는 안 된다.**

### 3-6. 오늘 새로 한 실험 (10/8): 이준만의 CSMG 대체안 탐색

**(a) 풀링 해상도 스윕** (`PoolLoc`: dense인데 특징 지도를 s배 평균 풀링해서 비교, NY 1 km 32궤적)

| s | 속도(ms/샘플)* | 주간 성공률 | 위치오차 중앙값 | L2 성공률 | L2 오차 | L2, s=1 대비 p |
|---|---|---|---|---|---|---|
| 1 | 24.6 | 100% | 0.8 m | 40.6% | 120 m | — |
| 2 | 7.3 | 100% | 1.6 m | 25.0% | 116 m | 0.27 |
| 4 | 4.6 | 96.9% | 4.1 m | 12.5% | 235 m | 0.012 |
| 8 | 6.5 | 43.8% | 15.6 m | 12.5% | 261 m | 0.023 |
| 11 | 3.9 | 43.8% | 26.3 m | — | — | — |
| 22 | 4.5 | 12.5% | 423 m | — | — | — |

*속도는 Colab GPU, 같은 블록 하나로 잰 것(블록 특징 캐시 후). 아래 (c) 참고.
- s=1의 L2 40.6%가 이전 결과와 정확히 같아서 파이프라인과 시드가 재현됐다.
- 주간에서 s=8, 11의 하락은 오차 중앙값(15.6, 26.3 m)이 풀링 후 한 칸의 크기(16 m, 22 m)와 비슷해서 **해상도 한계**로 보이고, s=22(423 m)는 진짜 정보 손실로 보인다. 정황일 뿐 검증하지 않았다.
- 어두운 조건에서는 **거칠게 풀링할수록 오히려 나빠졌다**(내 예상, "노이즈가 평균되어 버틴다"가 틀렸다). 3번 비교했고 노이즈 난수가 달라서 p를 그대로 믿지 말고 방향만 본다.

**(b) coarse-to-fine** (`CoarseFineLoc`: 거친 해상도로 후보 위치를 찾고, 그 주변 ±r 칸만 s=1로 정밀 확인)

| 설정 | 속도 | 주간 | L2 | L2, dense 대비 p |
|---|---|---|---|---|
| CF s=4, r=4 | 21.7 ms | 100% | 34.4% | 0.80 |
| CF s=4, r=8 | 19.2 ms | 100% | 31.2% | 0.58 |
| CF s=8, r=8 | 19.2 ms | 68.8% | 25.0% | 0.18 (주간 p=0.002) |

주간 정확도는 유지했지만 **속도 이득은 거의 없었다**. 어두운 조건에서는 개선이 없다.

**(c) 연산량·파라미터 (MAC 수 직접 계산, `FlopCounterMode`와 해석적 계산이 일치)**

| 항목 | 값 |
|---|---|
| 파라미터 | 백본 7.64 M, CSMG 1.31 M, 전체 17.13 M |
| UAV 크롭 256의 연산량 | 백본 18.23 GMACs, CSMG 1.41 GMACs |
| Bearing 한 스텝(UAV쪽) | 19.66 GMACs (백본 + CSMG + head 0.02) |
| dense 한 스텝 | 9.08 GMACs (템플릿 백본 8.62 + 상관 0.458) |
| 상관 비중 | dense 한 스텝의 약 5% |
| 풀링 상관 MACs | s=1: 458 M, s=2: 30 M, s=4: 1.84 M, CF s=4 r=4: 21.9 M |
| 블록 특징 계산(백본 512) | 72.93 GMACs (Bearing도 타일 4장에 비슷한 규모) |
| 실측 속도 | Bearing 10.6 ms/샘플, dense 26.2 ms/샘플 |

**해석 (추정 포함)**
- CSMG 자체의 연산은 한 스텝의 약 7%로 가볍다. 연산량 때문에 CSMG를 바꾸는 게 아니다.
- dense가 MAC 기준으로 가벼운 것은 대부분 **더 작은 입력(176 크롭)**으로 백본을 돌리기 때문이고, CSMG를 뺀 효과가 아니다.
- 연산량은 적은데 실제 시간이 2.5배 느린 것은 22×22 커널 합성곱의 비효율일 가능성이 크다(**미검증**). 그래서 앞서 말한 "풀링으로 3~5배 빨라진다"는 알고리즘의 이점이 아니라 이 구현의 특성일 수 있다. **정정한 내용이다.**
- **지도 저장량 추정(미측정)**: 256 px 타일 하나의 특징 지도(512×32×32, fp32) 약 2.1 MB, 1 km²(256 타일) 약 0.54 GB. Bearing은 타일마다 설명자 벡터(256×4 = 1024개 값, 약 4 KB)라 약 500배 작다. 연산이 아니라 **지도 저장량**이 dense의 진짜 비용일 수 있고 장거리일수록 커진다.

**(d) 대체 백본 비교 (학습 없음, 같은 dense 매칭에 특징만 교체) → 결과 대기 중**
사용자가 `FeatLoc`으로 VGG conv3_3(st4), ResNet18 layer2(st8), ResNet18 layer3(st16), ResNet50 layer2(st8)를 주간과 L2로 돌리는 중이다(첫 시도는 `Resnet18_Weights` 오타로 에러 → 수정본 전달). 기준은 VGG conv4_3의 주간 100%, L2 40.6%. 결과는 `bearinguav_cache/nav_backbone_NY.txt`에 저장된다. **아직 결과를 받지 못했다.**

---

## 4. 확인된 것 / 가설 / 틀렸던 것

### 4-1. 확인됨 (위의 조건 안에서)
- 낯선 도시 장거리 실패는 블록 선택 실패다(oracle 95.3%).
- 모델 내부 confidence(PSG, CA)는 약하다.
- 이웃 재탐색 분류기는 학습한 도시에서만 통하고 낯선 도시와 dense 위에서는 효과가 없다.
- 학습 없는 dense는 주간 낯선 도시(2D)에서 항법을 풀지만 L2에서 무너진다.
- 피크 점수는 틀린 스텝을 잘 가려내지만(AUROC 0.874), 재탐색으로 복구되지 않는다.
- 해상도를 줄이는 것은 어두운 조건을 개선하지 못한다.

### 4-2. 가설 (미확인)
- 어두운 조건의 실패 원인이 VGG 특징 자체일 수 있다(대체 백본 실험 결과 대기).
- 어두운 조건에서 블록 안 위치 읽기가 블록 이탈보다 먼저 나빠질 수 있다.
- 정적 측정(희윤: L2 dense 타일 판별 91~97%)과 닫힌 루프(40.6%)가 어긋나는 원인은 **모른다**.
- 22×22 커널 합성곱의 비효율이 dense의 느린 속도의 원인일 수 있다.

### 4-3. 내가 틀렸거나 실수한 것 (같은 실수 반복 금지)
- `pmax` 부호를 뒤집으라고 제안했는데 그냥 노이즈였다(0.160).
- 직선 3000 m 경로를 제안했지만 1 km 지도에서는 불가능했다(7구간×430 m로 대체).
- 이웃 재탐색 코드에서 블록 인덱스 상한을 `n_block-2`(13)로 둔 오류(14까지 가능). **3-3의 Tokyo·NY 재탐색 결과에 섞였을 수 있고, 영향은 확인하지 못했다.** dense 코드에서는 고쳤다.
- 풀링이 어두운 노이즈를 평균해서 도움이 될 거라고 예상했는데 틀렸다.
- "dense가 CSMG를 교체했다"고 쓴 적이 있다. 틀린 표현이다. dense(B2)는 baseline이다.
- 속도를 알고리즘 이득으로 해석했다가 구현 특성일 가능성을 발견하고 정정했다.
- `ResNet18_Weights` 철자 오류(`capitalize()` 때문).
- `nav.py`의 디코드가 유닛당 256 px(라벨은 128 px)이라는 버그가 있고, 이것이 증상을 우연히 가린다.

### 4-4. 주의할 비교 불일치
- **Tokyo baseline**: 이준 31.2%(32궤적, CI 18~49) vs 희윤 문서 45.8%(96궤적, CI 36~56). CI는 겹치나 같은 궤적인지 모른다. 희윤에게 확인해야 한다.
- 이준이 말한 "기존 모델"은 **공개 가중치(누수 있음, B0)**이고 희윤의 B1은 누수 없이 재학습한 모델이다. 낯선 도시 성공률은 둘 다 20~30%대로 비슷하다.
- 공식 테스트 분할은 누수가 있다(테스트 블록 100%가 학습에 포함).
- 어두움 쌍 비교는 노이즈 난수가 달라서 엄밀하지 않다.

---

## 5. 팀원 문서 (GLUF 진단) 요약과 우리와의 연결

내용: 저조도·장거리에서 VGG16 특징 변화(저조도 0.2686, 장거리 0.5402), GLUF 서술자 변화, Non-local 제거(성능 거의 변화 없음), Cluster aggregation 제거(크게 악화), Partial restoration(GLUF 복원이 VGG16 복원보다 회복이 큼), paired bootstrap, 결론 "GLUF가 downstream bottleneck, cluster aggregation을 더 강건한 방법으로 교체".

코드 구조 확인 결과: **GLUF = VGG16(conv4_3) + CSMG**, CSMG 안에 Non-local과 클러스터 집계가 있다. 즉 문서의 "GLUF 교체"는 결국 CSMG 교체다. 문서에는 **무엇으로 교체할지가 없다.**

**그 팀원에게 물어볼 것** (중요한 순서):
1. Partial restoration의 "partial"이 정확히 무엇인가? 모듈별로 같은 비율로 맞췄나? (맞추지 않았다면 "GLUF 복원이 더 낫다"는 비교는 성립하지 않는다. VGG 출력을 통째로 복원하면 GLUF 이후가 모두 깨끗해져 100%가 되어야 한다.)
2. 표의 숫자(0.4837 등)가 오차인가 정확도인가? recovery %의 기준은?
3. Non-local OFF가 오히려 약간 좋아졌는데(0.4837→0.4463) 어떻게 해석하나?
4. Cluster OFF의 악화는 재학습 없이 뺐기 때문 아닌가? 강건성의 증거가 되는가?
5. 샘플 수, 데이터 겹침 여부, 저조도·장거리 합성 방식은?
6. VGG16 vs ResNet18 feature change는 어떤 거리로 쟀나? (백본 간 스케일이 달라 직접 비교는 곤란)
7. 정적 예측만 봤는데 닫힌 루프 항법에서도 같은 병목이 나오나?
8. 교체안이 따로 있나?

**우리 결과와의 연결**: 같은 VGG를 쓰는 dense가 어두운 조건에서 무너진 것은 "VGG16이 upstream 취약점"이라는 이 문서와 맞닿는다(가설). 그러면 **CSMG만 바꿔서는 어두운 조건이 안 풀릴 수 있다.** 단, 사용자가 그 팀원에게 보낼 메시지 초안에서 "희윤이 CSMG를 dense로 교체했다"는 문장이 있었다면 정정이 필요하다(baseline이다).

---

## 6. 앞으로 할 일

### 6-1. 바로 할 것
1. **대체 백본 실험 결과(3-6 (d))를 받아서 해석한다.** L2에서 VGG conv4_3(40.6%)보다 의미 있게 오르는 후보가 있는지가 핵심이다. 오르면 "어두운 조건의 원인은 VGG이고 백본 교체가 답"이라는 이야기가 되고, 안 오르면 어두운 합성 자체의 한계일 수 있다.
2. **이준만의 CSMG 대체안을 정한다.** 지금까지의 선택지는 다음과 같다.
   - A. dense의 특징 교체(백본 교체): 어두운 조건 진단에는 유용하나 "CSMG 대체"는 아니다.
   - B. 공간 구조를 유지한 가벼운 서술자(풀링 등): 학습 없이 독립 위치추정. 새 알고리즘은 아니다.
   - C. 얼린 VGG 위에 작은 헤드를 약하게 학습: CSMG 형식에 맞춰 끼워 넣을 수 있는 유일한 후보. GPU 비용과 "단순 재학습 금지" 제약이 있다.
   - 현재까지의 결과는 "정확도는 유지하면서 연산이 비슷하거나 적고, 어두운 조건은 못 풀며, 지도 저장량이 크게 늘어난다"에 가깝다. 새 방법을 제안한 결과가 아니라 **dense의 비용이 문제가 아니라는 확인과 한계의 발견**이다. 솔직하게 발표해야 한다.
3. **지도 저장량(약 0.54 GB/km²)을 실제로 측정한다.** 위의 추정은 계산일 뿐이다.
4. **느린 속도의 원인 검증**: 상관을 `unfold + matmul` 또는 FFT로 구현해 시간이 줄어드는지 본다.
5. **팀 공유 문서 업데이트**: `팀공유_이준파트_단독.md`(이준이 한 것만)에 오늘의 풀링·CF·연산량·백본 결과를 반영한다.

### 6-2. 시간이 남으면
- 어두운 조건에서 블록 안 오차(19%)가 가장자리에서만 나는지, 어디서나 나는지 구분(`trace_dense`로 위치별 오차 분석).
- 어두운 영상 전처리(밝기 보정·노이즈 제거)를 VGG 앞에 넣어 L2 회복을 본다(학습 없음, 셀 하나). 단 입력 쪽 보정이라 CSMG 대체와는 결이 다르다.
- 다른 낯선 도시(Paris, London 등)와 궤적 수를 늘려 신뢰구간을 좁힌다.
- 3D 시점 항법 여부를 교수님께 확인한다(지금까지 모두 2D, 희윤의 3D 평가에서 B2가 62~89%로 내려갔다).
- heading을 모를 때(16각도 탐색, 26배 느림)의 성공률.
- 물리 기반 저조도 합성(현재는 단순 8bit).

### 6-3. 희윤에게 확인
- Tokyo baseline에 쓴 궤적 세트(위 4-4).
- L2/L3 닫힌 루프 결과가 있는지.
- 풀링 계열(`diag_gluf.py`의 GAP/GeM)을 어떻게 시험했고 결과가 어땠는지(중복 확인용).
- dense 연산 비용을 누가 재는지(이제 이준이 MAC을 쟀으니 공유 가능).

### 6-4. 교수님께 여쭐 것
1. dense 같은 특징 지도 매칭이 드론 탑재 조건에서 허용되는 연산량·저장량인가?
2. 낯선 도시 블록 선택은 dense로 풀리는데, 어두운 조건에서 무너지는 부분을 다음 hidden problem으로 잡아도 되는가?
3. 이 단계에서 3D 시점이나 실제 야간 영상 검증이 필요한가, 2D 시뮬레이션으로 충분한가?

---

## 7. 코드 (이번 세션에서 만든 것)

### 7-1. 어두움 래퍼
```python
from diag.diag_modules import darken, DARK
class DarkWrap:
    """loc 앞에서 UAV 크롭만 어둡게 (위성 타일은 그대로)"""
    def __init__(self, loc, lvl, seed=0):
        self.loc, self.lvl, self.rng = loc, lvl, np.random.default_rng(seed)
    def __call__(self, crops, headings, blocks):
        return self.loc([darken(c, self.lvl, self.rng) for c in crops], headings, blocks)
```
사용: `nav_run(DarkWrap(loc, 'L2', 0), img, starts, goals, decode_px=128)`. 이전 L2 결과와 맞추려면 시드 0.

### 7-2. PoolLoc (풀링 dense)
```python
class PoolLoc(DenseLoc):
    def __init__(self, m, img, s):
        super().__init__(m, img, compass=True); self.s = s
    def _pool(self, x): return F.avg_pool2d(x, self.s, self.s) if self.s > 1 else x
    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.c:
            sx, sy = bx*256, by*256
            Bf = self.m.backbone(to_tensor(self.img[sy:sy+512, sx:sx+512]).unsqueeze(0).to(DEV))
            Bf = self._pool(Bf)[0]; mu = Bf.mean((1, 2), keepdim=True)
            self.c[(bx, by)] = (F.normalize(Bf - mu, dim=0), mu)
        return self.c[(bx, by)]
    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        tpl = torch.stack([to_tensor(rot_tpl(c, h)) for c, h in zip(crops, headings)])
        Uf = torch.cat([self.m.backbone(tpl[k:k+64].to(DEV)) for k in range(0, len(tpl), 64)])
        Uf = self._pool(Uf); hp = Uf.shape[-1]; out = []
        for j, (bx, by) in enumerate(blocks):
            Bf, mu = self.block(bx, by)
            U = F.normalize(Uf[j].unsqueeze(0) - mu.unsqueeze(0), dim=1)
            corr = F.conv2d(Bf.unsqueeze(0), U)[0] / (hp*hp)
            _, Hc, Wc = corr.shape
            iy, ix = divmod(int(corr.flatten().argmax()) % (Hc*Wc), Wc)
            stride = 8 * self.s; teff = hp * stride
            out.append(((ix*stride + teff/2 - 256)/128, (iy*stride + teff/2 - 256)/128))
        return np.array(out)
```

### 7-3. CoarseFineLoc
```python
class CoarseFineLoc(DenseLoc):
    def __init__(self, m, img, s, r):
        super().__init__(m, img, compass=True); self.s, self.r = s, r
    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.c:
            sx, sy = bx*256, by*256
            raw = self.m.backbone(to_tensor(self.img[sy:sy+512, sx:sx+512]).unsqueeze(0).to(DEV))
            mu_f = raw[0].mean((1, 2), keepdim=True); fine = F.normalize(raw[0] - mu_f, dim=0)
            rc = F.avg_pool2d(raw, self.s, self.s)[0]
            mu_c = rc.mean((1, 2), keepdim=True); coarse = F.normalize(rc - mu_c, dim=0)
            self.c[(bx, by)] = (fine, mu_f, coarse, mu_c)
        return self.c[(bx, by)]
    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        tpl = torch.stack([to_tensor(rot_tpl(c, h)) for c, h in zip(crops, headings)])
        Uf = torch.cat([self.m.backbone(tpl[k:k+64].to(DEV)) for k in range(0, len(tpl), 64)])
        Uc = F.avg_pool2d(Uf, self.s, self.s)
        h = Uf.shape[-1]; hp = Uc.shape[-1]; s, r = self.s, self.r; out = []
        for j, (bx, by) in enumerate(blocks):
            fine, mu_f, coarse, mu_c = self.block(bx, by)
            Uj = F.normalize(Uc[j].unsqueeze(0) - mu_c.unsqueeze(0), dim=1)
            cc = F.conv2d(coarse.unsqueeze(0), Uj)[0] / (hp*hp)
            _, Hc, Wc = cc.shape
            iy, ix = divmod(int(cc.flatten().argmax()) % (Hc*Wc), Wc)
            H = fine.shape[-1]
            x0 = min(max(0, ix*s - r), H - h); x1 = min(H - h, ix*s + r)
            y0 = min(max(0, iy*s - r), H - h); y1 = min(H - h, iy*s + r)
            patch = fine[:, y0:y1 + h, x0:x1 + h].unsqueeze(0)
            Uj = F.normalize(Uf[j].unsqueeze(0) - mu_f.unsqueeze(0), dim=1)
            cf = F.conv2d(patch, Uj)[0] / (h*h)
            _, Hf, Wf = cf.shape
            fy, fx = divmod(int(cf.flatten().argmax()) % (Hf*Wf), Wf)
            fx += x0; fy += y0
            out.append(((fx*8 + h*8/2 - 256)/128, (fy*8 + h*8/2 - 256)/128))
        return np.array(out)
```

### 7-4. FeatLoc (백본 교체) + 후보
```python
import torchvision
class FeatLoc:
    def __init__(self, fe, st, img): self.fe, self.st, self.img, self.c = fe, st, img, {}
    @torch.no_grad()
    def block(self, bx, by):
        if (bx, by) not in self.c:
            sx, sy = bx*256, by*256
            B = self.fe(to_tensor(self.img[sy:sy+512, sx:sx+512]).unsqueeze(0).to(DEV))[0]
            mu = B.mean((1, 2), keepdim=True)
            self.c[(bx, by)] = (F.normalize(B - mu, dim=0), mu)
        return self.c[(bx, by)]
    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        tpl = torch.stack([to_tensor(rot_tpl(c, h)) for c, h in zip(crops, headings)])
        Uf = torch.cat([self.fe(tpl[k:k+64].to(DEV)) for k in range(0, len(tpl), 64)])
        hp = Uf.shape[-1]; out = []
        for j, (bx, by) in enumerate(blocks):
            Bf, mu = self.block(bx, by)
            U = F.normalize(Uf[j].unsqueeze(0) - mu.unsqueeze(0), dim=1)
            corr = F.conv2d(Bf.unsqueeze(0), U)[0] / (hp*hp)
            _, Hc, Wc = corr.shape
            iy, ix = divmod(int(corr.flatten().argmax()) % (Hc*Wc), Wc)
            out.append(((ix*self.st + hp*self.st/2 - 256)/128, (iy*self.st + hp*self.st/2 - 256)/128))
        return np.array(out)

def resnet_fe(name, upto):
    r = (torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) if name == 'resnet18'
         else torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)).eval().to(DEV)
    layers = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2] + ([r.layer3] if upto == 3 else [])
    return torch.nn.Sequential(*layers).eval()
# 후보: vgg_conv3_3 = m2.backbone[:16] (st4), resnet18 layer2 (st8), resnet18 layer3 (st16), resnet50 layer2 (st8)
```
메모리 주의: `vgg_conv3_3`은 블록 특징 지도가 커서(256×128×128) 블록 캐시가 GPU 메모리를 많이 쓴다. 실행마다 `del loc; torch.cuda.empty_cache()`.

### 7-5. 연산량 측정
```python
from torch.utils.flop_counter import FlopCounterMode
def gmacs(fn):
    with FlopCounterMode(display=False) as fc, torch.no_grad(): fn()
    return fc.get_total_flops() / 2 / 1e9
```
합성곱·행렬곱 위주로 세므로 풀링·정규화 같은 작은 연산은 빠진다.

---

## 8. 만들어 둔 문서 (`/mnt/user-data/outputs/`)

| 파일 | 상태 |
|---|---|
| `팀공유_이준파트_정리.md` | 전체 버전 (희윤 정정 반영: dense = baseline, 3D 수치 추가). 오늘의 풀링·CF·연산량 결과는 **미반영** |
| `팀공유_이준파트_단독.md` | 이준이 직접 한 것만. 오늘의 새 실험 **미반영** |
| `이준_연구정리_v2.md` | 교수님 피드백(10/1) 후 정리. 일부 구식 |
| `교수님_미팅자료_팀_전체정리.md` | 4명 아이디어 한 문서 (지민 수치는 예상치로 표기) |
| `문제재정의_및_역할분담.md` | 일부 구식 (희윤과의 합의 후 변경됨) |
| `팀회의_방향비교_및_논의사항.md`, `진행상황_및_다음계획_공유.md`, `이준_아이디에이션_장거리위치추정.md`, `GPS-free_드론_위치추정_연구_제안.md` | 초기 문서 |
| `이준_연구정리.md` | **폐기** (VO + confidence 융합) |
| `diag_modules.py`, `diag_nav.py`, `diag_gluf.py`, `diag_phase0.py` | 희윤의 코드 그대로 복사 |

사용자 Drive `bearinguav_cache/`의 결과 파일: `nav_result_Tokyo.txt`, `nav_result_NY.txt`, `nav_dense_NY.txt`, `nav_long_NY.txt`, `nav_dark_NY.txt`, `nav_peak_NY.txt`, `nav_cond_NY.txt`, `nav_pool_NY.txt`, `nav_pool_NY_big.txt`, `nav_pool_dark_NY.txt`, `nav_cf_NY.txt`, `nav_backbone_NY.txt`(대기).
재현 시드: 궤적 2026(평가), 7(임계값 결정), 분류기 학습 rng 0, 어두움 노이즈 0.

---

## 9. 이 프로젝트의 한계 (발표나 문서에서 반드시 밝힐 것)
- 2D 위성 크롭 + 나침반 heading 가정. 3D 시점 항법은 하지 않았다(2D는 상한선).
- 낯선 도시는 NY 하나, 궤적 16~64개.
- 어두움은 단순 합성(8bit)이고 위성 타일은 밝은 채 UAV 크롭만 어둡다.
- 긴 경로는 같은 1 km² 지도 안을 돌아다닌 것이다.
- 속도 측정은 Colab GPU의 상대 비교이며 구현 특성이 섞여 있다. 드론 하드웨어 수치가 아니다.
- 조건부 재탐색은 임계값 0.269 하나만 시험했다.
- dense 매칭과 항법 시뮬레이터, 모듈별 진단은 **희윤의 작업**이다.
