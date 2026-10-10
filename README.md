# Bearing-UAV: CSMG → Grid Pooling 실험

Trone 팀 · 작성 이준 · 2026-10-09 (10/10 업데이트: 대표 모델을 8×8로 변경)

Bearing-UAV의 CSMG 모듈을 **Grid Pooling**(특징 지도를 N×N 칸으로 나눠 칸마다 평균을 내는 방식)으로 바꾸는 실험 코드입니다. 학습, 미학습 도시 1 km 닫힌 루프 항법 평가, 실험 환경 설치를 모두 포함합니다. 컴퓨터 종류(NVIDIA GPU가 있는 PC, Mac, Colab)와 관계없이 같은 명령으로 실행할 수 있습니다.

## 지금까지의 결과

미학습 도시 1 km 항법 성공률입니다(NY 96 + Paris 32 + London 32 = 160궤적, 모두 4 epoch, 시드 0).

| 모델 | NY | Paris | London | **합계** |
| --- | --- | --- | --- | --- |
| 공개 가중치 (원래 Bearing, CSMG, 100 ep) | 32.3% | 37.5% | 28.1% | 32.5% |
| CSMG 재학습 (같은 설정) | 24.0% | 21.9% | 21.9% | 23.1% |
| Grid 1×1 (`grid1`) | 21.9% | 28.1% | 37.5% | 26.3% |
| Grid 2×2 (`grid2`) | 34.4% | 37.5% | 34.4% | 35.0% |
| Grid 4×4 (`grid`) | 41.7% | 43.8% | 37.5% | 41.3% |
| **Grid 8×8 (`grid8`)** | **46.9%** | **59.4%** | **65.6%** | **53.1%** |

- **칸이 많을수록 좋아집니다.** 1×1 → 8×8로 갈수록 성공률이 단조 증가합니다.
- **대표 모델은 8×8입니다.** 원래 Bearing보다 유의하게 높고(+50/−17, p=0.00007), 4×4보다도 유의하게 높습니다(+33/−14, p=0.008). (McNemar 정확검정, 세 도시 합산)
- 학습 시드 하나로 한 번씩만 학습한 결과입니다. 아래 실험으로 검증합니다.

---

## 실험 분담 (~10/14)

각자 **자기 컴퓨터에서 학습부터 평가까지 독립적으로** 진행하고, 끝나면 결과 표를 공유합니다.

| | 질문 | 담당 |
| --- | --- | --- |
| **A** | 시드를 바꿔 다시 학습해도 Grid 8×8이 CSMG보다 나은가? | **김재홍** |
| **B** | Grid 8×8을 오래(20 epoch) 학습하면 얼마나 더 좋아지는가? | **김재홍** |
| C | 격자 칸 수(1×1, 2×2, 8×8, 16×16)에 따라 성능이 어떻게 달라지는가? (8×8까지 완료) | 이준 |
| D | 물리 카메라 모델로 만든 저조도 영상을 섞어 학습하면 어두운 조건의 성능이 좋아지는가? | 이준 |

### 공통 규칙
1. **학습 설정은 바꾸지 않습니다.** 코드 기본값(batch 64, AdamW 3e-4, 90K 전체)을 그대로 사용하고, 명령어의 **tag**와 **epoch 수**만 바꿉니다.
2. **비교는 같은 컴퓨터에서 평가한 숫자끼리만 합니다.** 컴퓨터가 다르면 같은 모델도 항법 결과가 조금 달라질 수 있습니다. 비교 대상인 기존 모델(`models/grid8.pth`, `models/grid.pth`, `models/csmg.pth`, 시드 0)이 저장소에 들어 있어서, 각자 컴퓨터에서 다시 평가할 수 있습니다.
3. 학습 중에는 **컴퓨터가 잠자기 상태로 들어가지 않게** 해 주세요(노트북은 전원 연결, 덮개 열어 두기).
4. 코드를 고쳐야 하면 직접 수정하지 말고 이준에게 알려 주세요. 수정 후 `git pull`로 받으면 됩니다.

---

## 설치 (처음 한 번)

필요한 것: **Python 3.9 이상, PyTorch, torchvision, opencv-python, pandas, git**, 디스크 여유 공간 **약 15 GB**

```
git clone https://github.com/yijuuuun/trone-grid-pooling.git
cd trone-grid-pooling
python setup_local.py
```
- **이미 설치했다면** `git pull` 후 `python setup_local.py`를 한 번 더 실행해 주세요. 새로 추가된 비교 모델(`grid8.pth`)이 복사됩니다(데이터는 다시 받지 않습니다).
- `setup_local.py`는 원 저장소 코드, 모델 가중치, 데이터셋(6.3 GB)을 내려받아 풉니다. 20분~1시간 정도 걸리며, 중간에 끊기면 다시 실행하면 이어서 받습니다.
- 설치 위치는 기본적으로 홈 폴더의 `bearing`입니다. 다른 위치에 두려면 `python setup_local.py D:\bearing`처럼 경로를 지정합니다.
- 패키지가 없다는 에러가 나면 `pip install torch torchvision opencv-python pandas`를 실행해 주세요.

**설치 확인과 속도 측정** (약 5분)
```
python run.py check
```
마지막에 `CHECK_OK`가 나오면 준비가 끝난 것입니다. 출력된 "4 epoch 약 몇 시간"을 참고해 일정을 잡아 주세요.
- `cuda`(NVIDIA GPU): CSMG 4 epoch에 1시간 안팎
- `mps`(Mac): 7~8시간이라 하루에 하나씩 밤에 실행하는 것을 권장합니다.
- `cpu`: 너무 느려서 진행이 어렵습니다. 이 경우에는 알려 주세요.

> **Colab을 사용하는 경우:** 런타임을 T4 GPU로 바꾸고 아래 명령 앞에 `!`를 붙여 실행합니다. 예: `!git clone ...`, `%cd trone-grid-pooling`, `!python setup_local.py /content/bearing`. 무료 Colab은 하루 GPU 사용 한도가 약 2시간이므로 날짜를 나눠 실행해 주세요. 런타임이 초기화되면 설치부터 다시 해야 합니다.

---

## 실험 A: 시드 반복 (김재홍, 필수)

**질문:** 시드를 바꿔 다시 학습해도 Grid 8×8이 CSMG보다 나은가?

**① 학습** (모든 명령은 저장소 폴더에서 실행)
```
python run.py train csmg_s1 4
python run.py train csmg_s2 4
python run.py train grid8_s1 4
python run.py train grid8_s2 4
```
- 소요 시간: CSMG는 `check`에서 나온 시간이고, Grid는 그보다 훨씬 빠릅니다(Mac M3 기준 CSMG 약 7시간, Grid 약 3시간).

**② 평가** (시드 0 기존 모델까지 함께)
```
python run.py eval csmg csmg_s1 csmg_s2 grid8 grid8_s1 grid8_s2
```
- 공개 가중치(원래 Bearing)도 자동으로 함께 평가됩니다.
- 결과 파일: `bearing/out/eval_more_csmg_csmg_s1_csmg_s2_grid8_grid8_s1_grid8_s2.txt`

**③ 공유할 것:** 위 결과 파일과 아래 표

| 모델 | 시드 0 | 시드 1 | 시드 2 | 평균 |
| --- | --- | --- | --- | --- |
| 공개 가중치 | | — | — | — |
| CSMG 재학습 | | | | |
| Grid 8×8 | | | | |

- 각 칸에는 결과 파일의 **NY, Paris, London 성공 수를 더해 160으로 나눈 값**을 적어 주세요.
- 같은 시드끼리 Grid 8×8 대 CSMG의 McNemar 결과(`vs csmg +a/-b p=...`)도 함께 적어 주세요.

## 실험 B: 긴 학습 (김재홍, 권장, 실험 A 이후)

**질문:** Grid 8×8을 20 epoch 학습하면 4 epoch(53.1%)보다 얼마나 더 좋아지는가? 원래 Bearing(100 epoch, 32.5%)과의 차이는 얼마나 벌어지는가?

```
python run.py train grid8 20
python run.py eval grid8 grid8_e20
```
- 4 epoch이 아닌 경우 저장 이름에 epoch 수가 자동으로 붙어(`grid8_e20.pth`) 기존 모델을 덮어쓰지 않습니다.
- 소요 시간은 Grid 4 epoch의 약 5배입니다. 너무 오래 걸리면 10 epoch(`python run.py train grid8 10`, 저장 이름 `grid8_e10`)으로 진행해도 됩니다.

**공유할 것:** 결과 파일과 한 줄 요약 (예: "grid8_e20 합계 OO% vs grid8 OO% vs 공개 가중치 OO%, +a/-b p=...")

## 실험 C, D (이준)

```
# C: 격자 칸 수
python run.py train grid1 4      # 1×1 (전역 평균)
python run.py train grid2 4      # 2×2
python run.py train grid8 4      # 8×8
python run.py train grid16 4     # 16×16
python run.py eval grid1 grid2 grid grid8
python run.py eval grid grid8 grid16

# D: 저조도 학습 (학습 드론 영상의 50%를 물리 카메라 모델로 어둡게)
python run.py train grid_phys 4
python run.py train csmg_phys 4  # 대조군
python run.py eval --phys iso_64 grid csmg grid_phys csmg_phys
python run.py eval --phys auto_256 grid csmg grid_phys csmg_phys
```

- 저조도 영상은 **주희윤님의 물리 카메라 모델**(`diag/diag_lowlight.py`의 `capture()`)로 만듭니다. 센서 신호로 되돌린 뒤 노출·게인에 따른 광자 노이즈, 읽기 노이즈, 장노출 모션 블러, 12비트 변환을 거칩니다.
- 학습 시 빛의 양과 방식은 희윤님 `lowlight_train` 데이터셋과 같은 분포입니다(빛 1/400~1/8 로그 균등, iso/exposure/auto 균등). 데이터셋을 내려받지 않고 학습 중에 바로 만듭니다.
- 평가 조건 표기(`iso_64` = iso 방식, 빛 1/64)는 희윤님 `lowlight_eval`의 9조건(iso/exposure/auto × 1/16, 1/64, 1/256)과 같습니다.

---

## 일정 (김재홍)

| 날짜 | 공유할 것 |
| --- | --- |
| **10/11 (일)** | `python run.py check` 출력 (장치, 예상 시간) |
| **10/13 (화)** | 실험 A 결과 |
| **10/14 (수)** | 실험 B 결과 (시간이 부족하면 10 epoch으로 하거나 생략 가능) |


- 결과 파일(`eval_more_*.txt`)과 위의 표를 공유해 주세요.
- 학습 로그(`train_*.txt`)도 함께 부탁드립니다. 마지막 줄이 `# done ...`이면 정상 종료입니다.
- 에러가 나면 터미널 출력의 마지막 20줄을 보내 주세요.

---

## 참고

### tag 규칙
| tag | 의미 |
| --- | --- |
| `csmg` | 원래 CSMG를 같은 설정으로 재학습 (대조군) |
| `grid` | 4×4 Grid Pooling (제안 방법) |
| `gridN` | N×N Grid (`grid1`, `grid2`, `grid8`). 출력 크기 1024 유지 (채널 = 1024 / N²) |
| `grid_north` | 나침반 방향으로 북쪽 정렬 후 4×4 Grid |
| `_phys` | 학습 드론 영상의 50%를 물리 카메라 모델로 어둡게 (희윤님 `capture()`) |
| `_dark` | 학습 드론 영상의 50%를 단순 합성(L1~L3)으로 어둡게 (이전 방식) |
| `_sK` | 학습 시드 K (예: `grid_s1`). 없으면 시드 0 |
| `_eN` | N epoch 학습 (4가 아니면 자동으로 붙음, 예: `grid_e20`) |

### 명령 요약
| 명령 | 하는 일 |
| --- | --- |
| `python setup_local.py [설치위치]` | 환경 설치 (원 저장소, 가중치, 데이터셋) |
| `python run.py check` | 설치 확인과 속도 측정 |
| `python run.py train <tag> <epoch>` | 학습. 결과는 `bearing/out/<tag>.pth`, `train_<tag>.txt` |
| `python run.py eval [--l2 \| --phys <방식>_<d>] <tag> ...` | 미학습 도시 3곳(NY 96, Paris 32, London 32궤적) 1 km 항법 평가. 공개 가중치가 자동으로 포함됩니다. `--l2`는 단순 합성 어두움, `--phys iso_64`는 물리 카메라 모델(iso, 빛 1/64) |

### 파일 구성
| 파일 | 내용 |
| --- | --- |
| `swap_lib.py` | Grid Pooling 모듈, 모델 구성, 항법용 위치추정기 |
| `train_swap.py` | 학습 (VGG16 고정, 나머지 새로 초기화) |
| `eval_more.py` | 미학습 도시 항법 평가 |
| `eval_swap.py` | NY 주간·L2, Tokyo 평가 (32궤적) |
| `setup_local.py`, `run.py` | 설치와 실행 |
| `models/` | 기존 모델 `grid.pth`, `csmg.pth` (시드 0, 비교 대상) |
| `diag/` | **주희윤님의 코드** (항법 시뮬레이터, 진단, 저조도 카메라 모델). 수정하지 않습니다 |
| `legacy/` | 이전 Colab 전용 진단 스크립트 (참고용) |

### 출처
- 원 논문 코드: https://github.com/liukejia121/bearinguav
- 데이터와 가중치: Hugging Face `HaoyZhou/bearinguav`
- 저조도 카메라 모델: 주희윤님 `gmldbs3014/bearing-uav-lowlight`의 `code/diag_lowlight.py`
