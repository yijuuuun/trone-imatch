# R1 경량화 실험 결과 및 재현 가이드

## 1. 실험 목적

본 실험은 LoMa 기반 영상 매칭을 활용하는 항법 시스템의 처리 효율을 개선하기 위해 수행하였다.

주요 목표는 지도 특징 재사용과 PyTorch의 `torch.compile`을 적용하여 모델 실행 과정의 효율을 개선할 수 있는지 확인하는 것이다.

이번 실험은 다음 두 가지 방향으로 진행하였다.

1. **Map Feature Cache:** 지도 영상의 특징점을 미리 계산하여 반복적인 특징 추출 과정을 줄일 수 있는지 확인한다.
2. **Torch Compile 적용:** LoMa-Cache 및 일반 LoMa 모델에 `torch.compile`을 각각 적용하고 처리 시간과 항법 결과를 비교한다.

이 중 Torch Compile 적용을 핵심 실험으로 설정하고, Map Feature Cache는 추가적인 경량화 가능성을 확인하기 위한 보조 실험으로 다루었다.

## 2. 공통 실험 환경

| 항목 | 설정 |
|---|---|
| 실행 환경 | Kaggle Notebook |
| GPU | NVIDIA Tesla T4 |
| 기본 매칭 모델 | LoMa |
| 항법 평가 버전 | R1 |
| 평가 구간 | 92개 |
| 전체 걸음 수 | 1,020개 |
| 방향 탐색 | 4방향 |
| 최소 Inlier 조건 | 15 |
| Odometry | VIO |
| Random Seed | 2026 |

모든 GPU 성능 측정은 Kaggle 환경에서 수행하였다. Windows 로컬 환경은 코드 수정 및 실행 준비에 사용하고 GPU 성능 측정에는 사용하지 않았다.

## 3. Map Feature Cache 실험

### 3.1 실험 목적

Map Feature Cache는 동일한 지도 영상에 대한 특징 추출을 반복하지 않도록 미리 계산한 특징 정보를 재사용하는 방식이다.

이를 통해 지도 특징 추출 과정에서 발생하는 반복 연산을 줄이고 항법 처리 시간을 단축할 수 있는지 확인하였다.

### 3.2 실험 결과

| 평가 항목 | 결과 |
|---|---:|
| 평가 구간 | 92개 |
| 전체 걸음 수 | 1,020개 |
| 10m 정답률 | 18.7% |
| 20m 정답률 | 50.0% |
| 32m 정답률 | 77.4% |
| 걸음 오차 중앙값 | 20.0m |
| 마지막 오차 중앙값 | 20.8m |
| 마지막 40m 이내 | 75.0% |
| 모름(Unknown) | 48.7% |
| 블록 밖 | 32.5% |
| 걸음당 처리 시간 | 0.53초 |

Map Feature Cache를 적용한 조건에서 걸음당 처리 시간은 0.53초로 측정되었다.

해당 결과는 지도 특징 재사용에 따른 처리 시간 변화를 확인하기 위한 별도 실험 결과이다.

## 4. Torch Compile 적용 실험

Torch Compile 실험은 기존 LoMa-Cache 모델에 적용한 실험과 일반 LoMa 모델에 적용한 실험으로 구분하였다.

### 4.1 LoMa-Cache + Torch Compile

#### (1) 실험 목적

기존 LoMa-Cache 모델에 PyTorch의 `torch.compile`을 적용하여 반복적인 모델 연산의 실행 효율을 개선할 수 있는지 확인하였다.

Map Feature Cache와 Torch Compile을 함께 사용하는 조건에서 항법 결과와 처리 시간을 측정하고, 동일한 조건으로 3회 반복하여 실행 시간의 변동 폭을 확인하였다.

#### (2) 실험 설정

| 항목 | 설정 |
|---|---|
| Matcher | loma-cache |
| Torch Compile | 적용 |
| Map Feature Cache | 적용 |
| 평가 버전 | R1 |
| 반복 횟수 | 3회 |

#### (3) 코드 및 실행 구성

LoMa-Cache 컴파일 실험은 기존 Cache 기반 매칭 경로를 유지하면서 Torch Compile을 적용한 별도 실험이다.

실험 결과를 일반 LoMa 컴파일 결과와 구분하여 관리하였으며, 두 실험의 모델 구성과 Cache 적용 여부가 다르므로 결과를 하나로 합산하지 않았다.

세 회차(`r1-light-cache-compile_run1`, `_run2`, `-run3`)의 `src/imatch/matchers.py`는 `fc` 명령으로 비교하여 서로 동일함을 확인하였다. 따라서 세 회차는 같은 코드로 실행된 것으로 보고, 코드는 1회차(`r1-light-cache-compile_run1`)의 `src/`만 보관한다.

※ LoMa-Cache 컴파일을 위해 수정한 원본 파일과 구체적인 코드 변경 사항은 해당 실험의 최종 소스 파일을 기준으로 별도 확인해야 한다. 일반 LoMa에 적용한 코드와 동일한 수정이었다고 가정하지 않는다.

#### (4) 실험 결과

| 평가 항목 | 1회차 | 2회차 | 3회차 |
|---|---:|---:|---:|
| 10m 정답률 | 18.5% | 18.5% | 18.5% |
| 20m 정답률 | 49.2% | 49.2% | 49.2% |
| 32m 정답률 | 75.8% | 75.8% | 75.8% |
| 걸음 오차 중앙값 | 20.4m | 20.4m | 20.4m |
| 마지막 오차 중앙값 | 21.3m | 21.3m | 21.3m |
| 마지막 40m 이내 | 71.7% | 71.7% | 71.7% |
| 모름(Unknown) | 49.7% | 49.7% | 49.7% |
| 블록 밖 | 33.4% | 33.4% | 33.4% |
| 걸음당 처리 시간 | 0.47초 | 0.52초 | 0.48초 |

#### (5) 반복 측정 분석

| 처리 시간 통계 | 결과 |
|---|---:|
| 평균 | 0.49초 |
| 중앙값 | 0.48초 |
| 최솟값 | 0.47초 |
| 최댓값 | 0.52초 |

3회 반복 측정에서 정확도 관련 지표는 동일하게 나타났다. 걸음당 처리 시간은 평균 0.49초로 측정되었으며, 최소 0.47초에서 최대 0.52초 사이의 변동을 보였다.

이를 통해 동일한 항법 평가 조건에서 결과의 일관성을 확인하였다.

### 4.2 일반 LoMa + Torch Compile

#### (1) 실험 목적

LoMa-Cache가 아닌 일반 LoMa 모델에도 `torch.compile`을 적용하여 컴파일 기능의 적용 가능성과 항법 처리 결과를 확인하였다.

기존 LoMa-Cache 컴파일 실험과 구분하기 위해 별도의 커널을 생성하였으며, Map Feature Cache를 적용하지 않은 상태에서 측정하였다.

#### (2) 코드 수정 내용

일반 LoMa 컴파일 적용을 위해 `imatch/matchers.py`와 `scripts/kaggle/build_kernel.py`를 수정하였다.

**① `imatch/matchers.py` 수정**

기존 코드에서는 `LOMA_FP32` 환경변수가 설정된 경우에만 FP32 변환을 수행하도록 되어 있었다.

이와 별도로 `LOMA_COMPILE` 환경변수를 확인하는 조건문을 추가하여 일반 LoMa 모델에도 컴파일을 적용할 수 있도록 수정하였다.

추가한 코드는 다음과 같다.

```python
if name.startswith("loma") and os.environ.get("LOMA_COMPILE") == "1":
    self.m.matcher.compile()
```

코드의 동작 과정은 다음과 같다.

- `name.startswith("loma")`: 현재 Matcher 이름이 LoMa 계열인지 확인한다.
- `os.environ.get("LOMA_COMPILE") == "1"`: 컴파일 환경변수가 활성화되어 있는지 확인한다.
- `self.m.matcher.compile()`: 조건이 충족되면 실제 LoMa 매처 모델에 PyTorch Compile을 적용한다.

기존 FP32 설정 코드는 별도로 유지하였다.

```python
if name.startswith('loma') and os.environ.get('LOMA_FP32'):
    from vismatch.utils import force_float32
    force_float32(self.m.matcher)
```

따라서 FP32 강제 변환 여부와 Torch Compile 적용 여부를 서로 독립적으로 제어할 수 있도록 구성하였다.

이번 일반 LoMa 컴파일 실험에서는 `LOMA_COMPILE=1`을 설정하고 `LOMA_FP32`는 별도로 설정하지 않았다.

**② `scripts/kaggle/build_kernel.py` 수정**

커널을 생성할 때 실행되는 Launcher 코드에 컴파일 환경변수를 추가하였다.

`LAUNCHER = r'''` 내부의 import 문 다음에 아래 코드를 추가하였다.

```python
os.environ["LOMA_COMPILE"] = "1"
```

이렇게 수정한 이유는 로컬에서 커널을 생성하더라도 Kaggle에서 실제 실행되는 `run.py`에 컴파일 환경변수가 포함되도록 하기 위해서이다.

환경변수가 생성된 실행 파일에 포함되지 않으면 `matchers.py`의 컴파일 조건문이 실행되지 않을 수 있다.

**③ 생성된 `run.py` 확인**

커널 생성 후 실제 실행 파일에도 환경변수가 들어갔는지 확인하였다.

확인 명령:

```cmd
findstr /n /i "LOMA_COMPILE" kaggle_kernels\r1-light-loma-compile\run.py
```

확인 결과:

```text
6:os.environ["LOMA_COMPILE"] = "1"
```

이를 통해 Kaggle에서 실행되는 `run.py`에 컴파일 환경변수가 포함된 것을 확인하였다.

**④ 문법 검사**

수정한 매처 파일의 Python 문법을 확인하였다.

```cmd
python -m py_compile imatch\matchers.py
```

문법 검사가 정상적으로 완료되었다.

2회차(`r1-light-loma-compile-run2`)와 3회차(`r1-light-loma-compile-run3`)의 `src/imatch/matchers.py`는 `fc` 명령으로 비교하여 서로 동일함을 확인하였다. 따라서 두 회차는 같은 코드로 실행된 것으로 보고, 코드는 2회차(`r1-light-loma-compile-run2`)의 `src/`를 기준으로 한다.

#### (3) 실험 설정

| 항목 | 설정 |
|---|---|
| Matcher | loma |
| Torch Compile | 적용 |
| 환경변수 | `LOMA_COMPILE=1` |
| FP32 강제 설정 | 미적용 |
| Map Feature Cache | 미적용 |
| 방향 탐색 | 4방향 |
| 최소 Inlier 조건 | 15 |
| 평가 버전 | R1 |
| Odometry | VIO |
| Random Seed | 2026 |
| 반복 횟수 | 3회 |

#### (4) 커널 생성 및 실행 명령

**① CMD에서 프로젝트 폴더로 이동**

```cmd
cd C:\Users\user\Downloads\trone-imatch-main(1118)\trone-imatch-main
```

**② 커널 생성**

```cmd
python scripts/kaggle/build_kernel.py --user jimin562 --dataset heeyunju/visloc-seq-eval --slug r1-light-loma-compile --args "--matcher loma --angles 4 --stop-inl 15 --variants r1 --odo vio --seed 2026" --script run_visloc_nav
```

**③ Kaggle 커널 업로드**

```cmd
kaggle kernels push -p kaggle_kernels/r1-light-loma-compile --accelerator NvidiaTeslaT4
```

**④ 실험 결과 다운로드**

1회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run1 --file-pattern "out/.*"
```

2회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run2 --file-pattern "out/.*"
```

3회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run3 --file-pattern "out/.*"
```

**⑤ 실행 코드 다운로드**

실험별 `src` 파일도 별도로 보관하였다.

1회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run1 --file-pattern "src/.*"
```

2회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run2 --file-pattern "src/.*"
```

3회차:

```cmd
kaggle kernels output jimin562/r1-light-loma-compile -p outputs/r1-light-loma-compile-run3 --file-pattern "src/.*"
```

#### (5) 실험 결과

3회 반복 측정 결과는 다음과 같다.

| 평가 항목 | 1회차 | 2회차 | 3회차 |
|---|---:|---:|---:|
| 평가 구간 | 92 | 92 | 92 |
| 전체 걸음 수 | 1,020 | 1,020 | 1,020 |
| 10m 정답률 | 18.5% | 18.5% | 18.5% |
| 20m 정답률 | 49.2% | 49.2% | 49.2% |
| 32m 정답률 | 75.8% | 75.8% | 75.8% |
| 걸음 오차 중앙값 | 20.4m | 20.4m | 20.4m |
| 마지막 오차 중앙값 | 21.3m | 21.3m | 21.3m |
| 마지막 40m 이내 | 71.7% | 71.7% | 71.7% |
| 모름(Unknown) | 49.7% | 49.7% | 49.7% |
| 블록 밖 | 33.4% | 33.4% | 33.4% |
| 걸음당 처리 시간 | 0.70초 | 0.74초 | 0.72초 |

#### (6) 반복 측정 분석

| 처리 시간 통계 | 결과 |
|---|---:|
| 평균 | 0.72초 |
| 중앙값 | 0.72초 |
| 최솟값 | 0.70초 |
| 최댓값 | 0.74초 |

세 회차에서 정확도 관련 지표는 모두 동일하게 나타났다. 걸음당 처리 시간은 평균 0.72초로 측정되었으며, 최솟값과 최댓값의 차이는 0.04초였다.

이를 통해 일반 LoMa에 Torch Compile을 적용한 조건에서 반복 측정 결과의 일관성을 확인하였다.

#### (7) 실행 로그 및 주의 사항

실행 중 다음과 같은 경고가 발생하였다.

- `Not enough SMs to use max_autotune_gemm mode`
- `TRITON_DEFAULT_FP_FUSION is unset`

해당 메시지는 컴파일 과정에서 출력된 경고이며, 세 회차의 평가 결과는 정상적으로 출력되었다.

이번 실험에서는 `LOMA_COMPILE=1`을 적용하였으며, `LOMA_FP32`는 별도로 설정하지 않았다. 따라서 FP32 강제 변환을 적용한 실험으로 해석해서는 안 된다.

## 5. 실험 종합 비교

| 실험 | 적용 방식 | 걸음당 처리 시간 | 비고 |
|---|---|---:|---|
| Map Feature Cache | 지도 특징 재사용 | 0.53초 | 보조 실험 |
| LoMa-Cache + Torch Compile | Cache 모델 컴파일 | 평균 0.49초 | 3회 측정 완료 |
| 일반 LoMa + Torch Compile | 일반 모델 컴파일 | 평균 0.72초 | 3회 측정 완료 |

※ 각 실험은 모델 구성과 적용한 최적화 방식이 다르므로 처리 시간만으로 직접적인 성능 우열을 확정하지 않는다.

## 6. 종합 결론

이번 R1 경량화 실험에서는 LoMa 기반 항법 시스템의 처리 효율을 개선하기 위해 Map Feature Cache와 Torch Compile을 각각 시도하였다.

핵심 실험인 Torch Compile은 LoMa-Cache 모델과 일반 LoMa 모델에 각각 적용하여 구분해서 평가하였다.

LoMa-Cache + Torch Compile 실험에서는 3회 반복 측정 결과 걸음당 처리 시간이 평균 0.49초로 나타났으며, 정확도 관련 지표는 세 회차에서 동일하게 유지되었다.

일반 LoMa + Torch Compile 실험에서도 3회 반복 측정을 완료하였다. 정확도 관련 지표는 모든 회차에서 동일했으며, 걸음당 처리 시간은 평균 0.72초로 측정되었다. 최솟값은 0.70초, 최댓값은 0.74초였다.

일반 LoMa 컴파일 실험에서는 `imatch/matchers.py`에 컴파일 조건문을 추가하고, `scripts/kaggle/build_kernel.py`의 Launcher에 환경변수 설정을 추가하였다. 생성된 `run.py`에서도 환경변수가 포함되었는지 확인하여 실제 Kaggle 실행 환경에서 컴파일 설정이 적용되도록 하였다.

Map Feature Cache는 지도 특징 추출의 반복을 줄이기 위한 별도 최적화 방법으로 확인하였다.

따라서 이번 실험의 핵심은 **LoMa-Cache와 일반 LoMa에 각각 Torch Compile을 적용하고, 반복 측정을 통해 모델 구성별 처리 시간과 항법 결과의 일관성을 확인한 것**이다.

## 7. 결과 파일 및 재현 자료

실험별 결과와 실행 코드는 `2026-11-20_r1-light/` 폴더 아래에 실험별로 구분하여 보관한다. 컴파일 실험은 회차별로 폴더를 나누었다. 이 문서는 해당 폴더의 `README.md`이다.

`out/`에는 평가 결과(`*_segs.csv`: 구간별, `*_steps.csv`: 걸음별), `src/`에는 실행 당시의 코드를 보관한다. 파이썬이 자동 생성하는 `__pycache__` 폴더는 아래 구조에서 생략하였다.

```text
2026-11-20_r1-light/
├── README.md
│
├── r1-light-cache/                          # Map Feature Cache
│   ├── r1-light-cache.log
│   ├── out/
│   │   ├── visloc_loma-cache-r1_segs.csv
│   │   └── visloc_loma-cache-r1_steps.csv
│   └── src/
│       ├── imatch/                          # blocks, geometry, matchers, metrics, nav, pipeline, replay
│       └── scripts/                         # run, run_nav, run_nav_b1, run_visloc_nav
│
├── r1-light-cache-compile_run1/             # LoMa-Cache + Torch Compile 1회차
│   ├── out/
│   │   ├── visloc_loma-cache-compile-r1_segs.csv
│   │   └── visloc_loma-cache-compile-r1_steps.csv
│   └── src/
│       ├── imatch/
│       └── scripts/
│
├── r1-light-cache-compile_run2/             # LoMa-Cache + Torch Compile 2회차
│   ├── r1-light-cache-compile.log
│   └── out/
│       ├── visloc_loma-cache-r1_segs.csv
│       └── visloc_loma-cache-r1_steps.csv
│
├── r1-light-cache-compile-run3/             # LoMa-Cache + Torch Compile 3회차
│   ├── r1-light-cache-compile.log
│   └── out/
│       ├── visloc_loma-cache-r1_segs.csv
│       └── visloc_loma-cache-r1_steps.csv
│
├── r1-light-loma-compile-run1/              # 일반 LoMa + Torch Compile 1회차
│   ├── r1-light-loma-compile.log
│   ├── out/
│   │   ├── visloc_loma-r1_segs.csv
│   │   └── visloc_loma-r1_steps.csv
│   └── source/
│       └── r1-light-loma-compile.py
│
├── r1-light-loma-compile-run2/              # 일반 LoMa + Torch Compile 2회차
│   ├── r1-light-loma-compile.log
│   ├── out/
│   │   ├── visloc_loma-r1_segs.csv
│   │   └── visloc_loma-r1_steps.csv
│   └── src/
│       ├── imatch/
│       └── scripts/
│
└── r1-light-loma-compile-run3/              # 일반 LoMa + Torch Compile 3회차
    ├── r1-light-loma-compile.log
    └── out/
        ├── visloc_loma-r1_segs.csv
        └── visloc_loma-r1_steps.csv
```
