# 팀 저장소에서 재현하기

이 문서는 완료된 방법 B 실험을 재현하는 절차다. 제출용 CSV 분석에는 GPU나 이미지·가중치가 필요하지 않다. 모든 명령은 팀 저장소 루트에서 실행한다.

## 환경

기존 CUDA Python이 있으면 우선 사용한다. 새 환경이 필요하면 시스템 Python을 바꾸지 않고 아래처럼 별도로 만든다. 설치 명령을 기존 환경에 무조건 반복 실행하지 않는다.

```bash
python3.11 -m venv .venv-e3-e4
P="$PWD/.venv-e3-e4/bin/python"
E=experiments/2026-11-06_scale_ransac
"$P" -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
"$P" -m pip install -r "$E/requirements-loma-runtime.txt"
"$P" -m pip install --no-deps vismatch==1.4.0
"$P" "$E/test_protocol.py"
```

시스템 Toolkit 12.2와 PyTorch wheel의 CUDA runtime 12.6은 서로 다르다. 드라이버의 CUDA 13.0 표시는 Toolkit 13 설치 요구가 아니다. LoMa 전용 vismatch 설치 범위는 [환경](ENVIRONMENT.md)을 따른다.

## 완료된 CSV 재평가 (GPU 불필요)

```bash
E=experiments/2026-11-06_scale_ransac
A="$PWD/$E/results/method_b_20261102_r01"
# 새 출력 경로만 사용한다. 아래 경로가 있으면 다른 이름을 선택한다.
"$P" "$E/evaluate_existing.py" --run "$A/final" --split final --out /tmp/e3e4-team-evaluate-new.md
"$P" "$E/analyze.py" --runs "$A/final" --split final --out /tmp/e3e4-final-analysis-new
```

`analyze.py`는 팀 저장소의 `results/failure_tags.csv`를 과거 참고 부분집합 분석에만 사용한다. 제출 CSV의 새 Baseline과 짝지어 계산하며 과거 72.9%를 비교 기준으로 사용하지 않는다. `/tmp` 출력은 제출하거나 기존 결과에 덮어쓰지 않는다.

## 데이터와 가중치 (별도 준비)

원본 `Bearing_UAV_90K.zip`을 허가된 경로에 준비한다. 원래 k와 row는 팀 저장소의 `results/raw_capstone1/out_imatch/samples_1000.csv`에서 읽는다. UAV/타일 1,892개만 읽고 UAV 1,000장·위성 블록 1,000장을 재구성한다.

```bash
D="$PWD/data/im_t1_3d_1000_reconstructed"
W="$PWD/.cache/e3-e4/torch"
"$P" "$E/reconstruct_singapore.py" --zip data/Bearing_UAV_90K.zip \
  --report /tmp/e3e4-export-new.json --export "$D" --diagnostic-only
"$P" "$E/prepare_weights.py" --help
```

기존 결과의 `method_b_dataset.json`과 freeze는 원래 서버 경로·실행 identity를 고정한 역사적 기록이다. 다른 경로에서 GPU 재실행하려면 기존 기록을 보존한 별도 연구용 작업 사본에서 새 dataset seal을 만들고 smoke→tune→새 freeze→final 순서로 실행한다. `seal_method_b.py --out <별도 기록>`으로 먼저 검증할 수 있으나 실행기는 실험 폴더의 `method_b_dataset.json`을 읽는다. 기존 파일을 덮어쓰거나 기존 freeze를 새 실행에 재사용하지 않는다. 다른 경로로 새 seal을 만들 때는 별도 사본에서 기존 seal을 기록 폴더로 보존한 뒤 새 seal을 생성한다.

팀 최신 공통 코드가 실측 당시 코드와 달라지면 원래 source fingerprint와 다르다. 원래 측정값은 원래 코드에 대한 결과로 보존하며, 새 코드로 재실행한 것으로 표현하지 않는다. 새 실험의 후보·채택 규칙은 final 전에 다시 동결한다. GPU 실행 전체 명령·원래 데이터 해시·전처리는 [README](README.md)에 기록돼 있다.
