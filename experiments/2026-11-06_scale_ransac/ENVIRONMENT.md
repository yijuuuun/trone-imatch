# 실측 환경 — E3·E4 방법 B

2026-11-02~03 연구실 RTX A6000 GPU 0에서 실행한 환경이다. 아래 파일·경로는 준비 방법이며 데이터·가중치·가상환경 자체는 Git에 포함하지 않는다.

| 항목 | 실제 실행 값 |
| --- | --- |
| OS / Python | Linux / Python 3.11.3, 프로젝트 전용 `.venv-e3-e4` |
| GPU | NVIDIA RTX A6000 48GB, GPU 0 |
| NVIDIA driver / 시스템 Toolkit | 580.178.04 / 12.2.128 (변경 없음) |
| PyTorch / torchvision | 2.8.0+cu126 / 0.23.0+cu126 |
| PyTorch CUDA runtime | 12.6, 실제 CUDA 모델 로드·추론 완료 |
| OpenCV / vismatch | 4.12.0.88 / 1.4.0 |
| numpy / pandas / scipy | 2.2.6 / 2.2.3 / 1.15.3 |
| 실행 | batch 1, 조건별 순차 실행, tmux |
| threads | torch 4 / OpenCV 1 |

기본 Anaconda의 CPU torch 2.0.1과 시스템 CUDA는 변경하지 않았다. vismatch는 `--no-deps`로 설치한 LoMa 전용 구성이다. 다른 매처 의존성을 모두 설치하거나 검증하지 않았다.

가중치는 공식 `loma_B.pt`, `dad.pth`, `dinov2_vitl14_pretrain.pth`다. URL은 [공식 가중치 명세](official_weights.json), 다운로드 SHA256은 [검증 기록](validation/official_weights_download.json), 실제 설치 목록은 [환경 lock](validation/method_b_environment.lock.txt)을 따른다. 고정 코드의 requirements 주석은 준비 당시 기록이며 현재 실행 완료 여부는 이 문서와 README를 따른다.

실행 시간은 warm-up 및 이미지 I/O·모델 로드를 제외한 resize+매칭+기하 계산이다. 실제 GPU 메모리·샘플별 시간은 결과 CSV에 포함한다. 추가 설치·GPU 재실행은 이번 Git 제출 준비에서 수행하지 않았다.
