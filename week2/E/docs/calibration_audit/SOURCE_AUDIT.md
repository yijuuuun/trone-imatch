# SOURCE_AUDIT — 2026-11-10 KST

## 결론과 검색 범위

T4R의 외부 공식 모델명·약어 확장·공개 표준 출처는 확인하지 못했습니다. 사용자 기록은 Capstone1 내부 condition명과 구현 경로를 가리키지만 원본이 없어 **팀 내부 모델이라고 확정하는 것조차 보류**합니다. 공식 모델임을 주장할 근거는 없습니다. 현재 Mild/Severe는 T4R 재현도 NAT2021 보정 모델도 아닌 C등급 개발 합성입니다.

읽기 권한이 있는 gpu_02 홈/현재 프로젝트/공유 첨부/개인 아카이브에서 lowlight_real.py,diag_lowlight.py,night_real.md,degrade_real,CONDITIONS_REAL를 검색했습니다. /srv는 비어 있었고, 접근 불가 경로를 우회하지 않았습니다. gpu_03 홈은 검색하지 않았습니다. 개인 Private GitHub 연구 저장소의 lowlight_real/night_real/degrade_real/diag_lowlight/T4R/capture 검색은 결과가 없었고, 정확한 diag/lowlight_real.py 조회도404였습니다. 지정된 팀 Private 저장소 yijuuuun/trone-imatch의 diag 파일 조회도404였습니다. 404는 파일이 없다는 확증이 아니라 현재 연결의 미접근 결과입니다. source absence를 모델 부재의 증명으로 쓰지 않습니다.

## 공식 자료에서 확인한 사실

| 근거 | 확인한 내용 | 확인하지 못한 내용 |
|---|---|---|
| [공식 NAT2021 프로젝트](https://vision4robotics.github.io/NAT2021/) | 야간 도로·도시·캠퍼스, DJI Mavic Air2,30fps; test180/L-test23/train1400 시퀀스 | 실제 ISO/셔터/노출·카메라 gamma·센서 noise 수치 |
| [논문](https://arxiv.org/abs/2203.10541) PDF p5 §4.1/Table1 | test140815,L-test53564,train276081 frames; test/L-test 수동 주석 | 촬영별 sensor RAW 설정·노출 합성 파라미터 |
| 논문 PDF p6 §4.3 | 객체 중심 local region intensity의 프레임값을 시퀀스에 걸쳐 평균; LAI<20 | ROI 정확한 크기/shape,명시적인 RGB→intensity 변환식·물리 단위 |
| [공식 supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ye_Unsupervised_Domain_Adaptation_CVPR_2022_supplemental.pdf) p1 Table1,p2 Fig1 | IV의 local intensity max-min>40,LAI<20; LAI49/180 등의 속성 분포 | 전체 영상 pixel brightness histogram 또는 mean/std,센서 noise 분산 |
| [UDAT source](https://github.com/vision4robotics/UDAT) | 실제 야간 영상의 DCE 보정과 tracking/domain adaptation 코드 | 주간→야간 exposure/Poisson-Gaussian/WB 카메라 모델 |

현재 확인한 root LICENSE는 UDAT **코드**의 Apache2.0입니다. 이는 별도 Google Drive 원본 영상에 대한 명시적 라이선스 확인을 대신하지 않습니다. 프로젝트·README·논문 및 검사한 ZIP 중앙목록에서 데이터 전용 LICENSE/README/terms 파일은 찾지 못했습니다. 따라서 영상이 Apache2.0이라고 확정하지 않으며, 상업이용·재배포·외부 발표 원본 이미지 사용 범위도 확정하지 않습니다. 원저작자 제공의 공개 benchmark 다운로드에서 로컬 기술 통계를 계산했고 재배포/업로드하지 않았습니다. 권리 확인이 더 필요한 공유 용도는 저작자 데이터 조건을 별도로 확인해야 합니다. Google Drive HTML의 SPDX는 페이지 JavaScript의 라이선스이므로 데이터 이용 조건으로 사용하지 않았습니다.

## 공식 코드 감사

preprocessing/lowlight_enhancement.py는 이미 야간인 NAT2021 train 영상을 DCE 네트워크로 enhance합니다. scale_factor12는 처리 크기/네트워크 설정이며 노출 factor가 아닙니다. DCE/Epoch99.pth는 보정 모델 checkpoint이고 카메라 gamma/ISO 설정이 아닙니다. code search의 TRAIN.LOSS_GAMMA나 discriminator noise_strength는 학습/feature 변수이며 센서 noise의 근거로 쓰지 않았습니다. 이번 감사에서는 DCE 모델을 실행하거나 학습하지 않았습니다.

UDAT source revision: `dcb55cac37461095585506e4aadfcb2fc3b954f9`. ZIP 및 개별 근거 파일 SHA는 SOURCE_SHA256.json에 기록했습니다. 논문은 CVF fetch 실패 시 저자 arXiv PDF로 확인했고, supplement는 CVF 원본을 읽었습니다. Sources의 PDF/TXT/HTML/코드 스냅샷과 SOURCE_INVENTORY.csv를 함께 보존했습니다.

## 새로 계산한 통계의 출처

공식 링크가 가리키는 NAT2021.zip의 공개 HTTP Range 접근으로 원본60프레임을 확보했습니다. 4,966,716,421bytes ZIP 중앙목록141359항목,180시퀀스/140815 image frames를 확인했습니다. sequence ID를 정렬해 round(i*(179)/19),i0..19의20시퀀스를 고르고 각각 첫/중간/마지막을 선택했습니다. 선택은 밝기·GT·LoMa 결과를 보고 바꾸지 않았습니다. JPEG bytes는ZIP CRC와 길이 확인 후 SHA256을 저장했습니다. 전체 ZIP/모든 프레임을 다운로드하지 않았습니다.

60개는 모두1280×720 RGB JPEG이고 ICC/EXIF가 없었습니다. 실제 shutter/ISO/WB나 실제 카메라 응답곡선은 회수할 수 없습니다. 이 해상도는 **본 표본의 직접 관측**이며 모든 원본 영상의 공식 규격으로 일반화하지 않습니다. NAT2021_STATISTICS.csv는 저자가 발표한 통계가 아니라 본 감사의 직접 측정입니다. 외부 NAT2021-test 표본의 기술 분석이며 LoMa 시험 k100..999/잠긴 지역/궤적을 사용한 것이 아닙니다. NAT2021 표본으로 파라미터를 피팅하거나 LoMa benchmark 점수를 튜닝하지 않았습니다.
