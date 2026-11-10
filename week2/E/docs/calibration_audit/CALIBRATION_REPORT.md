# CALIBRATION_REPORT — 근거 감사 결과

작업일: 2026-11-10 KST. 기존 GPU 실측을 변경하지 않고 문헌·코드 출처와 CPU 영상 통계를 감사했습니다. **기존 Mild·Severe는 모두 C: Experimental입니다.**

## 1. T4R은 공식 카메라 모델인가, 팀 내부 설정인가?

**정체 미확정입니다. 공식 외부 카메라 모델임을 확인하지 못했습니다.** 사용자 기록은 Capstone 1 내부 구현을 가리키지만, 원본 없이 내부 설정이라고 확정하는 것도 보류합니다.

우선 검색한 대상은 `diag/lowlight_real.py`의 `degrade_real` / `CONDITIONS_REAL`, `diag/diag_lowlight.py`의 `capture`, `diag/night_real.md`입니다. `rn_1/500`, `rn_1/1500`, `rn_1/3333`과 2026-10-14의 capture→degrade_real 전환은 사용자 제공 단서입니다. 접근 가능한 로컬·개인 아카이브에서 원본을 찾지 못했고, 팀 저장소 조회는 404였습니다. 이 404로 파일 부재를 확정하지 않습니다.

조건 이름의 분모를 노출·ISO·noise 값으로 바꾸거나 T4R 코드를 임의 복원하지 않았습니다.

## 2. NAT2021에서 공식 확인 가능한 조건은?

[공식 프로젝트](https://vision4robotics.github.io/NAT2021/)와 [논문](https://arxiv.org/abs/2203.10541)은 DJI Mavic Air 2, 30fps, 도로·도시·캠퍼스 등 야간 장면을 명시합니다. 논문의 구성은 test 180시퀀스/140,815프레임, L-test 23/53,564, train 1,400/276,081입니다.

LAI는 **객체 중심 주변 영역의 밝기를 시퀀스에 걸쳐 평균한 값이 20 미만**인 속성입니다. 전체 영상 평균이 아닙니다. [보충자료 Table 1](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ye_Unsupervised_Domain_Adaptation_CVPR_2022_supplemental.pdf)에서 IV는 같은 local intensity의 최대–최소 차이가 40 초과인 속성입니다. ROI의 정확한 크기와 intensity 변환식은 검토한 자료에 명시되지 않았습니다.

실제 촬영 ISO·셔터·카메라 gamma·센서 noise 분산이나 주간→야간 합성 파라미터는 검토한 공식 자료에서 찾지 못했습니다. 공식 코드의 DCE 처리는 실제 야간 영상 보정이며 야간 카메라 합성 모델이 아닙니다.

직접 확보한 60개 표본은 모두 1280×720 RGB JPEG이고 EXIF/ICC가 없었습니다. 이 해상도는 본 표본의 관측이지 전체 데이터의 공식 규격을 추정한 값이 아닙니다. UDAT root의 Apache 2.0을 원본 영상의 확정 라이선스로 전용하지 않았습니다. 데이터 전용 이용·재배포 조건은 검사한 공식 자료에서 확인되지 않았습니다. 공개 배포본으로 로컬 통계를 계산했으며 원본 영상이나 연락판을 외부 업로드하지 않았습니다.

## 3. 기존 Mild/Severe가 실제 NAT2021 분포와 유사한가?

고정 규칙으로 선택한 20시퀀스·60프레임의 전체 영상 밝기 Y′ 평균은 **26.09/255**였습니다. 기존 개발20개에서 Mild는 **90.14**, Severe는 **50.87**로 각각 약 3.46배·1.95배 밝았습니다. Y′<0.05인 어두운 픽셀 비율은 NAT 표본 40.89%, Mild 1.68%, Severe 9.61%였습니다.

Severe의 any-channel=0 비율은 24.99%로 NAT 표본의 0.568%보다 높았습니다. 현재 합성이 본 NAT 표본의 밝기·색·endpoint 분포와 유사하다는 근거는 없습니다. 다만 서로 다른 장소·시점·FOV의 unpaired 비교이므로 센서 모델의 인과 검증이나 NAT 전체 모집단 검정으로 해석하지 않습니다.

본 감사의 평균을 저자가 발표한 NAT 공식 평균으로 부르지 않습니다. 전체 밝기를 LAI로 재분류하지 않았고, 실제 센서 noise 분산을 RGB에서 분리 추정하지도 않았습니다.

## 4. 노출·noise·gamma 설정의 출처를 추적할 수 있는가?

현재 코드·프로토콜·SHA까지는 추적됩니다. 그러나 그 수치가 T4R 원본이나 NAT 실측에서 나온 근거는 없습니다. 노출 계수 0.45/0.15, Poisson scale 1000/500, read σ 0.008/0.015, Gamma 0.6은 기존 개발값입니다.

sRGB EOTF/OETF 방향은 맞고 shot variance는 eL/P로 광량에 연결됩니다. 하지만 실제 센서 단위가 보정되지 않았으며 WB·색온도·국부 광원·ISP가 없습니다. read noise와 하단 clipping은 그림자의 밝기와 색을 바꿉니다.

| 등급 | 이번 판정 |
|---|---|
| A Source-verified | NAT 촬영·LAI/IV 정의, sRGB 변환 방식만 해당. 전체 합성 조건이 A인 것은 아님 |
| B Data-calibrated | 없음. NAT 통계는 계산했지만 파라미터를 피팅·보정하지 않음 |
| C Experimental | 기존 Mild·Severe, Gamma·CLAHE 개발 파라미터 |
| 미확정 단서 | T4R의 정체·원본 값, rn_*의 수치 매핑, capture의 ISO 기록 |

각 값과 단위·코드 위치·출처는 `LOWLIGHT_PARAMETER_PROVENANCE.csv`에 기록했습니다. 보정 exponent 0.6과 sRGB transfer의 2.4를 구분합니다.

## 5. 기존 저조도 결과는 어디까지 주장할 수 있는가?

다음 표현까지 가능합니다.

> 명시된 C등급 합성과 고정 seed를 사용한 Full LoMa-B 독립 Oracle T1 개발20쌍에서 Daylight 65%, Mild 70%, Severe 60%였고, Gamma·CLAHE의 위치 성공 회복을 확인하지 못했다.

기존 260건은 해당 조건에서의 실제 측정 기록입니다. 이를 T4R 원본 재현, NAT 통계로 보정한 카메라, 실제 야간 UAV 정확도 또는 팀 공식 평가로 확장할 수 없습니다. Gamma·CLAHE가 실제 야간에서 일반적으로 효과 없다는 결론도 아닙니다. NAT 표본에는 LoMa 위치 추론을 실행하지 않았습니다.

## 6. 재실험이 필요한가, 제한 표시로 충분한가?

**기존 C등급 Pilot 보고에는 근거 제한 표시로 충분합니다.** 기존 결과를 수정·삭제하거나 새 합성을 소급 적용할 필요는 없습니다.

목표가 Capstone 1의 T4R/degrade_real 재현 또는 실제 NAT에 부합하는 야간 검증이라면 기존 설정만으로는 부족합니다. 원본 파일·조건의 단위·근거 문서를 확보하고, 별도 버전으로 재현한 뒤 같은 개발 코호트에서 새 run으로 비교해야 합니다.

현재는 원본 `CONDITIONS_REAL`이 없고 unpaired 영상 통계만으로 물리 카메라 파라미터를 식별할 수 없습니다. 따라서 임의 보정값·T4R 재구현·추가 GPU 실험은 만들지 않았습니다. 팀의 잠긴 k=100~999, 지역 시험 및 항법 궤적도 사용하지 않았습니다.

## 산출물과 보존

요청된 6개 파일을 모두 작성했습니다. `NAT2021_STATISTICS.csv`는 직접 측정한 60행입니다. 추가로 표본 manifest/CRC/SHA, 공식 문헌·코드 스냅샷, 합성60행 통계, preclip40건 추적, 히스토그램, 시퀀스별·비공식 시각 그룹별 통계, noise proxy와 로컬 예시 영상을 보존했습니다.

기존 결과·코드·보고서 701파일의 해시가 유지됐습니다. 원본 `WEEK2_REPORT.md`, 260건 GPU 결과, 기존 CSV·가중치·데이터는 변경하지 않았습니다. 새 설정·모델 학습·Git Commit/Push/PR·권한 우회는 없습니다. 전량을 새 `calibration_audit/`에 저장했습니다.

재현 코드는 `read_nat_index.py`, `sample_nat.py`, `measure_statistics.py`, `supplement_statistics.py`입니다. 기존 출력이 있으면 덮어쓰기를 거부하므로 새 재현에는 별도 감사 폴더를 사용해야 합니다. 측정 환경은 기존 `.venv-gpu`, OpenCV 4.12.0, NumPy 2.2.6이며 통계는 CPU로만 계산했습니다. 문서 최초 생성본은 `drafts_before_editorial/`에도 보존했습니다.
