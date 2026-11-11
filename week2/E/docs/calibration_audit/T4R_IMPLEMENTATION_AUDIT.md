# T4R_IMPLEMENTATION_AUDIT

## 원본 구현: 미확보

사용자가 제시한 `diag/lowlight_real.py`, `diag/diag_lowlight.py`, `diag/night_real.md`와 `degrade_real`, `CONDITIONS_REAL`을 접근 가능한 작업 디렉터리·개인 아카이브에서 검색했지만 원본을 찾지 못했습니다. `rn_1/500`, `rn_1/1500`, `rn_1/3333`과 2026-10-14 전환 기록은 사용자 제공 단서입니다.

T4R의 약어 의미·논문·표준 지위를 확정할 수 없습니다. 내부 설정명일 가능성이 있지만 원본 없이 확정하지 않습니다. 과거 capture의 ISO 1/16~1/256도 변수의 뜻과 단위가 미확인입니다. 실제 센서 ISO나 현재 기본 설정으로 취급하지 않습니다. 이름만으로 원본 조건을 복원하지 않았습니다.

## 기존 코드의 8개 감사 항목

대상: `../../../standalone/lowlight.py`
SHA256: `8f5ada00939d57b04860b224ff5bf88bbcb26d729cd960bcc94c13cc2c93acc7`

| 항목 | 코드와 측정 결과 | 한계 |
|---|---|---|
| 1. 선형광 노출 | 29~30행 EOTF 후 32행 linear×exposure 적용 | 처리된 JPEG를 sRGB라고 가정한 값이며 RAW radiance 아님 |
| 2. 감마 방향 | 35~36행 OETF는 역변환. 보정 x^0.6은 0<x<1을 밝힘 | 보정 exponent는 센서 gamma·노출과 다름 |
| 3. 광량과 noise | Poisson(eLP)/P + Gaussian(0,σ²) | e/P/σ는 실측 센서 단위가 아님 |
| 4. clipping | 아래쪽 channel clipping 평균 Mild 1.475%, Severe 10.711%. 위쪽 둘 다 0% | 하단 절단이 그림자 bias와 색을 바꿈 |
| 5. 색온도·WB | 구현 없음. 모든 채널에 동일 노출 | 독립 채널 noise/clipping 때문에 출력색은 달라질 수 있음 |
| 6. 공간 조명 | 전체 영상에 균일한 exposure 적용 | 별도 가로등·국부 조명·flare·공간가변 노출 없음 |
| 7. 실제 야간과 밝기 | 본 NAT 평균 26.09/255, Mild 90.14, Severe 50.87 | unpaired 표본 비교. 전체 NAT나 센서 인과 검증 아님 |
| 8. 재현성 | default_rng, Pilot seed=20261110+k. 감사 trace40건과 원래 함수가 bit-exact 일치 | decode/shape/NumPy/호출 순서를 고정해야 함 |

[sRGB 변환 참고](https://www.w3.org/TR/css-color-4/#color-conversion-code). EOTF/OETF 방향은 맞지만 원본 JPEG의 실제 camera response가 sRGB와 같다고 검증한 것은 아닙니다. 두 조건의 seed가 같아도 Poisson 파라미터에 따라 난수 소비량이 달라 같은 noise realization까지 보장하지는 않습니다.

## 노이즈 식과 물리적 의미

clipping 전 Z=Poisson(eLP)/P+N(0,σ²)이므로 E[Z|L]=eL, Var[Z|L]=eL/P+σ²입니다. shot 성분은 줄어든 광량과 연결되고 read 성분은 고정됩니다. 이는 수학적인 signal-dependent 관계를 보여줄 뿐, 실제 ISO·electron/ADU·full-well·CFA·demosaic·FPN·ISP denoise/compression을 검증한 모델은 아닙니다.

기존 Mild는 e=0.45/P=1000/σ=0.008, Severe는 e=0.15/P=500/σ=0.015입니다. Severe에서 L<0.75이면 read 분산이 shot 분산보다 큽니다. 이 단위와 수치가 실제 DJI 센서에 맞는 근거는 없습니다.

## clipping·색 왜곡 진단

최종 출력의 any-channel=0 비율은 Mild 3.768%, Severe 24.992%, NAT 표본 0.568%였습니다. clipping 전 음수 비율과 최종 endpoint 비율은 서로 다른 통계입니다.

검정 입력에서는 하단 절단 후 채널 평균 코드가 Mild 8.37/255, Severe 12.74/255가 됐습니다. 음수 noise를 제거하므로 0 부근에 양의 bias가 생깁니다. `BLACK_INPUT_DIAGNOSTIC.json`은 이 CPU 진단이며 새 LoMa 조건이나 실제 센서 실측이 아닙니다.

## T4R 원본과 비교할 수 없는 부분

일반적인 선형광·Poisson–Gaussian·OETF 방식이 사용자 기록과 겹쳐도 동일 구현의 증거가 아닙니다. `CONDITIONS_REAL`, WB, black-level, 공간 효과, 단위가 없어 원본과 수치 parity를 확인하지 못했습니다. 원본 재현과 그 조건의 GPU 재비교는 미실시했습니다. 기존 코드는 변경하지 않았습니다.
