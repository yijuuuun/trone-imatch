# 긴 구간 항법 × 큰 이동 오차: R1 vs R1 + R4(v1 경량), VisLoc 시험용 481 km, vio4 (주희윤, 2026-11-22)

[2026-11-20_visloc_nav_full](../2026-11-20_visloc_nav_full/README.md)(#64)과 같은 실험을 이동 오차만 4배(`vio4`)로 바꿔 다시 돈다.

**결과는 아직 없다.** 아래 "미리 정한 것"은 시험용을 돌리기 전에 적었다. 결과를 보고 바꾸지 않는다.

## 1. 왜 하나

- #64(`vio`, 보통 이동 오차)에서는 R1 + R4의 이득이 작았다. R1이 길을 12번 잃고 11번을 혼자 되찾아 R4가 할 일이 적었다.
- #60은 1 km 개발용(`seq_calib`)에서 이동 오차를 4배로 키우면 R1 + R4가 유의하게 더 도착함을 보였다(새 seed 308번, 22 vs 4, p = 0.0005). 추측 위치가 블록 밖으로 많이 밀리면 옆 블록 재탐색이 쓸모 있어진다.
- 확인하지 않은 것: 시험용 데이터에서, 그리고 수십 km 긴 거리에서도 그런가. 긴 거리에서는 R1이 한 번 잃으면 못 돌아오는 경우가 늘 수 있어 차이가 1 km보다 클 수 있다.
- 가설은 #60의 개발용 결과에서 나왔고, 이번 시험용 결과(`vio`)를 보고 세운 것이 아니다. 시험용 `seq_eval_full`의 `vio4` 결과는 아직 아무도 보지 않았다.

## 2. 미리 정한 것 (시험용 실행 전, 2026-11-22)

1. **데이터와 조건**: #64와 같다. `seq_eval_full` 11구간(481 km), R1(`r1`) vs R1 + R4 v1 경량(`r4g0n5w20ac4`), `--matcher loma-cache --angles 4 --stop-inl 15 --seed 2026`. 바뀌는 것은 `--odo vio4` 하나다.
   - `vio4` = #60 `imatch/nav.py`와 같은 정의: 구간마다 방향 치우침 N(0, 12°), 걸음마다 N(0, 8°), 거리 N(1, 20%). 이 브랜치에 #60과 같은 글자로 넣었다.
   - seed는 #64와 같은 2026 하나, 한 번 돈다. 같은 seed라 바뀌는 것은 오차의 크기뿐이다.
2. **주 판정**: 구간별 "40 m 이내 걸음 비율"을 R1 + R4와 R1로 짝지어 Wilcoxon 부호 순위 검정(양측, 차이 0인 구간 뺌, `analyze.py`).
   - 차이 중앙값은 차이 0인 구간을 빼고 잰다(검정과 같은 구간).
   - p < 0.05이고 차이 중앙값 > 0이면 "긴 거리에서 이동 오차가 클 때 R1 + R4가 R1보다 유의하게 낫다"로 보고한다.
   - p < 0.05이고 차이 중앙값 < 0이면 "R1 + R4가 유의하게 나쁘다"로 보고한다.
   - 참고로 같은 검정을 #64(`vio`)에 대면 R1 + R4 높음 5, 낮음 3, 같음 3, p = 0.11이다(판정 기준을 정하며 코드가 도는지 확인하다 본 값. #64의 판정에는 쓰지 않음).
   - 아니면 "11구간으로는 차이가 확인되지 않음"으로 보고한다.
   - 지점별 McNemar는 구간이 11개라 힘이 약해 주 판정으로 쓰지 않는다(#64 2절 5번과 같은 이유).
3. **함께 적는 것** (판정에 쓰지 않음, 모두 적음): 5 / 10 / 20 / 50 km와 구간 끝 40 m 이내(McNemar 참고), 전체 40 m·20 m 이내 걸음, 길 잃음 → 되찾음, 끝까지 못 돌아온 구간, 걸음당 시간, 실시간 넘는 걸음. 추측 항법만(`dr`, `vio4`)은 CPU로 따로 돌려 참고로 적는다.
4. **실행 환경**: Kaggle T4에서 R1과 R1 + R4를 한 커널로 돌린다(처음에는 #64의 Colab 매칭 캐시를 이어 쓰려 했으나, 실행 전에 Kaggle GPU가 다시 생겨 바꿈. 결과를 보기 전의 변경).
   - 매칭은 처음부터 한다. 같은 커널 안에서 R1이 매칭한 (사진, 블록)은 R1 + R4가 다시 매칭하지 않는다(#64와 같음).
   - 시간은 이 실행 안의 R1과 R1 + R4끼리만 비교한다(#64 Colab 숫자와 나란히 놓지 않는다).
5. **예상** (틀려도 고치지 않음): R1은 #64보다 길을 더 자주 잃고, 끝까지 못 돌아오는 구간이 는다. R1 + R4는 그중 일부를 되찾아 40 m 이내 걸음 비율이 R1보다 높다. `visloc_09`는 매칭 문제라 두 조건 모두 실패한다.
6. **#64 결과는 바꾸지 않는다.** 이 실험은 #64의 `vio` 결론에 조건("이동 오차가 클 때")을 더할 뿐이다.

## 3. 실행

```
# 추측 항법만 (CPU, 참고)
python scripts/run_visloc_nav.py --data data/visloc/seq_eval_full --out experiments/2026-11-22_visloc_nav_full_vio4/run/test \
    --matcher loma-cache --variants dr --odo vio4 --seed 2026
# R1, R1 + R4: Kaggle T4
python scripts/kaggle/build_kernel.py --user heeyunju --dataset heeyunju/visloc-seq-eval-full --slug visloc-nav-full-vio4 \
    --script run_visloc_nav --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20ac4 --odo vio4 --seed 2026"
kaggle kernels push -p kaggle_kernels/visloc-nav-full-vio4 --accelerator NvidiaTeslaT4
kaggle kernels output heeyunju/visloc-nav-full-vio4 -p experiments/2026-11-22_visloc_nav_full_vio4/run/test --file-pattern 'out/.*'
# 표
python experiments/2026-11-20_visloc_nav_full/analyze.py --run experiments/2026-11-22_visloc_nav_full_vio4/run/test
```

## 4. 결과

(시험용 실행 뒤 채움)
