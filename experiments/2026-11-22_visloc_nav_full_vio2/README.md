# 긴 구간 항법 × 이동 오차 2배: R1 vs R1 + R4(v1 경량), VisLoc 시험용 481 km, vio2 (주희윤, 2026-11-22)

[#64 `vio`](../2026-11-20_visloc_nav_full/README.md)와 [`vio4`](../2026-11-22_visloc_nav_full_vio4/README.md)의 중간인 이동 오차 2배(`vio2`)를 같은 방법으로 돈다.

**결과는 아직 없다.** 아래 "미리 정한 것"은 시험용을 돌리기 전에 적었다. 결과를 보고 바꾸지 않는다.

## 1. 왜 하나

- 긴 구간에서 40 m 이내 걸음 비율(R1)은 `vio` 69%, `vio4` 14%로 크게 다르고, 두 조건 모두 R1 + R4의 이득은 작거나 없었다.
- `vio4`에서 R4가 못 도운 까닭은 추측 위치가 재탐색 범위(약 ±128 m)보다 훨씬 멀리(중앙값 약 480 m) 밀렸기 때문이다. `vio2`는 그 경계쯤이라, R4가 닿을 수 있는 거리에서 길을 잃는 경우가 많을 수 있다.
- #60(1 km 개발용)의 `vio2`는 R1 58, R1 + R4 경량 56(한쪽만 성공 3 vs 1, p = 0.63)으로 차이가 없었다.
- 이 실험은 `vio`와 `vio4` 시험용 결과를 본 뒤에 세웠다. `vio2` 시험용 결과는 아직 아무도 보지 않았다.

## 2. 미리 정한 것 (시험용 실행 전, 2026-11-22)

1. **데이터와 조건**: `vio4` 실험과 같고 `--odo vio2`만 다르다. `seq_eval_full` 11구간(481 km), R1(`r1`) vs R1 + R4 v1 경량(`r4g0n5w20ac4`), `--matcher loma-cache --angles 4 --stop-inl 15 --seed 2026`.
   - `vio2` = #60 `imatch/nav.py` 정의: 구간마다 방향 치우침 N(0, 6°), 걸음마다 N(0, 4°), 거리 N(1, 10%).
2. **주 판정** (`vio4`와 같음): 구간별 "40 m 이내 걸음 비율"을 R1 + R4와 R1로 짝지어 Wilcoxon 부호 순위 검정(양측, 차이 0인 구간 뺌, 차이 중앙값도 0인 구간을 빼고 잼).
   - p < 0.05이고 차이 중앙값 > 0이면 "이동 오차 2배에서 긴 거리를 날 때 R1 + R4가 R1보다 유의하게 낫다"로 보고한다.
   - p < 0.05이고 차이 중앙값 < 0이면 "R1 + R4가 유의하게 나쁘다"로 보고한다.
   - 아니면 "11구간으로는 차이가 확인되지 않음"으로 보고한다.
3. **함께 적는 것** (판정에 쓰지 않음, 모두 적음): 5 / 10 / 20 / 50 km와 구간 끝 40 m 이내(McNemar 참고), 전체 40 m·20 m 이내 걸음, 길 잃음 → 되찾음, 끝까지 못 돌아온 구간, 걸음당 시간, 실시간 넘는 걸음, "모름"일 때 추측 오차 중 128 m 이내 비율, 재탐색이 받아들인 걸음의 정확도. 추측 항법만(`dr`, `vio2`)은 CPU로 따로 돌려 참고로 적는다.
4. **실행 환경**: Kaggle T4 커널 하나에서 R1과 R1 + R4를 함께, 매칭은 처음부터. 시간은 이 실행 안의 두 조건끼리만 비교한다.
5. **예상** (틀려도 고치지 않음): 40 m 이내 걸음 비율은 `vio`와 `vio4` 사이다. R1 + R4가 R1보다 조금 높지만 11구간으로는 유의하지 않다.
6. `vio`(#64), `vio4` 결과는 바꾸지 않는다.

## 3. 실행

```
python scripts/run_visloc_nav.py --data data/visloc/seq_eval_full --out experiments/2026-11-22_visloc_nav_full_vio2/run/test \
    --matcher loma-cache --variants dr --odo vio2 --seed 2026
python scripts/kaggle/build_kernel.py --user heeyunju --dataset heeyunju/visloc-seq-eval-full --slug visloc-nav-full-vio2 \
    --script run_visloc_nav --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20ac4 --odo vio2 --seed 2026"
kaggle kernels push -p kaggle_kernels/visloc-nav-full-vio2 --accelerator NvidiaTeslaT4
python experiments/2026-11-20_visloc_nav_full/analyze.py --run experiments/2026-11-22_visloc_nav_full_vio2/run/test
```

## 4. 결과

(시험용 실행 뒤 채움)
