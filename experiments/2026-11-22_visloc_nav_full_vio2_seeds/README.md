# 확인 실험: 이동 오차 2배에서 긴 구간 R1 + R4 이득이 새 seed에서도 나오나 (주희윤, 2026-11-22)

[`vio2` seed 2026](../2026-11-22_visloc_nav_full_vio2/README.md)에서 R1 + R4가 R1보다 유의하게 나았다(Wilcoxon p = 0.027). 그 결과를 본 뒤 세운 가설이라 **새 seed로만** 확인한다(#60 9절과 같은 방식).

**결과는 아직 없다.** 아래 "미리 정한 것"은 새 seed를 돌리기 전에 적었다. 결과를 보고 바꾸지 않는다.

## 1. 가설

이동 오차 2배(`vio2`)로 VisLoc 시험용 긴 구간(11구간, 481 km)을 날 때, R1 + R4(v1 경량)가 R1보다 40 m 이내 걸음 비율이 높다.

## 2. 미리 정한 것 (새 seed 실행 전, 2026-11-22)

1. **데이터와 조건**: seed 2026 실험과 같고 seed만 다르다. `seq_eval_full` 11구간, `--matcher loma-cache --angles 4 --stop-inl 15 --odo vio2`, R1(`r1`) vs R1 + R4 v1 경량(`r4g0n5w20ac4`).
2. **새 seed**: 2027, 2028, 2029 세 개. seed마다 Kaggle T4 커널 하나(R1과 R1 + R4를 한 커널에서). 추측 항법만(`dr`)은 seed마다 CPU로 돌려 참고로 적는다.
3. **주 판정**: 구간마다 세 seed의 "40 m 이내 걸음 비율 차이(R1 + R4 − R1)"를 평균 내고, 그 11개 값으로 Wilcoxon 부호 순위 검정(양측, 차이 0인 구간 뺌).
   - 같은 구간을 seed마다 다시 날므로 seed × 구간 33개는 서로 독립이 아니다. 그래서 구간 단위(11개)로 판정한다.
   - p < 0.05이고 차이 중앙값(0 뺌) > 0이면 **"확인됨: 이동 오차 2배에서 긴 거리를 날 때 R1 + R4가 R1보다 유의하게 낫다"**.
   - p < 0.05이고 중앙값 < 0이면 "반대로 확인됨(R1 + R4가 나쁘다)".
   - 아니면 **"확인되지 않음"**. seed 2026 결과는 가설을 세운 데이터의 결과로만 남긴다.
4. **seed 2026**: 판정에서 뺀다. 2026을 넣은 네 seed 결과도 따로 적는다(판정에 쓰지 않음).
5. **함께 적는 것** (판정에 쓰지 않음): seed별 같은 표(`analyze.py`), seed별 Wilcoxon, seed × 구간 33개 Wilcoxon(참고), 전체 40 m·20 m 이내 걸음, 끝까지 못 돌아온 구간, 걸음당 시간.
6. **한 seed가 실패하면**(Kaggle 오류 등) 같은 seed로 다시 돌린다. seed를 바꾸거나 빼지 않는다.

## 3. 실행

```
for s in 2027 2028 2029; do
  python scripts/run_visloc_nav.py --data data/visloc/seq_eval_full --out experiments/2026-11-22_visloc_nav_full_vio2_seeds/run/s$s \
      --matcher loma-cache --variants dr --odo vio2 --seed $s
  python scripts/kaggle/build_kernel.py --user heeyunju --dataset heeyunju/visloc-seq-eval-full --slug visloc-nav-full-vio2-s$s \
      --script run_visloc_nav --args "--matcher loma-cache --angles 4 --stop-inl 15 --variants r1,r4g0n5w20ac4 --odo vio2 --seed $s"
  kaggle kernels push -p kaggle_kernels/visloc-nav-full-vio2-s$s --accelerator NvidiaTeslaT4
done
python experiments/2026-11-22_visloc_nav_full_vio2_seeds/analyze_seeds.py
```

## 4. 결과

(새 seed 실행 뒤 채움)
