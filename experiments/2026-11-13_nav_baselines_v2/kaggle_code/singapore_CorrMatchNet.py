import time
import numpy as np
import pandas as pd

# 실행 기록을 별도로 저장
corr_sg_log = []

t0 = time.time()

corr_sg_success, corr_sg_reason, corr_sg_med_errors = nav_run(
    corr_loc_sg,
    sg_frames,
    sg_starts,
    sg_goals,
    decode_px=128,
    oracle_block=False,
    step_m=15.,
    arrive_m=20.,
    log=corr_sg_log,
    odo=ODO["vio"],
    seed=7,
    reject_inl=0,
    gate_m=0.,
    recover=None
)

elapsed = time.time() - t0

# 결과 출력
n = len(sg_starts)

print("\n===== Singapore CorrMatchNet 3D + VIO =====")
print(f"성공: {sum(corr_sg_success)}/{n} ({100*np.mean(corr_sg_success):.2f}%)")
print("실패 사유:", pd.Series(corr_sg_reason).value_counts().to_dict())
print(f"평균 궤적별 중앙 위치오차: {np.nanmean(corr_sg_med_errors):.2f} m")
print(f"실행 시간: {elapsed:.1f}초")

# CSV 저장
pd.DataFrame(corr_sg_log).to_csv(
    "/kaggle/working/SG_CorrMatchNet_3D_VIO_seed7_steps.csv",
    index=False
)

pd.DataFrame({
    "success": corr_sg_success,
    "reason": corr_sg_reason,
    "median_error_m": corr_sg_med_errors
}).to_csv(
    "/kaggle/working/SG_CorrMatchNet_3D_VIO_seed7_traj.csv",
    index=False
)

print("결과 CSV 저장 완료")