import time
import pandas as pd
import numpy as np

# 동일 조건으로 CorrMatchNet 내비게이션 실행
corr_sf_log = []

t0 = time.time()

corr_sf_success, corr_sf_reason, corr_sf_med_errors = nav_run(
    loc=corr_loc_sf,
    frames=sf_frames,
    starts=sf_starts,
    goals=sf_goals,
    decode_px=128,
    oracle_block=False,
    step_m=15.,
    arrive_m=20.,
    n_block=15,
    log=corr_sf_log,
    odo=ODO["vio"],
    seed=7,
    reject_inl=0,
    gate_m=0.,
    recover=None
)

elapsed = time.time() - t0

# 걸음별 결과 저장
corr_sf_steps = pd.DataFrame(corr_sf_log)
steps_path = "/kaggle/working/SF_CorrMatchNet_3D_VIO_seed7_steps.csv"
corr_sf_steps.to_csv(steps_path, index=False)

# 궤적별 결과 저장
corr_sf_traj = (
    corr_sf_steps.groupby("traj")
    .agg(
        steps=("err_m", "count"),
        median_error_m=("err_m", "median"),
        mean_error_m=("err_m", "mean"),
    )
    .reset_index()
)

corr_sf_traj["success"] = corr_sf_traj["traj"].map(
    lambda i: bool(corr_sf_success[int(i)])
)
corr_sf_traj["reason"] = corr_sf_traj["traj"].map(
    lambda i: str(corr_sf_reason[int(i)])
)

traj_path = "/kaggle/working/SF_CorrMatchNet_3D_VIO_seed7_traj.csv"
corr_sf_traj.to_csv(traj_path, index=False)

# 결과 요약
print("===== SF CorrMatchNet 3D + VIO (32 trajectories) =====")
print(f"성공: {np.sum(corr_sf_success)}/32 "
      f"({np.mean(corr_sf_success)*100:.2f}%)")
print("\n종료 이유:")
print(pd.Series(corr_sf_reason).value_counts().to_string())

print(f"\n평균 궤적별 중앙 위치오차: "
      f"{np.mean(corr_sf_med_errors):.2f} m")
print(f"총 실행 시간: {elapsed:.1f}초")
print(f"기록된 걸음 수: {len(corr_sf_steps)}")

print("\n저장 완료:")
print(steps_path)
print(traj_path)