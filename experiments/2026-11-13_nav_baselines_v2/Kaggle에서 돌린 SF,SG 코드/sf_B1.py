sf_b1_log = []

sf_b1_success, sf_b1_reason, sf_b1_med_errors = nav_run(
    loc=b1_loc_sf,
    frames=sf_frames,
    starts=sf_starts,
    goals=sf_goals,
    decode_px=128,
    oracle_block=False,
    step_m=15.,
    arrive_m=20.,
    n_block=15,
    log=sf_b1_log,
    odo=ODO["vio"],
    seed=7,
    reject_inl=0,
    gate_m=0.,
    recover=None
)

print("\n===== SF B1 3D + VIO 결과 =====")
print(f"성공: {sf_b1_success.sum()}/32")
print(f"성공률: {sf_b1_success.mean()*100:.2f}%")
print("\n종료 이유:")
print(pd.Series(sf_b1_reason).value_counts())
print(f"\n평균 궤적별 중앙 위치오차: {np.mean(sf_b1_med_errors):.2f}m")

pd.DataFrame(sf_b1_log).to_csv(
    "/kaggle/working/SF_B1_3D_VIO_seed7_steps.csv",
    index=False
)

pd.DataFrame({
    "traj": range(32),
    "success": sf_b1_success.astype(int),
    "reason": sf_b1_reason,
    "median_error_m": sf_b1_med_errors
}).to_csv(
    "/kaggle/working/SF_B1_3D_VIO_seed7_traj.csv",
    index=False
)

print("\nCSV 저장 완료")