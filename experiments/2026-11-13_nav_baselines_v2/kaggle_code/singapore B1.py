import pandas as pd
import numpy as np
from imatch1.nav import make_traj, nav_run, ODO, Nearest3D

# 두 도시가 공유할 동일한 경로 생성
nav_starts, nav_goals = make_traj(
    n_traj=32,
    dist_m=1000,
    seed=7
)

print("경로 수:", len(nav_starts))
print("경로 길이 평균(m):",
      np.mean(np.linalg.norm(nav_goals - nav_starts, axis=1)) / 4)

# 싱가포르 B1 본 실험
sg_frames = Nearest3D(CAPSTONE_ROOT, "3d_ho37bc")
sg_log = []

sg_success, sg_reason, sg_errors = nav_run(
    loc=b1_loc_sg,
    frames=sg_frames,
    starts=nav_starts,
    goals=nav_goals,
    decode_px=128,
    step_m=15,
    arrive_m=20,
    n_block=15,
    odo=ODO["vio"],
    seed=7,
    log=sg_log,
)

sg_results = pd.DataFrame(sg_log)
sg_results.to_csv(
    "/kaggle/working/B1_3d_37bc_nav_log.csv",
    index=False
)

print("\n===== B1 Singapore 3D Navigation =====")
print("성공:", int(sg_success.sum()), "/", len(sg_success))
print("성공률:", round(sg_success.mean() * 100, 2), "%")
print("종료 이유:")
print(pd.Series(sg_reason).value_counts())
print("궤적별 오차 중앙값 평균:",
      round(float(np.mean(sg_errors)), 2), "m")
print("총 기록 걸음:", len(sg_log))
print("로그 저장 완료")