"""Export-only path binding. Never defaults to the original research folder."""
import os
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('WEEK2_WORKSPACE', str(PACKAGE/'.local'))).expanduser().resolve()
AUDIT=ROOT/'experiments/2026-11-13_lowlight/calibration_audit'
NAT=ROOT/'experiments/2026-11-13_lowlight/nat2021_frame_matching_v1'
