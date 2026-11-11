# Week 2 reproduction

Independent Full LoMa-B pretrained research. This is **not** the team imatch pipeline or official team evaluation. Development k=0..19 results use Oracle T1 (GT-containing map block). k=100..999 are not used by these commands. Matching/inliers/H consistency are not location accuracy.

Tested environment: Python 3.11.3, RTX A6000, driver 580.178.04, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 2.2.6, OpenCV 4.12.0.88. PyTorch/torchvision must be the matching CUDA121 wheels. Install the small runtime requirements separately; the source is imported directly, without resolving/upgrading the upstream pyproject requirements. Existing tested environments can be reused.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r configs/requirements-runtime.txt
git clone https://github.com/davnords/LoMa.git /path/to/external/LoMa
git -C /path/to/external/LoMa checkout 415caedde23a9ed479172ed174fe4ff7b4514d4b
export WEEK2_WORKSPACE=/path/to/new/external/workspace
python src/verify_export.py
python src/prepare_workspace.py --loma-source /path/to/external/LoMa --weights-dir /path/to/external/weights --bearing-zip /path/to/external/Bearing_UAV_90K.zip
python src/prepare_dev.py
```

Run commands from this branch's `week2/E` directory. The weights are external `loma_B.pt`, `dad.pth`, `dinov2_vitl14_pretrain.pth`; URLs, sizes and SHA256 are in `configs/weights.expected.json`. Obtain those files separately from the listed official hosts. prepare_workspace checks all hashes and links the external source/weights rather than copying them into Git. Bearing ZIP is only required for the synthetic/optimization pilot; omit --bearing-zip and prepare_dev for NAT-only runs. prepare_dev restores exactly the preserved development image identities and validates source and JPEG output hashes; it extracts location metadata from that external ZIP. Existing workspace is never initialized twice. Use a fresh workspace for a new run.

The original pilot uses 10 warm-up single-heading pairs per condition, then 20 measured pairs, plus 5 separate stage profiles per condition. Timed infer includes enhancement/preprocess, feature extraction, matching and geometry, excludes decoding/loading/GT scoring. Latencies depend on hardware/load; numerical and threshold changes across environments can change success labels. Heading selection is max valid-H inliers, then matches, then first angle. Keypoints2048, match threshold0.1, MAGSAC++3px; FP16 retains FP32 master weights and controls the original LoMa internal AMP flags. The profiling wrapper separately times features, Matching Transformer, MAGSAC and total.

Never point WEEK2_WORKSPACE to the original measured research directory. The scripts write new JSON/NPZ/CSV files there and refuse conflicting protocols/overwrites. Output artifacts stay outside Git. Smoke commands and helpers do not train models.

## Recorded results and commands

Synthetic Oracle T1 pilot20: Daylight65%, Mild70%, Severe60% operational Success@10m. Mild/Severe are **C Experimental**, not official NAT2021/T4R camera parameters. Gamma/CLAHE did not establish location-accuracy recovery. Full source audit and measured synthetic/real brightness distributions are separate in docs/calibration_audit and results/calibration_audit. No original T4R code was reconstructed or claimed verified.

NAT2021:20 sequences,40 temporal pairs (first/middle anchors, partner+5 frames),120 actual FP32 single-heading GPU inferences. Both temporal frames receive the chosen enhancement; this differs from synthetic UAV-only enhancement. No satellite/location GT, no Success@10m.

| Enhancement | Mean pair Inlier Ratio | Valid H |
|---|---:|---:|
| Original | 47.2% | 100% |
| Gamma0.6 | 38.7% | 100% |
| CLAHE LAB-L clip2/grid8 | 37.2% | 100% |

Mean pair ratio differs from pooled sum(inliers)/sum(matches). No H failures occurred, so weakest matches are not fabricated failure cases. Brightness/scene are confounded; a monotonic dark-scene degradation was not established. Gamma lowered inliers39/40, CLAHE40/40; higher visible brightness did not imply better matching. Native1280x720 resized1274x714 for LoMa; fit in original image coordinates. Homography overlap is a posthoc proxy, not GT. See docs/NAT2021_REPORT.md for conclusions and limitations.

Synthetic repeat (only after the above Bearing preparation):

```bash
CUDA_VISIBLE_DEVICES=0 python -u src/pilot_matrix.py --out "$WEEK2_WORKSPACE/runs/E_pilot20" --limit 20 --only F1 E_mild E_severe E_mild_gamma E_mild_clahe E_severe_gamma E_severe_clahe
python src/analyze_pilot.py --run "$WEEK2_WORKSPACE/runs/E_pilot20" --out "$WEEK2_WORKSPACE/analysis/E_pilot20" --limit 20
CUDA_VISIBLE_DEVICES=0 python -u src/extra_gamma_clahe.py --out "$WEEK2_WORKSPACE/runs/E_extra" --limit 20 --only E_severe_gamma_clahe
```

F1 is the unchanged Daylight FP16/8 baseline (published as E_daylight), reused without duplicate inference. The optional combined correction was historically measured as a separate follow-up, not folded into the core analysis.

Real-night repeat (external author-linked public original data; verify your applicable use permissions first; no frames redistributed here):

```bash
python src/sample_nat.py
python src/prepare_nat_pairs.py
CUDA_VISIBLE_DEVICES=0 python -u src/run_nat_matching.py
cp results/pilot20/enhancement_comparison.csv "$WEEK2_WORKSPACE/experiments/2026-11-13_lowlight/"
python src/report_nat_matching.py
```

sample_nat downloads a deterministic60-frame statistical sample from the original ZIP through bounded HTTP ranges and writes its manifest. prepare_nat_pairs reuses40 anchors and downloads40 partners; no output-based selection. Compare against configs/NAT2021_SAMPLE_MANIFEST.json and configs/nat2021_PAIR_MANIFEST.json to verify identities/hashes. Public host availability/range support is external; no archive copy is included. run_nat_matching writes120 records plus correspondences; report_nat_matching needs those new NPZs and local frames and creates figures only in your external workspace. For the report's separate Bearing reference table, copy `results/pilot20/enhancement_comparison.csv` into `$WEEK2_WORKSPACE/experiments/2026-11-13_lowlight/` before running report_nat_matching. Those frozen reference numbers are not comparable location accuracy for NAT.

Optional CPU statistical audit (after NAT sampling and Bearing preparation, before generating repeated audit outputs): `python src/measure_statistics.py`, then `python src/supplement_statistics.py`. sRGB-code and assumed-linear metrics are distinguished. RGB noise proxies are not calibrated sensor variance. LAI uses object-neighborhood criteria, not our whole-image brightness metric. Historical figures/examples referenced by copied reports are deliberately omitted because they contain original images; regenerate only locally. This also applies to the five real-night example images.
