# [E] Low-light robustness — verified experiment summary

Date: 2026-11-10 · Environment: linuxserver2 / gpu_02 · GPU: NVIDIA RTX A6000

**Provenance:** Values below are user-reported from completed experiments. Full source code, sample CSVs, original datasets, and model weights are still on the gpu_02 server and have **not** been uploaded to this GitHub branch.

## Synthetic low-light, Full LoMa-B
Independent Oracle T1 development pilot, 20 paired UAV/satellite samples; the team navigation evaluation is **not** reproduced here.

| Condition | Success@10m |
|---|---:|
| Daylight | 65% |
| Mild | 70% |
| Severe | 60% |

Gamma/CLAHE did not improve positional success in this pilot. Both Mild/Severe were classified **C: Experimental**, not calibrated to NAT2021. T4R original implementation and official camera parameters were not recovered.

## NAT2021 image-statistics audit
Official-distribution sample (20 sequences, 60 frames), not an official whole-dataset statistic. Scenes differ from the synthetic test.

| Condition | Mean sRGB brightness /255 | Dark pixel ratio (Y' < 0.05) |
|---|---:|---:|
| NAT2021 sample | 26.09 | 40.89% |
| Synthetic Mild | 90.14 | 1.68% |
| Synthetic Severe | 50.87 | 9.61% |

## NAT2021 real-night frame matching
20 sequences / 40 same-sequence frame pairs, 5-frame separation, **single heading FP32**, 120 GPU inferences across three conditions.

| Processing | Mean matches | Mean inliers | Mean per-pair inlier ratio | Valid homography | Mean latency |
|---|---:|---:|---:|---:|---:|
| Original | 1,355 | 654 | 47.2% | 100% | 1,367 ms |
| Gamma | 1,251 | 499 | 38.7% | 100% | 1,405 ms |
| CLAHE | 1,264 | 483 | 37.2% | 100% | 1,428 ms |

Under this **short-frame, high-overlap** setup, unprocessed night images had the best geometric consistency. An estimated valid homography is not proof of geolocation accuracy; NAT2021 has no satellite/geo ground truth for Success@10m in this evaluation.

## Server-side originals (not included in Git)
- `${WEEK2_WORKSPACE}/experiments/2026-11-13_lowlight/`
- `.../calibration_audit/CALIBRATION_REPORT.md`
- `.../nat2021_frame_matching_v1/REPORT.md`
- `.../nat2021_frame_matching_v1/samples.csv`

**Pending:** Upload only reviewed research code and small derived CSV/Markdown files. Do not commit NAT2021 raw frames, Bearing-UAV raw images, large weights, environment directories, credentials, or unreviewed copyrighted material. Get team code access for official location/navigation evaluation.


## Reproducible source export

The code, selected numerical CSVs, protocols and provenance documents are now under [E/README.md](E/README.md). This section supersedes historical statements above that source/CSV upload is pending. No source images or weights are included. The original independent/Oracle limitations still apply.
