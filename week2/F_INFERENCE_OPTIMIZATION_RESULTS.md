# [F] Full LoMa-B inference optimization — verified experiment summary

Date: 2026-11-10 · Environment: linuxserver2 / gpu_02 · GPU: NVIDIA RTX A6000

**Provenance:** Values below are user-reported from independent GPU experiments. This commit contains an archival research summary **only**, not the server code, sample CSVs, official model weights, or dataset.

## Matched development pilot
20 paired samples, independent Oracle T1 scoring; rejected cases count as failures. **Not the official team navigation score.**

| Precision / headings | Mean end-to-end latency | Success@10m | Speedup vs FP32/8 |
|---|---:|---:|---:|
| FP32 / 8 | 1,828.1 ms | 70% (14/20) | 1.00× |
| FP16 / 8 | 832.7 ms | 65% (13/20) | 2.20× |
| FP32 / 4 | 930.9 ms | 70% (14/20) | 1.96× |
| FP16 / 4 | 415.2 ms | 65% (13/20) | 4.40× |

The FP32/8 → FP32/4 paired comparison had **zero success→failure transitions** in these 20 samples. FP16/4 lost one success, at k4. A 5 percentage-point change here represents one sample; the sample size is insufficient for a firm accuracy-equivalence claim.

Half-resolution input (0.5×) produced 50% success and was rejected. Stage-level latency and matching/geometry analyses were also completed on the server; original tables are not yet in this branch.

**Optimization, not structural pruning:** Heading count and numerical precision reduce inference work; they do not reduce Full LoMa-B parameter count. Earlier B128 pruning work is separate.

## Next steps
1. Retest FP32/8, FP32/4, FP16/4 on development k=0…99.
2. Integrate the shared team `imatch` geospatial and heading evaluation when the authorized source is accessible.
3. Keep held-out/test city data unused until team evaluation.
4. Preserve all experiment files and publish reviewed scripts and small result tables only.

## Server-side originals (not included in Git)
- `${WEEK2_WORKSPACE}/experiments/2026-11-13_light2/`
- `${WEEK2_WORKSPACE}/meeting_2026-11-11_v1/TEAM_INTEGRATION.md`
- `${WEEK2_WORKSPACE}/meeting_2026-11-11_v1/paired_transition_summary.csv`

**Do not upload:** raw UAV/satellite dataset imagery, raw NAT2021 frames, pretrained checkpoints, secrets, virtual environments, logs or all 6,000+ workspace files.


## Reproducible source export

The code, selected numerical CSVs, protocols and provenance documents are now under [F/README.md](F/README.md). This section supersedes historical statements above that source/CSV upload is pending. No source images or weights are included. The original independent/Oracle limitations still apply.
