# Export provenance and scope

The Python implementations were copied from the actually executed server sources. `configs/source_manifest.json` records original relative paths and SHA256 and exported SHA256. Only workspace/source path bindings, development-only restoration bounds (100), and branch-selective analysis were adapted. Low-light formulas, precision control, heading selection, coordinate restoration, geometry fitting and GT scoring are retained.

`src/runtime_paths.py`, `prepare_workspace.py` and `verify_export.py` are new packaging helpers, not research models. Historical config source hashes refer to the original executed code, while source_manifest records the portable copies. Historical commands and document links retain original relative structure; README below gives the exported commands. Absolute server paths in exported text were replaced with `${WEEK2_WORKSPACE}` or `<external-path>`. CSV numeric values were not changed. Source bytes and all prior measurements remain untouched.

Official LoMa: https://github.com/davnords/LoMa commit `415caedde23a9ed479172ed174fe4ff7b4514d4b`; 53 local files matched the official Git blob hashes. Install no new matcher model: obtain this source separately and pass its directory to prepare_workspace. Config contains per-file SHA256 for Python/TOML validation. Follow upstream MIT/Apache notices in the repository THIRD_PARTY_NOTICES.md. We do not redistribute upstream model source or weights in this export.

Excluded: all NAT2021/Bearing-UAV image/video/ZIP files, image-containing plots and match visualizations, all model weights, tensor correspondence NPZs, caches/virtualenvs, historical logs, full project/archive copies and credentials. Dataset manifest file names/hashes and numeric measurements are included; frames are not. Source dataset permissions must be obtained independently. UDAT code license is not a confirmed grant for the NAT2021 videos. No dataset license transfer is asserted.

The existing repository already has historical archives; this upload preserves them and only adds/updates the branch-specific week2 payload. No PR or main change is requested or performed.
