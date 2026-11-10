# Packaging validation

All exported Python files parsed and all modules imported in an isolated interpreter using the existing tested environment. Core import and deterministic preprocessing smoke passed. Original/export lowlight transforms match byte-for-byte for 240 light/correction/seed cases per branch.

A new external workspace passed official-source and three weight SHA checks. Exactly100 development pairs were restored from the external ZIP; every source image and reconstructed JPEG hash passed. Source metadata coordinate checks: max3.11e-9px discrepancy. No locked test data was restored or evaluated.

Actual GPU smoke: official341,523,567-parameter Full LoMa-B strict loading passed. Exported E NAT original first pair produced1511 matches/1013 inliers and valid H, exactly matching the frozen original counts. Exported F FP16/4 k0 produced280 matches/53 inliers, exactly matching frozen original counts. GPU smoke is limited to one pair per path, not a fresh20/120-case replication or a new latency benchmark. Common backend code is identical in both branch exports.

Every included result CSV matches the corresponding original parsed rows, including all numerical values. Original research code/report/result hashes were checked after packaging; zero changed. Detailed checks and CSV row counts are in VALIDATION.json. No training, PR or main modification is part of this export.
