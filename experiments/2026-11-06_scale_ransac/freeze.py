"""Record a development-only decision before evaluating k=100..999."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze import IDENTITY_KEYS, read_runs
from common import protocol, sha256, source_fingerprint, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tune-run', type=Path, required=True)
    ap.add_argument('--tune-analysis', type=Path, required=True)
    ap.add_argument('--candidate', choices=[c['name'] for c in protocol()['conditions']], required=True)
    ap.add_argument('--reason', required=True, help='decision based only on k=0..99, include resource trade-off')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if json.loads((a.tune_run / 'provenance.json').read_text())['split'] != 'tune':
        raise ValueError('Expected a tune run only')
    _, manifests, identity = read_runs([a.tune_run], 'tune')
    analysis = json.loads((a.tune_analysis / 'analysis_manifest.json').read_text())
    if analysis['split'] != 'tune' or analysis['runs'] != manifests:
        raise ValueError('Analysis is not from this development run')
    if not analysis['baseline_e4_3_geometry_identical']:
        raise ValueError('Baseline reproducibility check failed; investigate before final evaluation')
    fingerprint, _ = source_fingerprint()
    if fingerprint != identity['source_fingerprint']:
        raise ValueError('Code changed after tune run')
    write_json(a.out, {'utc_frozen': datetime.now(timezone.utc).isoformat(),
        'candidate': a.candidate, 'reason': a.reason, 'source_fingerprint': fingerprint,
        'protocol': protocol(), 'execution_identity': {k: identity[k] for k in IDENTITY_KEYS},
        'tune_run_provenance_sha256': sha256(a.tune_run / 'provenance.json'),
        'tune_report_sha256': sha256(a.tune_analysis / 'REPORT.md'),
        'rule': 'Compare all five predeclared conditions once on final; do not tune based on final outcomes.'})
    print(a.out)


if __name__ == '__main__':
    main()
