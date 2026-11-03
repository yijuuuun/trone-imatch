"""Estimate sequential server time/VRAM from measured smoke data; no GPU inference."""
import argparse
import json
from pathlib import Path

from analyze import read_runs
from common import protocol, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if json.loads((a.smoke_run / 'provenance.json').read_text())['split'] != 'smoke':
        raise ValueError('Expected a completed smoke run')
    measured, _, _ = read_runs([a.smoke_run], 'smoke')
    rows = []
    for c in protocol()['conditions']:
        r = measured[c['name']]
        meta = json.loads((a.smoke_run / c['name'] / 'completed.json').read_text())
        rows.append({'condition': c['name'], 'mean_inference_sec': float(r.sec.mean()),
            'observed_peak_allocated_mib': float(r.gpu_peak_allocated_mib.max()),
            'observed_peak_reserved_mib': float(r.gpu_peak_reserved_mib.max()),
            'estimated_tune_inference_sec': float(r.sec.mean()*100),
            'estimated_final_inference_sec': float(r.sec.mean()*900),
            'measured_warmup_sec': meta['warmup_sec']})
    result = {'smoke_provenance_sha256': sha256(a.smoke_run / 'provenance.json'), 'conditions': rows,
        'estimated_tune_sec_excluding_setup_io': sum(r['estimated_tune_inference_sec']+r['measured_warmup_sec'] for r in rows),
        'estimated_final_sec_excluding_setup_io': sum(r['estimated_final_inference_sec']+r['measured_warmup_sec'] for r in rows),
        'caveat': '5 smoke samples cannot guarantee peak VRAM or runtime of all samples. Reserve headroom; batch=1; one GPU; conditions sequential. Setup, image IO, hashing and process initialization excluded.'}
    write_json(a.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
