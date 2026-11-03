"""Validate the original Singapore 1000-pair layout/GT; never fabricate or change input data."""
import argparse
import csv
import json
import math
from pathlib import Path

from common import ROOT, SET, write_json


def audit(data):
    report = {'data_root': str(data.resolve()), 'set': SET, 'expected_k': '0..999',
              'samples_csv_exists': (data / 'samples.csv').is_file(), 'ready': False, 'issues': []}
    if not report['samples_csv_exists']:
        report['issues'].append('samples.csv not found; cannot establish GT/sample mapping')
        return report
    with (data / 'samples.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {'set', 'k', 'x_norm', 'y_norm', 'theta'} <= fields:
            report['issues'].append('Missing required manifest columns')
            return report
        rows = [r for r in reader if r['set'] == SET]
    report['singapore_rows'] = len(rows)
    try:
        keys = [int(r['k']) for r in rows]
        if sorted(keys) != list(range(1000)):
            report['issues'].append('Expected one unique row for each k=0..999')
        for r in rows:
            if not all(math.isfinite(float(r[k])) for k in ['x_norm', 'y_norm', 'theta']):
                raise ValueError('Non-finite GT')
    except (ValueError, TypeError):
        report['issues'].append('Invalid k or GT values')
        return report
    missing, bad_dimensions = [], []
    try:
        from PIL import Image
    except ImportError:
        Image = None
        report['issues'].append('Pillow missing; image decoding/dimensions cannot be validated')
    for k in range(1000):
        for kind, size in [('uav', 256), ('block', 512)]:
            p = data / SET / f'k{k:03d}_{kind}.jpg'
            if not p.is_file():
                missing.append(str(p))
            elif Image is not None:
                try:
                    with Image.open(p) as im:
                        im.load()
                        if im.size != (size, size):
                            bad_dimensions.append({'path': str(p), 'size': list(im.size)})
                except (OSError, ValueError) as e:
                    bad_dimensions.append({'path': str(p), 'decode_error': str(e)})
    report['missing_image_count'] = len(missing)
    report['missing_image_examples'] = missing[:10]
    report['invalid_images'] = bad_dimensions
    if missing or bad_dimensions:
        report['issues'].append('Missing, undecodable or unexpected-size images')
    # Match the historical (set,k) GT so a different Singapore export cannot silently replace it.
    with (ROOT / 'results/loma_8dir.csv').open(newline='') as f:
        expected = {int(r['k']): r for r in csv.DictReader(f) if r['set'] == SET}
    mismatches = []
    for r in rows:
        k = int(r['k'])
        ref = expected.get(k)
        if ref is None or any(not math.isclose(float(r[a]), float(ref[b]), rel_tol=0, abs_tol=1e-10)
                              for a, b in [('x_norm', 'gt_x'), ('y_norm', 'gt_y'), ('theta', 'gt_theta')]):
            mismatches.append(k)
    report['historical_gt_mismatch_k'] = mismatches
    if mismatches:
        report['issues'].append('GT differs from historical sample identities; do not relabel or reorder automatically')
    report['ready'] = not report['issues']
    report['uav_pattern'] = str(data / SET / 'k###_uav.jpg')
    report['satellite_block_pattern'] = str(data / SET / 'k###_block.jpg')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    report = audit(a.data)
    if a.out:
        write_json(a.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
