"""Audit the uploaded ZIP against persisted sample identities, optionally export diagnostic data.

No sampling, coordinate fitting, whole-archive extraction or changes to original inputs.
The original load_rows implementation/prepared-image hashes are unavailable: this tool
does not certify byte-equivalence to the historic prepared dataset.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from common import ROOT, SET, sha256, write_json

PREFIX = 'Bearing_UAV_90K/'
METADATA = PREFIX + 'c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'


def csv_rows(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def member_name(path):
    name = path.removeprefix('./')
    if not name.startswith(PREFIX + 'cityc/') or '..' in Path(name).parts or '\\' in name:
        raise ValueError(f'Unexpected Singapore source path: {path}')
    return name


def verify_mapping(manifest, historical, metadata):
    rows = sorted([r for r in manifest if r['set'] == SET], key=lambda r: int(r['k']))
    refs = sorted([r for r in historical if r['set'] == SET], key=lambda r: int(r['k']))
    if [int(r['k']) for r in rows] != list(range(1000)) or [int(r['k']) for r in refs] != list(range(1000)):
        raise ValueError('Expected unique historical Singapore k=0..999; no reindexing allowed')
    max_diff = 0.
    mapping = []
    for s, ref in zip(rows, refs):
        row = int(s['row'])
        if row not in metadata:
            raise ValueError(f'Persisted row {row} missing in ZIP metadata')
        raw = metadata[row]
        for original, result in [('x_norm', 'gt_x'), ('y_norm', 'gt_y'), ('theta', 'gt_theta')]:
            values = [float(s[original]), float(ref[result]), float(raw[original])]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Non-finite original GT')
            delta = max(abs(values[0]-values[1]), abs(values[0]-values[2]))
            max_diff = max(max_diff, delta)
            if delta > 1e-12:
                raise ValueError(f'GT mismatch k={s["k"]}, persisted row={row}, field={original}')
        paths = {key: member_name(raw[key]) for key in ['p1_path', 'p2_path', 'p3_path', 'p4_path', 'target_path']}
        if not paths['target_path'].endswith('_v3d.jpg'):
            raise ValueError('Expected actual 3D UAV, not a 2D satellite crop')
        mapping.append({'sample': s, 'row': row, 'k': int(s['k']), 'sources': paths})
    if len({m['sources']['target_path'] for m in mapping}) != 1000:
        raise ValueError('Historical rows do not identify 1000 distinct UAV images')
    return mapping, max_diff


def audit(z, manifest_path):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ZIP member names; refusing ambiguous reads')
    members = set(names)
    if METADATA not in members:
        raise ValueError('Original 90K 3D metadata CSV not found')
    samples, fields = csv_rows(manifest_path)
    refs, _ = csv_rows(ROOT / 'results/loma_8dir.csv')
    selected = [r for r in samples if r['set'] == SET]
    wanted = {int(r['row']) for r in selected}
    metadata_bytes = z.read(METADATA)
    metadata, city_counts = {}, collections.Counter()
    reader = csv.DictReader(io.StringIO(metadata_bytes.decode('utf-8-sig')))
    total_rows = 0
    for i, row in enumerate(reader):
        total_rows += 1
        city_counts['/'.join(row['target_path'].removeprefix('./').split('/')[:2])] += 1
        if i in wanted:
            metadata[i] = row
    mapping, max_diff = verify_mapping(samples, refs, metadata)
    # Three independent existing baselines retain the original global row identities.
    baselines, _ = csv_rows(ROOT / 'results/baselines_3d_k0-999.csv')
    baseline_rows = {}
    for r in baselines:
        if r['set'] == SET:
            baseline_rows.setdefault(r['method'], {})[int(r['k'])] = int(r['row'])
    for method, indexed in baseline_rows.items():
        if [indexed.get(k) for k in range(1000)] != [m['row'] for m in mapping]:
            raise ValueError(f'Baseline row mismatch: {method}')
    prefix, _ = csv_rows(ROOT / 'results/raw_capstone1/out_imatch/samples.csv')
    prefix = [r for r in prefix if r['set'] == SET]
    for r in prefix:
        s = mapping[int(r['k'])]['sample']
        if any(r[k] != s[k] for k in ['row', 'x_norm', 'y_norm', 'theta']):
            raise ValueError('Stage-1 prefix differs from persisted full manifest')
    required = sorted({p for m in mapping for p in m['sources'].values()})
    missing = [p for p in required if p not in members]
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected image members absent: {missing[:3]}')
    info = {i.filename: i for i in z.infolist()}
    report = {'zip_entries': len(names), 'zip_total_uncompressed_bytes': sum(i.file_size for i in info.values()),
        'zip_top_levels': dict(collections.Counter('/'.join(n.split('/')[:2]) for n in names)),
        'prepared_eval_files_present': any(n.endswith('samples.csv') or '/3d_ho37bc/' in n or 'im_t1_3d_1000' in n for n in names),
        'metadata_member': METADATA, 'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'metadata_rows': total_rows, 'metadata_city_counts': dict(city_counts),
        'historical_manifest': str(manifest_path), 'historical_manifest_sha256': sha256(manifest_path),
        'set': SET, 'n': len(mapping), 'kmin': 0, 'kmax': 999,
        'historical_gt_max_abs_difference': max_diff,
        'existing_baseline_row_checks': sorted(baseline_rows), 'stage1_prefix_n_verified': len(prefix),
        'source_identity_verified': True, 'selected_unique_source_images': len(required),
        'selected_image_bytes': sum(info[n].file_size for n in required),
        'original_load_rows_available': False, 'original_prepared_image_hashes_available': False,
        'preprocessing_verified': False, 'eligible_for_final_evaluation': False,
        'known_export_steps': 'RGB images; block=[[p1,p2],[p3,p4]]; cv2 JPEG quality=97; no resize/augmentation added',
        'limitation': 'Original load_rows code and exported image hashes unavailable. Raw identities/GT are verified; original pixel preprocessing is not certified.'}
    return mapping, fields, report


def export(z, mapping, fields, report, out):
    import cv2
    import numpy as np
    from PIL import Image
    out.mkdir(parents=True, exist_ok=False)
    folder = out / SET
    folder.mkdir()
    cache, source_hashes, output_hashes = {}, {}, {}
    decoder_differences = []
    def read_rgb(name):
        if name in cache:
            return cache[name]
        raw = z.read(name)  # zipfile verifies CRC only for the selected members.
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (256, 256, 3):
            raise ValueError(f'Unexpected selected source dimensions: {name}')
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        with Image.open(io.BytesIO(raw)) as pil:
            if not np.array_equal(rgb, np.asarray(pil.convert('RGB'))):
                decoder_differences.append(name)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        if '/sat_' in name:
            cache[name] = rgb
        return rgb
    mappings = []
    for m in mapping:
        source = m['sources']
        p1, p2, p3, p4 = [read_rgb(source[f'p{i}_path']) for i in range(1, 5)]
        block = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
        uav = read_rgb(source['target_path'])
        for kind, rgb in [('uav', uav), ('block', block)]:
            dest = folder / f'k{m["k"]:03d}_{kind}.jpg'
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
            if not ok:
                raise RuntimeError(f'JPEG encoding failed: {dest}')
            raw = encoded.tobytes()
            with dest.open('xb') as f:
                f.write(raw)
            output_hashes[str(dest.relative_to(out))] = hashlib.sha256(raw).hexdigest()
        mappings.append({'set': SET, 'k': m['k'], 'row': m['row'], **source})
        if (m['k'] + 1) % 200 == 0:
            print(f'Exported {m["k"]+1}/1000 verified source identities', flush=True)
    with (out / 'samples.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(m['sample'] for m in mapping)
    with (out / 'source_mapping.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(mappings[0]))
        writer.writeheader()
        writer.writerows(mappings)
    write_json(out / 'export_provenance.json', {**report, 'opencv_version': cv2.__version__,
        'source_image_sha256': source_hashes, 'exported_image_sha256': output_hashes,
        'opencv_vs_pillow_decode_difference_members': sorted(set(decoder_differences)),
        'status': 'source-aligned diagnostic reconstruction; not certified as the original prepared image bytes'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=ROOT / 'results/raw_capstone1/out_imatch/samples_1000.csv')
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--export', type=Path, help='new folder; only the selected pairs are decoded/re-encoded')
    ap.add_argument('--diagnostic-only', action='store_true', help='acknowledge missing original preprocessing verification')
    a = ap.parse_args()
    if a.export and not a.diagnostic_only:
        ap.error('Original load_rows is unavailable; export requires --diagnostic-only and cannot enter final evaluation')
    with zipfile.ZipFile(a.zip) as z:
        mapping, fields, report = audit(z, a.manifest)
        report['zip_path'] = str(a.zip.resolve())
        report['zip_bytes'] = a.zip.stat().st_size
        report['zip_sha256'] = sha256(a.zip)
        write_json(a.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if a.export:
            export(z, mapping, fields, report, a.export)


if __name__ == '__main__':
    main()
