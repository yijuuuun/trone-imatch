"""Fix source-verified reconstructed samples/preprocessing for a fresh Method B baseline."""
import argparse
import hashlib
import json
from pathlib import Path

from check_assets import audit
from common import HERE, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=HERE / 'method_b_dataset.json')
    a = ap.parse_args()
    check = audit(a.data)
    if not check['ready']:
        raise ValueError(check['issues'])
    prov = json.loads((a.data / 'export_provenance.json').read_text())
    if not prov['source_identity_verified'] or prov['n'] != 1000:
        raise ValueError('Expected original 1000 Singapore source identities')
    images = prov['exported_image_sha256']
    if len(images) != 2000:
        raise ValueError('Expected exactly 2000 exported pair images')
    for name, digest in images.items():
        if sha256(a.data / name) != digest:
            raise ValueError(f'Image hash mismatch: {name}')
    write_json(a.out, {'mode': 'method_b_reconstructed_fresh_baseline',
        'dataset_root': str(a.data.resolve()), 'n': 1000, 'kmin': 0, 'kmax': 999,
        'source_identity_verified': True, 'historical_pixel_identity_verified': False,
        'within_protocol_preprocessing_fixed': True,
        'manifest_hashes': {name: sha256(a.data/name) for name in ['samples.csv', 'source_mapping.csv', 'export_provenance.json']},
        'image_hashes_digest': hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest(),
        'source_zip_sha256': prov['zip_sha256'],
        'preprocessing': {'decode': 'OpenCV RGB uint8', 'source_size_px': [256, 256],
            'uav_size_px': [256, 256], 'block_size_px': [512, 512],
            'block_layout': [['p1', 'p2'], ['p3', 'p4']], 'jpeg_quality': 97,
            'resize_at_reconstruction': None, 'crop_at_reconstruction': None,
            'interpolation_at_reconstruction': None, 'augmentation': None,
            'opencv_version': prov['opencv_version']},
        'coordinates': 'Block center (0,0); 128 px/unit; 32 m/unit; original 512px map block',
        'evaluation': 'Ground-truth block T1; clipped [-1,1], missing pose=center; success error<10m',
        'historical_reference': '72.9% is a different prior protocol and is not a paired baseline here'})
    print(a.out)


if __name__ == '__main__':
    main()
