"""Read-only server audit; exits nonzero when GPU inference prerequisites are absent."""
import argparse
import json
import os
from pathlib import Path

from common import ROOT, environment, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT / 'data/im_t1_3d_1000')
    ap.add_argument('--gpu', default='0', help='physical GPU index or UUID')
    ap.add_argument('--torch-home', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    if a.torch_home:
        os.environ['TORCH_HOME'] = str(a.torch_home.resolve())
    r = environment(a.data, a.gpu)
    missing = []
    for name in ['numpy', 'pandas', 'scipy', 'torch', 'vismatch']:
        if not r['packages'][name]:
            missing.append(name)
    import importlib.util
    if not importlib.util.find_spec('cv2'):
        missing.append('cv2')
    if not r['torch'].get('cuda_available'):
        missing.append('CUDA-capable PyTorch/device access')
    if r['nvidia_smi']['returncode']:
        missing.append('nvidia-smi/driver access')
    if not r['samples_csv_exists']:
        missing.append('dataset samples.csv/images')
    from common import HERE
    official = json.loads((HERE / 'official_weights.json').read_text())
    r['required_loma_weights'] = [{**w, 'exists': (Path(r['weights_cache']) / w['filename']).is_file()}
                                  for w in official['weights']]
    for w in r['required_loma_weights']:
        if not w['exists']:
            missing.append('cached official weight: ' + w['filename'])
    r['weight_compatibility_status'] = 'Not validated by presence check; offline model initialization must pass state_dict checks'
    r['blocking_prerequisites'] = missing
    if a.out:
        write_json(a.out, r)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return int(bool(missing))


if __name__ == '__main__':
    raise SystemExit(main())
