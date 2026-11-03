"""Shared protocol and provenance; no GPU imports at module load."""
import hashlib
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SET = '3d_ho37bc'
SPLITS = {'smoke': (0, 5), 'tune': (0, 100), 'final': (100, 1000), 'all': (0, 1000)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def command(args):
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {'returncode': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'returncode': -1, 'stdout': '', 'stderr': str(e)}


def protocol():
    return json.loads((HERE / 'conditions.json').read_text())


def source_fingerprint():
    files = sorted(HERE.glob('*.py')) + [HERE / 'conditions.json']
    files += [HERE / 'official_weights.json', HERE / 'requirements-loma-runtime.txt']
    if (HERE / 'method_b_dataset.json').is_file():
        files.append(HERE / 'method_b_dataset.json')
    files += sorted((ROOT / 'imatch').glob('*.py'))
    files += [ROOT / 'scripts' / 'evaluate.py', ROOT / 'requirements.txt']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def environment(data, gpu='0'):
    nvcc_on_path = shutil.which('nvcc')
    nvcc = nvcc_on_path
    if nvcc is None:
        candidates = [Path(os.environ[k]) / 'bin/nvcc' for k in ['CUDA_HOME', 'CUDA_PATH'] if os.environ.get(k)]
        candidates.append(Path('/usr/local/cuda/bin/nvcc'))
        nvcc = next((str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'torch', 'opencv-python', 'opencv-python-headless', 'vismatch']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        'os': platform.platform(), 'hostname': platform.node(), 'python': sys.version, 'executable': sys.executable,
        'virtual_env': os.environ.get('VIRTUAL_ENV'), 'conda_prefix': os.environ.get('CONDA_PREFIX'),
        'ssh_connected': bool(os.environ.get('SSH_CONNECTION')), 'tmux': shutil.which('tmux'),
        'gpu_selection': gpu, 'packages': packages, 'data': str(Path(data).resolve()),
        'samples_csv_exists': (Path(data) / 'samples.csv').is_file(),
        'nvidia_smi': command(['nvidia-smi']),
        'gpu_inventory': command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
        'gpu_processes': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']),
        'nvcc_on_path': nvcc_on_path, 'nvcc_resolved': nvcc,
        'nvcc': command([nvcc or 'nvcc', '--version']),
        'git_status': command(['git', '-C', str(ROOT), 'status', '--short', '--branch']),
        'pip_freeze': command([sys.executable, '-m', 'pip', 'freeze']),
    }
    # Resolve nvidia-smi physical indices to UUIDs before CUDA initialization.
    # CUDA's numeric enumeration need not match nvidia-smi on a mixed-GPU server.
    if result['gpu_inventory']['returncode'] == 0:
        rows = csv.reader(result['gpu_inventory']['stdout'].splitlines(), skipinitialspace=True)
        selected = [r for r in rows if len(r) >= 2 and (r[0] == gpu or r[1] == gpu)]
        if len(selected) == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = selected[0][1].strip()
    result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        import torch
        result['torch'] = {'version': torch.__version__, 'build_cuda': torch.version.cuda,
                           'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}
        cache = Path(torch.hub.get_dir()) / 'checkpoints'
    except Exception as e:
        result['torch'] = {'error': repr(e)}
        cache = Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch')) / 'hub/checkpoints'
    result['weights_cache'] = str(cache)
    result['cached_weights'] = [{'path': str(p), 'bytes': p.stat().st_size}
                               for p in sorted(cache.glob('*')) if p.is_file()]
    return result


def validate_samples(data, split):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(Path(data) / 'samples.csv')
    required = {'set', 'k', 'x_norm', 'y_norm', 'theta'}
    if not required.issubset(df.columns):
        raise ValueError(f'samples.csv missing {required - set(df.columns)}')
    lo, hi = SPLITS[split]
    df = df[(df['set'] == SET) & (df.k >= lo) & (df.k < hi)].sort_values('k')
    if list(df.k) != list(range(lo, hi)):
        raise ValueError(f'Expected exactly one Singapore row for each k={lo}..{hi-1}')
    if not np.isfinite(df[['x_norm', 'y_norm', 'theta']].to_numpy(float)).all():
        raise ValueError('non-finite ground truth')
    files = {}
    for k in df.k:
        for kind in ['uav', 'block']:
            p = Path(data) / SET / f'k{k:03d}_{kind}.jpg'
            if not p.is_file():
                raise FileNotFoundError(p)
            files[str(p.relative_to(data))] = sha256(p)
    return df, files


def validate_method_b_dataset(data):
    """Seal represents a fresh baseline protocol, not historical pixel identity."""
    seal = json.loads((HERE / 'method_b_dataset.json').read_text())
    if seal['dataset_root'] != str(Path(data).resolve()):
        raise ValueError('Dataset differs from the sealed Method B root')
    for name, expected in seal['manifest_hashes'].items():
        if sha256(Path(data) / name) != expected:
            raise ValueError(f'Method B manifest changed: {name}')
    prov = json.loads((Path(data) / 'export_provenance.json').read_text())
    if not prov['source_identity_verified']:
        raise ValueError('Original sample identities not verified')
    for name, expected in prov['exported_image_sha256'].items():
        if sha256(Path(data) / name) != expected:
            raise ValueError(f'Method B image changed: {name}')
    return seal
