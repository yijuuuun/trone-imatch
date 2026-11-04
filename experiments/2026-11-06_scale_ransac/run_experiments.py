"""Sequential LoMa 1-direction E3/E4 runner. No installation, downloads or overwrites."""
import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from common import (HERE, ROOT, SET, SPLITS, environment, protocol, sha256,
                    source_fingerprint, validate_samples, validate_method_b_dataset, write_json)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='new run directory, must not exist')
    ap.add_argument('--split', choices=['smoke', 'tune', 'final'], required=True)
    ap.add_argument('--gpu', default='0', help='physical GPU index or UUID; mapped to cuda:0')
    ap.add_argument('--torch-home', type=Path)
    ap.add_argument('--freeze', type=Path, help='required for final; created after tune analysis')
    ap.add_argument('--max-used-mib', type=int, default=1024, help='refuse occupied GPU above this limit')
    ap.add_argument('--min-free-mib', type=int, default=4096, help='minimum free VRAM, not a fit guarantee')
    ap.add_argument('--allow-shared-gpu', action='store_true', help='use only after coordinating server allocation')
    ap.add_argument('--dry-run', action='store_true', help='print protocol without importing GPU dependencies')
    return ap.parse_args(argv)


def seed_all(seed, torch, cv2, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)


class BasePixelLoMa:
    """Reuse the existing backend; estimate H at base pixels, return H at input pixels.

    The library's matched_kpts* are pre-RANSAC (not inlier_kpts*).
    Keep its default internal RANSAC for compatibility; its H/inliers are ignored.
    """
    def __init__(self, existing, scale):
        self.existing, self.scale = existing, scale

    def __call__(self, im0, im1):
        import numpy as np
        from imatch.geometry import magsac
        torch = self.existing.torch
        t = lambda im: torch.from_numpy(im).permute(2, 0, 1).float() / 255.
        with torch.inference_mode():
            r = self.existing.m(t(im0), t(im1))
        p0 = np.asarray(r['matched_kpts0'], dtype=np.float32).reshape(-1, 2)
        p1 = np.asarray(r['matched_kpts1'], dtype=np.float32).reshape(-1, 2)
        if p0.shape != p1.shape or not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            raise ValueError('Invalid raw correspondence arrays')
        _, _, H, inl = magsac(p0 / self.scale, p1 / self.scale)
        if H is not None:
            S = np.diag([self.scale, self.scale, 1.])
            H = S @ H @ np.linalg.inv(S)
        return p0, p1, H, inl


def monitor_gpu(gpu, path, stop):
    with open(path, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['utc', 'gpu_uuid', 'memory_used_mib', 'memory_free_mib', 'utilization_pct', 'query_status'])
        while not stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi', '-i', gpu,
                    '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True, capture_output=True, timeout=5)
                vals = next(csv.reader([r.stdout.strip()])) if r.returncode == 0 else ['', '', '', '']
                writer.writerow([datetime.now(timezone.utc).isoformat(), *vals, r.returncode])
                f.flush()
            except (OSError, subprocess.TimeoutExpired):
                writer.writerow([datetime.now(timezone.utc).isoformat(), '', '', '', '', 'unavailable'])
                f.flush()
            stop.wait(2)


def assert_free_gpu(a, env):
    inv, processes = env['gpu_inventory'], env['gpu_processes']
    if inv['returncode'] or processes['returncode']:
        raise RuntimeError('Cannot establish GPU occupancy; see environment.json')
    rows = list(csv.reader(inv['stdout'].splitlines(), skipinitialspace=True))
    selected = [r for r in rows if r[0] == a.gpu or r[1] == a.gpu]
    if len(selected) != 1:
        raise RuntimeError('Select one GPU by physical index or full UUID')
    row = selected[0]
    used, free = float(row[4]), float(row[5])
    busy = any(r and r[0].strip() == row[1].strip()
               for r in csv.reader(processes['stdout'].splitlines(), skipinitialspace=True))
    if free < a.min_free_mib:
        raise RuntimeError(f'GPU has only {free} MiB free')
    if (busy or used > a.max_used_mib) and not a.allow_shared_gpu:
        raise RuntimeError('GPU occupied; select an allocated idle GPU')


def main(argv=None):
    a = parse_args(argv)
    cfg = protocol()
    if a.dry_run:
        lo, hi = SPLITS[a.split]
        print(json.dumps({'split': a.split, 'kmin': lo, 'kmax_exclusive': hi, 'gpu': a.gpu,
                          'out': str(a.out), **cfg}, ensure_ascii=False, indent=2))
        return
    if a.split == 'final' and not a.freeze:
        raise ValueError('--freeze is required before opening the final evaluation split')
    export_provenance = a.data / 'export_provenance.json'
    if export_provenance.is_file():
        data_info = json.loads(export_provenance.read_text())
        if not data_info.get('source_identity_verified'):
            raise ValueError('Reconstructed source sample identities are not verified')
        if cfg.get('dataset_policy') == 'method_b_reconstructed_fresh_baseline':
            validate_method_b_dataset(a.data)
        elif not data_info.get('preprocessing_verified') and a.split != 'smoke':
            raise ValueError('Original pixel preprocessing is unverified; reconstruction is restricted to diagnostic smoke')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if a.torch_home:
        os.environ['TORCH_HOME'] = str(a.torch_home.resolve())
    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / 'console.log', 'x', buffering=1) as log:
        class Tee:
            def __init__(self, stream): self.stream = stream
            def write(self, s): self.stream.write(s); log.write(s)
            def flush(self): self.stream.flush(); log.flush()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout), Tee(stderr)
        try:
            run(a, cfg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            write_json(a.out / 'FAILED.json', {'error': repr(exc), 'complete': False,
                'note': 'Partial files are not evaluation results. Use a new output path after resolving the cause.'})
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def run(a, cfg):
    env = environment(a.data, a.gpu)
    write_json(a.out / 'environment.json', env)
    if not env['torch'].get('cuda_available'):
        raise RuntimeError('CUDA unavailable; GPU experiments have not run')
    assert_free_gpu(a, env)
    import cv2
    import numpy as np
    import torch
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from imatch import geometry, get_matcher, localize
    from imatch.metrics import errors
    df, image_hashes = validate_samples(a.data, a.split)
    # Warm-up always uses a development sample, even on the final split.
    for kind in ['uav', 'block']:
        if not (a.data / SET / f'k000_{kind}.jpg').is_file():
            raise FileNotFoundError('Warm-up requires Singapore k000 images')
    fingerprint, source_hashes = source_fingerprint()
    if a.freeze:
        frozen = json.loads(a.freeze.read_text())
        if frozen['source_fingerprint'] != fingerprint or frozen['protocol'] != cfg:
            raise ValueError('Code or protocol changed since freeze; review using tune split first')
        write_json(a.out / 'freeze.json', frozen)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed_all(cfg['seed'], torch, cv2, np)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    loaded_weights = {}
    original_loader = torch.hub.load_state_dict_from_url

    def cached_only(url, *args, **kwargs):
        cache = Path(kwargs.get('model_dir') or Path(torch.hub.get_dir()) / 'checkpoints')
        name = kwargs.get('file_name') or os.path.basename(urlparse(url).path)
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(f'Offline model weight missing: {path}; URL={url}')
        loaded_weights[str(path.resolve())] = sha256(path)
        return original_loader(url, *args, **kwargs)

    torch.hub.load_state_dict_from_url = cached_only
    matcher = get_matcher('loma', 'cuda')
    if not loaded_weights:
        raise RuntimeError('No torch.hub weights observed; audit this vismatch version before using this protocol')
    backend_source = inspect.getsourcefile(type(matcher.m))
    base_source = inspect.getsourcefile(type(matcher.m).__mro__[1])
    provenance = {'source_fingerprint': fingerprint, 'source_hashes': source_hashes,
        'weights': loaded_weights, 'packages': env['packages'],
        'backend_source_sha256': sha256(backend_source), 'base_matcher_source_sha256': sha256(base_source),
        'samples_csv_sha256': sha256(a.data / 'samples.csv'), 'image_sha256': image_hashes,
        'protocol': cfg, 'split': a.split, 'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
        'argv': sys.argv, 'utc_start': datetime.now(timezone.utc).isoformat()}
    data_info_path = a.data / 'export_provenance.json'
    if data_info_path.is_file():
        data_info = json.loads(data_info_path.read_text())
        provenance['dataset_export_provenance_sha256'] = sha256(data_info_path)
        provenance['dataset_preprocessing_verified'] = data_info['preprocessing_verified']
        provenance['dataset_limitation'] = data_info['limitation']
    if cfg.get('dataset_policy') == 'method_b_reconstructed_fresh_baseline':
        provenance['dataset_policy'] = cfg['dataset_policy']
        provenance['dataset_seal_sha256'] = sha256(HERE / 'method_b_dataset.json')
    # Archive installed wrappers for the coordinate/pre-RANSAC audit.
    (a.out / 'installed_loma_wrapper.py.txt').write_text(Path(backend_source).read_text())
    (a.out / 'installed_base_matcher.py.txt').write_text(Path(base_source).read_text())
    if a.freeze:
        expected = frozen['execution_identity']
        for key in expected:
            if provenance[key] != expected[key]:
                raise ValueError(f'Final execution differs from frozen tune environment: {key}')
    write_json(a.out / 'provenance.json', provenance)
    df.to_csv(a.out / 'samples_selected.csv', index=False, mode='x')

    def read(k, kind):
        p = a.data / SET / f'k{k:03d}_{kind}.jpg'
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f'Cannot decode {p}')
        expected = 256 if kind == 'uav' else 512
        if im.shape[:2] != (expected, expected):
            raise ValueError(f'Unexpected image dimensions {p}: {im.shape}')
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    for condition in cfg['conditions']:
        name, scale, threshold = condition['name'], condition['scale'], condition['magsac_threshold']
        folder = a.out / name
        folder.mkdir()
        geometry.set_ransac_px(threshold)
        adapter = BasePixelLoMa(matcher, scale)
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(a.gpu, folder / 'gpu.csv', stop), daemon=True)
        monitor.start()
        rows = []
        condition_start = time.perf_counter()
        condition_log = open(folder / 'inference.log', 'x', buffering=1)
        try:
            condition_log.write(json.dumps(condition) + '\n')
            # Exclude one warmed inference per condition from all sample timings.
            seed_all(cfg['seed'], torch, cv2, np)
            localize(adapter, read(0, 'uav'), read(0, 'block'), (0, 0, 0), scale, [0.])
            torch.cuda.synchronize()
            warmup_sec = time.perf_counter() - condition_start
            condition_log.write(f'warmup_sec={warmup_sec:.3f}\n')
            with open(folder / 'samples.partial.csv', 'x', newline='') as f:
                writer = None
                for s in df.itertuples():
                    uav, block = read(s.k, 'uav'), read(s.k, 'block')
                    seed_all(cfg['seed'] + int(s.k), torch, cv2, np)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    recs, best = localize(adapter, uav, block, (s.x_norm, s.y_norm, s.theta), scale, [0.])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    b = best or {}
                    px, py, heading = b.get('px', np.nan), b.get('py', np.nan), b.get('head', np.nan)
                    if best and not np.isfinite([px, py, heading]).all():
                        raise ValueError(f'Non-finite pose at k={s.k}')
                    err, fail = errors([px], [py], [s.x_norm], [s.y_norm])
                    row = dict(sample_id=f'{SET}:{s.k:03d}', city='Singapore', set=SET, k=int(s.k),
                        experiment_name=name, matcher='loma', scale=scale, magsac_threshold=threshold,
                        threshold_frame='base_block_px', effective_resized_threshold_px=threshold * scale,
                        gt_x=s.x_norm, gt_y=s.y_norm, gt_theta=s.theta,
                        pred_x=px, pred_y=py, pred_theta=heading, angle=0.,
                        n_inl=int(recs[0]['n_inl']), n_match=int(recs[0]['n_match']),
                        position_error_m=float(err[0]), success_10m=bool(err[0] < 10),
                        pipeline_failure=bool(fail[0]), status='ok' if best else 'no_homography',
                        dataset_preprocessing_verified=provenance.get('dataset_preprocessing_verified', 'not_recorded'),
                        # err_m retains scripts/run.py's un-clipped recording convention.
                        err_m=float(np.hypot((px if np.isfinite(px) else 0)-s.x_norm,
                                             (py if np.isfinite(py) else 0)-s.y_norm)*32),
                        head_err=float(geometry.wrap((heading if np.isfinite(heading) else 0)-s.theta)),
                        sec=elapsed, runtime_sec=elapsed, matcher_sec=recs[0]['sec'],
                        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    message = (f'{name} k={s.k} inlier={row["n_inl"]} error={err[0]:.2f}m '
                               f'time={elapsed:.3f}s peak={row["gpu_peak_allocated_mib"]:.0f}MiB')
                    print(message, flush=True)
                    condition_log.write(message + '\n')
            (folder / 'samples.partial.csv').rename(folder / 'samples.csv')
            write_json(folder / 'completed.json', {'condition': condition, 'n': len(rows),
                'warmup_sec': warmup_sec, 'wall_sec': time.perf_counter()-condition_start,
                'gpu_peak_allocated_mib': max(r['gpu_peak_allocated_mib'] for r in rows),
                'gpu_peak_reserved_mib': max(r['gpu_peak_reserved_mib'] for r in rows)})
        except Exception:
            import traceback
            condition_log.write(traceback.format_exc())
            raise
        finally:
            condition_log.close()
            stop.set()
            monitor.join(timeout=6)
    write_json(a.out / 'completed.json', {'split': a.split, 'n_per_condition': len(df),
        'conditions': [c['name'] for c in cfg['conditions']], 'wall_sec': time.perf_counter()-start})
    print(f'Completed {a.split}: {a.out}', flush=True)


if __name__ == '__main__':
    main()
