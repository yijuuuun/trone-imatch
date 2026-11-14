"""Common single-image localization protocol. No ground truth reaches adapters."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import math
import random
import time
import csv
import io
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def discover_sample_csv(root, extract_dir=None, fallback_root=None):
    """Find by schema, not Kaggle slug layout. Refuse ambiguous datasets."""
    required = {'set', 'k', 'x_norm', 'y_norm'}
    root = Path(root)
    def compatible(header):
        return required.issubset(next(csv.reader([header.lstrip('\ufeff')]), []))
    def scan(folder):
        candidates = []
        for p in sorted(folder.rglob('*.csv')):
            try:
                with p.open(encoding='utf-8-sig') as f:
                    if compatible(f.readline()):
                        candidates.append(p)
            except (UnicodeError, OSError, csv.Error):
                continue
        return candidates
    paths = scan(root) if root.exists() else []
    if not paths and fallback_root and Path(fallback_root).exists() and root != Path(fallback_root):
        print(f'No compatible CSV under {root}; searching {fallback_root}')
        root = Path(fallback_root)
        paths = scan(root)
    if len(paths) > 1:
        raise ValueError(f'Multiple compatible CSVs. Set SAMPLE_CSV explicitly: {[str(p) for p in paths]}')
    if paths:
        return paths[0]
    archives = []
    for archive in sorted(root.rglob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.filename.lower().endswith('.csv'):
                    try:
                        with z.open(member) as f:
                            header = f.readline(65536).decode('utf-8-sig')
                        if compatible(header):
                            archives.append((archive, member.filename))
                    except (UnicodeError, csv.Error):
                        continue
    if len(archives) > 1:
        raise ValueError(f'Multiple datasets inside ZIPs; select/extract one explicitly: {archives}')
    if archives:
        if extract_dir is None:
            raise ValueError(f'Dataset is zipped: {archives[0]}. Set extract_dir to a writable folder.')
        archive, member_name = archives[0]
        destination = Path(extract_dir) / sha256(archive)[:16]
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f'Unsafe ZIP member: {member.filename}')
            z.extractall(destination)
        return destination / member_name
    inventory = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()][:60]
    raise FileNotFoundError(f'No CSV with columns {sorted(required)} under {root}. '
                            f'Input files (first 60): {inventory}. Raw VisLoc data needs conversion; labels are not inferred.')


def read_dataset(root, sample_csv=None, limit=None, extract_dir=None, fallback_root=None):
    root = Path(root)
    paths = [Path(sample_csv)] if sample_csv else [discover_sample_csv(root, extract_dir, fallback_root)]
    table = pd.read_csv(paths[0])
    required = {'set', 'k', 'x_norm', 'y_norm'}
    if not required.issubset(table.columns):
        raise ValueError(f'Missing columns: {required - set(table.columns)}')
    if table.empty or table.duplicated(['set', 'k']).any():
        raise ValueError('Empty dataset or duplicate (set,k) identifiers')
    table = table.sort_values(['set', 'k']).reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        table = table.iloc[:limit].copy()
    if not np.isfinite(table[['x_norm', 'y_norm']].to_numpy(float)).all():
        raise ValueError('Nonfinite ground truth')
    base = paths[0].parent
    records = []
    for r in table.to_dict('records'):
        item = {'sample_id': f"{r['set']}:{int(r['k'])}", 'set': r['set'], 'k': int(r['k']),
                'gt_x': float(r['x_norm']), 'gt_y': float(r['y_norm'])}
        for kind in ('uav', 'block'):
            p = base / str(r.get(f'{kind}_path', f"{r['set']}/k{int(r['k']):03d}_{kind}.jpg"))
            if not p.is_file():
                raise FileNotFoundError(p)
            item[f'{kind}_path'] = str(p)
            item[f'{kind}_sha256'] = sha256(p)
        records.append(item)
    return pd.DataFrame(records), paths[0]


def dataset_audit(manifest, meters_per_unit):
    e = np.hypot(manifest.gt_x, manifest.gt_y) * meters_per_unit
    centered = bool(np.allclose(manifest[['gt_x', 'gt_y']], 0, atol=1e-6))
    return {'n': len(manifest), 'all_labels_at_center': centered,
            'center_baseline_accuracy_10m_pct': float((e < 10).mean() * 100),
            'position_ranking_valid': not centered,
            'note': 'GPS-centered crops reveal the target location; position ranking is diagnostic only.' if centered
                    else 'Nonconstant labels do not prove absence of leakage; audit crop generation and train/test overlap.'}


def load_pair(row):
    images = []
    for key, shape in [('uav_path', (256, 256)), ('block_path', (512, 512))]:
        bgr = cv2.imread(row[key])
        if bgr is None or bgr.shape[:2] != shape:
            raise ValueError(f'{row[key]}: expected {shape}, got {None if bgr is None else bgr.shape}')
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


def position_error(prediction, gx, gy, meters_per_unit):
    x, y = float(prediction.get('x', float('nan'))), float(prediction.get('y', float('nan')))
    valid = bool(prediction.get('valid', True)) and math.isfinite(x) and math.isfinite(y)
    # Missing/rejected predictions are failures, including when the GT is exactly (0,0).
    # No clipping: clipping toward GT can inflate accuracy on wrong map blocks.
    error = math.hypot(x - gx, y - gy) * meters_per_unit if valid else float('inf')
    return x, y, valid, error


def summarize(predictions, params, audit, metadata):
    e = predictions.error_m.to_numpy(float)
    times = predictions.seconds.to_numpy(float)
    return {**metadata, 'status': 'measured', 'n': len(e),
            'accuracy_10m_pct': float((e < 10).mean() * 100),
            'accuracy_20m_pct': float((e < 20).mean() * 100),
            'accuracy_32m_pct': float((e < 32).mean() * 100),
            'valid_pct': float(predictions.valid.mean() * 100),
            'mean_error_valid_m': float(e[np.isfinite(e)].mean()) if np.isfinite(e).any() else np.nan,
            'mean_ms': float(times.mean() * 1000), 'p50_ms': float(np.median(times) * 1000),
            'p95_ms': float(np.percentile(times, 95) * 1000), 'fps': float(1 / times.mean()),
            'total_parameters': params, 'parameters_M': params / 1e6,
            'position_ranking_valid': audit['position_ranking_valid']}


def all_modules(roots):
    """Find nn.Modules in wrappers/plain containers too; shared objects counted once."""
    seen, modules = set(), []
    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            modules.append(obj)
            for value in vars(obj).values():
                visit(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                visit(value)
    visit(roots)
    return modules


def parameter_count(roots):
    unique = {}
    for module in all_modules(roots):
        for p in module.parameters(recurse=False):
            unique[id(p)] = p
    if not unique:
        raise ValueError('No parameters found; provide every learned component in adapter.modules')
    return sum(p.numel() for p in unique.values())


def fp32_audit(roots):
    """Observe Conv/Linear computation during warmup, including hidden modules."""
    observed, handles = set(), []
    def capture(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            observed.add(str(output.dtype))
    for module in all_modules(roots):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(capture))
    return observed, handles


def enforce_fp32(roots):
    """Disable model-internal AMP as well as converting parameter storage."""
    from dataclasses import fields, is_dataclass, replace
    for module in all_modules(roots):
        module.float()
        for name in ('amp', 'use_amp'):
            if hasattr(module, name) and isinstance(getattr(module, name), bool):
                setattr(module, name, False)
        if hasattr(module, 'amp_dtype'):
            module.amp_dtype = torch.float32
        cfg = getattr(module, 'cfg', None)
        if is_dataclass(cfg):
            changes = {f.name: False for f in fields(cfg) if f.name in ('mp', 'amp', 'use_amp')}
            if changes:
                module.cfg = replace(cfg, **changes)


class IMatchAdapter:
    def __init__(self, device, angles=8, stop_inl=15, checkpoint=''):
        import os
        os.environ['LOMA_FP32'] = '1'
        from imatch.matchers import get_matcher
        from imatch.pipeline import localize
        self.match = get_matcher('loma', device)
        self.localize = localize
        self.modules = [self.match.m.matcher]
        # Full LoMa model, including lazy detector/descriptor, is initialized by warmup.
        from vismatch.utils import set_device_globals
        set_device_globals('loma', device, amp_dtype=torch.float32)
        enforce_fp32(self.modules)
        if checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = state.get('model_state_dict', state)
            self.modules[0].load_state_dict(state, strict=True)
        self.angles = list(np.arange(0, 360, 360 / angles))
        self.stop_inl = stop_inl
        self.metadata = {'model': 'iMatch', 'variant': f'LoMa-B/{angles}dir/stop={stop_inl}',
                         'weights': str(checkpoint) or 'official LoMa pretrained (no task-specific training)',
                         'checkpoint_sha256': sha256(checkpoint) if checkpoint else '',
                         'training_regime': 'public-pretrained' if not checkpoint else 'user-checkpoint: audit training split'}

    def predict(self, uav, block):
        # GT argument is deliberately constant and is used only by legacy match diagnostics.
        records, best = self.localize(self.match, uav, block, (0., 0., 0.),
                                      angles=self.angles, stop_inl=self.stop_inl)
        if best is None:
            return {'valid': False}
        # Single-image benchmark: no R1 dead reckoning; low inlier count is recorded, not silently imputed.
        return {'x': best['px'], 'y': best['py'], 'valid': True,
                'inliers': best['n_inl'], 'n_angles': len(records)}


class BearingAdapter:
    def __init__(self, repo, weights_dir, device):
        import sys
        sys.path.insert(0, str(repo))
        # Official utils_transform accesses np.complex_ during import on NumPy 2.
        for name, value in {'complex_': np.complex128, 'float_': np.float64}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        # imgaug 0.4 imports np.sctypes, removed in NumPy 2. This compatibility alias
        # affects import-time augmentation setup; evaluation uses only ToTensor/Normalize.
        if not hasattr(np, 'sctypes'):
            np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64],
                          'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                          'float': [np.float16, np.float32, np.float64],
                          'complex': [np.complex64, np.complex128], 'others': [bool, object, str, bytes]}
        from cvphr.models.posaglreg.models import load_config_and_model
        cls, kw = load_config_and_model(str(weights_dir))
        self.model = cls(**kw)
        path = Path(weights_dir) / 'best_model.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.model = self.model.float().to(device).eval()
        self.device, self.modules = device, [self.model]
        self.metadata = {'model': 'Bearing-UAV', 'variant': 'official cross_view',
                         'weights': str(path), 'checkpoint_sha256': sha256(path),
                         'training_regime': 'public-pretrained: training-city overlap must be audited'}

    def predict(self, uav, block):
        # Official p1,p2,p3,p4 order = TL,TR,BL,BR; fifth image = UAV.
        patches = [block[y:y+256, x:x+256] for x, y in [(0, 0), (256, 0), (0, 256), (256, 256)]] + [uav]
        x = np.stack(patches).astype(np.float32) / 255
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())[None].to(self.device)
        pos, _direction = self.model(x)
        xy = pos[0].float().cpu().numpy()
        return {'x': float(xy[0]), 'y': float(xy[1]), 'valid': True}


def load_external_adapter(path, config, device):
    spec = importlib.util.spec_from_file_location('external_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.build(config, device)
    if not callable(getattr(adapter, 'predict', None)) or not getattr(adapter, 'modules', None):
        raise TypeError('Adapter must implement predict(uav_rgb, block_rgb) and expose modules')
    for field in ('model', 'variant', 'weights', 'training_regime'):
        if not adapter.metadata.get(field):
            raise ValueError(f'Missing provenance: {field}')
    return adapter


def benchmark(adapter, manifest, out, audit, device='cuda', meters_per_unit=32., warmup=5, repeats=3):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if warmup < 1 or repeats < 1:
        raise ValueError('warmup/repeats must be positive')
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
    first = load_pair(manifest.iloc[0])
    with torch.inference_mode():
        adapter.predict(*first)  # lazy initialization excluded from timing and parameter count
        enforce_fp32(adapter.modules)
        for module in all_modules(adapter.modules):
            module.eval()
        for _ in range(warmup):
            adapter.predict(*first)
        observed, handles = fp32_audit(adapter.modules)
        try:
            adapter.predict(*first)
        finally:
            for handle in handles:
                handle.remove()
        if observed != {'torch.float32'}:
            raise RuntimeError(f'FP32 compute audit failed: {observed}')
        params = parameter_count(adapter.modules)
        save_json(out / 'model.json', {**adapter.metadata, 'parameters': params,
                                      'observed_compute_dtypes': sorted(observed)})
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        journal = out / 'predictions.partial.csv'
        if journal.exists():
            journal.unlink()
        for j, row in manifest.iterrows():
            print(f"{adapter.metadata['model']}: starting sample {j+1}/{len(manifest)} ({row.sample_id})", flush=True)
            uav, block = load_pair(row)  # disk I/O excluded identically for all methods
            for repeat in range(repeats):
                seed = 2026 + int(j) * repeats + repeat
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); cv2.setRNGSeed(seed)
                sync()
                start = time.perf_counter()
                pred = adapter.predict(uav, block)
                sync()
                seconds = time.perf_counter() - start
                x, y, valid, error = position_error(pred, row.gt_x, row.gt_y, meters_per_unit)
                rows.append({'sample_id': row.sample_id, 'set': row['set'], 'repeat': repeat,
                             'pred_x': x, 'pred_y': y, 'valid': valid, 'error_m': error,
                             'seconds': seconds, 'inliers': pred.get('inliers'), 'n_angles': pred.get('n_angles')})
                pd.DataFrame([rows[-1]]).to_csv(journal, mode='a', header=not journal.exists(), index=False)
                print(f"  repeat {repeat+1}/{repeats}: {seconds:.2f}s, angles={pred.get('n_angles', '-')}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / 'predictions.csv', index=False)
    # One fixed run per image for accuracy; all repeats for latency. No best-of-repeat selection.
    first_run = df[df.repeat == 0]
    summary = summarize(first_run, params, audit, adapter.metadata)
    t = df.seconds.to_numpy()
    summary.update(mean_ms=float(t.mean()*1000), p50_ms=float(np.median(t)*1000),
                   p95_ms=float(np.percentile(t, 95)*1000), fps=float(1/t.mean()),
                   latency_repeats=repeats, precision='fp32-audited', batch_size=1,
                   peak_allocated_MB=torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith('cuda') else None)
    pd.DataFrame([summary]).to_csv(out / 'summary.csv', index=False)
    groups = [summarize(g, params, audit, adapter.metadata) | {'set': name}
              for name, g in first_run.groupby('set')]
    pd.DataFrame(groups).to_csv(out / 'per_set.csv', index=False)
    return summary
