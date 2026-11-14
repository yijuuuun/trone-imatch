"""Isolated official Sample4Geo/MEAN inference; launched by additive notebook cells."""
import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def distance_m(q, g, mode):
    if mode == 'wgs84':
        lat1, lat2 = math.radians(q['lat']), math.radians(g['lat'])
        dlat = lat2 - lat1
        dlon = math.radians(g['lon'] - q['lon'])
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371008.8 * 2 * math.asin(math.sqrt(min(1., max(0., a))))
    if q['frame_id'] != g['frame_id']:
        raise ValueError('Query and retrieved Gallery have different metric coordinate frames')
    return math.hypot(q['east_m']-g['east_m'], q['north_m']-g['north_m'])


def load_coordinates(qcsv, gallery, queries, approved, provenance):
    if not approved:
        return None, None, 'Coordinate provenance not approved; location accuracy N/A'
    if not provenance.strip() or not qcsv:
        return None, None, 'Missing coordinate provenance or Query coordinate CSV'
    q = pd.read_csv(qcsv, dtype={'sample_id': str})
    if q.sample_id.duplicated().any() or set(q.sample_id) != set(queries.sample_id):
        raise ValueError('Query coordinates must match all original sample IDs exactly')
    q = q.set_index('sample_id').loc[queries.sample_id].reset_index()
    if {'lat','lon'}.issubset(q.columns) and {'lat','lon'}.issubset(gallery.columns):
        cols, mode = ['lat','lon'], 'wgs84'
        for t in (q, gallery):
            if not np.isfinite(t[cols].to_numpy(float)).all() or (t.lat.abs()>90).any() or (t.lon.abs()>180).any():
                raise ValueError('Invalid WGS84 coordinates')
    elif {'east_m','north_m','frame_id'}.issubset(q.columns) and {'east_m','north_m','frame_id'}.issubset(gallery.columns):
        cols, mode = ['east_m','north_m'], 'metric'
        for t in (q, gallery):
            if not np.isfinite(t[cols].to_numpy(float)).all() or t.frame_id.isna().any():
                raise ValueError('Invalid metric coordinates/frame IDs')
        # Search must not use query GT to choose a coordinate frame or restrict candidates.
        if len(set(q.frame_id) | set(gallery.frame_id)) != 1:
            raise ValueError('Use a shared metric CRS for the entire bank, or WGS84. No GT-based filtering.')
    else:
        return None, None, 'No shared WGS84 or metric coordinates; local gt_x/gt_y cannot locate other blocks'
    return q, mode, ''


def load_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected an official tensor state_dict checkpoint')
    keys = list(state)
    if all(k.startswith('module.') for k in keys):
        state = {k[7:]: v for k,v in state.items()}
    return state


def official_transform(repo, model):
    # Execute only the upstream validation assignment, avoiding training augmentation
    # construction incompatible with modern albumentations. No interpolation/normalization changes.
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    tree = ast.parse((repo/'sample4geo/dataset/university.py').read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_transforms')
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'val_transforms' for t in n.targets))
    cfg = model.get_config()
    ns = dict(A=A, cv2=cv2, ToTensorV2=ToTensorV2, img_size=(384,384), mean=cfg['mean'], std=cfg['std'])
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<official_validation_transform>', 'exec'), ns)
    return ns['val_transforms']


def load_model(name, repo, checkpoint):
    sys.path.insert(0, str(repo))
    state = load_state(checkpoint)  # missing/invalid weights fail before construction
    if name == 'Sample4Geo':
        from sample4geo.model import TimmModel
        model = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, img_size=384)
    else:
        from sample4geo.hand_convnext.model import make_model
        # Official registration replaces timm convnext_tiny with the repository implementation.
        # Suppress redundant ImageNet initialization downloads only; full strict checkpoint load follows.
        module = importlib.import_module('sample4geo.hand_convnext.ConvNext.make_model')
        original = module.create_model
        def no_download(*args, **kwargs):
            kwargs['pretrained'] = False
            return original(*args, **kwargs)
        module.create_model = no_download
        try:
            model = make_model(SimpleNamespace(views=2, nclasses=701, block=2, triplet_loss=.3, resnet=False))
        finally:
            module.create_model = original
    model.load_state_dict(state, strict=True)
    del state
    model = model.float().cuda().eval()
    transform = official_transform(repo, model)
    return model, transform


def rgb(path):
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def sync():
    torch.cuda.synchronize()


def feature(model, transform, image, name):
    x = transform(image=image)['image'].unsqueeze(0).to('cuda', dtype=torch.float32)
    sync()
    start = time.perf_counter()
    with torch.autocast('cuda', enabled=False):
        output = model(x)
    sync()
    forward_ms = (time.perf_counter()-start)*1000
    if name == 'MEAN':
        output = output[-2]  # official trainer.predict: global feature, NOT part feature map
    if output.ndim != 2 or output.shape[0] != 1 or output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError(f'Invalid official embedding: {output.shape}, {output.dtype}')
    return F.normalize(output, dim=-1), forward_ms


def evaluate(config):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU substitution')
    torch.cuda.set_device(config['gpu_index'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2026)
    name = config['name']
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out/'summary.json'
    config_signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
    if result_path.exists():
        saved=json.loads(result_path.read_text())
        if (saved.get('config_signature') != config_signature or saved.get('checkpoint_sha256') != digest(config['checkpoint']) or
            saved.get('queries_sha256') != digest(config['queries']) or
            saved.get('gallery_sha256') != digest(config['gallery'])):
            raise ValueError('Existing result belongs to different weights/data. Start a new addon run; preserved old result.')
        print('Preserved completed result:', result_path, flush=True)
        return saved
    queries = pd.read_csv(config['queries'], dtype={'sample_id': str})
    gallery = pd.read_csv(config['gallery'], dtype={'gallery_id': str})
    if queries.empty or queries.sample_id.duplicated().any():
        raise ValueError('Expected nonempty unique queries')
    if gallery.empty or gallery.gallery_id.duplicated().any():
        raise ValueError('Gallery must have unique IDs and at least one image')
    qgeo, mode, reason = load_coordinates(config['query_geo'], gallery, queries,
                                         config['coordinates_verified'], config['coordinate_provenance'])
    smoke = config['smoke']
    if smoke:
        queries = queries.head(3).copy()
        gallery = gallery.head(3).copy()
        qgeo, mode, reason = None, None, 'Smoke test only: three Queries/Gallery images; accuracy N/A'
    model, transform = load_model(name, Path(config['repo']), config['checkpoint'])
    params = sum(p.numel() for p in model.parameters())
    observed, hooks = set(), []
    def capture(module, args, output):
        if isinstance(output, torch.Tensor):
            observed.add(str(output.dtype))
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(capture))
    with torch.inference_mode():
        first = rgb(queries.iloc[0].uav_path)
        feature(model, transform, first, name)
        for hook in hooks:
            hook.remove()
        if observed != {'torch.float32'}:
            raise RuntimeError(f'Compute dtype audit failed: {observed}')
        for _ in range(config['warmup']):
            feature(model, transform, first, name)
        torch.cuda.reset_peak_memory_stats()
        gallery_rows, embeddings = [], []
        offline_start = time.perf_counter()
        for i, row in gallery.iterrows():
            image = rgb(row.image_path)  # gallery wall time includes disk I/O, separately labeled
            sync(); start = time.perf_counter()
            emb, forward_ms = feature(model, transform, image, name)
            sync(); sec = time.perf_counter()-start
            embeddings.append(emb.cpu())
            gallery_rows.append(dict(gallery_id=row.gallery_id, seconds=sec, forward_ms=forward_ms))
            if (i+1)%100 == 0:
                print(f'{name}: Gallery {i+1}/{len(gallery)}', flush=True)
        bank = torch.cat(embeddings).to('cuda')
        sync()
        offline_wall = time.perf_counter()-offline_start
        del embeddings
        pd.DataFrame(gallery_rows).to_csv(out/'gallery_timing.csv', index=False)
        torch.save(bank.cpu(), out/'gallery_features.pt')
        # Search warmup uses the full bank, same as online queries.
        for _ in range(config['warmup']):
            emb, _ = feature(model, transform, first, name)
            (emb @ bank.T).argmax().item()
        sync()
        rows = []
        journal = out/'predictions.partial.csv'
        if journal.exists():
            journal.unlink()  # failed attempt only; never overwrite a completed summary
        for i, row in queries.reset_index(drop=True).iterrows():
            image = rgb(row.uav_path)  # same I/O exclusion as previous benchmark
            print(f'{name}: Query {i+1}/{len(queries)}', flush=True)
            for repeat in range(config['repeats']):
                sync(); start = time.perf_counter()
                emb, forward_ms = feature(model, transform, image, name)
                sync(); search_start = time.perf_counter()
                pick = int((emb @ bank.T).argmax().item())
                sync(); stop = time.perf_counter()
                g = gallery.iloc[pick]  # coordinate lookup only AFTER image-only search
                err = distance_m(qgeo.iloc[i], g, mode) if mode else np.nan
                record = dict(sample_id=row.sample_id, repeat=repeat, gallery_id=g.gallery_id,
                              error_m=err, seconds=stop-start, forward_ms=forward_ms,
                              search_ms=(stop-search_start)*1000)
                if mode:
                    for column in (('lat','lon') if mode=='wgs84' else ('east_m','north_m','frame_id')):
                        record['pred_'+column] = g[column]
                        record['gt_'+column] = qgeo.iloc[i][column]
                rows.append(record)
                pd.DataFrame([record]).to_csv(journal, mode='a', index=False, header=not journal.exists())
        df = pd.DataFrame(rows)
        df.to_csv(out/'predictions.csv', index=False)
        times = df.seconds.to_numpy()
        first_run = df[df.repeat == 0]
        valid_ranking = bool(config['position_ranking_valid'] and mode and not smoke)
        summary = dict(model=name, status='smoke_passed' if smoke else 'measured', n=len(queries),
                       valid_pct=100., mean_ms=float(times.mean()*1000), p50_ms=float(np.median(times)*1000),
                       p95_ms=float(np.percentile(times,95)*1000), fps=float(1/times.mean()),
                       total_parameters=params, parameters_M=params/1e6,
                       peak_allocated_MB=torch.cuda.max_memory_allocated()/1024**2,
                       precision='fp32-audited', batch_size=1, latency_repeats=config['repeats'],
                       position_ranking_valid=valid_ranking, reason=reason,
                       latency_scope='preprocess+H2D+query_model+L2+GPU_cosine_top1; no image decode; precomputed GPU gallery',
                       gallery_n=len(gallery), gallery_precompute_wall_s=offline_wall,
                       gallery_precompute_encode_s=sum(r['seconds'] for r in gallery_rows),
                       gallery_precompute_scope='wall includes decode, encode, D2H, bank upload; encode_s excludes decode/D2H',
                       forward_mean_ms=float(df.forward_ms.mean()), search_mean_ms=float(df.search_ms.mean()),
                       checkpoint_sha256=digest(config['checkpoint']), repo_commit=config['repo_commit'],
                       queries_sha256=digest(config['queries']), gallery_sha256=digest(config['gallery']),
                       gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                       coordinate_provenance=config['coordinate_provenance'],
                       coordinate_distance='haversine WGS84' if mode=='wgs84' else mode,
                       retrieval_recall1_pct=None, retrieval_recall_reason='No independently verified retrieval relevance labels supplied')
        for threshold in (10,20,32):
            summary[f'accuracy_{threshold}m_pct'] = float((first_run.error_m < threshold).mean()*100) if mode else None
        # Mathematical distance may be computed with verified coordinates even if the original
        # experiment is diagnostic; position_ranking_valid stays False and must accompany it.
        summary['config_signature'] = config_signature
        result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        pd.DataFrame([summary]).to_csv(out/'summary.csv', index=False)
        (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    evaluate(json.loads(Path(args.config).read_text(encoding='utf-8')))
