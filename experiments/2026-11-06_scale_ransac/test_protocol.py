"""CPU smoke tests. Synthetic data stay in temporary directories, never research outputs."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from common import HERE, ROOT, SET, protocol, validate_samples, write_json
sys.path.insert(0, str(ROOT))
from analyze import IDENTITY_KEYS, analyze, holm, paired_stats, read_runs
from imatch.geometry import pose_from_homography
from imatch.metrics import errors
from run_experiments import BasePixelLoMa, parse_args


def rows(condition):
    # Three geometry outcomes: valid, missing estimate, estimate outside block.
    px = [0., np.nan, 1.2]
    py = [0., np.nan, 0.]
    gx = [0., .5, .8]
    gy = [0., 0., 0.]
    e, fail = errors(px, py, gx, gy)
    return pd.DataFrame({'set': [SET]*3, 'k': [0, 1, 2], 'gt_x': gx, 'gt_y': gy, 'gt_theta': [0.]*3,
        'experiment_name': condition['name'], 'scale': condition['scale'],
        'magsac_threshold': condition['magsac_threshold'], 'threshold_frame': 'base_block_px',
        'pred_x': px, 'pred_y': py, 'pred_theta': [0., np.nan, 0.], 'n_inl': [10, 0, 8], 'n_match': [12, 0, 9],
        'sec': [.1, .2, .3], 'position_error_m': e, 'success_10m': e < 10,
        'pipeline_failure': fail, 'status': ['ok', 'no_homography', 'ok'],
        'gpu_peak_allocated_mib': [100.]*3, 'gpu_peak_reserved_mib': [120.]*3})


def synthetic_run(path):
    path.mkdir()
    cfg = protocol()
    meta = {k: f'test-{k}' for k in IDENTITY_KEYS}
    meta.update(protocol=cfg, split='smoke')
    write_json(path / 'provenance.json', meta)
    write_json(path / 'completed.json', {'split': 'smoke'})
    for c in cfg['conditions']:
        p = path / c['name']
        p.mkdir()
        data = rows(c)
        data = pd.concat([data, data.iloc[:2]], ignore_index=True)
        data['k'] = np.arange(5)
        data.to_csv(p / 'samples.csv', index=False)
        write_json(p / 'completed.json', {'condition': c, 'n': 5})


class ProtocolTests(unittest.TestCase):
    def test_team_fallback_clip_and_strict_success_boundary(self):
        e, fail = errors([np.nan, 2., 10/32], [np.nan, 0, 0], [.5, 1., 0], [0, 0, 0])
        np.testing.assert_allclose(e, [16, 0, 10])
        np.testing.assert_array_equal(fail, [True, True, False])
        np.testing.assert_array_equal(e < 10, [False, True, False])

    def test_scale_homography_returns_identical_base_pose(self):
        Hbase = np.array([[1., .1, 140], [-.1, 1., 105], [.0001, -.0002, 1.]])
        expected = pose_from_homography(Hbase, 1., 0., [128, 128])
        for scale in [1., 1.5, 2.]:
            S = np.diag([scale, scale, 1.])
            got = pose_from_homography(S @ Hbase @ np.linalg.inv(S), scale, 0., [128, 128])
            np.testing.assert_allclose(list(got.values()), list(expected.values()), atol=1e-12)

    def test_adapter_normalizes_points_before_magsac(self):
        import torch
        from types import SimpleNamespace
        pts = np.array([[2, 4], [10, 20], [30, 50], [70, 90]], np.float32)
        existing = SimpleNamespace(torch=torch, m=lambda a, b: {'matched_kpts0': pts*1.5, 'matched_kpts1': (pts+100)*1.5})
        H = np.array([[1., 0, 100], [0, 1., 100], [0, 0, 1.]])
        def fake_magsac(p0, p1):
            np.testing.assert_allclose(p0, pts)
            np.testing.assert_allclose(p1, pts+100)
            return p0, p1, H, np.ones(4, bool)
        with patch('imatch.geometry.magsac', side_effect=fake_magsac):
            _, _, resized, mask = BasePixelLoMa(existing, 1.5)(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(mask.sum(), 4)
        self.assertEqual(resized[0, 2], 150.)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV absent; real MAGSAC smoke requires existing GPU env')
    def test_real_magsac_and_threshold_parameter(self):
        from imatch import geometry
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 200, (80, 2)).astype(np.float32)
        dst = points + [100., 80.]
        old = geometry.RANSAC_PX
        try:
            for threshold in [3., 8.]:
                geometry.set_ransac_px(threshold)
                _, _, H, mask = geometry.magsac(points, dst.astype(np.float32))
                self.assertGreater(mask.sum(), 70)
                pose = geometry.pose_from_homography(H, 1, 0, [128, 128])
                self.assertAlmostEqual(pose['px'], (228-256)/128, places=5)
        finally:
            geometry.set_ransac_px(old)

    def test_paired_recovery_regression_and_exact_p(self):
        b = rows(protocol()['conditions'][0])
        b['position_error_m'] = [20., 20., 0.]
        c = b.copy()
        c['position_error_m'] = [0., 0., 20.]
        r = paired_stats(b, c, n_boot=100)
        self.assertEqual((r['recovered'], r['regressed']), (2, 1))
        self.assertEqual(r['mcnemar_p'], 1.)
        self.assertAlmostEqual(r['error_delta_mean'], -20/3)
        np.testing.assert_allclose(holm([.01, .04, .03, 1.]), [.04, .09, .09, 1.])

    def test_end_to_end_report_and_missing_duplicate_gt_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            synthetic_run(run)
            out = Path(tmp) / 'analysis'
            analyze([run], 'smoke', out, n_boot=100)
            self.assertTrue((out / 'REPORT.md').is_file())
            self.assertTrue(json.loads((out / 'analysis_manifest.json').read_text())['baseline_e4_3_geometry_identical'])
            r = run / 'e3_scale1p5/samples.csv'
            original = pd.read_csv(r)
            for mutated in [original.iloc[:2], pd.concat([original, original.iloc[:1]]), original.assign(gt_x=0.)]:
                mutated.to_csv(r, index=False)
                with self.assertRaises(ValueError):
                    read_runs([run], 'smoke')
            original.to_csv(r, index=False)
            with self.assertRaises(ValueError):
                read_runs([run, run], 'smoke')
            with self.assertRaises(FileExistsError):
                analyze([run], 'smoke', out, n_boot=100)

    def test_validate_sample_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            pd.DataFrame({'set':[SET, SET], 'k':[0, 2], 'x_norm':[0., 0.],
                          'y_norm':[0., 0.], 'theta':[0., 0.]}).to_csv(data/'samples.csv', index=False)
            with self.assertRaises(ValueError):
                validate_samples(data, 'smoke')

    def test_cli_dry_run_and_final_gate(self):
        a = parse_args(['--data', '/absent', '--out', '/absent', '--split', 'final', '--dry-run'])
        self.assertEqual(a.gpu, '0')
        p = subprocess.run([sys.executable, str(HERE/'run_experiments.py'), '--data', '/absent',
                            '--out', '/absent', '--split', 'final', '--dry-run'], text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['kmin'], 100)
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, str(HERE/'run_experiments.py'), '--data', '/absent',
                                '--out', str(Path(tmp)/'out'), '--split', 'final'], text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse((Path(tmp)/'out').exists())

    def test_freeze_requires_completed_development_analysis_and_identity(self):
        from common import source_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tune = path/'tune'
            tune.mkdir()
            cfg = protocol()
            identity = {k: f'test-{k}' for k in IDENTITY_KEYS}
            identity['source_fingerprint'] = source_fingerprint()[0]
            write_json(tune/'provenance.json', {**identity, 'split':'tune', 'protocol':cfg})
            write_json(tune/'completed.json', {'split':'tune'})
            for c in cfg['conditions']:
                dest = tune/c['name']
                dest.mkdir()
                template = rows(c).iloc[[0]].copy()
                data = pd.concat([template]*100, ignore_index=True)
                data['k'] = np.arange(100)
                data.to_csv(dest/'samples.csv', index=False)
                write_json(dest/'completed.json', {'n':100, 'condition':c})
            analysis = path/'analysis'
            analyze([tune], 'tune', analysis, n_boot=100)
            command = [sys.executable, str(HERE/'freeze.py'), '--tune-run', str(tune),
                       '--tune-analysis', str(analysis), '--candidate', 'baseline', '--reason',
                       'synthetic test only', '--out', str(path/'freeze.json')]
            p = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads((path/'freeze.json').read_text())['candidate'], 'baseline')
            p = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0, 'must not overwrite an existing freeze')

    def test_gpu_guard_rejects_busy_device(self):
        from types import SimpleNamespace
        from run_experiments import assert_free_gpu
        args = SimpleNamespace(gpu='0', min_free_mib=4096, max_used_mib=1024, allow_shared_gpu=False)
        env = {'gpu_inventory': {'returncode':0, 'stdout':'0, GPU-test, Test GPU, 24000, 0, 24000, 0'},
               'gpu_processes': {'returncode':0, 'stdout':''}}
        assert_free_gpu(args, env)
        env['gpu_processes']['stdout'] = 'GPU-test, 12345, 500'
        with self.assertRaises(RuntimeError):
            assert_free_gpu(args, env)
        args.allow_shared_gpu = True
        assert_free_gpu(args, env)
        env['gpu_inventory']['stdout'] = '0, GPU-test, Test GPU, 24000, 23000, 1000, 99'
        with self.assertRaises(RuntimeError):
            assert_free_gpu(args, env)

    def test_asset_audit_rejects_metadata_without_images(self):
        from check_assets import audit
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            pd.DataFrame({'set':[SET], 'k':[0], 'x_norm':[0.], 'y_norm':[0.],
                          'theta':[0.]}).to_csv(data/'samples.csv', index=False)
            report = audit(data)
            self.assertFalse(report['ready'])
            self.assertEqual(report['singapore_rows'], 1)
            self.assertEqual(report['missing_image_count'], 2000)
            self.assertIn('Expected one unique row for each k=0..999', report['issues'])

    def test_weight_listing_does_not_create_cache_or_claim_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)/'nonexistent-cache'
            p = subprocess.run([sys.executable, str(HERE/'prepare_weights.py'), '--torch-home', str(cache)],
                               text=True, capture_output=True)
            self.assertEqual(p.returncode, 1)
            report = json.loads(p.stdout)
            self.assertEqual(len(report['weights']), 3)
            self.assertFalse(report['compatibility_validated'])
            self.assertFalse(report['all_present'])
            self.assertFalse(cache.exists())

    def test_new_baseline_failure_tags_priority_and_success(self):
        from analyze import fresh_failure_tags
        base = pd.DataFrame({'pred_theta':[90., 90., 0., 0., 90.], 'gt_theta':[0.]*5,
            'pipeline_failure':[True, False, False, False, True], 'n_inl':[0, 1, 4, 50, 0],
            'position_error_m':[20., 20., 20., 20., 1.]})
        tags, _ = fresh_failure_tags(base)
        self.assertEqual(tags.tolist(), ['블록 밖', '회전', '짝 부족', '기타', '성공'])

    def test_opencv_receives_actual_threshold(self):
        if not importlib.util.find_spec('cv2'):
            self.skipTest('OpenCV unavailable')
        import cv2
        from imatch import geometry
        pts = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]], dtype=np.float32)
        old = geometry.RANSAC_PX
        try:
            for px in [3.,8.]:
                geometry.set_ransac_px(px)
                with patch.object(cv2,'findHomography',return_value=(np.eye(3),np.ones((4,1),np.uint8))) as spy:
                    geometry.magsac(pts,pts)
                    self.assertEqual(spy.call_args.args[2],cv2.USAC_MAGSAC)
                    self.assertEqual(spy.call_args.args[3],px)
        finally:
            geometry.set_ransac_px(old)


if __name__ == '__main__':
    unittest.main(verbosity=2)
