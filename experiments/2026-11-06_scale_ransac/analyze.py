"""Strict paired analysis using the team's errors(), plus bootstrap/Wilcoxon/McNemar."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from common import HERE, ROOT, SET, SPLITS, protocol, sha256, write_json

sys.path.insert(0, str(ROOT))
from imatch.metrics import errors
from imatch.geometry import wrap

IDENTITY_KEYS = ['source_fingerprint', 'weights', 'packages', 'backend_source_sha256',
                 'base_matcher_source_sha256', 'samples_csv_sha256', 'gpu_name', 'gpu_total_mib']


def paired_stats(base, candidate, seed=20261102, n_boot=10000):
    """All arrays already joined on identical (set,k), including failures via team fallback."""
    b, c = base.position_error_m.to_numpy(), candidate.position_error_m.to_numpy()
    bok, cok = b < 10, c < 10
    gained, lost = int((~bok & cok).sum()), int((bok & ~cok).sum())
    r = {'n': len(b), 'baseline_success_pct': float(bok.mean()*100),
         'success_pct': float(cok.mean()*100), 'success_delta_pp': float((cok.mean()-bok.mean())*100),
         'recovered': gained, 'regressed': lost,
         'mcnemar_p': float(binomtest(gained, gained+lost).pvalue) if gained+lost else 1.,
         'error_mean_m': float(c.mean()), 'error_median_m': float(np.median(c)),
         'inlier_mean': float(candidate.n_inl.mean()), 'inlier_median': float(candidate.n_inl.median()),
         'failure_ge10m_count': int((~cok).sum()),
         'pipeline_failure_count': int(candidate.pipeline_failure.sum()),
         'runtime_mean_sec': float(candidate.sec.mean()), 'runtime_median_sec': float(candidate.sec.median()),
         'gpu_peak_allocated_mib': float(candidate.gpu_peak_allocated_mib.max()),
         'gpu_peak_reserved_mib': float(candidate.gpu_peak_reserved_mib.max()),
         'error_improved_count': int((c < b).sum()),
         'inlier_improved_count': int((candidate.n_inl.to_numpy() > base.n_inl.to_numpy()).sum())}
    r['both_error_and_inlier_improved_count'] = int(((c < b) & (candidate.n_inl.to_numpy() > base.n_inl.to_numpy())).sum())
    rng = np.random.default_rng(seed)
    for name, d in [('error', c-b), ('inlier', candidate.n_inl.to_numpy()-base.n_inl.to_numpy()),
                    ('runtime', candidate.sec.to_numpy()-base.sec.to_numpy())]:
        means = []
        for offset in range(0, n_boot, 256):
            idx = rng.integers(0, len(d), size=(min(256, n_boot-offset), len(d)))
            means.extend(d[idx].mean(axis=1))
        ci = np.quantile(means, [.025, .975])
        r.update({f'{name}_delta_mean': float(d.mean()), f'{name}_delta_median': float(np.median(d)),
                  f'{name}_delta_mean_ci95_low': float(ci[0]), f'{name}_delta_mean_ci95_high': float(ci[1])})
        if name != 'runtime':
            # scipy auto handles exact small samples; ties/zeroes use its approximation as applicable.
            r[f'{name}_wilcoxon_p'] = float(wilcoxon(d, method='auto').pvalue) if np.any(d != 0) else 1.
    return r


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def fresh_failure_tags(base):
    heading_error = wrap(base.pred_theta.fillna(0).to_numpy()-base.gt_theta.to_numpy())
    labels = np.select([base.pipeline_failure.to_numpy(), heading_error > 45, base.n_inl.to_numpy() < 30],
                       ['블록 밖', '회전', '짝 부족'], default='기타')
    assigned = pd.Series(np.where(base.position_error_m.to_numpy() >= 10, labels, '성공'), index=base.index)
    return assigned, heading_error


def read_runs(paths, split):
    cfg = protocol()
    collections = {c['name']: [] for c in cfg['conditions']}
    identity = None
    manifests = []
    for path in paths:
        meta = json.loads((path / 'provenance.json').read_text())
        complete = json.loads((path / 'completed.json').read_text())
        if meta['protocol'] != cfg:
            raise ValueError('Run protocol differs from conditions.json')
        current = {k: meta[k] for k in IDENTITY_KEYS}
        if identity is not None and current != identity:
            raise ValueError('Run code, weights, dataset manifest, hardware or dependencies differ')
        identity = current
        manifests.append({'path': str(path.resolve()), 'split': meta['split'],
                          'provenance_sha256': sha256(path / 'provenance.json')})
        if meta.get('dataset_preprocessing_verified') is False:
            manifests[-1]['dataset_limitation'] = meta['dataset_limitation']
        if complete['split'] != meta['split']:
            raise ValueError('completion/provenance split mismatch')
        for c in cfg['conditions']:
            name = c['name']
            condition_complete = json.loads((path / name / 'completed.json').read_text())
            r = pd.read_csv(path / name / 'samples.csv')
            lo, hi = SPLITS[meta['split']]
            if r['set'].ne(SET).any() or sorted(r.k) != list(range(lo, hi)):
                raise ValueError(f'Incomplete or duplicate sample keys in {name}')
            if condition_complete['condition'] != c or condition_complete['n'] != len(r):
                raise ValueError(f'Condition metadata mismatch: {name}')
            if r.scale.ne(c['scale']).any() or r.magsac_threshold.ne(c['magsac_threshold']).any():
                raise ValueError(f'Wrong settings in {name}')
            if r.experiment_name.ne(name).any() or r.threshold_frame.ne('base_block_px').any():
                raise ValueError('Wrong condition or coordinate frame')
            if not r.status.isin(['ok', 'no_homography']).all():
                raise ValueError('Runtime errors cannot be evaluated as ordinary matching failures')
            numeric = ['gt_x', 'gt_y', 'gt_theta', 'n_inl', 'sec', 'gpu_peak_allocated_mib', 'gpu_peak_reserved_mib']
            if not np.isfinite(r[numeric].to_numpy(float)).all() or (r[numeric[3:]] < 0).any().any():
                raise ValueError('Missing/non-finite measurement')
            if np.isinf(r[['pred_x', 'pred_y', 'pred_theta']].to_numpy(float)).any():
                raise ValueError('Infinite prediction is invalid')
            has_pose = np.isfinite(r[['pred_x', 'pred_y', 'pred_theta']].to_numpy(float)).all(axis=1)
            if not np.array_equal(has_pose, r.status.eq('ok').to_numpy()):
                raise ValueError('Pose/status mismatch')
            e, fail = errors(r.pred_x, r.pred_y, r.gt_x, r.gt_y)
            if not np.allclose(e, r.position_error_m, atol=1e-8):
                raise ValueError('Recorded error disagrees with imatch.metrics.errors')
            if not np.array_equal(e < 10, r.success_10m.to_numpy()):
                raise ValueError('Recorded success disagrees with team <10 m rule')
            r['position_error_m'], r['pipeline_failure'] = e, fail
            collections[name].append(r)
    lo, hi = SPLITS[split]
    combined = {}
    for name, frames in collections.items():
        r = pd.concat(frames, ignore_index=True)
        if r.duplicated(['set', 'k']).any():
            raise ValueError('Overlapping run samples; no silent deduplication')
        r = r[r.k.between(lo, hi-1)].set_index(['set', 'k']).sort_index()
        if list(r.index) != [(SET, k) for k in range(lo, hi)]:
            raise ValueError(f'Expected complete {split} k={lo}..{hi-1} for {name}')
        combined[name] = r
    base = combined['baseline']
    for name, r in combined.items():
        if not np.allclose(r[['gt_x', 'gt_y', 'gt_theta']], base[['gt_x', 'gt_y', 'gt_theta']], rtol=0, atol=1e-12):
            raise ValueError(f'Ground truth mismatch in {name}')
    return combined, manifests, identity


def analyze(paths, split, out, n_boot=10000):
    conditions, manifests, identity = read_runs(paths, split)
    tags = pd.read_csv(ROOT / 'results/failure_tags.csv')
    if tags.duplicated(['set', 'k']).any():
        raise ValueError('Duplicate historical failure tags')
    tags = tags[tags['set'] == SET].set_index(['set', 'k']).tag
    base = conditions['baseline']
    groups = {'all_singapore': np.ones(len(base), bool)}
    historical = tags.reindex(base.index)
    assigned, heading_error = fresh_failure_tags(base)
    for source, series in [('fresh', assigned), ('historical', historical)]:
        for tag in ['블록 밖', '짝 부족', '회전', '기타']:
            groups[f'{source}:{tag}'] = series.eq(tag).to_numpy()
        groups[f'{source}:블록 밖+짝 부족'] = series.isin(['블록 밖', '짝 부족']).to_numpy()
    stats, differences = [], []
    for group, mask in groups.items():
        if not mask.any():
            continue
        for name, candidate in conditions.items():
            b, c = base[mask], candidate[mask]
            s = paired_stats(b, c, n_boot=n_boot)
            s.update(split=split, group=group, experiment_name=name,
                     tag_source='fresh Method B baseline' if group.startswith('fresh:') else 'historical reference only' if group.startswith('historical:') else 'all samples',
                     comparison_baseline='fresh LoMa 1-direction scale1 MAGSAC3')
            stats.append(s)
            if group == 'all_singapore':
                d = pd.DataFrame({'baseline_error_m': b.position_error_m, 'candidate_error_m': c.position_error_m,
                    'error_delta_m': c.position_error_m-b.position_error_m,
                    'baseline_inlier': b.n_inl, 'candidate_inlier': c.n_inl, 'inlier_delta': c.n_inl-b.n_inl,
                    'baseline_runtime_sec': b.sec, 'candidate_runtime_sec': c.sec,
                    'baseline_success': b.position_error_m < 10, 'candidate_success': c.position_error_m < 10,
                    'fresh_baseline_failure_tag': assigned,
                    'historical_8dir_failure_tag': historical, 'experiment_name': name})
                differences.append(d.reset_index())
    table = pd.DataFrame(stats)
    table['mcnemar_p_holm4'] = np.nan
    primary = table.group.eq('all_singapore') & table.experiment_name.ne('baseline')
    table.loc[primary, 'mcnemar_p_holm4'] = holm(table.loc[primary, 'mcnemar_p'])
    # Identical Baseline/E4-A must produce identical geometry and inliers; timings may differ.
    repeat = conditions['e4_magsac3']
    cols = ['pred_x', 'pred_y', 'pred_theta', 'n_inl', 'n_match']
    duplicate_ok = np.allclose(base[cols], repeat[cols], rtol=0, atol=1e-10, equal_nan=True)
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / 'comparison.csv', index=False, mode='x')
    pd.concat(differences, ignore_index=True).to_csv(out / 'paired_differences.csv', index=False, mode='x')
    fresh_tags = base.assign(tag=assigned, head_err_recomputed=heading_error)
    fresh_tags[fresh_tags.tag.ne('성공')].reset_index().to_csv(out / 'baseline_failure_tags.csv', index=False, mode='x')
    assigned[assigned.ne('성공')].value_counts().rename_axis('tag').rename('n').to_csv(out / 'baseline_failure_tag_counts.csv', mode='x')
    write_json(out / 'analysis_manifest.json', {'split': split, 'runs': manifests, 'identity': identity,
        'bootstrap_repetitions': n_boot, 'seed': 20261102, 'baseline_e4_3_geometry_identical': bool(duplicate_ok),
        'failure_tags_sha256': sha256(ROOT / 'results/failure_tags.csv'),
        'primary_family': '4 candidate McNemar tests on all_singapore; Holm correction includes duplicate E4-A',
        'tag_and_continuous_tests': 'exploratory; unadjusted, do not claim confirmatory significance',
        'source_of_tags': 'fresh baseline failures recalculated per split; historical 8-direction tags reference only'})
    lines = [f'# E3·E4 {split} 결과', '', f'k={SPLITS[split][0]}~{SPLITS[split][1]-1}; 새 1방향 Baseline과 paired comparison.', '',
        '성공: 기존 팀 규칙 `<10 m`. 실패 수(≥10 m)는 추정 없음/블록 밖 실패와 별도.', '',
        '| 조건 | n | 성공률 | inlier 평균 / 중앙값 | 오차 평균 / 중앙값(m) | ≥10m 실패 | 복구 / 악화 | 시간 평균(s) | GPU allocated / reserved peak(MiB) | McNemar p / Holm p |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    if any('dataset_limitation' in m for m in manifests):
        lines[2:2] = ['**방법 B 자체 Baseline:** k·원본 row·GT와 본 프로토콜의 입력 해시는 검증·고정했다. 과거 prepared 이미지의 픽셀 동일성은 미검증이다. 모든 비교는 새 Baseline 기준이며 과거 72.9%와 직접 비교하지 않는다.', '']
    for r in table[table.group.eq('all_singapore')].itertuples():
        adj = f'{r.mcnemar_p_holm4:.4g}' if np.isfinite(r.mcnemar_p_holm4) else '—'
        lines.append(f'| {r.experiment_name} | {r.n} | {r.success_pct:.2f}% | {r.inlier_mean:.2f} / {r.inlier_median:.2f} '
            f'| {r.error_mean_m:.3f} / {r.error_median_m:.3f} | {r.failure_ge10m_count} | {r.recovered} / {r.regressed} '
            f'| {r.runtime_mean_sec:.4f} | {r.gpu_peak_allocated_mib:.0f} / {r.gpu_peak_reserved_mib:.0f} '
            f'| {r.mcnemar_p:.4g} / {adj} |')
    lines += ['', '## 실패 태그별 분석', '',
        '`fresh:`는 새 Baseline의 ≥10m 실패를 재분류했다. 우선순위: 추정 없음/블록 밖 > heading오차>45° > inlier<30 > 기타.',
        '`historical:`는 과거 8방향 태그의 고정 부분집합을 참고로만 표시한다. 회전 태그는 heading오차 기반 진단값이며 인과 원인 확정이 아니다.', '',
        '| 태그 | 조건 | n | 새 1방향 baseline 성공률 → 조건 성공률 | 복구 / 악화 | 오차 변화 평균(m) | inlier 변화 평균 | 둘 다 개선 | 탐색적 McNemar p |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in table[table.group.ne('all_singapore')].itertuples():
        lines.append(f'| {r.group} | {r.experiment_name} | {r.n} | {r.baseline_success_pct:.2f}% → {r.success_pct:.2f}% '
            f'| {r.recovered} / {r.regressed} | {r.error_delta_mean:.3f} | {r.inlier_delta_mean:.3f} '
            f'| {r.both_error_and_inlier_improved_count} | {r.mcnemar_p:.4g} |')
    lines += ['', '## 연속량 짝지은 통계', '',
        '음수 오차 차이는 개선, 양수 inlier 차이는 증가. Bootstrap은 샘플 쌍을 재표집한 평균 차이 percentile 95% CI.', '',
        '| 조건 | 오차 차이 평균 [95% CI] / 중앙값(m) | inlier 차이 평균 [95% CI] / 중앙값 | Wilcoxon p(오차 / inlier) | 시간 차이 평균 [95% CI](s) |',
        '| --- | --- | --- | --- | --- |']
    for r in table[table.group.eq('all_singapore')].itertuples():
        lines.append(f'| {r.experiment_name} | {r.error_delta_mean:.3f} [{r.error_delta_mean_ci95_low:.3f}, {r.error_delta_mean_ci95_high:.3f}] / {r.error_delta_median:.3f} '
            f'| {r.inlier_delta_mean:.3f} [{r.inlier_delta_mean_ci95_low:.3f}, {r.inlier_delta_mean_ci95_high:.3f}] / {r.inlier_delta_median:.3f} '
            f'| {r.error_wilcoxon_p:.4g} / {r.inlier_wilcoxon_p:.4g} '
            f'| {r.runtime_delta_mean:.4f} [{r.runtime_delta_mean_ci95_low:.4f}, {r.runtime_delta_mean_ci95_high:.4f}] |')
    lines += ['', f'Baseline와 E4 3px의 기하·inlier 동일성: **{bool(duplicate_ok)}**.',
        'False이면 비결정성/환경을 조사하고 채택 결론을 보류한다. 연속량 및 태그 p는 탐색적이다.',
        '유의하지 않은 차이를 개선으로 확정하지 않는다. 최종 평가 결과로 설정을 재탐색하지 않는다.',
        '미래 프레임의 상관을 확인하지 못했으므로 샘플 독립성 가정과 단일 seed의 한계를 함께 보고한다.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'REPORT.md')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--split', choices=list(SPLITS), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    if a.bootstrap < 100:
        ap.error('--bootstrap must be >=100')
    analyze(a.runs, a.split, a.out, a.bootstrap)


if __name__ == '__main__':
    main()
