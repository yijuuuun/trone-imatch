"""List expected official files; downloading requires explicit --download. No model load."""
import argparse
import json
import os
from pathlib import Path
import urllib.request

from common import HERE, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--torch-home', type=Path, required=True)
    ap.add_argument('--download', action='store_true', help='fetch only the three official LoMa-B assets')
    ap.add_argument('--out', type=Path, help='new JSON manifest; no overwrite')
    a = ap.parse_args()
    official = json.loads((HERE / 'official_weights.json').read_text())
    cache = a.torch_home.resolve() / 'hub/checkpoints'
    report = {'architecture': 'LoMa-B', 'cache': str(cache), 'weights': [],
              'compatibility_validated': False,
              'note': 'Presence/hash alone do not establish compatibility. Offline model initialization must pass state_dict checks.'}
    for spec in official['weights']:
        target = cache / spec['filename']
        downloaded = False
        if a.download and not target.exists():
            cache.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + '.partial')
            with partial.open('xb') as f:
                with urllib.request.urlopen(spec['url'], timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if partial.stat().st_size < 1024:
                raise ValueError(f'Download unexpectedly small: {partial}')
            # Exclusive hard-link creation prevents races/overwrites of existing weights.
            os.link(partial, target)
            partial.unlink()
            downloaded = True
        exists = target.is_file()
        report['weights'].append({**spec, 'path': str(target), 'exists': exists,
            'downloaded_this_call': downloaded, 'bytes': target.stat().st_size if exists else None,
            'observed_sha256': sha256(target) if exists else None})
    report['all_present'] = all(w['exists'] for w in report['weights'])
    if a.out:
        write_json(a.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['all_present'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
