"""Apply strict coverage checks, then run the original scripts/evaluate.py unchanged."""
import argparse
import subprocess
import sys
from pathlib import Path

from analyze import read_runs
from common import ROOT, SPLITS, protocol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--split', choices=['smoke', 'tune', 'final'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    read_runs([a.run], a.split)
    lo, hi = SPLITS[a.split]
    files = [str(a.run / c['name'] / 'samples.csv') for c in protocol()['conditions']]
    command = [sys.executable, str(ROOT / 'scripts/evaluate.py'), *files, '--kmin', str(lo), '--n', str(hi)]
    # Exclusive creation prevents overwriting an earlier evaluation log.
    with open(a.out, 'x') as f:
        completed = subprocess.run(command, text=True, stdout=f, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(a.out)


if __name__ == '__main__':
    main()
