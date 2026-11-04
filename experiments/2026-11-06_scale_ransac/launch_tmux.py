"""Launch the runner with this exact Python and safely quoted arguments in tmux."""
import argparse
import shlex
import subprocess
import sys

from common import HERE, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session', required=True)
    ap.add_argument('runner_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.runner_args[1:] if a.runner_args[:1] == ['--'] else a.runner_args
    if not args:
        ap.error('pass runner arguments after --')
    command = [sys.executable, '-u', str(HERE / 'run_experiments.py'), *args]
    print('Python:', sys.executable)
    print('Run:', shlex.join(command), flush=True)
    subprocess.run(['tmux', 'new-session', '-d', '-s', a.session, '-c', str(ROOT), shlex.join(command)], check=True)
    print('Attach:', shlex.join(['tmux', 'attach', '-t', a.session]))


if __name__ == '__main__':
    main()
