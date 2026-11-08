"""Read campaign progress without starting or changing an experiment."""
from pathlib import Path
import argparse
import json
import os

from common import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--active', action='store_true', help='List epoch or grid progress for active jobs')
    args = parser.parse_args()
    if not (HERE / 'status.json').exists():
        print('Campaign has not started.'); return
    state = json.loads((HERE / 'status.json').read_text())
    try:
        os.kill(state['controller_pid'], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    state['controller_alive'] = alive
    if args.json:
        print(json.dumps(state, indent=2)); return
    print(f"State: {state['state']}; controller alive: {alive}; updated UTC: {state['updated_utc']}")
    print(f"Completed {state['completed']}/{state['total']}; running {state['running']}; pending {state['pending']}; failed {state['failed']}")
    print(f"{'Family / classifier':24} {'Completed':>10} {'Running':>8} {'Pending':>8} {'Failed':>7}")
    for name, row in state['groups'].items():
        print(f"{name:24} {row['completed']:4}/{row['total']:<5} {row['running']:8} {row['pending']:8} {row['failed']:7}")
    if args.active:
        for job in state['active_jobs']:
            path = HERE / 'runs' / job['id'] / 'progress.json'
            if not path.exists():
                print(f"{job['id']}: starting"); continue
            progress = json.loads(path.read_text())
            detail = (f"epoch {progress['epoch']}/{progress['epochs']}" if 'epoch' in progress else
                      f"candidates {progress.get('grid_candidates_completed', 0)}/{progress.get('grid_candidates_total', '?')}")
            print(f"{job['id']}: {progress['phase']}, {detail}")


if __name__ == '__main__':
    main()
