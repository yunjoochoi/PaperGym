"""Materialize each registered task's dataset into repo-local dev/test splits
(data/datasets/<task>/{dev,test}.jsonl). Pulls HF ONCE here so the run loop
never touches the network, and so test LABELS live only on the host (never in
the sandbox image)."""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from papergym.execution.task import TASKS, DEFAULT_DATA_ROOT

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--tasks", nargs="*", default=None)
    args = p.parse_args(argv)
    for tid in (args.tasks or list(TASKS)):
        counts = TASKS[tid].materialize(data_root=args.data_root,
                                        n_test=args.n_test, n_dev=args.n_dev)
        print(f"materialized {tid}: {counts}")


if __name__ == "__main__":
    main()
