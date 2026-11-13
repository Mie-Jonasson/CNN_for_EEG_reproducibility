#!/usr/bin/env python3
"""Run the whole Table 2 pipeline on this machine: download, train, summarize.

Finished runs are skipped, so an interrupted pipeline can be started again.
Options not listed below are passed on to training.

  # full reproduction (both datasets, both bands, every subject)
  python run_all.py

  # smoke test: 1 subject, few epochs
  python run_all.py --datasets BCIC_IV_2a --low-cuts 4 --subjects 1 \
      --max-epochs 4 --max-increase-epochs 2
"""

import argparse
import time

import reproduce_table2_deepconvnet as table2
from data_loading import download


def format_wall(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=("BCIC_IV_2a", "HGD"),
                        default=["BCIC_IV_2a", "HGD"])
    parser.add_argument("--low-cuts", type=int, nargs="+", choices=table2.LOW_CUTS,
                        default=list(table2.LOW_CUTS))
    parser.add_argument("--subjects", type=int, nargs="+",
                        help="subject ids (default: all of each dataset)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-download", action="store_true")
    args, train_extra = parser.parse_known_args()

    try:
        if not args.skip_download:
            for name in args.datasets:
                download(name, args.subjects)

        for name in args.datasets:
            for low_cut in args.low_cuts:
                argv = ["--results-dir", args.results_dir, "train",
                        "--dataset", name, "--low-cut", str(low_cut), *train_extra]
                if args.subjects:
                    argv += ["--subjects", *map(str, args.subjects)]
                print(f"\n=== {name} {low_cut}-fend Hz ===", flush=True)
                table2.cmd_train(table2.parse_args(argv))

        table2.cmd_summarize(table2.parse_args(
            ["--results-dir", args.results_dir, "summarize"]))
    finally:
        print(f"Wall clock {format_wall(time.perf_counter() - started)}", flush=True)


if __name__ == "__main__":
    main()
