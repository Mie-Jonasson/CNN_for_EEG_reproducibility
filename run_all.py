#!/usr/bin/env python3
"""Run the whole Table 2 pipeline with one command: download, train, summarize.

Default (HPC): run on the login node from the project folder, after creating
the venv (see hpc/table2_array.sbatch). It
1. downloads BCIC IV 2a and HGD into the MOABB cache on the login node
   (compute nodes may have no internet; cached files are skipped),
2. submits hpc/table2_array.sbatch as a Slurm array, one task per
   (dataset, band, subject) run, and
3. submits a summarize job that starts once the whole array has finished and
   writes results/summary.csv.

--local runs the same steps one after another on this machine instead
(e.g. for a smoke test). Finished runs are skipped in both modes, so an
interrupted pipeline can simply be started again.

Options not listed below are passed on to training, e.g.

  # full reproduction on the HPC (46 array tasks)
  python run_all.py --sbatch-args="--partition=gpu --account=myproject"

  # HPC smoke test: 1 subject, few epochs
  python run_all.py --datasets BCIC_IV_2a --low-cuts 4 --subjects 1 \
      --max-epochs 4 --max-increase-epochs 2

  # local smoke test
  python run_all.py --local --datasets BCIC_IV_2a --low-cuts 4 --subjects 1 \
      --max-epochs 4 --max-increase-epochs 2
"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import reproduce_table2_deepconvnet as table2
from data_loading import download

SBATCH_SCRIPT = "hpc/table2_array.sbatch"


def sbatch(args):
    """Submit a Slurm job and return its job id."""
    cmd = ["sbatch", "--parsable", *args]
    print("$", shlex.join(cmd), flush=True)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return out.strip().split(";")[0]


def submit_hpc(args, train_extra):
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch not found; run this on the HPC login node "
                         "or use --local")
    # Array task ids are indices into all_jobs(); keep only the selected runs
    indices = [i for i, (name, low_cut, subject) in enumerate(table2.all_jobs())
               if name in args.datasets and low_cut in args.low_cuts
               and (not args.subjects or subject in args.subjects)]
    if not indices:
        raise SystemExit("No runs match the selected datasets/bands/subjects")

    Path("logs").mkdir(exist_ok=True)
    extra_sbatch = shlex.split(args.sbatch_args)
    export = f"--export=ALL,RESULTS_DIR={args.results_dir}"
    array_id = sbatch([*extra_sbatch, export,
                       f"--array={','.join(map(str, indices))}",
                       SBATCH_SCRIPT, *train_extra])

    python = shlex.quote(str(Path(".venv/bin/python")))
    summarize_cmd = (f"{python} reproduce_table2_deepconvnet.py "
                     f"--results-dir {shlex.quote(args.results_dir)} summarize")
    # afterany: summarize even if some tasks failed, so partial results show up
    summary_id = sbatch([*extra_sbatch, f"--dependency=afterany:{array_id}",
                         "--job-name=table2-summary", "--time=00:15:00",
                         "--output=logs/%x_%j.out", f"--wrap={summarize_cmd}"])

    print(f"\nSubmitted {len(indices)} training tasks (job {array_id}) and the "
          f"summary (job {summary_id}).\nCheck progress with: squeue -u {os.environ.get('USER', '$USER')}\n"
          f"Summary will be in {args.results_dir}/summary.csv and "
          f"logs/table2-summary_{summary_id}.out")


def run_local(args, train_extra):
    for name in args.datasets:
        for low_cut in args.low_cuts:
            argv = ["--results-dir", args.results_dir, "train",
                    "--dataset", name, "--low-cut", str(low_cut), *train_extra]
            if args.subjects:
                argv += ["--subjects", *map(str, args.subjects)]
            table2.cmd_train(table2.parse_args(argv))
    table2.cmd_summarize(table2.parse_args(["--results-dir", args.results_dir,
                                            "summarize"]))


def main():
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
    parser.add_argument("--local", action="store_true",
                        help="train here one run after another instead of on Slurm")
    parser.add_argument("--sbatch-args", default="",
                        help='extra sbatch options for both jobs, e.g. '
                             '"--partition=gpu --account=myproject"')
    args, train_extra = parser.parse_known_args()

    if not args.skip_download:
        for name in args.datasets:
            download(name, args.subjects)

    if args.local:
        run_local(args, train_extra)
    else:
        submit_hpc(args, train_extra)


if __name__ == "__main__":
    main()
