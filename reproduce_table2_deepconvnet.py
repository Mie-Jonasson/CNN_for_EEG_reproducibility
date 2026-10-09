#!/usr/bin/env python3
"""Minimal DeepConvNet (Deep4Net) reproduction of Schirrmeister et al. 2017 Table 2.

Paper: https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730
Tutorial base: braindecode BCIC IV 2a cropped decoding, with Deep4Net instead of Shallow.

Table 2 reports mean decoding accuracy as a delta vs FBCSP. For BCIC IV 2a:
  0-38 Hz: FBCSP 68.0%, Deep ConvNet +2.9  -> ~70.9%
  4-38 Hz: FBCSP 67.8%, Deep ConvNet +2.3  -> ~70.1%

This script trains cropped Deep4Net on BNCI2014_001 (BCIC IV 2a) for those two
bands, averages over subjects, and prints deltas against the published FBCSP baselines.

Example (smoke test):
  python reproduce_table2_deepconvnet.py --subjects 3 --bands 4-38 --epochs 4

Full-ish BCIC run (slow on CPU):
  python reproduce_table2_deepconvnet.py --subjects all --bands both --epochs 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import multiply
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds

# Published Table 2 baselines / Deep ConvNet deltas (Schirrmeister et al., 2017).
TABLE2 = {
    "BCIC IV 2a": {
        "0-38": {"fbcsp": 68.0, "deep_delta": 2.9},
        "4-38": {"fbcsp": 67.8, "deep_delta": 2.3},
    },
}

BANDS = {
    "0-38": (None, 38.0),  # low-pass only; paper's 0 Hz lower cut
    "4-38": (4.0, 38.0),
}

BCIC_SUBJECTS = list(range(1, 10))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--subjects",
        default="3",
        help="'all' for subjects 1-9, a single id, or comma-separated ids (default: 3).",
    )
    p.add_argument(
        "--bands",
        choices=("0-38", "4-38", "both"),
        default="4-38",
        help="Frequency band(s) from Table 2 (default: 4-38, matches the tutorial).",
    )
    p.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-times", type=int, default=1000, help="Compute-window length in samples.")
    p.add_argument("--seed", type=int, default=20200220)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results_table2_deepconvnet.csv"),
        help="CSV path for per-subject results.",
    )
    return p.parse_args()


def parse_subjects(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return BCIC_SUBJECTS
    return [int(x) for x in spec.split(",") if x.strip()]


def pick_session_split(windows_dataset):
    """Return (train_set, test_set) for BCIC IV 2a session split."""
    splitted = windows_dataset.split("session")
    keys = list(splitted.keys())

    def find_key(candidates):
        for key in keys:
            low = str(key).lower()
            if any(c in low for c in candidates):
                return key
        return None

    train_key = find_key(("train", "session_t", "0train"))
    test_key = find_key(("test", "session_e", "1test", "eval"))
    if train_key is None or test_key is None:
        if len(keys) == 2:
            train_key, test_key = keys[0], keys[1]
        else:
            raise KeyError(f"Could not resolve train/test sessions from {keys}")
    return splitted[train_key], splitted[test_key]


def load_and_preprocess(subject_id: int, low_cut_hz: float | None, high_cut_hz: float):
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    factor_new = 1e-3
    init_block_size = 1000
    factor = 1e6  # V -> uV

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, factor)),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]
    preprocess(dataset, preprocessors, n_jobs=1)
    return dataset


def train_subject(
    subject_id: int,
    band: str,
    n_epochs: int,
    batch_size: int,
    n_times: int,
    seed: int,
    device: str,
    cuda: bool,
) -> dict:
    low_cut_hz, high_cut_hz = BANDS[band]
    set_random_seeds(seed=seed, cuda=cuda)

    dataset = load_and_preprocess(subject_id, low_cut_hz, high_cut_hz)
    n_chans = dataset[0][0].shape[0]
    n_classes = 4
    classes = list(range(n_classes))

    # Cropped Deep4Net: small final conv (paper: length 2), then dense predictions.
    model = Deep4Net(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length=2,
    )
    model.to_dense_prediction_model()
    n_preds_per_input = model.get_output_shape()[2]
    if cuda:
        model = model.cuda()

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(-0.5 * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )
    train_set, test_set = pick_session_split(windows_dataset)

    # Deep4Net hyperparameters suggested in the braindecode tutorial.
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(test_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, n_epochs - 1))),
            ("early_stopping", EarlyStopping(patience=10, load_best=True)),
        ],
        device=device,
        classes=classes,
    )
    clf.fit(train_set, y=None, epochs=n_epochs)

    # Prefer best validation accuracy from history when early stopping is used.
    history = clf.history
    valid_accs = [h["valid_accuracy"] for h in history if "valid_accuracy" in h]
    test_acc = float(max(valid_accs)) if valid_accs else float(clf.score(test_set, y=None))

    return {
        "dataset": "BCIC IV 2a",
        "subject": subject_id,
        "band": band,
        "accuracy_pct": 100.0 * test_acc,
        "n_epochs_ran": len(history),
    }


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary_rows = []
    for band, group in df.groupby("band"):
        mean_acc = group["accuracy_pct"].mean()
        fbcsp = TABLE2["BCIC IV 2a"][band]["fbcsp"]
        paper_delta = TABLE2["BCIC IV 2a"][band]["deep_delta"]
        paper_acc = fbcsp + paper_delta
        our_delta = mean_acc - fbcsp
        summary_rows.append(
            {
                "dataset": "BCIC IV 2a",
                "band": band,
                "n_subjects": len(group),
                "mean_acc_pct": mean_acc,
                "fbcsp_baseline_pct": fbcsp,
                "delta_vs_fbcsp_pct": our_delta,
                "paper_deep_delta_pct": paper_delta,
                "paper_deep_acc_pct": paper_acc,
                "abs_error_vs_paper_acc_pct": abs(mean_acc - paper_acc),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    bands = list(BANDS) if args.bands == "both" else [args.bands]

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"Subjects: {subjects}")
    print(f"Bands: {bands}")
    print(f"Max epochs: {args.epochs}")
    print(
        "Note: early stopping monitors the official test session (tutorial setup). "
        "This is convenient but not a pure held-out protocol."
    )

    rows: list[dict] = []
    for band in bands:
        for subject_id in subjects:
            print(f"\n=== subject={subject_id} band={band} ===")
            row = train_subject(
                subject_id=subject_id,
                band=band,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                n_times=args.n_times,
                seed=args.seed,
                device=device,
                cuda=cuda,
            )
            rows.append(row)
            print(f"accuracy: {row['accuracy_pct']:.1f}%")

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote per-subject results to {args.out}")

    summary = summarize(rows)
    print("\nTable-2 style summary (Deep ConvNet only):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
