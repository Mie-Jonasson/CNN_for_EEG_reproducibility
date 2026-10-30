#!/usr/bin/env python3
"""Deep ConvNet (Deep4Net) reproduction of Schirrmeister et al. (2017), Table 2.

Trains one cropped Deep4Net per dataset, frequency band and subject with the
paper's two-phase early stopping, and evaluates it once on the test set.
Section references are to arXiv v5 (https://arxiv.org/abs/1703.05051v5).

Training:
- Cropped training (Sec. 2.5.4): crops of 522 samples (~2 s). Each whole
  trial (1125 samples, -0.5 s to 4 s) goes through the dense-prediction model
  in one pass, giving 604 crop predictions per trial; the trial prediction is
  their mean. The paper passes ~1000-sample inputs (~500 crops) instead; since
  Deep4Net uses unpadded convolutions the crops and their predictions are the
  same, only grouped differently into forward passes.
- Loss: per-crop cross-entropy plus the tied sample loss (Sec. 2.5.4, Eq. 3).
- Optimizer: Adam (Sec. 2.5.5).
- Two-phase early stopping (Sec. A.4):
  1. Split the training set into a training and a validation fold. Train on
     the training fold until validation accuracy has not improved for
     --max-increase-epochs epochs, or --max-epochs is reached.
  2. Reset to the parameters with the best validation accuracy and continue on
     training + validation fold until the validation fold loss drops to the
     training fold loss at that best epoch.
- The test set is only used once, after training.

Not stated in the paper; defaults follow the authors' original braindecode
experiment code: batch size 60, Adam learning rate 1e-3, last 20% of the
training trials as validation fold, phase 1 for at most 800 epochs with
patience 80, phase 2 stops at the latest after as many epochs as phase 1's
best epoch.

Usage:
  # one dataset/band, some subjects
  python reproduce_table2_deepconvnet.py train --dataset BCIC_IV_2a --low-cut 4 --subjects 1 2
  # one of the 46 (dataset, band, subject) runs, e.g. from a Slurm array
  python reproduce_table2_deepconvnet.py train --job-index 0
  # Table 2 style summary of all finished runs
  python reproduce_table2_deepconvnet.py summarize
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from braindecode.models import Deep4Net
from braindecode.util import set_random_seeds

from data_loading import SUBJECTS, load_dataset, split_train_test
from preprocessing import create_trial_windows, preprocess_dataset

# Table 2: FBCSP accuracy and Deep ConvNet difference to it, in %
TABLE2 = {
    ("BCIC_IV_2a", 0): (68.0, 2.9),
    ("BCIC_IV_2a", 4): (67.8, 2.3),
    ("HGD", 0): (91.2, 1.3),
    ("HGD", 4): (90.9, 0.5),
    ("Combined", 0): (82.1, 1.9),
    ("Combined", 4): (81.9, 1.2),
}
LOW_CUTS = (0, 4)
N_CLASSES = 4
# Input size the model is built for (Sec. 2.5.4: "about 1000 input samples");
# with final_conv_length=2 this gives crops of 522 samples (~2 s at 250 Hz)
N_TIMES = 1000
FINAL_CONV_LENGTH = 2


def all_jobs():
    """All (dataset, low cut, subject) runs needed for Table 2."""
    return [
        (name, low_cut, subject)
        for name in ("BCIC_IV_2a", "HGD")
        for low_cut in LOW_CUTS
        for subject in SUBJECTS[name]
    ]


def load_trials(name, subject, low_cut_hz):
    """Preprocessed train and test trials as (X, y) arrays in time order."""
    dataset = preprocess_dataset(load_dataset(name, [subject]), name, low_cut_hz)
    train_set, test_set = split_train_test(create_trial_windows(dataset, name))

    def to_arrays(windows):
        X = np.stack([windows[i][0] for i in range(len(windows))])
        y = np.array([windows[i][1] for i in range(len(windows))])
        return X.astype(np.float32), y.astype(np.int64)

    return to_arrays(train_set), to_arrays(test_set)


def loss_fn(logits, y, tied_loss):
    """Mean per-crop cross-entropy, plus the tied sample loss (Eq. 3).

    The tied sample loss is the cross-entropy between the predictions of each
    pair of neighbouring crops (Sec. 2.5.4).
    """
    log_probs = F.log_softmax(logits, dim=1)  # (batch, classes, crops)
    targets = y[:, None].expand(-1, log_probs.shape[2])
    loss = F.nll_loss(log_probs, targets)
    if tied_loss:
        later_probs = log_probs[:, :, 1:].exp()
        loss = loss - (later_probs * log_probs[:, :, :-1]).sum(dim=1).mean()
    return loss


def train_epoch(model, optimizer, X, y, args, rng):
    model.train()
    for batch in np.array_split(rng.permutation(len(X)), max(1, len(X) // args.batch_size)):
        batch = torch.as_tensor(batch, device=X.device)
        optimizer.zero_grad()
        loss = loss_fn(model(X[batch]), y[batch], args.tied_loss)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, X, y, args):
    """Loss and trial accuracy (mean of the crop predictions per trial)."""
    model.eval()
    losses, correct = [], 0
    for start in range(0, len(X), args.batch_size):
        logits = model(X[start:start + args.batch_size])
        yb = y[start:start + args.batch_size]
        losses.append(loss_fn(logits, yb, args.tied_loss).item() * len(yb))
        trial_probs = F.softmax(logits, dim=1).mean(dim=2)
        correct += (trial_probs.argmax(dim=1) == yb).sum().item()
    return sum(losses) / len(X), correct / len(X)


def train_subject(name, low_cut, subject, args, device):
    set_random_seeds(seed=args.seed, cuda=device == "cuda")
    rng = np.random.RandomState(args.seed)
    (X_train, y_train), (X_test, y_test) = load_trials(name, subject, low_cut)

    # Sec. A.4: split the training set into a training and validation fold
    n_valid = int(round(len(X_train) * args.valid_fraction))
    to_tensor = lambda a: torch.as_tensor(a, device=device)
    X_fit, y_fit = to_tensor(X_train[:-n_valid]), to_tensor(y_train[:-n_valid])
    X_valid, y_valid = to_tensor(X_train[-n_valid:]), to_tensor(y_train[-n_valid:])
    X_all, y_all = to_tensor(X_train), to_tensor(y_train)
    X_test, y_test = to_tensor(X_test), to_tensor(y_test)

    model = Deep4Net(
        n_chans=X_train.shape[1],
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=FINAL_CONV_LENGTH,
    )
    model.to_dense_prediction_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []

    def log(phase, epoch):
        train_loss, train_acc = evaluate(model, X_fit, y_fit, args)
        valid_loss, valid_acc = evaluate(model, X_valid, y_valid, args)
        history.append(dict(phase=phase, epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, valid_loss=valid_loss,
                            valid_acc=valid_acc))
        if args.verbose:
            print(f"  phase {phase} epoch {epoch}: train loss {train_loss:.3f} "
                  f"acc {train_acc:.3f} | valid loss {valid_loss:.3f} "
                  f"acc {valid_acc:.3f}", flush=True)
        return history[-1]

    # Phase 1: train on the training fold until validation accuracy stops
    # improving, remembering the best parameters
    best = dict(epoch=0, valid_acc=-1.0)
    for epoch in range(1, args.max_epochs + 1):
        train_epoch(model, optimizer, X_fit, y_fit, args, rng)
        row = log(1, epoch)
        if row["valid_acc"] > best["valid_acc"]:
            best = dict(row, model=copy.deepcopy(model.state_dict()),
                        optimizer=copy.deepcopy(optimizer.state_dict()))
        if epoch - best["epoch"] >= args.max_increase_epochs:
            break

    # Phase 2: from the best parameters, train on training + validation fold
    # until the validation fold loss reaches the training fold loss at the
    # best epoch
    model.load_state_dict(best["model"])
    optimizer.load_state_dict(best["optimizer"])
    loss_to_reach = best["train_loss"]
    phase2_epochs = 0
    for epoch in range(1, best["epoch"] + 1):
        train_epoch(model, optimizer, X_all, y_all, args, rng)
        phase2_epochs = epoch
        if log(2, epoch)["valid_loss"] < loss_to_reach:
            break

    test_loss, test_acc = evaluate(model, X_test, y_test, args)
    result = dict(
        dataset=name, low_cut_hz=low_cut, subject=subject,
        test_acc=test_acc, test_loss=test_loss,
        n_train_trials=len(X_train), n_valid_trials=n_valid,
        n_test_trials=len(X_test),
        phase1_epochs=len([h for h in history if h["phase"] == 1]),
        best_epoch=best["epoch"], best_valid_acc=best["valid_acc"],
        loss_to_reach=loss_to_reach, phase2_epochs=phase2_epochs,
        args={k: v for k, v in vars(args).items() if k != "func"},
    )
    return result, pd.DataFrame(history)


def run_dir(results_dir, name, low_cut):
    return Path(results_dir) / f"{name}_{low_cut}Hz"


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.job_index is not None:
        jobs = [all_jobs()[args.job_index]]
    else:
        subjects = args.subjects or SUBJECTS[args.dataset]
        jobs = [(args.dataset, args.low_cut, s) for s in subjects]

    for name, low_cut, subject in jobs:
        out_dir = run_dir(args.results_dir, name, low_cut)
        result_path = out_dir / f"subject_{subject:02d}.json"
        if result_path.exists() and not args.overwrite:
            print(f"{result_path} exists, skipping")
            continue
        print(f"=== {name}, {low_cut}-f_end Hz, subject {subject} ({device}) ===",
              flush=True)
        start = time.time()
        result, history = train_subject(name, low_cut, subject, args, device)
        result["train_time_s"] = time.time() - start
        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_dir / f"subject_{subject:02d}_history.csv", index=False)
        result_path.write_text(json.dumps(result, indent=1))
        print(f"test accuracy {100 * result['test_acc']:.1f}% (best epoch "
              f"{result['best_epoch']}, phase 2 epochs {result['phase2_epochs']}, "
              f"{result['train_time_s'] / 60:.1f} min)", flush=True)


def cmd_summarize(args):
    rows = [json.loads(p.read_text())
            for p in Path(args.results_dir).glob("*/subject_*.json")]
    if not rows:
        print(f"No results in {args.results_dir}")
        return
    df = pd.DataFrame(rows)
    per_subject = df.pivot_table(index=["dataset", "subject"], columns="low_cut_hz",
                                 values="test_acc") * 100
    print("Test accuracy per subject (%):")
    print(per_subject.round(1).to_string(), "\n")

    summary = []
    for low_cut in LOW_CUTS:
        for name in ("BCIC_IV_2a", "HGD", "Combined"):
            # Combined: mean over the subjects of both datasets (Table 2)
            names = ("BCIC_IV_2a", "HGD") if name == "Combined" else (name,)
            accs = df[df.dataset.isin(names) & (df.low_cut_hz == low_cut)].test_acc
            n_expected = sum(len(SUBJECTS[n]) for n in names)
            fbcsp, paper_delta = TABLE2[(name, low_cut)]
            mean_acc = 100 * accs.mean() if len(accs) else np.nan
            summary.append(dict(
                dataset=name, band=f"{low_cut}-f_end", subjects=f"{len(accs)}/{n_expected}",
                acc=mean_acc, paper_acc=fbcsp + paper_delta,
                delta_vs_fbcsp=mean_acc - fbcsp, paper_delta_vs_fbcsp=paper_delta,
            ))
    summary = pd.DataFrame(summary)
    print("Table 2, Deep ConvNet (%):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    summary_path = Path(args.results_dir) / "summary.csv"
    summary.to_csv(summary_path, index=False)
    df.drop(columns="args").to_csv(Path(args.results_dir) / "per_subject.csv", index=False)
    print(f"\nWrote {summary_path} and per_subject.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    sub = parser.add_subparsers(required=True)

    train = sub.add_parser("train", help="train and evaluate subjects")
    train.set_defaults(func=cmd_train)
    train.add_argument("--dataset", choices=("BCIC_IV_2a", "HGD"), default="BCIC_IV_2a")
    train.add_argument("--low-cut", type=int, choices=LOW_CUTS, default=4,
                       help="0 or 4 Hz, the two frequency ranges of Table 2")
    train.add_argument("--subjects", type=int, nargs="+",
                       help="subject ids (default: all of the dataset)")
    train.add_argument("--job-index", type=int,
                       help=f"run only job N of the {len(all_jobs())} Table 2 runs "
                            "(overrides --dataset/--low-cut/--subjects)")
    train.add_argument("--max-epochs", type=int, default=800)
    train.add_argument("--max-increase-epochs", type=int, default=80,
                       help="phase 1 patience in epochs")
    train.add_argument("--valid-fraction", type=float, default=0.2)
    train.add_argument("--batch-size", type=int, default=60)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--no-tied-loss", dest="tied_loss", action="store_false")
    train.add_argument("--seed", type=int, default=20170629)
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--verbose", action="store_true", help="print every epoch")

    summarize = sub.add_parser("summarize", help="Table 2 style summary")
    summarize.set_defaults(func=cmd_summarize)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
