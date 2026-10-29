"""Dataset loading for the Schirrmeister et al. (2017) Table 2 reproduction.

Both datasets are fetched through MOABB (via braindecode's MOABBDataset) and
cached in the project's datasets/ folder on first use:

- BCIC IV 2a: MOABB ``BNCI2014_001`` (9 subjects, 22 EEG channels, 250 Hz).
  Downloaded from the public BNCI Horizon 2020 server.
- HGD: MOABB ``Schirrmeister2017`` (14 subjects, 128 EEG channels, 500 Hz).
  Downloaded from the G-Node GIN repository linked in MOABB.

Each subject has a predefined train/test split ("0train"/"1test" runs for HGD,
"0train"/"1test" sessions for BCIC IV 2a), which is the split the paper uses.

Note on the "Combined" rows of Table 2: they are not a model trained on pooled
data, but the subject-weighted mean over all 9 + 14 = 23 subjects, e.g.
FBCSP (9 * 68.0 + 14 * 91.2) / 23 = 82.1. See ``combined_accuracy``.
"""

import os
from pathlib import Path

from braindecode.datasets import MOABBDataset

# Store data in the project instead of ~/mne_data. Environment variables take
# precedence over paths saved in the global MNE config (~/.mne).
DATA_DIR = Path(__file__).resolve().parent / "datasets"
DATA_DIR.mkdir(exist_ok=True)
for key in ("MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_SCHIRRMEISTER2017_PATH"):
    os.environ[key] = str(DATA_DIR)

DATASETS = {
    "BCIC_IV_2a": "BNCI2014_001",
    "HGD": "Schirrmeister2017",
}
SUBJECTS = {
    "BCIC_IV_2a": list(range(1, 10)),
    "HGD": list(range(1, 15)),
}


def load_dataset(name, subject_ids=None):
    """Load raw recordings for one dataset as a braindecode BaseConcatDataset.

    Parameters
    ----------
    name : str
        "BCIC_IV_2a" or "HGD".
    subject_ids : list of int | None
        Subjects to load. None loads all subjects of the dataset.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}, choose from {list(DATASETS)}")
    if subject_ids is None:
        subject_ids = SUBJECTS[name]
    return MOABBDataset(dataset_name=DATASETS[name], subject_ids=subject_ids)


def download(name, subject_ids=None, max_attempts=10):
    """Download subjects into the MOABB cache, retrying failed downloads.

    The HGD server (G-Node GIN) is slow and MOABB gives up after a 30 s read
    timeout, so retry each subject until it is cached. Files that are already
    cached are not downloaded again.
    """
    if subject_ids is None:
        subject_ids = SUBJECTS[name]
    for subject in subject_ids:
        for attempt in range(1, max_attempts + 1):
            try:
                load_dataset(name, [subject])
                print(f"{name} subject {subject}: ok", flush=True)
                break
            except Exception as exc:
                print(f"{name} subject {subject}: attempt {attempt} failed ({exc})",
                      flush=True)
        else:
            raise RuntimeError(f"Could not download {name} subject {subject}")


def split_train_test(dataset):
    """Split into the predefined train and test parts used in the paper.

    Sec. A.6 (arXiv v5): BCIC IV 2a trains on session 1 and tests on session 2;
    HGD tests on the last 2 runs (approx. 160 trials) and trains on the rest.
    """
    # BCIC IV 2a encodes the split as sessions, HGD as runs within one session
    by = "session" if "0train" in dataset.description["session"].values else "run"
    splitted = dataset.split(by)
    return splitted["0train"], splitted["1test"]


def combined_accuracy(bcic_accuracies, hgd_accuracies):
    """Subject-weighted mean accuracy over both datasets (Table 2 "Combined")."""
    accs = list(bcic_accuracies) + list(hgd_accuracies)
    return sum(accs) / len(accs)
