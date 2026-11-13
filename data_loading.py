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

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
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
# MOABB requests this URL. GIN redirects it and often answers 503.
HGD_GIN = "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
# Same EDFs, hosted by NEMAR (dataset nm000172, version 1.0.3).
HGD_NEMAR = "https://data.nemar.org/nm000172/v1.0.3/sourcedata"
# Smallest NEMAR test file is about 154 MB. Anything smaller is a failed download.
HGD_MIN_BYTES = 100 * 1024 * 1024
_DEBUG_LOG = Path(
    "/Users/jonasson/Documents/MSc Computer Science/ComputerScienceMasters"
    "/SeminarsInDataScience/CNN_for_EEG_reproducibility/.cursor/debug-a802bb.log"
)


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


def _agent_log(hypothesis_id, message, data):
    # region agent log
    record = {
        "sessionId": "a802bb",
        "hypothesisId": hypothesis_id,
        "location": "data_loading.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _DEBUG_LOG.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass
    # endregion


def _hgd_cache_path(kind, subject):
    """Path pooch uses for MOABB's GIN url. A file here skips the GIN download."""
    url = f"{HGD_GIN}/{kind}/{subject}.edf"
    relative = urlparse(url).path.lstrip("/")
    return DATA_DIR / "MNE-schirrmeister2017-data" / relative


def _hgd_final_path(kind, subject):
    return DATA_DIR / "MNE-schirrmeister2017-data" / kind / f"{subject}.edf"


def _complete(path):
    return path.is_file() and path.stat().st_size >= HGD_MIN_BYTES


def _fetch_url(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    size = partial.stat().st_size
    if size < HGD_MIN_BYTES:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"short file from {url} ({size} bytes)")
    partial.replace(dest)
    return size


def _ensure_hgd_subject(subject):
    """Put train and test EDFs where MOABB looks, using the NEMAR mirror."""
    for kind in ("train", "test"):
        final = _hgd_final_path(kind, subject)
        cache = _hgd_cache_path(kind, subject)
        url = f"{HGD_NEMAR}/{kind}/{subject}.edf"
        if not _complete(final):
            print(f"HGD subject {subject}: fetching {kind} from NEMAR", flush=True)
            size = _fetch_url(url, final)
        else:
            size = final.stat().st_size
        # A partial GIN download must not be treated as a finished file.
        if cache.exists() and not _complete(cache):
            cache.unlink()
        if not _complete(cache):
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists() or cache.is_symlink():
                cache.unlink()
            cache.symlink_to(final)
        _agent_log(
            "A",
            "hgd file ready",
            {
                "subject": subject,
                "kind": kind,
                "bytes": size,
                "final": str(final),
                "cache": str(cache),
                "cache_is_link": cache.is_symlink(),
            },
        )


def download(name, subject_ids=None, max_attempts=10):
    """Download subjects into the MOABB cache, retrying failed downloads.

    HGD is fetched from the NEMAR mirror. GIN, the host MOABB requests,
    answers 503 for these files. A symlink keeps the mirror file on the path
    pooch checks, so a later training run does not call GIN again.
    """
    if subject_ids is None:
        subject_ids = SUBJECTS[name]
    for subject in subject_ids:
        for attempt in range(1, max_attempts + 1):
            try:
                if name == "HGD":
                    _ensure_hgd_subject(subject)
                load_dataset(name, [subject])
                print(f"{name} subject {subject}: ok", flush=True)
                _agent_log("B", "subject loaded", {"name": name, "subject": subject})
                break
            except Exception as exc:
                print(f"{name} subject {subject}: attempt {attempt} failed ({exc})",
                      flush=True)
                _agent_log(
                    "C",
                    "download attempt failed",
                    {
                        "name": name,
                        "subject": subject,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Could not download {name} subject {subject}"
                    ) from exc
                time.sleep(min(60, 5 * attempt))


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
