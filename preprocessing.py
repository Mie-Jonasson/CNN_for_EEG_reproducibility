"""Preprocessing following Schirrmeister et al. (2017), Table 2 setup.

Same code as Preprocessing.ipynb, as a module so training scripts can import
it. The notebook explains each step and quotes the paper.

Section references are to the arXiv version of the paper
(https://arxiv.org/abs/1703.05051v5), whose supplementary material is
included as Appendix A:

- Sec. 2.5.3  Trial-wise training (trial window)
- Sec. 2.7.1  EEG preprocessing and evaluating different frequency bands
- Sec. A.6    Datasets (channels, train/test split)
- Sec. A.7    EEG preprocessing (resampling, low-pass, standardization,
              cleaning, HGD channel selection)

The paper does not state the order of cleaning, resampling and filtering;
cleaning is done here on the unfiltered uV data.
"""

import numpy as np
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)

# Sec. A.7: "for the High-Gamma Dataset, we used only those sensors covering
# the motor cortex: all central electrodes (45), except the Cz electrode which
# served as the recording reference electrode."
HGD_MOTOR_CHANNELS = [
    "FC5", "FC1", "FC2", "FC6", "C3", "C4", "CP5", "CP1", "CP2", "CP6",
    "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4",
    "FFC5h", "FFC3h", "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h",
    "CCP5h", "CCP3h", "CCP4h", "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h",
    "FFC1h", "FFC2h", "FCC1h", "FCC2h", "CCP1h", "CCP2h", "CPP1h", "CPP2h",
]
# Sec. A.7: "we low-pass filtered the BCI Competition Dataset to below 38 Hz".
# HGD gets no low-pass, so f_end is the 125 Hz Nyquist frequency after resampling.
HIGH_CUT_HZ = {"BCIC_IV_2a": 38.0, "HGD": None}  # None: no low-pass
# Sec. A.7: "We resampled the High-Gamma Dataset to 250 Hz"
SFREQ = 250.0
# Sec. 2.7.1: "Filtering was done with a causal 3rd order Butterworth filter."
FILTER_PARAMS = dict(
    method="iir", iir_params=dict(order=3, ftype="butter"), phase="forward"
)
# Sec. A.7: "removed trials where at least one channel had a value outside
# +-800 uV"
BAD_TRIAL_THRESHOLD_UV = 800.0
# Sec. A.7: exponential moving standardization "with a decay factor of 0.999",
# initialized with "the mean and the variance of the first 1000 samples"
FACTOR_NEW = 1e-3  # 1 - decay factor of 0.999
INIT_BLOCK_SIZE = 1000
# Sec. 2.5.3: "4.5-second trials (from 500 ms before trial start cue until
# trial end cue ...)"
TRIAL_START_OFFSET_S = -0.5
# Sec. A.6: four classes per dataset
CLASS_MAPPING = {
    "BCIC_IV_2a": {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3},
    "HGD": {"right_hand": 0, "left_hand": 1, "rest": 2, "feet": 3},
}
BAD_TRIAL = "bad_trial"


def mark_high_amplitude_trials(raw, threshold_uv=BAD_TRIAL_THRESHOLD_UV):
    """Relabel trials whose absolute amplitude exceeds the threshold (Sec. A.7).

    Expects data in uV. Relabelled trials are not in CLASS_MAPPING, so
    create_windows_from_events skips them.
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    descriptions = raw.annotations.description.copy()
    for i, (onset, duration) in enumerate(
        zip(raw.annotations.onset, raw.annotations.duration)
    ):
        start = int(round((onset - raw.first_time) * sfreq))
        stop = start + int(round(duration * sfreq))
        if np.abs(data[:, start:stop]).max() > threshold_uv:
            descriptions[i] = BAD_TRIAL
    raw.annotations.description = descriptions
    return raw


def preprocess_dataset(dataset, name, low_cut_hz, n_jobs=1):
    """Preprocess a dataset from data_loading.load_dataset in place.

    low_cut_hz is 4.0 for the 4-f_end rows of Table 2 and None (or 0) for the
    0-f_end rows (Sec. 2.7.1).
    """
    low_cut_hz = low_cut_hz or None
    high_cut_hz = HIGH_CUT_HZ[name]

    if name == "HGD":
        preprocessors = [
            Preprocessor("pick", picks=HGD_MOTOR_CHANNELS),  # Sec. A.7
            Preprocessor(lambda data: np.multiply(data, 1e6)),  # V -> uV
            Preprocessor("resample", sfreq=SFREQ),  # Sec. A.7
        ]
    else:
        preprocessors = [
            # 22 EEG electrodes (Sec. A.6), dropping the EOG and stim channels
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: np.multiply(data, 1e6)),  # V -> uV
        ]
    preprocessors.append(  # Sec. A.7
        Preprocessor(mark_high_amplitude_trials, apply_on_array=False)
    )
    if low_cut_hz is not None or high_cut_hz is not None:  # Sec. 2.7.1, A.7
        preprocessors.append(
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz, **FILTER_PARAMS
            )
        )
    preprocessors.append(  # Sec. A.7, Eqs. 4-6
        Preprocessor(
            exponential_moving_standardize,
            factor_new=FACTOR_NEW,
            init_block_size=INIT_BLOCK_SIZE,
        )
    )
    preprocess(dataset, preprocessors, n_jobs=n_jobs)
    return dataset


def create_trial_windows(dataset, name, trial_start_offset_s=TRIAL_START_OFFSET_S,
                         **kwargs):
    """Cut trials from 0.5 s before cue onset to the trial end (Sec. 2.5.3).

    Extra keyword arguments are passed on to create_windows_from_events.
    """
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all(ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets)
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(trial_start_offset_s * sfreq),
        trial_stop_offset_samples=0,
        mapping=CLASS_MAPPING[name],
        preload=True,
        **kwargs,
    )
