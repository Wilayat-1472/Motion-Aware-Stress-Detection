"""Native-rate physiological features and SQI used in the research notebook."""

import numpy as np
from scipy.signal import find_peaks

from .config import FEATURE_NAMES as FEATURE_NAMES  # Public feature-order contract.


def motion_sqi(motion_std: float) -> float:
    if not np.isfinite(motion_std) or motion_std < 0:
        raise ValueError("Motion standard deviation must be finite and nonnegative.")
    return float(np.exp(-0.08 * motion_std))


def extract_features(bvp, eda, temperature, acc_magnitude):
    """Extract the notebook's 18 biomarkers from one filtered 30-second window."""
    peaks, _ = find_peaks(bvp, distance=int(0.4 * 64), prominence=np.std(bvp) * 0.4)
    if len(peaks) >= 4:
        rr_ms = np.diff(peaks) / 64.0 * 1000.0
        bpm = float(60000.0 / np.mean(rr_ms))
        sdnn = float(np.std(rr_ms))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    else:
        bpm, sdnn, rmssd = 75.0, 40.0, 30.0
    eda_diff = np.diff(eda)
    acc_std = float(np.std(acc_magnitude))
    sqi = motion_sqi(acc_std)
    return np.array(
        [
            bpm,
            sdnn,
            rmssd,
            float(np.std(bvp)),
            float(np.mean(eda)),
            float(np.std(eda)),
            float(eda[-1] - eda[0]),
            float(np.sum(eda_diff > 0.01)),
            float(np.max(eda_diff)),
            float(np.mean(temperature)),
            float(np.std(temperature)),
            float((temperature[-1] - temperature[0]) / 0.5),
            float(np.mean(acc_magnitude)),
            acc_std,
            float(np.max(acc_magnitude) - np.min(acc_magnitude)),
            sqi,
            bpm * sqi,
            rmssd * sqi,
        ],
        dtype=np.float32,
    )
