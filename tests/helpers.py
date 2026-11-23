"""Small deterministic sensor recordings for software checks, not research scores."""

import numpy as np

from stress_detection.dataset import WindowDataset
from stress_detection.preprocessing import extract_recording
import pandas as pd


def recording(subject_index=0):
    duration = 180
    rng = np.random.default_rng(40 + subject_index)
    bvp_time = np.arange(duration * 64) / 64
    eda_time = np.arange(duration * 4) / 4
    acc_time = np.arange(duration * 32) / 32
    bvp = np.sin(2 * np.pi * 1.25 * bvp_time) + 0.05 * rng.normal(size=len(bvp_time))
    eda = 2 + 0.002 * eda_time + 0.03 * np.sin(eda_time / 3) + subject_index * 0.1
    temp = 32 + 0.02 * np.cos(eda_time / 9) + subject_index * 0.2
    acc = np.column_stack(
        [0.3 * np.sin(acc_time), 0.2 * np.cos(acc_time), np.ones(len(acc_time))]
    )
    acc += rng.normal(0, 0.02, size=acc.shape)
    return {
        "signal": {
            "wrist": {
                "BVP": bvp[:, None],
                "EDA": eda[:, None],
                "TEMP": temp[:, None],
                "ACC": acc,
            }
        },
        "label": np.repeat([1, 2, 3], 60 * 700),
    }


def fixture_dataset():
    parts = [
        extract_recording(recording(i), subject)
        for i, subject in enumerate(["S2", "S3", "S11"])
    ]
    return WindowDataset(
        np.concatenate([p.sequences for p in parts]),
        np.concatenate([p.features for p in parts]),
        pd.concat([p.metadata for p in parts], ignore_index=True),
    )
