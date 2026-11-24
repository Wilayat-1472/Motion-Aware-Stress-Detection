"""Resting calibration and training-fold scaling shared by training and inference."""

from dataclasses import dataclass

import numpy as np

from .config import BASELINE_WINDOWS, WINDOW_SECONDS
from .dataset import WindowDataset


@dataclass
class CalibratedWindows:
    sequences: np.ndarray
    features: np.ndarray


def baseline_indices(dataset, subject):
    indices = np.flatnonzero((dataset.subjects == subject) & (dataset.labels == 0))
    indices = indices[
        np.argsort(dataset.metadata.loc[indices, "start_sec"].to_numpy(), kind="stable")
    ]
    if len(indices) < BASELINE_WINDOWS:
        raise ValueError(
            f"Subject {subject} needs at least {BASELINE_WINDOWS} Baseline windows for calibration."
        )
    return indices[:BASELINE_WINDOWS]


def post_onboarding_mask(dataset):
    """Score only windows starting after the entire resting reference interval.

    Excluding just the three reference rows is insufficient: the 15s stride
    makes the next 30s window overlap the last onboarding window.
    """
    mask = np.zeros(len(dataset.labels), dtype=bool)
    starts = dataset.metadata['start_sec'].to_numpy()
    for subject in dataset.subject_ids:
        reference = baseline_indices(dataset, subject)
        end = float(starts[reference].max()) + WINDOW_SECONDS
        subject_mask = (dataset.subjects == subject) & (starts >= end)
        if not subject_mask.any():
            raise ValueError(f'Subject {subject} has no windows after resting onboarding.')
        mask |= subject_mask
    return mask


def calibrate_windows(
    dataset: WindowDataset, *, tabular_only=False
) -> CalibratedWindows:
    sequences = dataset.sequences.copy()
    features = dataset.features.copy()
    for subject in dataset.subject_ids:
        mask = dataset.subjects == subject
        rest = baseline_indices(dataset, subject)
        rest_features = dataset.features[rest, :12]
        features[mask, :12] = (features[mask, :12] - rest_features.mean(0)) / (
            rest_features.std(0) + 1e-4
        )
        if not tabular_only:
            rest_sequence = dataset.sequences[rest, :3, :]
            sequences[mask, :3, :] = (
                sequences[mask, :3, :] - rest_sequence.mean((0, 2), keepdims=True)
            ) / (rest_sequence.std((0, 2), keepdims=True) + 1e-4)
            acc = sequences[mask, 3:, :]
            sequences[mask, 3:, :] = (acc - acc.mean(2, keepdims=True)) / (
                acc.std(2, keepdims=True) + 1e-4
            )
    if not tabular_only:
        sequences = np.clip(sequences, -10.0, 10.0)
        features = np.clip(features, -10.0, 10.0)
    return CalibratedWindows(sequences, features)


def fit_feature_scaler(features, training_mask):
    training = features[training_mask]
    if not len(training):
        raise ValueError("The training split is empty.")
    return training.mean(0), training.std(0) + 1e-5
