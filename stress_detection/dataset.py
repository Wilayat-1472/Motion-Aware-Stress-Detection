"""Validated notebook-format arrays and metadata, loaded without pickle."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES


@dataclass
class WindowDataset:
    sequences: np.ndarray
    features: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self):
        self.sequences = np.asarray(self.sequences, dtype=np.float32)
        self.features = np.asarray(self.features, dtype=np.float32)
        self.metadata = self.metadata.copy().reset_index(drop=True)
        count = len(self.metadata)
        if (
            not count
            or self.sequences.shape != (count, 6, 120)
            or self.features.shape != (count, 18)
        ):
            raise ValueError(
                "Expected nonempty sequences [N, 6, 120], features [N, 18], and N metadata rows."
            )
        if (
            not np.isfinite(self.sequences).all()
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("Signals and features must contain only finite values.")
        if not {"subject", "start_sec", "label"}.issubset(self.metadata):
            raise ValueError("Metadata requires subject, start_sec, and label columns.")
        if self.metadata[["subject", "start_sec", "label"]].isna().any().any():
            raise ValueError("Required metadata cannot contain missing values.")
        self.metadata["subject"] = self.metadata["subject"].astype(str)
        if (self.metadata["subject"].str.strip() == "").any():
            raise ValueError("Subject identifiers cannot be empty.")
        labels = pd.to_numeric(self.metadata["label"], errors="raise")
        starts = pd.to_numeric(self.metadata["start_sec"], errors="raise")
        if not labels.isin([0, 1, 2]).all():
            raise ValueError("Labels must be 0=Baseline, 1=Stress, or 2=Amusement.")
        if not np.isfinite(starts).all() or (starts < 0).any():
            raise ValueError("Window start times must be finite and nonnegative.")
        self.metadata["label"] = labels.astype(int)
        self.metadata["start_sec"] = starts
        if self.metadata.duplicated(["subject", "start_sec"]).any():
            raise ValueError("Duplicate subject/window start times are not allowed.")
        self.metadata["state"] = [CLASSES[i] for i in self.labels]
        self.metadata["sqi"] = self.features[:, 15]
        self.metadata["bpm"] = self.features[:, 0]
        self.metadata["eda_mean"] = self.features[:, 4]
        self.metadata["temp_mean"] = self.features[:, 9]

    @property
    def labels(self):
        return self.metadata["label"].to_numpy(dtype=np.int64)

    @property
    def subjects(self):
        return self.metadata["subject"].to_numpy(dtype=str)

    @property
    def subject_ids(self):
        return sorted(self.metadata["subject"].unique().tolist())

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "wesad_windows.npz", X_seq=self.sequences, X_feat=self.features
        )
        self.metadata.to_csv(directory / "wesad_meta.csv", index=False)


def load_windows(windows, metadata) -> WindowDataset:
    """Accept paths or file-like objects, including Streamlit uploads."""
    with np.load(windows, allow_pickle=False) as arrays:
        if not {"X_seq", "X_feat"}.issubset(arrays.files):
            raise ValueError("Window archive must contain X_seq and X_feat arrays.")
        sequences, features = arrays["X_seq"], arrays["X_feat"]
    return WindowDataset(
        sequences, features, pd.read_csv(metadata, dtype={"subject": str})
    )
