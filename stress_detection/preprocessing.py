"""Offline WESAD wrist filtering and window extraction, matching notebook cell 3."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .config import LABEL_MAP, STRIDE_SECONDS, WINDOW_SECONDS
from .dataset import WindowDataset
from .features import extract_features


def load_recording(path):
    """Read an original WESAD pickle from a trusted local dataset."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def bandpass_bvp(signal, fs=64.0):
    b, a = butter(3, [0.7 / (0.5 * fs), 3.5 / (0.5 * fs)], btype="band")
    return filtfilt(b, a, signal)


def lowpass_eda(signal, fs=4.0):
    b, a = butter(2, 0.5 / (0.5 * fs), btype="low")
    return filtfilt(b, a, signal)


def extract_recording(recording, subject: str) -> WindowDataset:
    """Process an already-loaded WESAD recording; labels select pure target windows."""
    wrist = recording["signal"]["wrist"]
    signals = {
        name: np.asarray(wrist[name], dtype=float)
        for name in ("BVP", "ACC", "EDA", "TEMP")
    }
    for name in ("BVP", "EDA", "TEMP"):
        signals[name] = signals[name].reshape(-1)
    acc = signals["ACC"]
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("ACC recording must have shape [samples, 3].")
    labels = np.asarray(recording["label"]).reshape(-1)
    if not all(np.isfinite(x).all() for x in [*signals.values(), labels]):
        raise ValueError("Recording contains nonfinite samples.")
    total_seconds = int(
        min(
            len(signals["BVP"]) / 64,
            len(signals["EDA"]) / 4,
            len(signals["TEMP"]) / 4,
            len(acc) / 32,
            len(labels) / 700,
        )
    )
    if total_seconds < WINDOW_SECONDS:
        raise ValueError(
            "Recording must contain at least 30 seconds of aligned signals."
        )
    bvp = bandpass_bvp(signals["BVP"])
    eda = lowpass_eda(signals["EDA"])
    temperature = signals["TEMP"]
    magnitude = np.linalg.norm(acc, axis=1)
    sequences, features, rows = [], [], []
    for start in range(0, total_seconds - WINDOW_SECONDS + 1, STRIDE_SECONDS):
        end = start + WINDOW_SECONDS
        values, counts = np.unique(labels[start * 700 : end * 700], return_counts=True)
        dominant = values[counts.argmax()]
        if dominant not in LABEL_MAP or counts.max() / counts.sum() < 0.90:
            continue
        b = bvp[start * 64 : end * 64]
        e = eda[start * 4 : end * 4]
        t = temperature[start * 4 : end * 4]
        a = acc[start * 32 : end * 32]
        sequence = np.stack(
            [
                b.reshape(120, 16).mean(1),
                e,
                t,
                *[a[:, axis].reshape(120, 8).mean(1) for axis in range(3)],
            ]
        )
        sequences.append(sequence.astype(np.float32))
        features.append(extract_features(b, e, t, magnitude[start * 32 : end * 32]))
        rows.append(
            {"subject": subject, "start_sec": start, "label": LABEL_MAP[dominant]}
        )
    if not rows:
        raise ValueError(f"No eligible windows for subject {subject}.")
    return WindowDataset(np.stack(sequences), np.stack(features), pd.DataFrame(rows))


def prepare_dataset(data_directory, output_directory):
    """Read trusted WESAD pickle files, then save safe NPZ/CSV window artifacts."""
    root = Path(data_directory)
    if (root / "WESAD").is_dir():
        root = root / "WESAD"
    paths = sorted(root.glob("S*/S*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No WESAD subject files found under {root}.")
    datasets = []
    for path in paths:
        if path.stem != path.parent.name:
            continue
        print(f"Extracting {path.stem}...", flush=True)
        # WESAD's original data format is pickle. Only trusted local datasets belong here.
        recording = load_recording(path)
        datasets.append(extract_recording(recording, path.stem))
    if not datasets:
        raise ValueError("No matching subject recording names found.")
    combined = WindowDataset(
        np.concatenate([d.sequences for d in datasets]),
        np.concatenate([d.features for d in datasets]),
        pd.concat([d.metadata for d in datasets], ignore_index=True),
    )
    combined.save(output_directory)
    return combined
