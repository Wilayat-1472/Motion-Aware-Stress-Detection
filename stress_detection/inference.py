"""Notebook-compatible calibrated window inference and trusted bundle loading."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .calibration import calibrate_windows
from .config import CLASSES
from .model import SQIGatedMultimodalNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "wesad_bundle.joblib"
DIAGNOSTICS_PATH = PROJECT_ROOT / "assets" / "wesad_loso_diagnostics.png"
REFERENCE_RESULTS_PATH = PROJECT_ROOT / "assets" / "reference_results.json"


@dataclass
class ModelBundle:
    model: SQIGatedMultimodalNet
    feature_mean: np.ndarray
    feature_std: np.ndarray
    classes: tuple[str, ...] = CLASSES
    provenance: dict = field(default_factory=dict)


def load_bundle(path=DEFAULT_BUNDLE_PATH):
    """Joblib is pickle-based; load only trusted local artifacts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    data = joblib.load(path)
    required = {"config", "state_dict", "f_mu", "f_sd", "classes"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(
            "Bundle must contain config, state_dict, f_mu, f_sd, and classes."
        )
    if data["config"] != {"seq_channels": 6, "feat_dim": 18, "num_classes": 3}:
        raise ValueError("Expected 6 signal channels, 18 features, and 3 classes.")
    mean = np.asarray(data["f_mu"], dtype=np.float32)
    std = np.asarray(data["f_sd"], dtype=np.float32)
    if mean.shape != (18,) or std.shape != (18,):
        raise ValueError("Bundle scaler arrays must each have shape (18,).")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Bundle scaler must be finite with strictly positive scales.")
    if tuple(data["classes"]) != CLASSES:
        raise ValueError("Expected class order: Baseline, Stress, Amusement.")
    model = SQIGatedMultimodalNet(**data["config"])
    model.load_state_dict(data["state_dict"], strict=True)
    model.cpu().eval()
    return ModelBundle(model, mean, std, CLASSES, data.get("provenance", {}))


def predict_calibrated(bundle, sequences, features, batch_size=128):
    """Infer already resting-calibrated/clipped windows, using the bundle scaler."""
    sequences = np.asarray(sequences, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    count = len(sequences)
    if not count or sequences.shape != (count, 6, 120) or features.shape != (count, 18):
        raise ValueError(
            "Expected nonempty sequences [N, 6, 120] and features [N, 18]."
        )
    if not np.isfinite(sequences).all() or not np.isfinite(features).all():
        raise ValueError("Inference inputs must be finite.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    normalized = (features - bundle.feature_mean) / bundle.feature_std
    result = []
    loader = DataLoader(
        TensorDataset(
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(normalized, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    bundle.model.cpu().eval()
    with torch.inference_mode():
        for sequence, feature in loader:
            result.append(torch.softmax(bundle.model(sequence, feature), dim=1).numpy())
    return np.concatenate(result)


def predict_dataset(bundle, dataset):
    """Apply the research calibration to raw notebook window artifacts."""
    calibrated = calibrate_windows(dataset)
    scores = predict_calibrated(bundle, calibrated.sequences, calibrated.features)
    frame = dataset.metadata.copy()
    frame["predicted_state"] = [bundle.classes[i] for i in scores.argmax(1)]
    for index, name in enumerate(bundle.classes):
        frame[f"score_{name.lower()}"] = scores[:, index]
    return frame


def prediction_scores(frame: pd.DataFrame):
    return frame[[f"score_{name.lower()}" for name in CLASSES]].to_numpy()
