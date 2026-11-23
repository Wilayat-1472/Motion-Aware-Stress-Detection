"""Train the notebook network and export reference or all-subject models."""

from dataclasses import asdict
from pathlib import Path
import platform

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .calibration import calibrate_windows, fit_feature_scaler
from .config import CLASSES, NOTEBOOK_URL, PREPROCESSING_VERSION, TrainingConfig
from .inference import ModelBundle
from .model import SQIGatedMultimodalNet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_network(sequences, standardized_features, labels, config):
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("Every training split must contain all three classes.")
    device_name = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(standardized_features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = SQIGatedMultimodalNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    weights = torch.tensor(
        len(labels) / (3.0 * counts), dtype=torch.float32, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config.label_smoothing
    )
    model.train()
    for _ in range(config.epochs):
        for seq, feat, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(seq.to(device), feat.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
    return model.cpu().eval()


def train_reference(dataset, config=TrainingConfig(), holdout="S11"):
    if holdout is not None and holdout not in dataset.subject_ids:
        raise ValueError(f"Held-out subject {holdout} is not in the dataset.")
    calibrated = calibrate_windows(dataset)
    train = (
        dataset.subjects != holdout
        if holdout is not None
        else np.ones(len(dataset.labels), dtype=bool)
    )
    mean, std = fit_feature_scaler(calibrated.features, train)
    set_seed(config.seed)
    model = train_network(
        calibrated.sequences[train],
        (calibrated.features[train] - mean) / std,
        dataset.labels[train],
        config,
    )
    provenance = {
        "source_notebook": NOTEBOOK_URL,
        "preprocessing": PREPROCESSING_VERSION,
        "training_config": asdict(config),
        "training_subjects": sorted(set(dataset.subjects[train])),
        "held_out_subject": holdout,
        "model_role": "held_out_reference" if holdout else "all_subjects",
        "training_windows": int(train.sum()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    return ModelBundle(model, mean, std, CLASSES, provenance)


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "state_dict": {
                name: value.cpu() for name, value in bundle.model.state_dict().items()
            },
            "f_mu": bundle.feature_mean,
            "f_sd": bundle.feature_std,
            "classes": list(bundle.classes),
            "config": {"seq_channels": 6, "feat_dim": 18, "num_classes": 3},
            "provenance": bundle.provenance,
        },
        path,
        compress=3,
    )
