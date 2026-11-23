"""L1-penalized physiological counterfactuals with sequence and motion held fixed."""

import numpy as np
import torch

from .calibration import calibrate_windows
from .config import FEATURE_NAMES


def explain_window(
    bundle, dataset, index, *, steps=120, learning_rate=0.05, penalty=0.15
):
    if index < 0 or index >= len(dataset.labels):
        raise ValueError("Window index is outside the dataset.")
    if steps < 1 or learning_rate <= 0 or penalty < 0:
        raise ValueError("Invalid counterfactual optimizer settings.")
    calibrated = calibrate_windows(dataset)
    sequence = torch.tensor(
        calibrated.sequences[index : index + 1], dtype=torch.float32
    )
    original = torch.tensor(
        (calibrated.features[index : index + 1] - bundle.feature_mean)
        / bundle.feature_std,
        dtype=torch.float32,
    )
    bundle.model.cpu().eval()
    changed = original.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([changed], lr=learning_rate)
    with torch.no_grad():
        before = torch.softmax(bundle.model(sequence, original), dim=1)[0].numpy()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = bundle.model(sequence, changed)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
        loss = loss + penalty * torch.abs(changed[:, :12] - original[:, :12]).mean()
        # Compute only the input gradient; do not accumulate model parameter gradients.
        (changed.grad,) = torch.autograd.grad(loss, changed)
        optimizer.step()
        with torch.no_grad():
            changed[:, 12:] = original[:, 12:]
    with torch.no_grad():
        after = torch.softmax(bundle.model(sequence, changed), dim=1)[0].numpy()
    delta = (changed - original).detach().numpy()[0]
    row = dataset.metadata.iloc[index]
    return {
        "subject": str(row["subject"]),
        "window_index": int(index),
        "start_sec": float(row["start_sec"]),
        "true_state": str(row["state"]),
        "target_class": "Baseline",
        "target_reached": bool(after.argmax() == 0),
        "original_scores": dict(zip(bundle.classes, before.tolist())),
        "counterfactual_scores": dict(zip(bundle.classes, after.tolist())),
        "feature_deltas": dict(zip(FEATURE_NAMES, delta.tolist())),
        "delta_units": "fold-standardized feature units after resting calibration",
        "fixed_feature_indices": list(range(12, 18)),
        "sequence_fixed": True,
        "steps": steps,
        "learning_rate": learning_rate,
        "l1_penalty": penalty,
        "top_features": [
            FEATURE_NAMES[i] for i in np.argsort(np.abs(delta[:12]))[::-1][:6]
        ],
        "bundle_provenance": bundle.provenance,
    }
