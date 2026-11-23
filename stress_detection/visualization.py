"""Offline research figures generated from actual evaluation artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from .config import CLASSES


def plot_diagnostics(
    results_path, metadata_path, output_path, *, subject="S11", counterfactual=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = pd.read_csv(metadata_path, dtype={"subject": str})
    # The original notebook stores subject IDs as object arrays. Read numeric arrays
    # without pickle, and use the corresponding CSV for subject identifiers.
    with np.load(results_path, allow_pickle=False) as archive:
        scores = archive["oof_probs"]
        predictions = archive["oof_preds"]
        labels = archive["y_true"]
        if "start_sec" in archive.files:
            if not np.array_equal(
                archive["start_sec"], metadata["start_sec"].to_numpy()
            ):
                raise ValueError(
                    "Result archive and metadata window times do not match."
                )
            if not np.array_equal(
                archive["subjects"], metadata["subject"].to_numpy(dtype=str)
            ):
                raise ValueError(
                    "Result archive and metadata subject order do not match."
                )
    count = len(metadata)
    if (
        scores.shape != (count, 3)
        or labels.shape != (count,)
        or predictions.shape != (count,)
    ):
        raise ValueError("Result archive shapes do not match the metadata.")
    if not np.array_equal(labels, metadata["label"].to_numpy()):
        raise ValueError("Result labels and metadata order do not match.")
    if (
        not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.allclose(scores.sum(1), 1.0, atol=1e-5)
        or not np.array_equal(predictions, scores.argmax(1))
    ):
        raise ValueError("Invalid out-of-fold probabilities.")
    subjects = metadata["subject"].to_numpy()
    if subject not in subjects:
        raise ValueError(f"Timeline subject {subject} is not present.")
    ids = sorted(set(subjects))
    per_subject = [
        f1_score(labels[subjects == s], predictions[subjects == s], average="macro")
        for s in ids
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(ids, per_subject)
    axes[0, 0].axhline(
        np.mean(per_subject),
        color="black",
        linestyle="--",
        label=f"Mean F1 = {np.mean(per_subject):.3f}",
    )
    axes[0, 0].set(title="A. Subject-held-out Macro-F1", ylabel="Macro-F1", ylim=(0, 1))
    axes[0, 0].legend()
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2], normalize="true")
    axes[0, 1].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title=f"B. Normalized confusion matrix (N = {count:,})",
        xlabel="Predicted",
        ylabel="True",
    )
    for row in range(3):
        for col in range(3):
            axes[0, 1].text(
                col,
                row,
                f"{matrix[row, col]:.1%}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > 0.5 else "black",
            )
    mask = subjects == subject
    rows = metadata[mask]
    x = np.arange(mask.sum())
    axes[1, 0].plot(x, scores[mask, 1], color="#d62728", label="Stress score")
    axes[1, 0].plot(x, rows["sqi"], color="#2ca02c", linestyle="--", label="Motion SQI")
    axes[1, 0].fill_between(
        x,
        0,
        1,
        where=labels[mask] == 1,
        color="#d62728",
        alpha=0.15,
        label="True Stress",
    )
    axes[1, 0].set(
        title=f"C. {subject}: offline score and motion timeline",
        xlabel="Retained window index",
        ylabel="Score / SQI",
    )
    axes[1, 0].legend()
    if counterfactual:
        names = counterfactual["top_features"][::-1]
        values = [counterfactual["feature_deltas"][name] for name in names]
        axes[1, 1].barh(names, values)
        axes[1, 1].axvline(0, color="black")
        axes[1, 1].set(
            title=f"D. {counterfactual['subject']}: physiological counterfactual",
            xlabel="Change in fold-standardized feature units",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Run the explain command to add a counterfactual.",
            ha="center",
            va="center",
            wrap=True,
        )
        axes[1, 1].set_axis_off()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recording(recording, subject, output_path):
    """Plot protocol, EDA, temperature, and motion at their native sampling rates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrist = recording["signal"]["wrist"]
    labels = np.asarray(recording["label"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1)
    temperature = np.asarray(wrist["TEMP"]).reshape(-1)
    magnitude = np.linalg.norm(np.asarray(wrist["ACC"]), axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for axis, values, sampling_rate, ylabel in zip(
        axes,
        [labels, eda, temperature, magnitude],
        [700, 4, 4, 32],
        ["Protocol label", "EDA (µS)", "TEMP (°C)", "ACC magnitude"],
    ):
        axis.plot(np.arange(len(values)) / sampling_rate / 60, values, linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[0].set_title(f"{subject}: native-rate WESAD wrist signals")
    axes[-1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
