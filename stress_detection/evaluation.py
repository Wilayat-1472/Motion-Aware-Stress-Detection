"""Subject-held-out tabular benchmarks, neural evaluation, and saved reports."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .calibration import calibrate_windows, fit_feature_scaler
from .config import CLASSES, TrainingConfig
from .inference import ModelBundle, predict_calibrated
from .training import set_seed, train_network


def summarize(labels, predictions, subjects):
    per_subject = {
        subject: float(
            f1_score(
                labels[subjects == subject],
                predictions[subjects == subject],
                average="macro",
            )
        )
        for subject in sorted(set(subjects))
    }
    return {
        "windows": int(len(labels)),
        "subjects": len(per_subject),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mean_subject_macro_f1": float(np.mean(list(per_subject.values()))),
        "std_subject_macro_f1": float(np.std(list(per_subject.values()))),
        "per_subject_macro_f1": per_subject,
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
        "normalized_confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2], normalize="true"
        ).tolist(),
    }


def validate_loso(dataset):
    if len(dataset.subject_ids) < 2:
        raise ValueError("LOSO evaluation needs at least two participants.")
    for subject in dataset.subject_ids:
        if set(dataset.labels[dataset.subjects != subject]) != {0, 1, 2}:
            raise ValueError(
                f"Training fold excluding {subject} does not contain all three classes."
            )


def run_tabular_benchmarks(dataset):
    validate_loso(dataset)
    calibrated = calibrate_windows(dataset, tabular_only=True)
    reports = {}
    for name, features in (
        ("uncalibrated", dataset.features),
        ("resting_calibrated", calibrated.features),
    ):
        predictions = np.zeros(len(dataset.labels), dtype=int)
        for subject in dataset.subject_ids:
            test = dataset.subjects == subject
            classifier = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=42,
            )
            classifier.fit(features[~test], dataset.labels[~test])
            predictions[test] = classifier.predict(features[test])
            print(f"{name}: completed held-out subject {subject}", flush=True)
        reports[name] = summarize(dataset.labels, predictions, dataset.subjects)
    return reports


def run_loso(dataset, config=TrainingConfig()):
    validate_loso(dataset)
    calibrated = calibrate_windows(dataset)
    set_seed(config.seed)  # Seed once before the fold loop, matching the notebook.
    scores = np.zeros((len(dataset.labels), 3), dtype=np.float32)
    folds = []
    started = time.monotonic()
    for subject in dataset.subject_ids:
        test = dataset.subjects == subject
        mean, std = fit_feature_scaler(calibrated.features, ~test)
        model = train_network(
            calibrated.sequences[~test],
            (calibrated.features[~test] - mean) / std,
            dataset.labels[~test],
            config,
        )
        bundle = ModelBundle(model, mean, std)
        scores[test] = predict_calibrated(
            bundle,
            calibrated.sequences[test],
            calibrated.features[test],
            config.evaluation_batch_size,
        )
        fold_f1 = float(
            f1_score(dataset.labels[test], scores[test].argmax(1), average="macro")
        )
        folds.append(
            {
                "test_subject": subject,
                "training_subjects": sorted(set(dataset.subjects[~test])),
                "macro_f1": fold_f1,
                "training_windows": int((~test).sum()),
                "test_windows": int(test.sum()),
            }
        )
        print(f"Held out {subject}: Macro-F1 {fold_f1:.4f}", flush=True)
    report = summarize(dataset.labels, scores.argmax(1), dataset.subjects)
    report.update(
        {
            "training_config": asdict(config),
            "elapsed_seconds": time.monotonic() - started,
            "folds": folds,
        }
    )
    return scores, report


def save_evaluation(dataset, scores, report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "wesad_loso_results.npz",
        oof_probs=scores,
        oof_preds=scores.argmax(1),
        y_true=dataset.labels,
        subjects=dataset.subjects,
        start_sec=dataset.metadata["start_sec"].to_numpy(),
    )
    write_json(report, directory / "metrics.json")


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
