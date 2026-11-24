"""Command-line workflow extracted from the WESAD research notebook."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .dataset import load_windows
from .evaluation import run_loso, run_tabular_benchmarks, save_evaluation, write_json
from .explainability import explain_window
from .inference import DEFAULT_BUNDLE_PATH, load_bundle, predict_dataset
from .preprocessing import load_recording, prepare_dataset
from .packaging import package_project
from .training import save_bundle, train_reference
from .visualization import plot_diagnostics, plot_recording


def add_windows(parser):
    parser.add_argument(
        "--windows",
        default="data/processed/wesad_windows.npz",
        help="Raw notebook-format X_seq/X_feat archive.",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/wesad_meta.csv",
        help="Matching window metadata CSV.",
    )


def add_training(parser):
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "download", help="Download WESAD through KaggleHub and print its location."
    )
    inspect = commands.add_parser(
        "inspect", help="Plot native-rate signals from a trusted WESAD recording."
    )
    inspect.add_argument("--recording", required=True)
    inspect.add_argument("--output", default="outputs/recording.png")
    package = commands.add_parser(
        "package", help="Create a portable source/deployment archive."
    )
    package.add_argument("--output", default="outputs/wesad_project.zip")
    prepare = commands.add_parser(
        "prepare", help="Filter trusted WESAD recordings and extract windows."
    )
    prepare.add_argument(
        "--data-dir",
        required=True,
        help="WESAD root or its parent; trusted pickle inputs only.",
    )
    prepare.add_argument("--output-dir", default="data/processed")
    benchmark = commands.add_parser(
        "benchmark", help="Run raw and resting-calibrated tabular LOSO benchmarks."
    )
    add_windows(benchmark)
    benchmark.add_argument("--output", default="outputs/tabular_metrics.json")
    benchmark.add_argument('--include-onboarding', action='store_true', help='Reproduce historical scoring, including resting reference windows.')
    evaluate = commands.add_parser(
        "evaluate", help="Train a new neural model per held-out subject."
    )
    add_windows(evaluate)
    add_training(evaluate)
    evaluate.add_argument("--output-dir", default="outputs/loso")
    evaluate.add_argument('--include-onboarding', action='store_true', help='Reproduce historical scoring, including resting reference windows.')
    train = commands.add_parser(
        "train", help="Train and export one reference or all-subject model."
    )
    add_windows(train)
    add_training(train)
    group = train.add_mutually_exclusive_group()
    group.add_argument(
        "--holdout", default="S11", help="Exclude this subject (default: S11)."
    )
    group.add_argument(
        "--all-subjects",
        action="store_true",
        help="Fit a deployment model using every participant.",
    )
    train.add_argument("--output", default="outputs/reference_model.joblib")
    predict = commands.add_parser(
        "predict", help="Calibrate and predict actual recorded windows."
    )
    add_windows(predict)
    predict.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH))
    predict.add_argument(
        "--output", help="Prediction CSV path; otherwise print JSON records."
    )
    explain = commands.add_parser(
        "explain", help="Optimize a fixed-sequence physiological counterfactual."
    )
    add_windows(explain)
    explain.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH))
    explain.add_argument("--subject", default="S11")
    explain.add_argument(
        "--stress-window",
        type=int,
        default=5,
        help="Zero-based Stress-window ordinal (default: sixth).",
    )
    explain.add_argument("--steps", type=int, default=120)
    explain.add_argument("--output", default="outputs/counterfactual.json")
    plot = commands.add_parser(
        "plot", help="Regenerate a four-panel diagnostics figure."
    )
    plot.add_argument("--results", default="outputs/loso/wesad_loso_results.npz")
    plot.add_argument("--metadata", default="data/processed/wesad_meta.csv")
    plot.add_argument("--counterfactual", help="Optional JSON produced by explain.")
    plot.add_argument("--subject", default="S11")
    plot.add_argument("--output", default="outputs/wesad_loso_diagnostics.png")
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            print(f"Packaged project: {package_project(args.output)}")
            return
        if args.command == "inspect":
            plot_recording(
                load_recording(args.recording), Path(args.recording).stem, args.output
            )
            print(f"Saved recording plot: {args.output}")
            return
        if args.command == "download":
            try:
                import kagglehub
            except ImportError:
                raise ValueError(
                    "Install requirements-notebook.txt to enable KaggleHub downloads."
                ) from None
            print(
                kagglehub.dataset_download(
                    "orvile/wesad-wearable-stress-affect-detection-dataset"
                )
            )
            return
        if args.command == "prepare":
            data = prepare_dataset(args.data_dir, args.output_dir)
            print(
                f"Saved {len(data.labels)} windows from {len(data.subject_ids)} subjects to {args.output_dir}."
            )
            return
        if args.command == "plot":
            counterfactual = (
                json.loads(Path(args.counterfactual).read_text())
                if args.counterfactual
                else None
            )
            plot_diagnostics(
                args.results,
                args.metadata,
                args.output,
                subject=args.subject,
                counterfactual=counterfactual,
            )
            print(f"Saved diagnostics: {args.output}")
            return
        data = load_windows(args.windows, args.metadata)
        if args.command in ("train", "evaluate"):
            config = TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
            )
        if args.command == "benchmark":
            write_json(run_tabular_benchmarks(data, exclude_onboarding=not args.include_onboarding), args.output)
            print(f"Saved benchmarks: {args.output}")
        elif args.command == "evaluate":
            scores, report = run_loso(data, config, exclude_onboarding=not args.include_onboarding)
            save_evaluation(data, scores, report, args.output_dir)
            print(
                f"Macro-F1: {report['macro_f1']:.4f}; balanced accuracy: {report['balanced_accuracy']:.4f}"
            )
        elif args.command == "train":
            bundle = train_reference(
                data, config, holdout=None if args.all_subjects else args.holdout
            )
            save_bundle(bundle, args.output)
            print(f"Saved model: {args.output}")
        elif args.command == "predict":
            bundle = load_bundle(args.bundle)
            result = predict_dataset(bundle, data)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(path, index=False)
                print(f"Saved {len(result)} predictions: {path}")
            else:
                print(result.to_json(orient="records", indent=2))
        elif args.command == "explain":
            indices = np.flatnonzero(
                (data.subjects == args.subject) & (data.labels == 1)
            )
            indices = indices[
                np.argsort(
                    data.metadata.loc[indices, "start_sec"].to_numpy(), kind="stable"
                )
            ]
            if args.stress_window < 0 or args.stress_window >= len(indices):
                raise ValueError(
                    f"Subject {args.subject} has {len(indices)} Stress windows; choose a valid ordinal."
                )
            explanation = explain_window(
                load_bundle(args.bundle),
                data,
                int(indices[args.stress_window]),
                steps=args.steps,
            )
            write_json(explanation, args.output)
            print(
                f"Saved counterfactual: {args.output}; target reached: {explanation['target_reached']}"
            )
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{args.command} failed: {exc}\n")
