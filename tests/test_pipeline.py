"""Research-pipeline parity, split integrity, and inference contract checks."""

import ast
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from tests.helpers import fixture_dataset, recording
from stress_detection.calibration import calibrate_windows, fit_feature_scaler
from stress_detection.config import TrainingConfig
from stress_detection.dataset import WindowDataset, load_windows
from stress_detection.evaluation import (
    run_loso,
    run_tabular_benchmarks,
    save_evaluation,
)
from stress_detection.explainability import explain_window
from stress_detection.inference import (
    DEFAULT_BUNDLE_PATH,
    load_bundle,
    predict_dataset,
    prediction_scores,
)
from stress_detection.preprocessing import extract_recording
from stress_detection.training import save_bundle, train_reference
from stress_detection.visualization import plot_diagnostics

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/wesad_sqi_multimodal_loso.ipynb").read_text())


def notebook_cell(index):
    return ast.parse("".join(NOTEBOOK["cells"][index]["source"]))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.dataset = fixture_dataset()
        cls.bundle = load_bundle()

    def test_extraction_matches_original_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S2"
            path.mkdir()
            sample = recording()
            with (path / "S2.pkl").open("wb") as handle:
                pickle.dump(sample, handle)
            tree = notebook_cell(2)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "DATA_DIR" for t in node.targets
                ):
                    node.value = ast.Constant(directory)
            # Suppress original /content output writes; execute the original extraction unchanged.
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr in ("savez_compressed", "to_csv")
                )
            ]
            context = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    compile(
                        ast.fix_missing_locations(tree), "notebook-extraction", "exec"
                    ),
                    context,
                )
            result = extract_recording(sample, "S2")
            np.testing.assert_array_equal(result.sequences, context["X_seq"])
            np.testing.assert_array_equal(result.features, context["X_feat"])
            np.testing.assert_array_equal(
                result.labels, context["meta_df"]["label"].to_numpy()
            )

    def test_calibration_matches_original_notebook(self):
        data = self.dataset
        context = {
            "np": np,
            "X_seq_raw": data.sequences,
            "X_feat": data.features,
            "subjects": data.subjects,
            "y": data.labels,
            "unique_subjects": data.subject_ids,
        }
        nodes = []
        for node in notebook_cell(4).body:
            if isinstance(node, ast.ClassDef):
                break
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("X_seq_cal", "X_feat_cal")
                for t in node.targets
            ):
                nodes.append(node)
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "subj"
            ):
                nodes.append(node)
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), "notebook-calibration", "exec"
            ),
            context,
        )
        result = calibrate_windows(data)
        np.testing.assert_array_equal(result.sequences, context["X_seq_cal"])
        np.testing.assert_array_equal(result.features, context["X_feat_cal"])
        self.assertTrue((np.abs(result.features) <= 10).all())

    def test_calibration_uses_chronology_not_csv_order(self):
        permutation = np.random.default_rng(5).permutation(len(self.dataset.labels))
        shuffled = WindowDataset(
            self.dataset.sequences[permutation],
            self.dataset.features[permutation],
            self.dataset.metadata.iloc[permutation],
        )
        expected = calibrate_windows(self.dataset)
        actual = calibrate_windows(shuffled)
        np.testing.assert_array_equal(actual.features, expected.features[permutation])
        np.testing.assert_array_equal(actual.sequences, expected.sequences[permutation])

    def test_training_scaler_excludes_held_out_subject(self):
        data = self.dataset
        calibrated = calibrate_windows(data)
        mask = data.subjects != "S11"
        mean, std = fit_feature_scaler(calibrated.features, mask)
        modified = calibrated.features.copy()
        modified[~mask] = 100000
        other_mean, other_std = fit_feature_scaler(modified, mask)
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(std, other_std)

    def test_recorded_inference_matches_notebook_forward(self):
        data = self.dataset
        calibrated = calibrate_windows(data)
        with torch.no_grad():
            expected = torch.softmax(
                self.bundle.model(
                    torch.tensor(calibrated.sequences),
                    torch.tensor(
                        (calibrated.features - self.bundle.feature_mean)
                        / self.bundle.feature_std
                    ),
                ),
                1,
            ).numpy()
        frame = predict_dataset(self.bundle, data)
        np.testing.assert_allclose(prediction_scores(frame), expected, atol=1e-6)
        self.assertTrue(np.allclose(prediction_scores(frame).sum(1), 1))

    def test_export_load_and_counterfactual_locks_motion(self):
        bundle = train_reference(
            self.dataset, TrainingConfig(epochs=1, device="cpu"), holdout="S11"
        )
        self.assertNotIn("S11", bundle.provenance["training_subjects"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            save_bundle(bundle, path)
            restored = load_bundle(path)
            np.testing.assert_allclose(
                prediction_scores(predict_dataset(bundle, self.dataset)),
                prediction_scores(predict_dataset(restored, self.dataset)),
                atol=1e-6,
            )
        index = np.flatnonzero(
            (self.dataset.subjects == "S11") & (self.dataset.labels == 1)
        )[0]
        gradients = [
            None if parameter.grad is None else parameter.grad.clone()
            for parameter in bundle.model.parameters()
        ]
        explanation = explain_window(bundle, self.dataset, int(index), steps=3)
        self.assertTrue(explanation["sequence_fixed"])
        np.testing.assert_array_equal(
            list(explanation["feature_deltas"].values())[12:], np.zeros(6)
        )
        for parameter, previous in zip(bundle.model.parameters(), gradients):
            if previous is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, previous, rtol=0, atol=0)

    def test_small_loso_matches_original_notebook_training(self):
        data = self.dataset
        with tempfile.TemporaryDirectory() as directory:
            data.save(directory)
            tree = notebook_cell(4)

            class AdaptPathsAndEpochs(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, str) and node.value.startswith(
                        "/content/"
                    ):
                        return ast.copy_location(
                            ast.Constant(str(Path(directory) / Path(node.value).name)),
                            node,
                        )
                    return node

                def visit_Call(self, node):
                    self.generic_visit(node)
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "range"
                        and len(node.args) == 1
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == 25
                    ):
                        node.args[0] = ast.Constant(1)
                    return node

            tree = ast.fix_missing_locations(AdaptPathsAndEpochs().visit(tree))
            context = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tree, "notebook-one-epoch-loso", "exec"), context)
                scores, report = run_loso(data, TrainingConfig(epochs=1, device="cpu"))
            np.testing.assert_allclose(
                scores, context["oof_probs"], atol=1e-6, rtol=1e-5
            )
            self.assertEqual(report["subjects"], 3)
            for fold in report["folds"]:
                self.assertNotIn(fold["test_subject"], fold["training_subjects"])
            save_evaluation(data, scores, report, directory)
            with np.load(
                Path(directory) / "wesad_loso_results.npz", allow_pickle=False
            ) as arrays:
                np.testing.assert_array_equal(arrays["subjects"], data.subjects)
            plot = Path(directory) / "diagnostics.png"
            plot_diagnostics(
                Path(directory) / "wesad_loso_results.npz",
                Path(directory) / "wesad_meta.csv",
                plot,
            )
            self.assertGreater(plot.stat().st_size, 1000)

    def test_tabular_benchmarks_run_both_variants(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_tabular_benchmarks(self.dataset)
        self.assertEqual(set(result), {"uncalibrated", "resting_calibrated"})
        self.assertEqual(result["resting_calibrated"]["subjects"], 3)

    def test_validation_rejects_bad_data_and_missing_baseline(self):
        data = self.dataset
        with self.assertRaises(ValueError):
            WindowDataset(data.sequences[:, :, :-1], data.features, data.metadata)
        metadata = data.metadata.copy()
        metadata.loc[0, "label"] = 8
        with self.assertRaises(ValueError):
            WindowDataset(data.sequences, data.features, metadata)
        metadata = data.metadata.copy()
        metadata.loc[metadata["subject"] == "S11", "label"] = 1
        invalid = WindowDataset(data.sequences, data.features, metadata)
        with self.assertRaisesRegex(ValueError, "S11 needs at least 3 Baseline"):
            calibrate_windows(invalid)
        with self.assertRaises(ValueError):
            extract_recording(
                {
                    "signal": {
                        "wrist": {
                            "BVP": [0],
                            "EDA": [0],
                            "TEMP": [0],
                            "ACC": [[0, 0, 0]],
                        }
                    },
                    "label": [1],
                },
                "short",
            )

    def test_artifact_contract_and_window_roundtrip(self):
        with patch("stress_detection.inference.joblib.load", return_value={}):
            with self.assertRaises(ValueError):
                load_bundle()
        data = joblib.load(DEFAULT_BUNDLE_PATH)
        data["f_sd"] = np.zeros(18)
        with patch("stress_detection.inference.joblib.load", return_value=data):
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                load_bundle()
        with tempfile.TemporaryDirectory() as directory:
            self.dataset.save(directory)
            result = load_windows(
                Path(directory) / "wesad_windows.npz",
                Path(directory) / "wesad_meta.csv",
            )
            np.testing.assert_array_equal(result.features, self.dataset.features)
            with self.assertRaises(FileNotFoundError):
                load_bundle(Path(directory) / "missing.joblib")


if __name__ == "__main__":
    unittest.main()
