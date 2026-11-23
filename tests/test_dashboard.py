"""Check research navigation and real-window rendering without a browser."""

import unittest

import torch
from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_research_overview_and_upload_navigation(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 3)
        app.sidebar.radio[0].set_value("Recording inference").run()
        self.assertFalse(app.exception)
        self.assertIn(
            "Inference on recorded windows", [item.value for item in app.header]
        )
        self.assertEqual(len(app.get("file_uploader")), 2)

    def test_recording_predictions_and_explanation(self):
        app = AppTest.from_string(
            """
from tests.helpers import fixture_dataset
from stress_detection.dashboard import render_predictions
from stress_detection.inference import load_bundle
render_predictions(fixture_dataset(), load_bundle())
""",
            default_timeout=60,
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertIn("predicted_state", app.dataframe[0].value.columns)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.get("download_button")), 2)
