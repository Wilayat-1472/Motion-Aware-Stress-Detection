"""Notebook constants and training defaults."""

from dataclasses import dataclass

CLASSES = ("Baseline", "Stress", "Amusement")
LABEL_MAP = {1: 0, 2: 1, 3: 2}
WINDOW_SECONDS = 30
STRIDE_SECONDS = 15
BASELINE_WINDOWS = 3
FEATURE_NAMES = (
    "bpm",
    "sdnn",
    "rmssd",
    "bvp_std",
    "eda_mean",
    "eda_std",
    "eda_slope",
    "scr_rise_count",
    "eda_max_diff",
    "temp_mean",
    "temp_std",
    "temp_slope",
    "acc_mean",
    "acc_std",
    "acc_range",
    "sqi",
    "bpm_times_sqi",
    "rmssd_times_sqi",
)
NOTEBOOK_URL = (
    "https://colab.research.google.com/drive/1F5PR5qAale2RXWGzsKVGAan7MENQchvG"
)
PREPROCESSING_VERSION = "wesad-notebook-v1"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 64
    evaluation_batch_size: int = 128
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    seed: int = 42
    device: str = "auto"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.evaluation_batch_size) < 1:
            raise ValueError("Epochs and batch sizes must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("Label smoothing must be in [0, 1).")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda.")
