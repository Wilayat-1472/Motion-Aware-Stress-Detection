"""Extract historical metrics from saved notebook outputs without executing cells."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "notebooks/wesad_sqi_multimodal_loso.ipynb").read_text())


def text(cell):
    return "".join(
        "".join(output.get("text", []))
        for output in notebook["cells"][cell].get("outputs", [])
    )


raw, calibrated = text(3).split("--- 2.")
deep = text(4)
pipelines = {}
for name, output in (
    ("Uncalibrated tabular", raw),
    ("Resting-calibrated tabular", calibrated),
    ("SQI-gated multimodal CNN", deep),
):
    mean = re.search(r"Mean Per-Subject Macro-F1:\s+([\d.]+) ± ([\d.]+)", output)
    pipelines[name] = {
        "macro_f1": float(
            re.search(r"Overall 15-Subject LOSO Macro-F1:\s+([\d.]+)", output)[1]
        ),
        "balanced_accuracy": float(
            re.search(r"Overall 15-Subject LOSO Balanced Accuracy:\s+([\d.]+)", output)[
                1
            ]
        ),
        "mean_subject_macro_f1": float(mean[1]),
        "std_subject_macro_f1": float(mean[2]),
    }
per_subject = {
    subject: float(score)
    for subject, score in re.findall(
        r"Unseen Test Subject:\s+(S\d+)\s+\| Macro-F1: ([\d.]+)", deep
    )
}
classes = {}
for name, precision, recall, f1, support in re.findall(
    r"^\s*(Baseline|Stress|Amusement)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
    deep,
    re.M,
):
    classes[name] = dict(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support=int(support),
    )
result = {
    "source": "Saved outputs of the original Colab notebook; not a new evaluation run.",
    "notebook_url": "https://colab.research.google.com/drive/1F5PR5qAale2RXWGzsKVGAan7MENQchvG",
    "windows": 2140,
    "subjects": 15,
    "pipelines": pipelines,
    "per_subject_macro_f1": per_subject,
    "classification_report": classes,
}
path = ROOT / "assets/reference_results.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(
    f"Extracted {len(pipelines)} pipelines, {len(per_subject)} folds, and {len(classes)} class rows."
)
