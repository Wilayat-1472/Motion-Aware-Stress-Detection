"""Research overview and calibrated inference on actual WESAD window exports."""

import io
import json

import pandas as pd
import streamlit as st

from .config import CLASSES, NOTEBOOK_URL
from .dataset import load_windows
from .explainability import explain_window
from .inference import (
    DIAGNOSTICS_PATH,
    REFERENCE_RESULTS_PATH,
    load_bundle,
    predict_dataset,
)


@st.cache_resource
def cached_bundle():
    return load_bundle()


def research_overview():
    st.header("15-subject LOSO research results")
    reference = json.loads(REFERENCE_RESULTS_PATH.read_text())
    st.caption(reference["source"])
    final = reference["pipelines"]["SQI-gated multimodal CNN"]
    cols = st.columns(3)
    cols[0].metric("Pooled Macro-F1", f"{final['macro_f1']:.4f}")
    cols[1].metric("Balanced accuracy", f"{final['balanced_accuracy']:.4f}")
    cols[2].metric("Mean subject Macro-F1", f"{final['mean_subject_macro_f1']:.4f}")
    st.write(
        "2,140 wrist-only windows from 15 WESAD participants; 30-second windows with a 15-second stride."
    )
    st.subheader("Pipeline comparison")
    table = pd.DataFrame(reference["pipelines"]).T
    table.columns = [
        "Pooled Macro-F1",
        "Balanced accuracy",
        "Mean subject Macro-F1",
        "Subject F1 SD",
    ]
    st.dataframe(table, width="stretch")
    st.subheader("Held-out participant scores")
    st.bar_chart(pd.Series(reference["per_subject_macro_f1"], name="Macro-F1"))
    st.subheader("Per-class performance")
    st.dataframe(pd.DataFrame(reference["classification_report"]).T, width="stretch")
    if DIAGNOSTICS_PATH.is_file():
        st.image(str(DIAGNOSTICS_PATH), width="stretch")
    st.markdown(f"[Open the original research notebook]({NOTEBOOK_URL})")
    with st.expander("Method and evaluation context"):
        st.write(
            "The notebook filters BVP and EDA, extracts six signal channels and 18 features, "
            "calibrates each subject against their first three Baseline windows, and fits the "
            "feature scaler on training subjects inside each LOSO fold. Calibration windows "
            "remain in evaluation. These are offline research results; the local inference "
            "model is the separately trained S11-held-out reference model."
        )


def render_predictions(dataset, bundle):
    """Render the recording path separately so it can be tested without file uploads."""
    result = predict_dataset(bundle, dataset)
    subject = st.selectbox("Participant", dataset.subject_ids)
    mask = dataset.subjects == subject
    frame = result[mask].sort_values("start_sec")
    held_out = bundle.provenance.get("held_out_subject", "S11")
    if held_out is None:
        st.caption(
            "This bundle was fitted on all subjects listed in its training provenance."
        )
    elif subject != held_out:
        st.warning(
            f"The reference model held out {held_out}. This participant may have been used for training; these scores are not a LOSO evaluation."
        )
    else:
        st.caption(
            f"{subject} was held out when fitting the reference model described by this bundle."
        )
    st.subheader("Recorded window predictions")
    columns = [f"score_{name.lower()}" for name in CLASSES]
    st.line_chart(frame.set_index("start_sec")[columns])
    st.caption(
        "X-axis: window start time in seconds. Scores are softmax outputs, not calibrated confidence estimates."
    )
    st.line_chart(frame.set_index("start_sec")[["sqi"]])
    st.dataframe(frame, width="stretch")
    st.download_button(
        "Download predictions",
        frame.to_csv(index=False),
        "wesad_predictions.csv",
        "text/csv",
    )
    index = st.selectbox(
        "Inspect a recorded window",
        frame.index.tolist(),
        format_func=lambda i: f"{result.loc[i, 'start_sec']:g}s · {result.loc[i, 'state']}",
    )
    row = result.loc[index]
    st.info(
        f"Protocol label: **{row['state']}** · Predicted: **{row['predicted_state']}**"
    )
    st.line_chart(
        pd.DataFrame(
            dataset.sequences[index].T,
            columns=["BVP", "EDA", "TEMP", "ACC x", "ACC y", "ACC z"],
        )
    )
    if st.button("Explain this window toward Baseline"):
        with st.spinner(
            "Optimizing physiological features with sequence and motion fixed..."
        ):
            explanation = explain_window(bundle, dataset, int(index))
        delta = pd.Series(
            {
                name: explanation["feature_deltas"][name]
                for name in explanation["top_features"]
            }
        )
        st.bar_chart(delta)
        st.write(
            {
                "original_scores": explanation["original_scores"],
                "counterfactual_scores": explanation["counterfactual_scores"],
                "target_reached": explanation["target_reached"],
            }
        )
        st.caption(
            "Changes use fold-standardized feature units. They describe model behavior, not physiological advice."
        )
        st.download_button(
            "Download explanation",
            json.dumps(explanation, indent=2),
            "counterfactual.json",
            "application/json",
        )


def recording_inference():
    st.header("Inference on recorded windows")
    st.write(
        "Upload the matching wesad_windows.npz and wesad_meta.csv files generated by the notebook or the prepare command."
    )
    st.info(
        "Use raw extracted windows. The app applies resting calibration and clipping before the saved model scaler. Each participant needs at least three labeled Baseline windows."
    )
    windows = st.file_uploader("Window arrays (X_seq and X_feat)", type=["npz"])
    metadata = st.file_uploader("Matching window metadata", type=["csv"])
    if windows is None or metadata is None:
        st.code("python -m stress_detection prepare --data-dir /path/to/WESAD")
        return
    try:
        dataset = load_windows(
            io.BytesIO(windows.getvalue()), io.BytesIO(metadata.getvalue())
        )
        render_predictions(dataset, cached_bundle())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        st.error(f"Unable to process these recordings: {exc}")


def main():
    st.set_page_config(
        page_title="Motion-Aware WESAD Research", page_icon="🧠", layout="wide"
    )
    st.title("🧠 Motion-Aware Stress & Affect Detection")
    st.caption(
        "Wrist-only physiology · Resting baseline calibration · SQI-gated multimodal learning"
    )
    page = st.sidebar.radio("Workspace", ["Research overview", "Recording inference"])
    st.sidebar.markdown(f"[Research notebook]({NOTEBOOK_URL})")
    if page == "Research overview":
        research_overview()
    else:
        recording_inference()
