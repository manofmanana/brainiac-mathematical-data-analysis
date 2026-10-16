"""Smoke tests for the Streamlit app using Streamlit's built-in AppTest runner."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

MAIN = str(Path(__file__).resolve().parents[1] / "main.py")


def run_app(**sidebar):
    at = st_testing.AppTest.from_file(MAIN, default_timeout=120)
    at.run()
    for label, value in sidebar.items():
        widget = next(w for w in at.sidebar.selectbox if w.label == label)
        widget.set_value(value)
    if sidebar:
        at.run()
    return at


def test_default_app_renders_every_tab_without_errors():
    at = run_app()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    labels = [m.label for m in at.metric]
    assert "Test accuracy" in labels
    assert "Relative error" in labels
    verdict = next(m for m in at.metric if m.label == "Verdict")
    assert verdict.value == "✅ Correct"


@pytest.mark.parametrize(
    "dataset", ["Iris (4 features, 3 species)", "Noisy sine wave (regression)", "Spirals (3 classes)"]
)
def test_each_dataset_renders(dataset):
    at = run_app(Dataset=dataset)
    assert not at.exception, at.exception
    assert not at.error


def test_experiment_form_runs():
    at = run_app()
    at.multiselect[0].set_value(["relu", "tanh"])
    at.multiselect[1].set_value(["he"])
    next(b for b in at.button if "Run experiment" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 2  # gradcheck table + experiment results


def test_extreme_settings_do_not_crash():
    # SGD at lr=1 with poor init may diverge; the app must report it, not raise.
    at = st_testing.AppTest.from_file(MAIN, default_timeout=120)
    at.run()
    next(w for w in at.sidebar.selectbox if w.label == "Optimiser").set_value("sgd")
    next(w for w in at.sidebar.select_slider if w.label == "Learning rate").set_value(1.0)
    next(w for w in at.sidebar.selectbox if w.label == "Activation").set_value("relu")
    next(w for w in at.sidebar.selectbox if w.label == "Initialisation").set_value("normal")
    at.run()
    assert not at.exception, at.exception
