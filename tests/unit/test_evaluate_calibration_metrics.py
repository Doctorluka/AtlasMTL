import numpy as np
import pandas as pd

from atlasmtl.core.evaluate import evaluate_predictions


def test_evaluate_adds_ece_brier_aurc_when_conf_present():
    true_df = pd.DataFrame({"lvl": ["A", "A", "B", "B"]})
    pred_df = pd.DataFrame(
        {
            "pred_lvl": ["A", "Unknown", "B", "A"],
            "conf_lvl": [0.9, 0.2, 0.8, 0.6],
        }
    )
    out = evaluate_predictions(pred_df, true_df, ["lvl"], n_bins=5)["lvl"]
    assert "ece" in out
    assert "brier" in out
    assert "aurc" in out
    assert 0.0 <= out["ece"] <= 1.0
    assert 0.0 <= out["brier"] <= 1.0
    assert 0.0 <= out["aurc"] <= 1.0


def test_evaluate_skips_calibration_metrics_when_conf_missing():
    true_df = pd.DataFrame({"lvl": ["A", "B"]})
    pred_df = pd.DataFrame({"pred_lvl": ["A", "B"]})
    out = evaluate_predictions(pred_df, true_df, ["lvl"])["lvl"]
    assert "ece" not in out
    assert "brier" not in out
    assert "aurc" not in out

