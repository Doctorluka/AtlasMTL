from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from atlasmtl import build_model, predict
from atlasmtl.core.evaluate import evaluate_hierarchy_metrics, evaluate_prediction_behavior, evaluate_predictions
from atlasmtl.preprocess import load_feature_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--query-h5ad", required=True)
    parser.add_argument("--hierarchy-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-panel-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = ad.read_h5ad(args.reference_h5ad)
    query = ad.read_h5ad(args.query_h5ad)
    hierarchy_rules = json.loads(Path(args.hierarchy_json).read_text(encoding="utf-8"))
    label_columns = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]

    model = build_model(
        adata=ref,
        label_columns=label_columns,
        hidden_sizes=[256, 128],
        dropout_rate=0.2,
        batch_size=256,
        num_epochs=8,
        learning_rate=1e-3,
        input_transform="binary",
        val_fraction=0.1,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        random_state=2026,
        calibration_method=None,
        reference_storage="external",
        reference_path=str(output_dir / "atlasmtl_reference.pkl"),
        device="cpu",
        show_progress=False,
        show_summary=False,
    )
    model_path = output_dir / "atlasmtl_multilevel_model.pth"
    model.save(str(model_path))

    result = predict(
        model,
        query,
        knn_correction="off",
        confidence_high=0.7,
        confidence_low=0.4,
        margin_threshold=0.2,
        input_transform="binary",
        hierarchy_rules=hierarchy_rules,
        enforce_hierarchy=True,
        batch_size=256,
        device="cpu",
        show_progress=False,
        show_summary=False,
    )

    query_with_preds = result.to_adata(query.copy(), mode="full", include_coords=False, include_metadata=True)
    query_out = output_dir / "query_with_predictions.h5ad"
    query_with_preds.write_h5ad(query_out)
    prediction_csv = output_dir / "predictions_full.csv"
    result.to_csv(prediction_csv, mode="full")

    true_df = query.obs.loc[:, label_columns].copy()
    metrics = evaluate_predictions(result.predictions, true_df, label_columns)
    behavior = evaluate_prediction_behavior(result.predictions, true_df, label_columns)
    hierarchy_metrics = evaluate_hierarchy_metrics(
        result.predictions,
        true_df,
        label_columns,
        hierarchy_rules=hierarchy_rules,
    )

    payload = {
        "label_columns": label_columns,
        "feature_panel_path": str(Path(args.feature_panel_json).resolve()) if args.feature_panel_json else None,
        "feature_panel": load_feature_panel(args.feature_panel_json).to_dict() if args.feature_panel_json else None,
        "metrics": metrics,
        "behavior_metrics": behavior,
        "hierarchy_metrics": hierarchy_metrics,
        "train_usage": model.get_resource_usage(),
        "predict_usage": result.get_resource_usage(),
        "prediction_metadata": result.metadata,
        "artifacts": {
            "model_path": str(model_path),
            "annotated_query_h5ad": str(query_out),
            "predictions_csv": str(prediction_csv),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(metrics).T.to_csv(output_dir / "per_level_metrics.csv", index=True)
    pd.DataFrame(behavior).T.to_csv(output_dir / "behavior_metrics.csv", index=True)
    pd.DataFrame(hierarchy_metrics).T.to_csv(output_dir / "hierarchy_metrics.csv", index=True)
    print(json.dumps(payload["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
