from __future__ import annotations

from benchmark.reports.export_paper_tables import (
    _atlasmtl_ablation_accuracy_table,
    _atlasmtl_ablation_resource_table,
    _atlasmtl_ablation_tradeoff_table,
    _protocol_table,
    _runtime_resource_table,
)


def test_protocol_table_exports_input_contract_columns():
    results = [
        {
            "method": "atlasmtl",
            "variant_name": "atlasmtl_cpu_whole_binary_uniform",
            "label_columns": ["anno_lv1", "anno_lv2"],
            "input_contract": {
                "backend": "atlasmtl",
                "label_scope": "multi_level",
                "reference_matrix_source": "preprocessed_h5ad:X",
                "query_matrix_source": "preprocessed_h5ad:X",
                "counts_layer": "counts",
                "normalization_mode": "atlasmtl_core_no_default_lognorm",
                "feature_alignment": "reference_feature_panel_exact_order",
            },
        }
    ]

    table = _protocol_table(results, None)

    assert list(table["method"]) == ["atlasmtl"]
    assert list(table["target_label_column"]) == ["anno_lv2"]
    assert list(table["counts_layer"]) == ["counts"]
    assert list(table["feature_alignment"]) == ["reference_feature_panel_exact_order"]
    assert list(table["variant_name"]) == ["atlasmtl_cpu_whole_binary_uniform"]


def test_runtime_resource_table_exports_extended_resource_columns():
    results = [
        {
            "method": "atlasmtl",
            "variant_name": "atlasmtl_cpu_whole_binary_uniform",
            "input_contract": {"backend": "atlasmtl"},
            "train_usage": {
                "elapsed_seconds": 1.5,
                "items_per_second": 100.0,
                "cpu_percent_avg": 240.0,
                "cpu_core_equiv_avg": 2.4,
                "process_avg_rss_gb": 1.1,
                "process_peak_rss_gb": 1.4,
                "gpu_avg_memory_gb": None,
                "gpu_peak_memory_gb": None,
                "device_used": "cpu",
                "num_threads_used": 4,
            },
            "predict_usage": {
                "elapsed_seconds": 0.5,
                "items_per_second": 200.0,
                "cpu_percent_avg": 120.0,
                "cpu_core_equiv_avg": 1.2,
                "process_avg_rss_gb": 1.0,
                "process_peak_rss_gb": 1.3,
                "gpu_avg_memory_gb": None,
                "gpu_peak_memory_gb": None,
                "device_used": "cpu",
                "num_threads_used": 4,
            },
        }
    ]

    table = _runtime_resource_table(results)

    assert list(table["method"]) == ["atlasmtl"]
    assert list(table["variant_name"]) == ["atlasmtl_cpu_whole_binary_uniform"]
    assert list(table["device_used"]) == ["cpu"]
    assert list(table["num_threads_used"]) == [4]
    assert list(table["train_cpu_core_equiv_avg"]) == [2.4]
    assert list(table["predict_process_avg_rss_gb"]) == [1.0]


def test_atlasmtl_ablation_tables_export_variant_metrics() -> None:
    results = [
        {
            "method": "atlasmtl",
            "variant_name": "atlasmtl_cpu_hvg3000_float_phmap",
            "ablation_config": {
                "device": "cpu",
                "feature_space": "hvg",
                "n_top_genes": 3000,
                "input_transform": "float",
                "task_weight_scheme": "phmap",
            },
            "metrics": {
                "anno_lv1": {"accuracy": 0.9, "macro_f1": 0.9, "balanced_accuracy": 0.9},
                "anno_lv4": {"accuracy": 0.7, "macro_f1": 0.68, "balanced_accuracy": 0.69},
            },
            "hierarchy_metrics": {
                "summary": {"path_consistency_rate": 1.0, "full_path_accuracy": 0.65}
            },
            "train_usage": {"elapsed_seconds": 10.0, "process_peak_rss_gb": 1.5, "cpu_core_equiv_avg": 2.0},
            "predict_usage": {"elapsed_seconds": 1.0, "process_peak_rss_gb": 1.2, "cpu_core_equiv_avg": 1.0},
        }
    ]

    accuracy = _atlasmtl_ablation_accuracy_table(results)
    resources = _atlasmtl_ablation_resource_table(results)
    tradeoff = _atlasmtl_ablation_tradeoff_table(results)

    assert list(accuracy["variant_name"]) == ["atlasmtl_cpu_hvg3000_float_phmap"]
    assert list(accuracy["anno_lv4_accuracy"]) == [0.7]
    assert list(resources["train_process_peak_rss_gb"]) == [1.5]
    assert list(tradeoff["target_level_accuracy"]) == [0.7]
