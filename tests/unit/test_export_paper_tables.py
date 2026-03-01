from __future__ import annotations

from benchmark.reports.export_paper_tables import _protocol_table, _runtime_resource_table


def test_protocol_table_exports_input_contract_columns():
    results = [
        {
            "method": "atlasmtl",
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


def test_runtime_resource_table_exports_extended_resource_columns():
    results = [
        {
            "method": "atlasmtl",
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
    assert list(table["device_used"]) == ["cpu"]
    assert list(table["num_threads_used"]) == [4]
    assert list(table["train_cpu_core_equiv_avg"]) == [2.4]
    assert list(table["predict_process_avg_rss_gb"]) == [1.0]
