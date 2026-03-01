from __future__ import annotations

from benchmark.reports.export_paper_tables import _protocol_table


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
