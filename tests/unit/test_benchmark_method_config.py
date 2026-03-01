from __future__ import annotations

from benchmark.methods.config import resolve_counts_layer, resolve_reference_query_layers


def test_resolve_counts_layer_prefers_method_override():
    manifest = {"counts_layer": "raw_counts"}
    method_cfg = {"counts_layer": "method_counts"}
    assert resolve_counts_layer(manifest, method_cfg) == "method_counts"


def test_resolve_reference_query_layers_fall_back_to_manifest_counts_layer():
    manifest = {"counts_layer": "raw_counts"}
    method_cfg = {}
    assert resolve_reference_query_layers(manifest, method_cfg) == ("raw_counts", "raw_counts")


def test_resolve_reference_query_layers_allow_per_side_override():
    manifest = {"counts_layer": "raw_counts"}
    method_cfg = {"reference_layer": "ref_counts", "query_layer": "query_counts"}
    assert resolve_reference_query_layers(manifest, method_cfg) == ("ref_counts", "query_counts")
