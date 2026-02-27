def test_import_smoke():
    import atlasmtl

    assert hasattr(atlasmtl, "build_model")
    assert hasattr(atlasmtl, "predict")
