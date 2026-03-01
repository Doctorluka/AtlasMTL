import numpy as np
from atlasmtl.models.reference_store import ReferenceData, load_reference_data, save_reference_data


def test_reference_store_supports_gzip(tmp_path):
    ref = ReferenceData(
        coords={"X_ref_latent": np.array([[0.0, 0.0]], dtype=np.float32)},
        labels={"celltype": np.array(["A"], dtype=object)},
    )
    path = tmp_path / "ref.pkl.gz"
    save_reference_data(ref, str(path))
    loaded = load_reference_data(str(path))
    assert np.allclose(loaded.coords["X_ref_latent"], ref.coords["X_ref_latent"])
    assert loaded.labels["celltype"][0] == "A"

