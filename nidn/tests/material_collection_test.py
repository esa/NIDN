from ..materials.material_collection import MaterialCollection


def test_material_collection_init():
    """Tests if the material collection can be initialized successfully."""
    target_frequencies = [9.5, 1.0, 0.1, 0.01]
    mc = MaterialCollection(target_frequencies)
    assert len(mc.material_names) > 0
    assert mc.target_frequencies == target_frequencies
    assert mc.epsilon_matrix is not None
    assert mc.epsilon_matrix.shape[0] == len(mc.material_names)
    assert mc.epsilon_matrix.shape[1] == len(target_frequencies)


if __name__ == "__main__":
    test_material_collection_init()
