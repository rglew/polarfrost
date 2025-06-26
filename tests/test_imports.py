"""Test basic imports."""

def test_import():
    """Test that the package can be imported."""
    import polarfrost
    assert polarfrost.__version__ == "0.1.0"
    assert hasattr(polarfrost, 'mondrian_k_anonymity')
    assert hasattr(polarfrost, 'clustering_k_anonymity')
