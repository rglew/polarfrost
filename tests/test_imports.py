"""Test basic imports."""

def test_import():
    """Test that the package can be imported and has the expected attributes."""
    import polarfrost
    assert polarfrost.__version__ == "0.1.0"
    assert hasattr(polarfrost, 'mondrian_k_anonymity')
    assert hasattr(polarfrost, 'mondrian_k_anonymity_polars')
    assert hasattr(polarfrost, 'mondrian_k_anonymity_spark')
