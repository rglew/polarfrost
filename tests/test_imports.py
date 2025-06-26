"""Test basic imports."""

def test_import():
    """Test that the package can be imported."""
    import frost
    assert frost.__version__ == "0.1.0"
    assert hasattr(frost, 'mondrian_k_anonymity')
    assert hasattr(frost, 'clustering_k_anonymity')
