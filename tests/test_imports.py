"""Test basic imports."""


def test_import() -> None:
    """Test that the package can be imported.

    Verifies the package can be imported and has the expected attributes.
    """
    import polarfrost  # noqa: E402

    assert (
        polarfrost.__version__ == "0.2.0"
    )
    assert hasattr(
        polarfrost,
        "mondrian_k_anonymity"
    )
    assert hasattr(
        polarfrost,
        "mondrian_k_anonymity_polars"
    )
    assert hasattr(
        polarfrost,
        "mondrian_k_anonymity_spark"
    )
