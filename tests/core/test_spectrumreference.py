import pytest
import numpy as np
from bioshift.core.spectrumreference import SpectrumReference


@pytest.fixture
def valid_reference_data():
    return {
        "spectrum_shape": np.array([256, 512]),
        "spectral_width": np.array([1000.0, 2000.0]),
        "spectrometer_frequency": np.array([500.0, 600.0]),
        "ref_coord": np.array([128.0, 256.0]),
        "ref_ppm": np.array([4.7, 117.0])
    }


def make_reference():
    return SpectrumReference(
        spectrum_shape=np.array([256, 512]),
        spectral_width=np.array([1024.0, 2048.0]),
        spectrometer_frequency=np.array([500.0, 600.0]),
        ref_coord=np.array([128.0, 256.0]),
        ref_ppm=np.array([4.7, 117.0])
    )


def test_valid_reference_initialization(valid_reference_data):
    ref = SpectrumReference(**valid_reference_data)
    for key, expected in valid_reference_data.items():
        actual = getattr(ref, key)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad_field, bad_value", [
    ("spectrum_shape", np.array([256])),
    ("spectral_width", np.array([1000.0])),
    ("spectrometer_frequency", np.array([500.0, 600.0, 700.0])),
    ("ref_coord", np.array([128.0])),
    ("ref_ppm", np.array([4.7]))
])
def test_shape_mismatch_raises(valid_reference_data, bad_field, bad_value):
    valid_reference_data[bad_field] = bad_value
    with pytest.raises(ValueError):
        SpectrumReference(**valid_reference_data)


def test_non_1d_array_raises(valid_reference_data):
    valid_reference_data["spectrum_shape"] = np.array([[256, 512]])
    with pytest.raises(ValueError):
        SpectrumReference(**valid_reference_data)


def test_zero_frequency_raises(valid_reference_data):
    valid_reference_data["spectrometer_frequency"] = np.array([500.0, 0.0])
    with pytest.raises(ValueError):
        SpectrumReference(**valid_reference_data)


def test_negative_spectral_width_raises(valid_reference_data):
    valid_reference_data["spectral_width"] = np.array([-100.0, 2000.0])
    with pytest.raises(ValueError):
        SpectrumReference(**valid_reference_data)


def test_zero_shape_dimension_raises(valid_reference_data):
    valid_reference_data["spectrum_shape"] = np.array([256, 0])
    with pytest.raises(ValueError):
        SpectrumReference(**valid_reference_data)


def test_transform_matches_expected():
    ref = make_reference()
    transform = ref.transform()

    expected_scaling = ref.spectral_width / \
        (ref.spectrum_shape * ref.spectrometer_frequency)
    expected_offset = ref.ref_ppm - expected_scaling * ref.ref_coord

    np.testing.assert_allclose(transform.scaling, expected_scaling)
    np.testing.assert_allclose(transform.offset, expected_offset)


def test_transform_and_inverse_round_trip():
    ref = make_reference()
    transform = ref.transform()
    inverse = transform.inverse

    point = np.array([10.5, 20.5])
    ppm = transform.apply(point)
    round_trip = inverse.apply(ppm)

    np.testing.assert_allclose(round_trip, point, rtol=1e-7, atol=1e-10)


def test_transform_shape_matches_reference_ndim():
    ref = make_reference()
    transform = ref.transform()
    assert transform.shape == ref.spectrum_shape.shape
