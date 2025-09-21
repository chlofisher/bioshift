import pytest
import numpy as np
from bioshift.core.spectrumtransform import SpectrumTransform


def random_transform_data(n=10, seed=123):
    output = []

    for i in range(n):
        ndim = np.random.randint(1, 5)
        transform_data = (np.random.rand(ndim), np.random.rand(ndim))
        if all(x != 0 for x in transform_data[0]):
            output.append(transform_data)

    return output


def random_transform_data_singular(n=10, seed=123):
    output = []

    for i in range(n):
        ndim = np.random.randint(1, 5)
        scaling = np.random.rand(ndim)

        num_zeros = np.random.randint(1, ndim + 1)
        zero_indices = np.random.choice(ndim, size=num_zeros, replace=False)

        scaling[zero_indices] = 0

        transform_data = (scaling, np.random.rand(ndim))

        output.append(transform_data)

    return output


def random_transform_data_mismatched(n=10, seed=123):
    output = []

    for i in range(n):
        ndim1 = 0
        ndim2 = 0

        while ndim1 == ndim2:
            ndim1 = np.random.randint(1, 5)
            ndim2 = np.random.randint(1, 5)

        transform_data = (np.random.rand(ndim1), np.random.rand(ndim2))
        if all(x != 0 for x in transform_data[0]):
            output.append(transform_data)

    return output


def random_points(n=10, seed=123):
    return [np.random.rand(np.random.randint(1, 5)) for i in range(n)]


@pytest.mark.parametrize("transform_data", random_transform_data())
def test_create_transform_valid(transform_data):
    scaling, offset = transform_data
    transform = SpectrumTransform(scaling, offset)

    assert all(transform.scaling == scaling)
    assert all(transform.offset == offset)


@pytest.mark.parametrize("transform_data", random_transform_data_singular())
def test_create_transform_singular(transform_data):
    scaling, offset = transform_data
    with pytest.raises(ValueError):
        SpectrumTransform(scaling, offset)


@pytest.mark.parametrize(
    "mismatched_transform_data", random_transform_data_mismatched()
)
def test_create_transform_mismatched_shapes(mismatched_transform_data):
    scaling, offset = mismatched_transform_data
    with pytest.raises(ValueError):
        SpectrumTransform(scaling, offset)


@pytest.mark.parametrize("transform_data", random_transform_data())
def test_inverse(transform_data):
    scaling, offset = transform_data
    transform = SpectrumTransform(scaling, offset)

    assert transform == transform.inverse.inverse


@pytest.mark.parametrize("transform_data", random_transform_data())
@pytest.mark.parametrize("point", random_points())
def test_apply(transform_data, point):
    scaling, offset = transform_data
    transform = SpectrumTransform(scaling, offset)

    if point.shape == transform.shape:
        assert np.allclose(transform.apply(point), point * scaling + offset)
    else:
        with pytest.raises(ValueError):
            transform.apply(point)


@pytest.mark.parametrize("transform_data", random_transform_data())
@pytest.mark.parametrize("point", random_points())
def test_apply_inverse(transform_data, point):
    scaling, offset = transform_data
    transform = SpectrumTransform(scaling, offset)

    if point.shape == transform.inverse.shape:
        assert np.allclose(transform.inverse.apply(point), (point - offset) / scaling)
    else:
        with pytest.raises(ValueError):
            transform.inverse.apply(point)
