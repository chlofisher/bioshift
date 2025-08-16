---
title: API
---

<a id="bioshift.core.spectrum"></a>

# bioshift.core.spectrum

<a id="bioshift.core.spectrum.Spectrum"></a>

## Spectrum Objects

```python
class Spectrum()
```

NMR spectrum object

**Attributes**:

- `data_source` - Object responsible for lazy-loading and parsing spectrum data.
- `nuclei` - The type of nucleus (13C, 1H, etc.) associated with each axis.
- `transform` - Object storing the transformation from array coordinate space to chemical shift space.
  
  Properties:
- `data` - N-dimensional numpy array containing the raw spectrum.
- `ndim` - Number of dimensions of the spectrum.
- `shape` - Number of data points along each axis of the spectrum.

<a id="bioshift.core.spectrum.Spectrum.add"></a>

#### add

```python
def add(other: Self) -> Self
```

Return a new spectrum equal to the pointwise sum of two spectra.

**Arguments**:

- `other` - The spectrum to add.

**Returns**:

- `Spectrum` - A new spectrum whose values are the sum of those of the two previous spectra.

**Raises**:

- `ValueError` - If the shapes of the two spectra do not match

<a id="bioshift.core.spectrum.Spectrum.subtract"></a>

#### subtract

```python
def subtract(other: Self) -> Self
```

Return a new spectrum equal to the pointwise difference of two spectra.

**Arguments**:

- `other` - The spectrum to subtract.

**Returns**:

- `Spectrum` - A new spectrum whose values are the difference of those of the two previous spectra.

**Raises**:

- `ValueError` - If the shapes of the two spectra do not match

<a id="bioshift.core.spectrum.Spectrum.__neg__"></a>

#### \_\_neg\_\_

```python
def __neg__() -> Self
```

Implements the `-` operator.

**Returns**:

- `Spectrum` - A new spectrum with negated values.

<a id="bioshift.core.spectrum.Spectrum.multiply"></a>

#### multiply

```python
def multiply(other) -> Self
```

Return a new spectrum equal to the pointwise product of two spectra.

**Arguments**:

- `other` - The spectrum to multiply by.

**Returns**:

- `Spectrum` - A new spectrum whose values are the product of those of the two previous spectra.

**Raises**:

- `ValueError` - If the shapes of the two spectra do not match

<a id="bioshift.core.spectrum.Spectrum.shift_to_coord"></a>

#### shift\_to\_coord

```python
def shift_to_coord(shift: NDArray) -> NDArray
```

Convert between chemical shift and grid coordinate systems. Index coordinates are interpolated between integer values.

**Arguments**:

- `shift` - ND array of chemical shifts.

**Returns**:

- `NDArray` - ND array of grid coordinates.

<a id="bioshift.core.spectrum.Spectrum.coord_to_shift"></a>

#### coord\_to\_shift

```python
def coord_to_shift(coord: NDArray) -> NDArray
```

Convert between grid and chemical shift coordinate systems.

**Arguments**:

- `shift` - ND array of grid coordinates.

**Returns**:

- `NDArray` - ND array of chemical shifts.

