---
title: API
---

<a id="bioshift"></a>

# bioshift

<a id="bioshift.fileio.ucsf"></a>

# bioshift.fileio.ucsf

<a id="bioshift.fileio.ucsf.UCSFSpectrumReader"></a>

## UCSFSpectrumReader Objects

```python
class UCSFSpectrumReader(SpectrumReader)
```

<a id="bioshift.fileio.ucsf.UCSFSpectrumReader.get_params"></a>

#### get\_params

```python
def get_params() -> dict[str, Any]
```

Read the spectrum metadata from the .ucsf file header.

**Returns**:

  Dictionary of params. Must always have keys 'ndim', 'header_size',
  'shape', 'block_shape', 'nuclei', 'ref_ppm', 'ref_coord',
  'spectrometer_frequency', 'spectral_width'.
  May also have keys 'integer', 'swap', 'endianness'.

<a id="bioshift.fileio.loadspectrum"></a>

# bioshift.fileio.loadspectrum

<a id="bioshift.fileio.blockedspectrum"></a>

# bioshift.fileio.blockedspectrum

<a id="bioshift.fileio.azara"></a>

# bioshift.fileio.azara

<a id="bioshift.fileio.azara.AzaraSpectrumReader"></a>

## AzaraSpectrumReader Objects

```python
class AzaraSpectrumReader(SpectrumReader)
```

<a id="bioshift.fileio.azara.AzaraSpectrumReader.spc_from_par"></a>

#### spc\_from\_par

```python
@classmethod
def spc_from_par(cls, par_path: Path) -> Path
```

Finds a corresponding .spc file from a .par file. First checks for
a .spc file specified in the .par file, failing that checks for a .spc
with a matching name.

**Arguments**:

- `par_path` - Path to the .par file.
  

**Returns**:

  Path to a .spc file.
  

**Raises**:

- `FileNotFoundError` - if a .spc file can not be found

<a id="bioshift.fileio.azara.AzaraSpectrumReader.par_from_spc"></a>

#### par\_from\_spc

```python
@classmethod
def par_from_spc(cls, spc_path: Path) -> Path
```

Finds a corresponding .par file from a .spc file. Given
spectrum.spc, checks spectrum.spc.par and spectrum.par.

**Arguments**:

- `spc_path` - Path to the .spc file.
  

**Returns**:

  Path to a .par file.
  

**Raises**:

- `FileNotFoundError` - if a .par file can not be found

<a id="bioshift.fileio.azara.AzaraSpectrumReader.get_params"></a>

#### get\_params

```python
def get_params() -> dict[str, Any]
```

Get a dictionary of parameters from the .par file for use in
constructing the spectrum.

**Returns**:

  Dictionary of params. Must always have keys 'ndim', 'header_size',
  'shape', 'block_shape', 'nuclei', 'ref_ppm', 'ref_coord',
  'spectrometer_frequency', 'spectral_width'.
  May also have keys 'integer', 'swap', 'endianness'.
  

**Raises**:

- `ValueError` - if unsupported keys 'varian', 'blocks', 'sigmas',
  'params' are detected. .par files containing these keys are
  intended for internal use within Azara only.

<a id="bioshift.fileio"></a>

# bioshift.fileio

<a id="bioshift.fileio.nmrpipe"></a>

# bioshift.fileio.nmrpipe

<a id="bioshift.fileio.spectrumreader"></a>

# bioshift.fileio.spectrumreader

<a id="bioshift.fileio.spectrumreader.SpectrumReader"></a>

## SpectrumReader Objects

```python
class SpectrumReader(ABC)
```

Base class for spectrum readers. Specifies an interface for producing a
SpectrumParams and a SpectrumDataSource from different file formats
implemented by concrete SpectrumReaders.

<a id="bioshift.fileio.spectrumreader.SpectrumReader.read"></a>

#### read

```python
def read() -> Spectrum
```

Creates a new spectrum from a SpectrumParams and SpectrumDataSource
constructed by reading from self.path.

**Returns**:

  New spectrum object.

<a id="bioshift.fileio.spectrumreader.SpectrumReader.can_read"></a>

#### can\_read

```python
@classmethod
@abstractmethod
def can_read(cls, path: Path) -> bool
```

Specifies whether or not a given path can be read by a particular
concrete SpectrumReader implementation. Used to dynamically determine
which concrete SpectrumReader to use to read from a given path.

**Returns**:

  True if the given path can be read by the concrete SpectrumReader
  class.

<a id="bioshift.analysis.gaussianpeakfitting"></a>

# bioshift.analysis.gaussianpeakfitting

<a id="bioshift.analysis.node"></a>

# bioshift.analysis.node

<a id="bioshift.analysis.filters"></a>

# bioshift.analysis.filters

<a id="bioshift.analysis.peakpicker"></a>

# bioshift.analysis.peakpicker

<a id="bioshift.analysis.pipeline"></a>

# bioshift.analysis.pipeline

<a id="bioshift.analysis.nodes.math"></a>

# bioshift.analysis.nodes.math

<a id="bioshift.analysis.noderegistry"></a>

# bioshift.analysis.noderegistry

<a id="bioshift.core.mock"></a>

# bioshift.core.mock

<a id="bioshift.core.spectrumtransform"></a>

# bioshift.core.spectrumtransform

<a id="bioshift.core.spectrumtransform.SpectrumTransform"></a>

## SpectrumTransform Objects

```python
class SpectrumTransform()
```

Represents a diagonal affine transformation used to map between
coordinates in the raw spectrum array and chemical shift values.

**Attributes**:

- `scaling` - Vector of scaling factors for each axis.
- `offset` - Offset vector.
  
  Properties:
- `inverse` - The inverse of the transform.

<a id="bioshift.core.spectrumtransform.SpectrumTransform.__init__"></a>

#### \_\_init\_\_

```python
def __init__(ndim, shape, scaling, offset)
```

Initialise the SpectrumTransform. Performs validation logic and
copies the input arrays to ensure immutability.

**Raises**:

- `ValueError` - If any of the scaling values are zero,
  to enforce that the transformation matrix is non-singular.
- `ValueError` - If the shape of the scaling and the offset vectors do
  not match.

<a id="bioshift.core.nucleus"></a>

# bioshift.core.nucleus

<a id="bioshift.core.peak"></a>

# bioshift.core.peak

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

<a id="bioshift.core.spectrumdatasource"></a>

# bioshift.core.spectrumdatasource

<a id="bioshift.core"></a>

# bioshift.core

