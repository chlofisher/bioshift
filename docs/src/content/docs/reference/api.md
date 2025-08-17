---
title: bioshift
---



---
## Spectrum
```python
class Spectrum:
```
NMR spectrum object


###### Attributes:
 - **data_source:**  Object responsible for lazy-loading and parsing spectrum data.
 - **nuclei:**  The type of nucleus (13C, 1H, etc.) associated with each axis.
 - **transform:**  Object storing the transformation from array coordinate space to chemical shift space.



###### Properties:
> data: N-dimensional numpy array containing the raw spectrum.
> ndim: Number of dimensions of the spectrum.
> shape: Number of data points along each axis of the spectrum.

#### name
```python
name: str
```

#### ndim
```python
ndim: int
```

#### nuclei
```python
nuclei: tuple[bioshift.core.nucleus.NMRNucleus, ...]
```

#### data_source
```python
data_source: bioshift.core.spectrumdatasource.SpectrumDataSource
```

#### transform
```python
transform: bioshift.core.spectrumtransform.SpectrumTransform
```

#### data
```python
data: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### shape
```python
shape: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### add
```python
def add(self, other: Self) -> Self:
```
Return a new spectrum equal to the pointwise sum of two spectra.


###### Arguments:
 - **other:**  The spectrum to add.



###### Returns:
> Spectrum: A new spectrum whose values are the sum of those of the two previous spectra.


###### Raises:
 - **ValueError:**  If the shapes of the two spectra do not match


#### subtract
```python
def subtract(self, other: Self) -> Self:
```
Return a new spectrum equal to the pointwise difference of two spectra.


###### Arguments:
 - **other:**  The spectrum to subtract.



###### Returns:
> Spectrum: A new spectrum whose values are the difference of those of the two previous spectra.


###### Raises:
 - **ValueError:**  If the shapes of the two spectra do not match


#### multiply
```python
def multiply(self, other) -> Self:
```
Return a new spectrum equal to the pointwise product of two spectra.


###### Arguments:
 - **other:**  The spectrum to multiply by.



###### Returns:
> Spectrum: A new spectrum whose values are the product of those of the two previous spectra.


###### Raises:
 - **ValueError:**  If the shapes of the two spectra do not match


#### shift_to_coord
```python
def shift_to_coord(
    self,
    shift: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
) -> numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]:
```
Convert between chemical shift and grid coordinate systems. Index coordinates are interpolated between integer values.


###### Arguments:
 - **shift:**  ND array of chemical shifts.



###### Returns:
> NDArray: ND array of grid coordinates.

#### coord_to_shift
```python
def coord_to_shift(
    self,
    coord: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
) -> numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]:
```
Convert between grid and chemical shift coordinate systems. 


###### Arguments:
 - **shift:**  ND array of grid coordinates.



###### Returns:
> NDArray: ND array of chemical shifts.


---
## NMRNucleus
```python
class NMRNucleus(enum.Enum):
```

#### HYDROGEN
```python
HYDROGEN = <NMRNucleus.HYDROGEN: '1H'>
```

#### NITROGEN
```python
NITROGEN = <NMRNucleus.NITROGEN: '15N'>
```

#### CARBON
```python
CARBON = <NMRNucleus.CARBON: '13C'>
```


---
## SpectrumTransform
```python
class SpectrumTransform:
```
Represents a diagonal affine transformation used to map between
coordinates in the raw spectrum array and chemical shift values.


###### Attributes:
 - **scaling:**  Vector of scaling factors for each axis.
 - **offset:**  Offset vector.



###### Properties:
> inverse: The inverse of the transform.

#### ndim
```python
ndim: int
```

#### shape
```python
shape: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### bounds
```python
bounds
```

#### scaling
```python
scaling: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### offset
```python
offset: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### inverse_scaling
```python
inverse_scaling
```

#### inverse_offset
```python
inverse_offset
```

#### grid_to_shift
```python
def grid_to_shift(self, x):
```

#### shift_to_grid
```python
def shift_to_grid(self, x):
```

#### from_reference
```python
@classmethod
def from_reference(
    cls,
    shape,
    spectral_width,
    spectrometer_frequency,
    ref_coord,
    ref_shift
):
```


---
## Peak
```python
class Peak:
```

#### position
```python
position: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### width
```python
width: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```


---
## PeakList
```python
class PeakList:
```

#### positions
```python
positions: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### widths
```python
widths: numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]
```

#### write_csv
```python
def write_csv(self, path):
```

#### from_csv
```python
def from_csv(path):
```


---
## load_spectrum
```python
def load_spectrum(path: str | os.PathLike) -> bioshift.core.spectrum.Spectrum:
```

