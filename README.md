<p align="center"> 
<img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/> 
<img src="https://img.shields.io/badge/python-3.11%2b-blue" alt="Python Version"/>
<img src="https://img.shields.io/pypi/v/bioshift?color=green" alt="PyPI version"/>
</p>

<h1 align="center">BioShift</h1>

**BioShift** is an open source python package for reading and analysing biomolecular NMR spectra.

## Features
- Load NMR spectra from multiple formats (.ucsf, .spc)
- Locate peaks in your spectra and identify spin systems
- Assign peptide residues using triple-resonance backbone assignment and statistical predictions from chemical shift values

## Quick Start
Install from PyPI in a virtual environment using `pip` or your python package manager of choice (`uv`, `poetry`, etc.). 

```bash
pip install bioshift
```

```python
import bioshift

spec = bioshift.Spectrum.load("./example/n15hsqc.ucsf")

bioshift.plot.heatmap(spec, show=True)
```

## Documentation
https://bioshift.pages.dev

References
---
Hoch JC, Baskaran K, Burr H, Chin J, Eghbalnia HR, Fujiwara T, Gryk MR, Iwata T, Kojima C, Kurisu G *et al* (2023) Biological Magnetic Resonance Data Bank. *Nucleic Acids Research* 51: D368-76. doi: [10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050)
