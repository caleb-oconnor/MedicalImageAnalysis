# Installation

## Create an environment

MedicalImageAnalysis depends on a number of scientific and imaging libraries,
so it is best installed in a dedicated Python environment.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## Install the package from this repository

From the repository root:

```bash
pip install -e .
```

That installs the package using the dependencies declared in `requirements.txt`.

## Install documentation dependencies

If you also want to build this website locally:

```bash
pip install -r docs/requirements.txt
```

## Build the HTML site

On Windows:

```bat
docs\make.bat html
```

Or with `sphinx-build` directly:

```bash
sphinx-build -b html docs docs/_build/html
```

## Notes

- Some project dependencies are large binary packages such as VTK, PyVista, and
  Open3D.
- If you build documentation in a lightweight environment, `conf.py` mocks
  several heavy imports so API pages can still render.
