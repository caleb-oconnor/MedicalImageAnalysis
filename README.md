# MedicalImageAnalysis

MedicalImageAnalysis is a Python toolkit for reading medical imaging datasets,
organizing them into reusable in-memory objects, and building workflows around
ROIs, meshes, rigid transforms, and deformable registrations.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://caleb-oconnor.github.io/MedicalImageAnalysis/)

## Documentation

- [Getting Started](https://caleb-oconnor.github.io/MedicalImageAnalysis/getting-started/index.html)
- [User Guide](https://caleb-oconnor.github.io/MedicalImageAnalysis/user-guide/index.html)
- [Examples](https://caleb-oconnor.github.io/MedicalImageAnalysis/examples/index.html)
- [API Reference](https://caleb-oconnor.github.io/MedicalImageAnalysis/api/index.html)
- [Overview](https://caleb-oconnor.github.io/MedicalImageAnalysis/overview/index.html)

## Why use MedicalImageAnalysis?

- **Reader-first workflows** for loading DICOM, MetaImage, and mesh data
- **Shared `Data` registry** for images, dose, ROIs, POIs, rigid transforms, and deformables
- **Structure-first design** so image-centric workflows live on `Image`, `Roi`, `Rigid`, and `Deformable`
- **Direct utility access** for users who want lower-level control over conversions, meshing, and registration

## Installation
[README.md](../../GitHub/MedicalImageAnalysis/README.md)
```bash
pip install medicalimageanalysis
```

Or install from source:

```bash
git clone https://github.com/caleb-oconnor/MedicalImageAnalysis.git
cd MedicalImageAnalysis
pip install -e .
```

## Quick start

```python
import medicalimageanalysis as mia

mia.read_dicoms(folder_path="C:/Data/Patient01")

print(mia.Data.image_list())
print(mia.Data.roi_list())

image = mia.Data.image[mia.Data.image_list()[0]]
print(image.array.shape)
```

## Common workflows

### Read a DICOM study

```python
import medicalimageanalysis as mia

files = mia.file_parser(folder_path="C:/Data/Patient01")
mia.read_dicoms(file_list=files["Dicom"])
```

### Build a mesh from an ROI

```python
import medicalimageanalysis as mia

image = mia.Data.image["CT 01"]
roi = image.roi["Liver"]
roi.create_mesh(reduction=0.5)

print(roi.mesh.n_points)
```

### Run a rigid registration

```python
import medicalimageanalysis as mia

rigid = mia.Rigid(reference_name="CT 01", moving_name="MR 01")
rigid.compute_registration(
    method="icp_point",
    reference_rois=["Liver"],
    moving_rois=["Liver"],
)
```

### Run a deformable registration

```python
import medicalimageanalysis as mia
a
deformable = mia.Deformable()
deformable.compute_registration(
    method="itk_fast_demons",
    reference_name="CT 01",
    moving_name="MR 01",
    reference_rois=["Liver"],
    moving_rois=["Liver"],
)
```

## Project structure

- `medicalimageanalysis.reader` — high-level loading entry points
- `medicalimageanalysis.read` — format-specific reader classes
- `medicalimageanalysis.structure` — image, ROI, dose, rigid, and deformable objects
- `medicalimageanalysis.utils` — lower-level conversion, meshing, and registration utilities
- `docs/` — Sphinx documentation source

## Documentation publishing note

The repository keeps the **documentation source** under `docs/`. The hosted
documentation site is generated from those files and published at:

https://caleb-oconnor.github.io/MedicalImageAnalysis/

## License

Licensed under the GNU Affero General Public License v3.

See [LICENSE.txt](LICENSE.txt) for details.