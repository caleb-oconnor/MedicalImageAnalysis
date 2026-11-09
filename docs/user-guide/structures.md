# Structures and the global registry

The package keeps track of loaded objects through the `Data` registry, which is
instantiated when `medicalimageanalysis` is imported.

## Data registry

`medicalimageanalysis.Data` provides convenience methods for discovering what
is currently loaded, such as:

- `image_list()`
- `deformable_list()`
- `dose_list()`
- `poi_list()`
- `rigid_list()`
- `roi_list()`

This pattern makes it straightforward to read data in one step and inspect or
post-process it in a later step.

## Core structure classes

The `medicalimageanalysis.structure` package exports the core object types:

- `Image`
- `Dose`
- `Roi`
- `Poi`
- `Rigid`
- `Deformable`

These classes represent loaded imaging data, dose volumes, annotations, and
associated transformations.

## Why this matters in the docs site

The registry and structure classes are the backbone of the package navigation
model. The user guide and API reference should both point to these same public
objects so the docs stay consistent as the codebase evolves.
