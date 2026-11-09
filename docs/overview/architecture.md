# Package architecture

The package can be thought of as four layers.

## 1. Readers

Reader functions and reader classes are responsible for understanding on-disk
formats such as DICOM, MHD, STL, VTK, and 3MF.

## 2. Shared registry

The `Data` object acts as a central registry so loaded data is available across
subsequent workflow steps.

## 3. Structures

The structure layer contains objects such as `Image`, `Rigid`, and
`Deformable`, which represent the domain entities manipulated by the package.

## 4. Utilities

The utility layer provides helpers for common computational tasks including:

- contour conversion
- mask generation
- mesh processing
- geometric measurements
- deformable registration support

This architecture works well for a documentation site because conceptual guides,
examples, and API pages can all point to the same public objects.
