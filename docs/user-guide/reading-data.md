# Reading data

MedicalImageAnalysis exposes both top-level convenience functions and
format-specific reader classes.

## Top-level reader functions

The main import surface is intentionally simple:

- `medicalimageanalysis.file_parser`
- `medicalimageanalysis.read_dicoms`
- `medicalimageanalysis.read_mhd`
- `medicalimageanalysis.read_3mf`

These are the best entry points for most workflows because they match how the
package is re-exported from `medicalimageanalysis.__init__`.

## Reader subpackage

The `medicalimageanalysis.read` package groups specialized reader classes:

- `DicomReader`
- `MhdReader`
- `ThreeMfReader`
- `StlReader`
- `VtkReader`

Use the class-based readers when you want lower-level control or when you are
documenting a format-specific pipeline.

## Typical workflow

1. Use `file_parser` to classify files in a study directory.
2. Pass the discovered file list into the appropriate reader.
3. Inspect loaded objects through the shared `Data` registry.
4. Continue with ROI conversion, meshing, or registration utilities.

## Related pages

- {doc}`structures`
- {doc}`utilities`
- {doc}`../api/index`
