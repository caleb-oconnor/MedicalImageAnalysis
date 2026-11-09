Top-level package
=================

.. currentmodule:: medicalimageanalysis

The top-level package is the main workflow entry point used throughout the
guides and examples:

.. code-block:: python

   import medicalimageanalysis as mia

Most API details are documented on the dedicated pages for the data registry,
reader modules, structure classes, and utilities.

Common root-level entry points include:

- :func:`medicalimageanalysis.file_parser`
- :func:`medicalimageanalysis.read_dicoms`
- :func:`medicalimageanalysis.read_mhd`
- :func:`medicalimageanalysis.read_3mf`

See also
--------

- :doc:`data` for the shared ``Data`` registry
- :doc:`reader` for the high-level reader functions
- :doc:`read` for format-specific reader classes
- :doc:`structure` for image, ROI, dose, and registration structures
- :doc:`utils/index` for geometry, conversion, and registration utilities





