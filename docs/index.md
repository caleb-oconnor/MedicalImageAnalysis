# MedicalImageAnalysis

MedicalImageAnalysis is a Python toolkit for reading medical imaging datasets,
organizing them into reusable in-memory objects, and building workflows around
ROIs, meshes, rigid transforms, and deformable registrations.

```{raw} html
<div class="hero-block">
  <p class="hero-kicker">Medical imaging workflows in Python</p>
  <h1>Read, structure, and process clinical imaging data.</h1>
  <p class="hero-lead">
    Explore DICOM and MetaImage readers, registry-driven data access,
    contour and mesh conversion helpers, and registration-aware structures
    from one package.
  </p>
</div>
```

::::{grid} 1 1 2 4
:gutter: 3

:::{grid-item-card} Overview
:link: overview/index
:link-type: doc

See what the package does, how the pieces fit together, and where to start.
:::

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install the package, import it cleanly, and run your first loading workflow.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

Learn the core concepts behind readers, the `Data` registry, ROI processing,
meshing, and registration.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Follow task-oriented examples for common imaging and geometry workflows.
:::
::::

## Why use MedicalImageAnalysis?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reader-first workflows
Use high-level entry points such as `file_parser`, `read_dicoms`, `read_mhd`,
and `read_3mf` to bring study data into Python quickly.
:::

:::{grid-item-card} Shared data registry
Loaded images, ROIs, dose objects, rigid transforms, and deformables are kept
in a common `Data` registry for multi-step workflows.
:::

:::{grid-item-card} Geometry and ROI tooling
Convert masks, contours, and meshes across representations with utilities
exposed from one namespace.
:::

:::{grid-item-card} Registration-aware structures
Work with rigid and deformable objects that stay connected to image-centric
processing pipelines.
:::
::::

## Documentation map

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Conceptual overview
:link: overview/index
:link-type: doc
Architecture, package design, and major capabilities.
:::

:::{grid-item-card} Acknowledgements
:link: overview/acknowledgements
:link-type: doc
Open-source tools that power DICOM IO, image processing, meshing, and visualization.
:::

:::{grid-item-card} Workflow guides
:link: user-guide/index
:link-type: doc
Task-oriented documentation for common data-processing patterns.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Cleaned-up API pages organized by concept rather than raw package paths.
:::

:::{grid-item-card} Examples gallery
:link: examples/index
:link-type: doc
Small examples that can later grow into a fuller gallery.
:::

:::{grid-item-card} Development docs
:link: development/index
:link-type: doc
Notes for contributing, editing docs, and growing the site.
:::

:::{grid-item-card} Project source
:link: https://github.com/caleb-oconnor/MedicalImageAnalysis

Browse the repository, packaging config, and implementation details on GitHub.
:::
::::

## Build the docs locally

```bash
pip install -r docs/requirements.txt
py -3 -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
overview/index
user-guide/index
examples/index
development/index
api/index
```



