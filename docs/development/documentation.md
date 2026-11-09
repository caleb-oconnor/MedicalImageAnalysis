# Working on the documentation

## Install docs dependencies

```bash
py -3 -m pip install -r docs/requirements.txt
```

## Build the site

```bash
py -3 -m sphinx -b html docs docs/_build/html
```

## Documentation design goals

- prefer short, human-readable section names
- organize API by concept rather than long dotted paths
- keep examples close to real workflows
- make the docs site feel like a real product website, not just generated API

## Updating docs when code changes

The documentation is a combination of hand-written guides and auto-generated
API pages.

### If you add a function or class to an already-documented module

1. Add or update the docstring in the Python source file.
2. If the object should be part of the public API, expose it from the relevant
   package `__init__.py`.
3. Rebuild the docs:

```bash
py -3 -m sphinx -b html docs docs/_build/html
```

If that module is already listed in the API docs, autosummary will regenerate
the page automatically.

### If you add a brand new module or file

You may also need to update one of the curated API pages:

- `docs/api/package.rst`
- `docs/api/read.rst`
- `docs/api/structure.rst`
- `docs/api/utils/index.rst`
- `docs/api/reader.rst`

If the new module belongs in a user guide or example workflow, add or update
the relevant Markdown page under `docs/user-guide/`, `docs/examples/`, or
`docs/overview/`.

### If you rename or delete code

1. Remove stale references from the relevant `docs/api/*.rst` file.
2. Rebuild the docs.
3. If stale generated pages remain under `docs/api/generated/`, delete them and
   rebuild again.

### Good maintenance rule

Whenever you add, rename, or remove public-facing code:

- update the docstring
- update exports in `__init__.py` if needed
- update the curated docs page if the object should appear in navigation
- rebuild the site and check for Sphinx warnings



