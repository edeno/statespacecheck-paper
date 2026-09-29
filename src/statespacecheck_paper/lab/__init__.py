"""Lab-only acquisition and export of the Figure-4 input from the Spyglass database.

These modules serve the Frank lab's data operations, not public reproduction:

- :mod:`~statespacecheck_paper.lab.spyglass_data` rebuilds the Figure-4 input
  file from the lab's Spyglass database and logs those fetches in a Spyglass
  export. It imports Spyglass only inside functions.
- :mod:`~statespacecheck_paper.lab.spyglass_pipeline` defines the Figure-4 decode
  and diagnostics as Spyglass tables. Importing it imports Spyglass, which
  connects to the lab database, so tests never import it.

Figure and analysis code never imports this package
(``tests/test_import_boundaries.py`` enforces this); the lab scripts
``scripts/fetch_figure04_inputs.py``, ``scripts/spyglass_export_figure04.py``,
and ``scripts/spyglass_pipeline_figure04.py`` call it. See
``docs/data-lineage.md`` and ``docs/spyglass-pipeline.md``.
"""
