"""The Frank lab's Spyglass pipeline for Figure 4, upstream of the paper's code.

Figure 4 analyzes one recording session that lives in the lab's Spyglass
database (Spyglass is the lab's DataJoint-based data-management system). The
paper's figures do not need that database: they read an archived input file
(:mod:`statespacecheck_paper.figure04_input`), which anyone can download.
This package is how that input was produced from the database, and how the
Figure-4 analysis can be run inside the lab's pipeline:

- :mod:`~statespacecheck_paper.spyglass_pipeline.figure04_input` rebuilds the
  Figure-4 input file (position and sorted spike times) from the Spyglass
  database, writes it, and compares it with the archived copy.
- :mod:`~statespacecheck_paper.spyglass_pipeline.paper_export` records that fetch
  in a Spyglass paper export, the log of database entries and files a paper used.
- :mod:`~statespacecheck_paper.spyglass_pipeline.figure04_schema` defines Figure
  4's decoding and per-spike diagnostics as Spyglass tables, so the lab can
  reproduce the figure's numbers from its own database. Importing it imports
  Spyglass, which connects to the lab database, so tests never import it.
- :mod:`~statespacecheck_paper.spyglass_pipeline.figure04_compute` holds what those
  tables compute, runnable and testable without a database.
- :mod:`~statespacecheck_paper.spyglass_pipeline.pickle_conversion` converts the
  five pickles the recording was first exported as into the input file.

The other modules import Spyglass only inside functions (the pickle conversion
not at all). Figure and analysis code never imports this package
(``tests/test_import_boundaries.py`` enforces this). The lab scripts
``scripts/fetch_figure04_inputs.py``, ``scripts/spyglass_export_figure04.py``,
``scripts/spyglass_pipeline_figure04.py``, and ``scripts/convert_figure04_pickles.py``
call it. See ``docs/data-lineage.md`` and ``docs/spyglass-pipeline.md``.
"""
