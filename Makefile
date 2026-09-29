# Public entry points; scientific recipes live in scripts/ and src/.
UV ?= uv
NODE ?= node
RUN = $(UV) run --frozen

.DEFAULT_GOAL := help
.PHONY: help sync sync-dev manuscript figures-simulated figures download-data \
        reported-values reproduce check check-python check-site

help:
	@echo "make manuscript         Build the PDF from committed figures and macros (LaTeX)"
	@echo "make sync               Install the locked analysis environment"
	@echo "make figures-simulated  Regenerate Figures 1-3 (no recording data)"
	@echo "make download-data      Download and verify the 75 MB Figure-4 input"
	@echo "make figures            Regenerate all figures (Figure-4 input required)"
	@echo "make reported-values    Update manuscript macros (offline)"
	@echo "make reproduce          Sync, download, regenerate figures/macros, build PDF"
	@echo "make sync-dev           Install locked development and viewer dependencies"
	@echo "make check              Run Python checks and website tests (Node 22+)"
	@echo "make check-python       Run formatting, lint, types, and default Python tests"
	@echo "make check-site         Run website metric parity tests (Node 22+)"

# --inexact installs the locked analysis environment without removing the
# development and viewer extras that sync-dev added.
sync:
	$(UV) sync --frozen --inexact

sync-dev:
	$(UV) sync --frozen --extra dev --extra interactive

manuscript:
	$(MAKE) -C manuscript

figures-simulated:
	$(RUN) python scripts/generate_figure01.py
	$(RUN) python scripts/generate_figure02.py
	$(RUN) python scripts/generate_figure03.py

download-data:
	$(RUN) python scripts/download_figure04_inputs.py

figures:
	$(RUN) python scripts/generate_all_figures.py

reported-values:
	$(RUN) python scripts/emit_reported_values.py

# Keep these dependent steps sequential even when invoked with make -j.
reproduce:
	$(MAKE) sync
	$(MAKE) download-data
	$(MAKE) figures
	$(MAKE) reported-values
	$(MAKE) manuscript

check: check-python check-site

check-python:
	$(RUN) ruff format --check .
	$(RUN) ruff check .
	$(RUN) mypy src/
	NODE="$(NODE)" $(RUN) pytest

check-site:
	$(MAKE) -C site test NODE="$(NODE)"
