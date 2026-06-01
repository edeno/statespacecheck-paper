.PHONY: all clean main supplement view help

# Default target
all: main.pdf supplement.pdf

# Convenience aliases so `make main` / `make supplement` build the PDFs
# (the phony names in .PHONY would otherwise be no-op targets).
main: main.pdf
supplement: supplement.pdf

# Figures bundled into the manuscript; changing any should force a rebuild.
FIGURES := $(wildcard figures/main/*.pdf figures/supplementary/*.pdf)

# Build main manuscript (self-contained: own preamble + embedded bibliography)
main.pdf: main.tex references.bib $(FIGURES)
	latexmk -pdf main.tex

# Build supplement
supplement.pdf: supplement.tex preamble.tex references.bib
	latexmk -pdf supplement.tex

# Clean all build artifacts
clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.synctex.gz *.run.xml *.bcf

# View main PDF
view: main.pdf
	open main.pdf

# Show available targets
help:
	@echo "Available targets:"
	@echo "  make          - Build both main.pdf and supplement.pdf"
	@echo "  make main     - Build main.pdf only"
	@echo "  make supplement - Build supplement.pdf only"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make view     - Open main.pdf"
	@echo "  make help     - Show this help message"

.DEFAULT_GOAL := all
