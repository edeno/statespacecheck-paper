.PHONY: all clean main view help

# Default target
all: main.pdf

# Convenience alias so `make main` builds the PDF
# (the phony name in .PHONY would otherwise be a no-op target).
main: main.pdf

# Figures bundled into the manuscript; changing any should force a rebuild.
FIGURES := $(wildcard figures/main/*.pdf figures/supplementary/*.pdf)

# Build main manuscript (self-contained: own preamble + embedded bibliography)
main.pdf: main.tex Local-GoF-Paper.bib $(FIGURES)
	latexmk -pdf main.tex

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
	@echo "  make          - Build main.pdf"
	@echo "  make main     - Build main.pdf only"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make view     - Open main.pdf"
	@echo "  make help     - Show this help message"

.DEFAULT_GOAL := all
