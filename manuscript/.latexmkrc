# Configuration for latexmk - automated LaTeX build tool
# https://www.ctan.org/pkg/latexmk

# Generate PDF using pdflatex
$pdf_mode = 1;

# Use biber for bibliography (modern biblatex backend)
$biber = 'biber %O %S';

# pdflatex command with options
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Additional file extensions to clean
$clean_ext = "synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk run tdo";

# PDF previewer (macOS) - for latexmk -pvc mode
$pdf_previewer = 'open -a Preview';

# Maximum number of compilation runs
$max_repeat = 5;
