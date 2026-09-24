# Assemble the deployable site: the page, its data, and the paper's Figure 1
# and PDF. Page data comes from `uv run python scripts/export_site_data.py`.
BUILD := _build

.PHONY: build serve test clean

build:
	rm -rf $(BUILD)
	mkdir -p $(BUILD)/figures
	cp -R index.html css js data $(BUILD)/
	cp ../manuscript/figures/main/figure01.png $(BUILD)/figures/
	cp ../manuscript/main.pdf $(BUILD)/paper.pdf
	touch $(BUILD)/.nojekyll

serve: build
	python3 -m http.server --directory $(BUILD) 8000

test:
	node --test tests/*.test.mjs

clean:
	rm -rf $(BUILD)
