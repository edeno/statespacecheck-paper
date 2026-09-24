# Assemble the deployable site: the page, its data, and the paper's Figure 1
# and PDF, with the reported numbers written into the HTML (needs Node 22+).
# Page data comes from `uv run python scripts/export_site_data.py`.
BUILD := _build
NODE ?= node

.PHONY: build serve test clean

build:
	rm -rf $(BUILD)
	mkdir -p $(BUILD)/figures
	cp -R index.html css js data $(BUILD)/
	cp ../manuscript/figures/main/figure01.png $(BUILD)/figures/
	cp ../manuscript/main.pdf $(BUILD)/paper.pdf
	$(NODE) tools/prerender.mjs $(BUILD)/index.html $(BUILD)/data/manifest.json
	touch $(BUILD)/.nojekyll

serve: build
	python3 -m http.server --directory $(BUILD) 8000

test:
	$(NODE) --test tests/*.test.mjs

clean:
	rm -rf $(BUILD)
