# Assemble the deployable site: the page, its data, and the paper's Figure 1
# and PDF, with the reported numbers written into the HTML (needs Node 22+).
# Page data comes from `uv run python scripts/export_site_data.py`; the session
# explorer's archive, named in data/recording_explorer.json, is taken from
# .explorer/ or downloaded there, checked, and extracted into the build.
BUILD := _build
NODE ?= node

.PHONY: build serve test test-browser clean

build:
	rm -rf $(BUILD)
	mkdir -p $(BUILD)/figures
	cp -R index.html css js data $(BUILD)/
	cp ../manuscript/figures/main/figure01.png $(BUILD)/figures/
	cp ../manuscript/main.pdf $(BUILD)/paper.pdf
	$(NODE) tools/prerender.mjs $(BUILD)/index.html $(BUILD)/data/manifest.json
	$(NODE) tools/explorer_data.mjs data/recording_explorer.json .explorer $(BUILD)/data
	touch $(BUILD)/.nojekyll

serve: build
	python3 -m http.server --directory $(BUILD) 8000

test:
	$(NODE) --test tests/*.test.mjs

# Accessibility and keyboard checks of the assembled page in headless Chromium.
# Needs `npm ci` and `npx playwright install chromium` once.
test-browser: build
	$(NODE) --test tests/browser/*.test.mjs

clean:
	rm -rf $(BUILD)
