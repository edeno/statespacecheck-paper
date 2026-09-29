"""Export the website's Figure-4 session explorer from the fingerprinted caches.

Writes the committed overview, ``site/data/recording_explorer.json``, and the
archive it names to ``site/.explorer/``. The archive is not committed: publish it
as an asset of the repository's ``site-data`` release with the printed command,
then commit the overview. Reuses the Figure-4 caches when they are current.
"""

import json

from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_workflow import prepare_figure04_render_data
from statespacecheck_paper.paths import (
    ANIMAL_DATE_EPOCH,
    DATA_PATH,
    FIGURE04_SUMMARY_PATH,
    SITE_DATA_DIR,
    SITE_EXPLORER_ARCHIVE_DIR,
)
from statespacecheck_paper.site_explorer_export import export_recording_explorer


def main() -> None:
    """Export the overview and the archive, and print how to publish the archive."""
    summary = json.loads(FIGURE04_SUMMARY_PATH.read_text(encoding="utf-8"))
    render_data = prepare_figure04_render_data(
        Figure4Config(), Figure4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH)
    )
    overview_path, archive_path = export_recording_explorer(
        render_data, summary, SITE_DATA_DIR, SITE_EXPLORER_ARCHIVE_DIR
    )
    print(f"Wrote {overview_path}")
    print(f"Wrote {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)")
    print("Publish it before committing the overview:")
    print(f"  gh release upload site-data {archive_path}")


if __name__ == "__main__":
    main()
