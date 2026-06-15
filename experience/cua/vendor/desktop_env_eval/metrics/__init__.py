# Vendored subset of OSWorld desktop_env/evaluators/metrics, pinned at
# xlang-ai/OSWorld 705623ca18e0055dd995fd5a350d6588cff2caf5.
#
# Only the modules exercised by the Phase-2 first-target domains (os,
# libreoffice_calc) are vendored; the upstream aggregator's chrome/docs/gimp/
# libreoffice/others/slides/thunderbird/vlc/vscode blocks are intentionally
# dropped (they pull heavy OCR/audio/CV deps and are added per-domain in later
# slices). The module bodies are byte-faithful copies; only absolute
# desktop_env.* imports were rewritten relative. ``infeasible`` is a sentinel —
# the dispatch handles it directly (action-history inspection), it is never
# called as a metric.

from .basic_os import (
    check_gnome_favorite_apps,
    is_utc_0,
    check_text_enlarged,
    check_moved_jpgs,
    is_in_vm_clickboard,
)
from .general import (
    check_csv,
    check_accessibility_tree,
    run_sqlite3,
    check_json,
    check_list,
    exact_match,
    match_in_list,
    is_in_list,
    fuzzy_match,
    check_include_exclude,
    check_direct_json_object,
    compare_time_in_speedtest_results,
    is_included_all_json_objects,
    is_gold_text_included_in_pdf,
    check_line_number,
    file_contains,
    compare_terminal_and_txt,
    fuzzy_place_math,
    compare_python_pure_text,
    diff_text_file,
    literal_match,
)
from .pdf import check_pdf_pages
from .table import (
    compare_table,
    compare_csv,
    compare_conference_city_in_order,
)


def infeasible():
    pass
