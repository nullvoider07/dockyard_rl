# Vendored OSWorld evaluator subset

Pinned subset of `desktop_env/evaluators` from **xlang-ai/OSWorld**, commit
`705623ca18e0055dd995fd5a350d6588cff2caf5`. Used by the control-center backend
(Phase 2), which has no `DesktopEnv` and must grade OSWorld tasks standalone.
The official backend does **not** use this — it grades via
`DesktopEnv.evaluate()` (the evaluators ship with the `desktop_env` image dep).

## What is vendored

Byte-faithful copies of the module bodies; the only edit is rewriting absolute
`desktop_env.evaluators.metrics.utils` imports to relative. Scope is the subset
the **os** and **libreoffice_calc** task domains exercise (sized from those
domains' task_config `evaluator` blocks):

- `getters/` — `file`, `general`, `info`, `misc`
  (serves types: `vm_command_line`, `vm_terminal_output`, `rule`,
  `list_directory`, `vm_file`, `cloud_file`)
- `metrics/` — `basic_os`, `general`, `table`, `pdf`, `utils`
  (serves funcs: `exact_match`, `check_include_exclude`, `is_utc_0`,
  `check_gnome_favorite_apps`, `check_moved_jpgs`, `compare_table`,
  `compare_csv`, `check_pdf_pages`; `infeasible` handled by the dispatch)

The aggregator `__init__.py` files are trimmed to these modules; the upstream
chrome/docs/gimp/libreoffice/others/slides/thunderbird/vlc/vscode blocks are
dropped and added per-domain in later slices (they pull heavy OCR/audio/CV
deps). `compare_pdfs` (used by 1 calc task) lives in the 134 KB upstream
`chrome.py` and is intentionally deferred.

## Not vendored (written fresh)

- `dispatch.py` — standalone `evaluate(task_config, env, setup_fn=None) -> float`
  replicating `DesktopEnv._set_task_info` + `.evaluate()`. One behavioural
  change vs upstream: in the multi-metric `or` branch a `FileNotFoundError`
  scores that metric 0 instead of leaving `result_state` unbound.
- `grading_env.py` — `GradingEnv` (the duck-typed `env` the getters require:
  `vm_ip`/`server_port`/`cache_dir`/`controller`/`action_history`) and the
  `GuestServerController` Protocol (the `:5000` client methods the getters call:
  `get_file`, `get_terminal_output`, `get_vm_directory_tree`,
  `get_vm_screen_size`, `get_vm_window_size`, `get_vm_wallpaper`,
  `get_accessibility_tree`, `execute_python_command`). Implemented by
  `clients/guest_server.py` (slice 2).

## Third-party deps (grading host / image)

`requests`, `pandas`, `numpy`, `openpyxl`, `pypdf`, `pdfplumber`, `PyYAML`,
`python-docx`, `lxml`, `rapidfuzz`, `formulas`, `tldextract`, `pytz`. These are
installed where grading runs; they are absent from the offline validation host,
so this subset is validated by `py_compile` + an isolated dispatch unit test
(fake metrics/getters), not a full import.

## Updating the pin

Re-copy the listed module files from the same commit and re-run the re-export
and dispatch checks. Do not edit the vendored module bodies beyond the relative
import rewrite; behavioural changes belong in `dispatch.py`.
