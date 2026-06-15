# Custom CUA task authoring

Custom task sets (Windows, macOS, or extra Linux tasks) reuse the **OSWorld
`task_config` JSON shape**, so the existing dataset loader
(`experience/cua/datasets/osworld.py`) and environment
(`experience/cua/environment.py`) run them **unchanged**. The only thing that
differs from the OSWorld suite is the `evaluator` block, which names a registered
custom evaluator (`experience/cua/grading`) instead of an OSWorld getter/metric
pair.

## Tree layout

Point the dataset's `data_dir` at a tree shaped exactly like OSWorld's
`evaluation_examples`:

```
<data_dir>/
  test_all.json                       # {domain: [task_id, ...]}
  examples/
    <domain>/
      <task_id>.json                  # one task_config per file
```

`custom_example/` in this directory is a minimal, valid reference tree.

## task_config shape

```jsonc
{
  "id": "win-open-notepad-0001",      // required, unique
  "instruction": "Open Notepad ...",  // required; shown to the agent
  "config": [],                       // optional setup steps (see below)
  "related_apps": ["notepad"],        // optional, informational
  "evaluator": {                      // required; drives grading
    "func": "screenshot_judge",       // a registered evaluator name, or a list
    "conj": "and",                    // optional, "and" (default) | "or" — for a func list
    "options": { "rubric": "..." }    // per-func options; dict, or a list aligned to func
  },
  "snapshot": "base",                 // optional, informational
  "proxy": false                      // optional, informational
}
```

### Evaluator

Grading goes through `experience/cua/grading.evaluate_custom`. The
action-history **FAIL gate** always applies first (same as OSWorld): a task whose
`func` is `"infeasible"` scores 1.0 iff the agent's last action was `FAIL`, and
any other task with a trailing `FAIL` scores 0.

Built-in evaluators (`experience/cua/grading/builtins.py`):

| `func`               | grades by | options |
| -------------------- | --------- | ------- |
| `infeasible`         | FAIL gate only — solved iff the agent signals `FAIL` | — |
| `manual`             | a fixed score (out-of-band / scaffolding) | `score` (default 1.0) |
| `final_action_match` | the agent's last recorded action | one+ of `equals`, `contains`, `control` (`done`/`fail`/`wait`) |
| `screenshot_judge`   | the final screen via an injected judge model | passed through to the judge (e.g. `rubric`) |

`func` may be a list with `conj` (`and` → mean, short-circuit 0; `or` → max,
short-circuit 1); `options` is then a list aligned to it (or a single dict
broadcast to every func).

Author task-set-specific evaluators next to the built-ins:

```python
from dockyard_rl.experience.cua.grading import register_evaluator

@register_evaluator("my_app_state")
def my_app_state(task_config, handle, *, options, judge=None) -> float:
    # handle is the live backend episode handle: read action_history / eye / cc /
    # host / task_config off it. Return a reward in [0, 1].
    ...
```

List the module(s) holding them under `env.osworld.evaluator_modules` so they
self-register before grading.

### Setup (`config`)

`config[]` steps (`{"type": ..., "parameters": {...}}`: `execute`, `sleep`,
`download`, `open`, `activate_window`) run through the OSWorld **guest server
(`:5000`)**, which only the Linux `control_center` image carries. The **Windows
and macOS guests do not run `:5000`**, so their tasks use `config: []` and rely
on the qcow2 overlay booting to a known state. Author setup into the guest image
(or a custom evaluator's preconditions) for those platforms.

### Grading channels (Windows / macOS)

Without the `:5000` guest server, the honest grading signals are: the FAIL gate,
the agent's final action (`final_action_match`), and the final screenshot routed
to a judge model (`screenshot_judge`). `screenshot_judge` requires a `judge`
callable — set `env.osworld.judge` to `callable(image_bytes, instruction,
**options) -> float`.

## Validate before a run

```python
from dockyard_rl.experience.cua.tasks import validate_tree

failures = validate_tree(
    "experience/cua/tasks/custom_example",
    evaluator_modules=["my_pkg.win_evals"],   # so custom funcs are registered
)
assert not failures, failures
```

`validate_tree` returns `{"<domain>/<task_id>": [problems]}` for every task that
does not validate (missing fields, an `evaluator.func` no registered evaluator
provides, a bad `conj`, a misaligned `options` list). `validate_task_config(tc)`
checks a single config.

## Wiring a custom run

In the run config (`examples/configs/grpo_osworld.yaml`):

```yaml
data:
  # point the OSWorld dataset at the custom tree
  dataset:
    data_dir: experience/cua/tasks/custom_example
    os_label: "Windows"            # or "macOS"; corrects the system prompt wording
env:
  osworld:
    backend: "windows"             # or "control_center" for a Linux custom tree
    grader: "custom"               # enable the registry-backed grader
    evaluator_modules: []          # modules holding @register_evaluator funcs
    # judge: injected programmatically for screenshot_judge
```
