# OSWorld — the computer-use agent

OSWorld is the computer-use (CUA) path: the agent operates a real desktop GUI —
taking screenshots, moving the mouse, typing — to complete tasks. It is the
largest environment, with its own subsystem under `experience/cua/`, and is
multimodal: the policy tokenizer loads an `AutoProcessor`
(`policy.tokenizer.use_processor=true`) so per-turn screenshots are encoded
through it.

## The episode

A turn is: observe (a screenshot, optionally an accessibility tree), the model
emits an action, the action is actuated on the guest, repeat — up to `max_turns`.
Actions are parsed (`experience/cua/actions/`, e.g. pyautogui-style) and actuated
through a backend.

## Backends

`env.osworld.backend` selects the actuation/observation stack:

| Backend | Guest |
| --- | --- |
| `official` | `desktop_env.DesktopEnv` (Docker KVM provider) — the offline-validated reference default. |
| `control_center` | In-house Linux stack (The-Eye + control-center + an OSWorld guest server for setup/grading). |
| `windows` / `macos` | Windows / macOS guests via control-center actuation. No built-in grading — supply a custom `env.osworld.grader` for non-Ubuntu task sets. |

The `official` backend's provider is `docker` (KVM-in-container, slow reset) or
`aws` (snapshot-revert, fast reset at scale). Each concurrent episode needs one
live container/VM, so actor concurrency is bounded by the provisioned host pool
(`experience/cua/provisioner.py`); exceeding it is intended backpressure.

## Grading

Ubuntu-authored OSWorld tasks grade through the guest server; custom Windows/macOS
tasks require a supplied grader. Grading material and helpers live under
`experience/cua/grading/` and `experience/cua/tasks/`.

Config: `examples/configs/grpo_osworld.yaml` (its `env.osworld` block documents
every backend, provider, and provisioning knob).
