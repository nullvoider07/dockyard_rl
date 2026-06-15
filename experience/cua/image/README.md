# CUA control-center image layer

`osworld-guest.dockerfile` overlays the user's `ubuntu-desktop-code` CUA image
with the OSWorld guest server on `:5000` — the evaluator/setup bridge for the
control-center backend (Phase 2, "option b"). The guest server is pinned to the
same OSWorld commit as `../vendor/desktop_env_eval`, so the guest-side
getters/setup and the host-side vendored evaluators share one contract.

The in-house stack keeps its roles: actuation → control-center `:50051`,
observation → The-Eye `:8080`. The guest server is used only for the OSWorld
task `config[]` setup steps and the evaluator getters/file pulls.

## Build

Requires a host with the base image pulled (cannot be validated offline; there
is no docker here and the base image is private):

```
docker build -f osworld-guest.dockerfile \
  --build-arg BASE_IMAGE=nullvoider/cua-ubuntu-24.04-code:v0.1 \
  -t cua-ubuntu-24.04-code-osworld:v0.1 .
```

The container then serves `:5000` alongside The-Eye `:8080`, control-center
`:50051`, and the Task Executor `:9090`. Point the backend at it via
`env.osworld.host` (see `examples/configs/grpo_osworld.yaml`).

## Notes / live-validation checklist

- The systemd unit runs as UID 1001 (the session user) with `DISPLAY=:0` and the
  session bus, so screenshots / pyautogui / AT-SPI run inside the live session.
  Confirm the UID and `XAUTHORITY` path match the base image after its
  per-container username rename.
- `/accessibility` needs `python3-pyatspi` + `gir1.2-atspi-2.0` (installed here);
  it is optional (the backend only fetches a11y when `include_a11y_text=true`).
- Publish/route `:5000` to wherever the backend runs (it connects to
  `env.osworld.host:5000`).
