# Getting started

Three documents take you from a clean machine to a running GRPO (Group Relative
Policy Optimization) job:

- [Installation](installation.md) — the runtime model (deps baked into the
  `ubuntu-swe` image; no `uv`, no virtualenvs at runtime), building the image,
  and local development.
- [Quickstart](quickstart.md) — launch a SWE-bench GRPO run end to end and what
  each stage of `setup()` wires together.
- [Configuration](configuration.md) — the `MasterConfig` model, the
  Hydra/OmegaConf override system, and the config family under
  `examples/configs/`.

```{toctree}
:hidden:
:maxdepth: 1

installation
quickstart
configuration
```
