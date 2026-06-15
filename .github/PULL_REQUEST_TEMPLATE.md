<!-- Keep PRs focused. Describe the change and how it was validated. -->

## Summary

<!-- What does this change and why? -->

## Validation

<!-- There is no live GPU/cluster CI. State what was run: -->

- [ ] `python -m compileall` / affected modules byte-compile
- [ ] `ruff check --select E9,F63,F7,F82 .` clean
- [ ] Relevant unit tests pass (note which; CPU-only)
- [ ] Pyright clean for touched files (run from the parent dir)
- [ ] Docs build (`sphinx-build -b html -W docs docs/_build/html`) if docs changed
- [ ] `infra/dockyard_k8s` tests + kubeconform pass if manifests/CLI changed

## Notes

<!-- Backend (torch/JAX), hardware-gated items (HV-* ledger), follow-ups. -->
