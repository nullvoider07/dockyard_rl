# Security Policy

## Supported versions

dockyard_rl is pre-1.0 and under active development. Security fixes are applied
to the `main` branch only; there are no maintained release branches yet.

## Reporting a vulnerability

Please report security issues privately — do **not** open a public issue.

Use GitHub's **private vulnerability reporting**: open the **Security** tab of
this repository and choose **Report a vulnerability**. This opens a private
advisory visible only to the maintainer.

When reporting, include:

- a description of the issue and its impact,
- the affected component (`algorithms/`, `models/`, `sandbox/`, `infra/`, …),
- reproduction steps or a proof of concept, and
- any suggested remediation.

Expect an initial acknowledgement within a few days. Coordinated disclosure is
preferred: please allow a fix to ship before any public discussion.

## Scope notes

The `sandbox/` and `ubuntu-base/` paths execute untrusted, model-generated
code by design. Reports about that execution surface are in scope only when
they describe a sandbox **escape** or a path that affects the host or the
training control plane — not the mere fact that arbitrary code runs inside the
isolated container.
