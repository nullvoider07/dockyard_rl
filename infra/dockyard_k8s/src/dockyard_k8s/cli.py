"""dockyard-k8s command-line entry point.

Hydra-style overrides (``infra.scheduler.queue=x``) are collected via
``click.UNPROCESSED`` — any ``key=value`` token after the recipe path is
passed to :func:`dockyard_k8s.config.load_recipe_with_infra` as an override.

Command surface:

  * ``check`` / ``render`` — load, validate, and emit manifests (no cluster).
  * ``run`` — submit a recipe: ephemeral RayJob (``launch.mode=rayjob``),
    long-lived RayCluster (``bringup``), or attach to an existing one.
  * ``status`` / ``logs`` — inspect the GPU cluster + sandbox pool.
  * ``cluster`` — up / down / list / dashboard for the GPU RayCluster + sandbox.
  * ``job`` — list / logs / stop Ray jobs on the cluster.
  * ``dev`` — a CPU dev pod (connect / stop / setup-secrets).
  * ``doctor`` — cluster-access preflight.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import click
import yaml
from kubernetes.client.exceptions import ApiException  # type: ignore[import-not-found]
from omegaconf import OmegaConf

from . import __version__
from .config import LoadedConfig, load_recipe_with_infra
from .manifest import sandbox_service_dns
from .render import render_manifests
from .schema import CodeSource, LaunchMode, RunMode, SubmitterMode

_RECIPE_ARG = click.argument(
    "recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
_OVERRIDES_ARG = click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
_INFRA_OPT = click.option(
    "--infra",
    "infra_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Standalone infra YAML (when recipe and infra are split).",
)
_MODE_CHOICE = click.Choice([m.value for m in RunMode])
_SUBMITTER_CHOICE = click.Choice([m.value for m in SubmitterMode])
_CODE_SOURCE_CHOICE = click.Choice([m.value for m in CodeSource])

# RunMode macro -> (submitter, codeSource, no_wait). --mode wins over
# infra.launch.runMode; explicit --submitter / --code-source / --wait win over both.
_MODE_DEFAULTS: dict[RunMode, tuple[SubmitterMode, CodeSource, bool]] = {
    RunMode.INTERACTIVE: (SubmitterMode.PORT_FORWARD, CodeSource.UPLOAD, False),
    RunMode.BATCH: (SubmitterMode.EXEC, CodeSource.IMAGE, True),
}

_DEV_POD_RBAC_YAML = """\
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: dockyard-k8s-edit
  labels:
    rbac.authorization.k8s.io/aggregate-to-edit: "true"
rules:
  - apiGroups: [ray.io]
    resources: [rayjobs, rayclusters]
    verbs: [get, list, watch, create, update, patch, delete]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: default-sa-dockyard-k8s-edit
  namespace: <NAMESPACE>
subjects:
  - kind: ServiceAccount
    name: default
    namespace: <NAMESPACE>
roleRef:
  kind: ClusterRole
  name: edit
  apiGroup: rbac.authorization.k8s.io"""


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="dockyard-k8s")
def main() -> None:
    """Kubernetes deployment for the dockyard_rl GRPO stack."""


# check / render

@main.command()
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the resolved bundle (infra + recipe + manifests) to this "
    "file. Format from the extension (.yaml / .json).",
)
def check(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    output: Path | None,
) -> None:
    """Load and validate a recipe/infra pair; print a summary or write a bundle."""
    loaded = _load(recipe, overrides, infra_path)
    manifests = render_manifests(loaded)
    if output is not None:
        _write_bundle(loaded, manifests, output)
        click.echo(f"wrote bundle ({len(manifests)} manifests) to {output}")
        return
    _print_summary(loaded, manifests)


@main.command()
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option(
    "--dry-run",
    is_flag=True,
    help="Pipe the rendered manifests through "
    "`kubectl apply --dry-run=client -f -` for structural validation.",
)
def render(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    dry_run: bool,
) -> None:
    """Render the K8s manifests as multi-document YAML."""
    loaded = _load(recipe, overrides, infra_path)
    manifests = render_manifests(loaded)
    doc = yaml.safe_dump_all(manifests, sort_keys=False, default_flow_style=False)
    if not dry_run:
        click.echo(doc, nl=False)
        return
    proc = subprocess.run(
        ["kubectl", "apply", "--dry-run=client", "-f", "-"],
        input=doc,
        text=True,
    )
    sys.exit(proc.returncode)


# run

def _mode_options(fn):
    """Shared --mode/--submitter/--code-source/--code-path/--run-id/--wait block."""
    fn = click.option(
        "--mode", "cli_mode", type=_MODE_CHOICE, default=None,
        help="Macro: interactive = port-forward + working_dir upload + tail; "
        "batch = kubectl exec + code from image + no wait. Overrides "
        "infra.launch.runMode.",
    )(fn)
    fn = click.option(
        "--submitter", "cli_submitter", type=_SUBMITTER_CHOICE, default=None,
        help="Transport for the training entrypoint. Overrides --mode's default.",
    )(fn)
    fn = click.option(
        "--code-source", "cli_code_source", type=_CODE_SOURCE_CHOICE, default=None,
        help="Where the code lives. `upload` stages a working_dir; `image` / "
        "`lustre` expect code on disk inside the pod.",
    )(fn)
    fn = click.option(
        "--code-path", "cli_code_path", type=str, default=None,
        help="Absolute container path for code when --code-source is image/lustre.",
    )(fn)
    fn = click.option(
        "--run-id", "cli_run_id", type=str, default=None,
        help="Human-readable tag for this run (Ray submission id / exec pidfile dir).",
    )(fn)
    fn = click.option(
        "--wait/--no-wait", "cli_wait", default=None,
        help="Override mode's wait default: --wait tails logs and exits on "
        "terminal state; --no-wait returns immediately after submit.",
    )(fn)
    return fn


@main.command()
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default="cwd",
    help="dockyard_rl repo root used to source files for the working_dir upload.",
)
@click.option("--replace", is_flag=True, help="Stop any running job before submitting (bringup/attach).")
@click.option("--recreate", is_flag=True, help="Delete + re-apply a drifted RayCluster (bringup).")
@click.option(
    "--rayjob-name", "rayjob_name", type=str, default=None,
    help="[rayjob mode] RayJob metadata name. Defaults to the cluster name.",
)
@click.option(
    "--shutdown/--no-shutdown", "rayjob_shutdown", default=True, show_default=True,
    help="[rayjob mode] Delete the ephemeral RayCluster once the job finishes.",
)
@click.option(
    "--ttl", "rayjob_ttl", type=int, default=3600, show_default=True,
    help="[rayjob mode] Seconds to keep the RayJob object after it finishes.",
)
@click.option(
    "--timeout", "rayjob_timeout", type=int, default=86400, show_default=True,
    help="[rayjob mode] Seconds to wait for a terminal state when waiting.",
)
@click.option("--dry-run", is_flag=True, help="[rayjob mode] Render the RayJob manifest and exit.")
@_mode_options
def run(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    repo_root: Path,
    replace: bool,
    recreate: bool,
    rayjob_name: str | None,
    rayjob_shutdown: bool,
    rayjob_ttl: int,
    rayjob_timeout: int,
    dry_run: bool,
    cli_mode: str | None,
    cli_submitter: str | None,
    cli_code_source: str | None,
    cli_code_path: str | None,
    cli_run_id: str | None,
    cli_wait: bool | None,
) -> None:
    """Submit a recipe to the cluster.

    Dispatch is on ``infra.launch.mode``:

    * ``rayjob`` (ephemeral) — apply a KubeRay RayJob; KubeRay creates the
      RayCluster, submits ``launch.entrypoint``, polls to terminal, tears down.
    * ``bringup`` (long-lived) — ensure the sandbox pool + GPU RayCluster, then
      submit training via the configured transport. The cluster stays up.
    * ``attach`` — submit onto the existing RayCluster ``launch.attach``.

    The sandbox executor pool (``infra.sandbox``) is applied before training in
    bringup/attach modes. ``--mode interactive`` (default) uses port-forward +
    upload and tails logs; ``--mode batch`` uses kubectl exec + in-image code.
    """
    from . import submit as submit_mod

    loaded = _load_or_exit(recipe, overrides, infra_path)
    if not loaded.infra.launch.entrypoint:
        _cli_error(
            "infra.launch.entrypoint is empty",
            hint="`dockyard-k8s run` requires infra.launch.entrypoint",
        )

    if loaded.infra.launch.mode is LaunchMode.RAYJOB:
        _run_rayjob(
            loaded,
            recipe=recipe,
            name=rayjob_name,
            shutdown_after=rayjob_shutdown,
            ttl_seconds=rayjob_ttl,
            timeout_s=rayjob_timeout,
            dry_run=dry_run,
            cli_wait=cli_wait,
        )
        return

    if dry_run:
        _cli_error("--dry-run is only valid in rayjob mode")

    from . import orchestrate

    if not submit_mod.is_in_cluster():
        _preflight_or_exit(loaded.infra.namespace)

    mode, submitter, code_src, no_wait = _resolve_mode_defaults(
        cli_mode=cli_mode,
        infra_mode=loaded.infra.launch.runMode,
        cli_submitter=cli_submitter,
        cli_code_source=cli_code_source,
        cli_wait=cli_wait,
    )
    _apply_mode_overrides(
        loaded, submitter=submitter, code_source=code_src, code_path=cli_code_path
    )
    click.echo(
        f"[run] launch.mode={loaded.infra.launch.mode.value} run-mode={mode.value} "
        f"submitter={submitter.value} code_source={code_src.value} "
        f"no_wait={no_wait} recreate={recreate}",
        err=True,
    )

    if loaded.infra.launch.mode is LaunchMode.BRINGUP and loaded.infra.kuberay is not None:
        _check_head_svc_collision(
            loaded.infra.kuberay.name, loaded.infra.namespace, creating="raycluster"
        )

    try:
        result = orchestrate.run(
            loaded,
            log=click.echo,
            repo_root=repo_root.resolve(),
            replace=replace,
            run_id=cli_run_id,
            recreate=recreate,
        )
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="run failed")

    _emit_handle(result.handle, recipe)
    if not no_wait:
        _follow_handle(result.handle)


def _run_rayjob(
    loaded: LoadedConfig,
    *,
    recipe: Path,
    name: str | None,
    shutdown_after: bool,
    ttl_seconds: int,
    timeout_s: int,
    dry_run: bool,
    cli_wait: bool | None,
) -> None:
    """``launch.mode=rayjob`` path — KubeRay owns the RayCluster lifecycle."""
    from . import k8s, orchestrate
    from . import submit as submit_mod
    from .rayjob import build_rayjob_manifest

    cluster = _pick_cluster_or_exit(loaded)
    entrypoint = loaded.infra.launch.entrypoint
    if not entrypoint:
        _cli_error("infra.launch.entrypoint is empty")
    manifest = build_rayjob_manifest(
        cluster,
        loaded.infra,
        entrypoint=entrypoint,
        name=name,
        shutdown_after_finishes=shutdown_after,
        ttl_seconds_after_finished=ttl_seconds,
    )
    job_name = manifest["metadata"]["name"]
    namespace = loaded.infra.namespace

    if dry_run:
        # Show the full manifest set (sandbox + RayJob) the run would apply.
        for m in render_manifests(loaded):
            click.echo(yaml.safe_dump(m, sort_keys=False).rstrip())
            click.echo("---")
        return

    if not submit_mod.is_in_cluster():
        _preflight_or_exit(namespace)

    _check_head_svc_collision(job_name, namespace, creating="rayjob")

    # Apply the sandbox pool first so the executors are reachable before the
    # training driver opens HTTP connections to them.
    if loaded.infra.sandbox is not None:
        try:
            orchestrate.ensure_sandbox(loaded, log=click.echo)
        except Exception as exc:  # noqa: BLE001
            _explain_and_exit(exc, context="sandbox bring-up failed")

    # DRA CRDs must exist before KubeRay schedules the workers that claim them.
    try:
        orchestrate.ensure_dra_resources(loaded, log=click.echo)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="DRA resource creation failed")

    click.echo(f"[run rayjob] applying RayJob {job_name} in {namespace}")
    try:
        k8s.apply_rayjob(manifest, namespace)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"rayjob {job_name} apply failed")

    click.echo(
        f"follow:  kubectl get rayjob {job_name} -n {namespace} -w\n"
        f"logs:    kubectl logs -n {namespace} -l ray.io/cluster={job_name} -f"
    )
    if cli_wait is False:
        return

    click.echo(f"[run rayjob] waiting for {job_name} to reach a terminal state ...")

    def _on_update(deployment: str | None, job: str | None) -> None:
        click.echo(f"[run rayjob] {job_name} deployment={deployment} job={job}")

    try:
        final = k8s.wait_for_rayjob_terminal(
            job_name, namespace, timeout_s=timeout_s, on_update=_on_update
        )
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"rayjob {job_name} wait failed")

    status = final.get("status") or {}
    dep = status.get("jobDeploymentStatus")
    job_status = status.get("jobStatus")
    message = (status.get("message") or "").strip()
    click.echo(f"[run rayjob] {job_name} finished: deployment={dep} job={job_status}")
    if message:
        click.echo(f"[run rayjob] message: {message}")
    # KubeRay's shutdownAfterJobFinishes tears down the RayCluster but not the
    # separate DRA CRs the CLI created — clean those up to match.
    if shutdown_after:
        orchestrate.delete_dra_resources(loaded, log=click.echo)
    sys.exit(0 if dep == "Complete" else 1)


# status / logs

@main.command()
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
def status(recipe: Path, overrides: tuple[str, ...], infra_path: Path | None) -> None:
    """Summarise the GPU RayCluster + sandbox pool declared in the recipe."""
    from . import inspect as ins

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster = ins.collect_cluster_status(loaded)
    sandbox = ins.collect_sandbox_status(loaded)

    if cluster is None and sandbox is None:
        click.echo("(no kuberay cluster or sandbox declared in recipe)")
        return

    if cluster is not None:
        workers = ",".join(p or "—" for p in cluster.worker_phases) or "—"
        click.echo("KUBERAY")
        click.echo(f"  {cluster.name}  state={cluster.state}")
        click.echo(f"    head    {cluster.head_pod or '—'} ({cluster.head_phase or '—'})")
        click.echo(f"    workers {workers}")
    if sandbox is not None:
        click.echo("SANDBOX")
        avail = "ready" if sandbox.available else "not-ready"
        click.echo(f"  {sandbox.name}  {sandbox.ready}/{sandbox.replicas} {avail}")


@main.command()
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option(
    "--source", type=click.Choice(["head", "worker", "sandbox"]), default="head",
    show_default=True, help="Which pod's logs to stream.",
)
@click.option("-f", "--follow", is_flag=True, help="Stream new output until Ctrl+C.")
@click.option("--tail", "tail_lines", type=int, default=200, show_default=True,
              help="Trailing lines to show before following.")
def logs(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    source: str,
    follow: bool,
    tail_lines: int,
) -> None:
    """Stream logs from the GPU cluster head/worker or a sandbox pod."""
    from . import inspect as ins
    from . import k8s

    loaded = _load_or_exit(recipe, overrides, infra_path)
    namespace = loaded.infra.namespace

    if source == "sandbox":
        if loaded.infra.sandbox is None:
            _cli_error("no sandbox declared in the recipe")
        pods = k8s.list_pods_by_label(
            f"app.kubernetes.io/name={loaded.infra.sandbox.name}", namespace
        )
        if not pods:
            _cli_error(f"no sandbox pods found for {loaded.infra.sandbox.name}")
        pod_name = pods[0].metadata.name
    else:
        cluster = _pick_cluster_or_exit(loaded)
        if source == "head":
            pod_name = ins.head_pod_name(cluster.name, namespace)
        else:
            pod_name = _first_worker_pod_or_exit(cluster.name, namespace)

    for line in ins.stream_pod_logs(
        pod_name, namespace, follow=follow, tail_lines=tail_lines
    ):
        click.echo(line, nl=False)


# cluster group

@main.group()
def cluster() -> None:
    """Manage the long-lived GPU RayCluster and sandbox pool."""


@cluster.command("up")
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option("--wait/--no-wait", default=True, help="Wait for readiness before returning.")
@click.option("--timeout", default=900, show_default=True, help="Readiness timeout (s).")
@click.option("--recreate", is_flag=True, help="Delete + re-apply a drifted RayCluster.")
@click.option("--dry-run", is_flag=True, help="Render manifests and print them; do not apply.")
def cluster_up(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    wait: bool,
    timeout: int,
    recreate: bool,
    dry_run: bool,
) -> None:
    """Bring up the sandbox pool and GPU RayCluster declared in the recipe."""
    from . import orchestrate

    loaded = _load_or_exit(recipe, overrides, infra_path)

    if dry_run:
        for m in render_manifests(loaded):
            click.echo(yaml.safe_dump(m, sort_keys=False).rstrip())
            click.echo("---")
        return

    try:
        if loaded.infra.sandbox is not None:
            orchestrate.ensure_sandbox(loaded, log=click.echo)
        if loaded.infra.kuberay is not None:
            _check_head_svc_collision(
                loaded.infra.kuberay.name, loaded.infra.namespace, creating="raycluster"
            )
            orchestrate.ensure_cluster(
                loaded, log=click.echo, recreate=recreate, wait_ready=wait,
                ready_timeout_s=timeout,
            )
    except ApiException as exc:
        if exc.status == 403:
            _cli_error(
                f"forbidden to create resources in {loaded.infra.namespace}",
                hint="missing RBAC — ask an admin to grant the edit role.",
            )
        _explain_and_exit(exc, context="cluster up failed")
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="cluster up failed")


@cluster.command("down")
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option("--wait/--no-wait", default=True, help="Wait for the RayCluster to disappear.")
def cluster_down(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    wait: bool,
) -> None:
    """Delete the GPU RayCluster and sandbox pool."""
    from . import k8s, orchestrate

    loaded = _load_or_exit(recipe, overrides, infra_path)
    namespace = loaded.infra.namespace

    if loaded.infra.kuberay is not None:
        name = loaded.infra.kuberay.name
        click.echo(f"deleting RayCluster {name} in {namespace} ...")
        k8s.delete_raycluster(name, namespace)
        if wait:
            k8s.wait_for_raycluster_gone(name, namespace)
        click.echo(f"RayCluster {name} deleted.")
        orchestrate.delete_cluster_pdb(loaded, log=click.echo)
        orchestrate.delete_dra_resources(loaded, log=click.echo)
    orchestrate.delete_sandbox(loaded, log=click.echo)


@cluster.command("list")
@click.option("--namespace", "-n", default=None, help="Namespace (default: kube context).")
def cluster_list(namespace: str | None) -> None:
    """List RayClusters in a namespace and their state."""
    from . import k8s
    from .config import _infer_kube_namespace

    ns = namespace or _infer_kube_namespace()
    rows = k8s.list_rayclusters(ns)
    if not rows:
        click.echo(f"(no RayClusters in {ns})")
        return
    for obj in rows:
        name = obj["metadata"]["name"]
        state = obj.get("status", {}).get("state", "—")
        click.echo(f"{name}\t{state}")


@cluster.command("dashboard")
@click.argument("name")
@click.option("--namespace", "-n", default=None, help="Namespace (default: kube context).")
@click.option("--port", "local_port", type=int, default=8265, show_default=True,
              help="Local port to bind the forward to.")
@click.option("--open/--no-open", "open_browser", default=True, show_default=True,
              help="Open the dashboard URL in a browser once the forward is up.")
def cluster_dashboard(
    name: str,
    namespace: str | None,
    local_port: int,
    open_browser: bool,
) -> None:
    """Port-forward a RayCluster's dashboard. NAME is the RayCluster name."""
    import time
    import webbrowser

    from . import submit as submit_mod
    from .config import _infer_kube_namespace

    ns = namespace or _infer_kube_namespace()
    if not submit_mod.is_in_cluster():
        _preflight_or_exit(ns)

    url = f"http://localhost:{local_port}"
    pf = submit_mod._PortForward(name, ns, local_port)
    click.echo(f"[dashboard] forwarding {name} head :8265 -> {url}")
    try:
        pf.start()
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="dashboard port-forward failed")
    if open_browser:
        webbrowser.open(url)
    click.echo("[dashboard] Ctrl+C to stop.")
    try:
        while pf.alive():
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n[dashboard] stopping forward.")
    finally:
        pf.stop()


# job group

@main.group()
def job() -> None:
    """Inspect and control Ray jobs on the GPU cluster."""


@job.command("list")
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
def job_list(recipe: Path, overrides: tuple[str, ...], infra_path: Path | None) -> None:
    """List Ray Jobs currently registered on the cluster."""
    from ray.job_submission import JobSubmissionClient

    from . import submit

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster_name = _cluster_name_for_jobs(loaded)

    try:
        with submit.dashboard_url(cluster_name, loaded.infra.namespace) as dash:
            jobs = JobSubmissionClient(dash).list_jobs()
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="list jobs failed")

    if not jobs:
        click.echo(f"(no Ray jobs on {cluster_name})")
        return
    click.echo(f"{'SUBMISSION':<40} {'STATUS':<12} ENTRYPOINT")
    for j in jobs:
        entry = (j.entrypoint or "").splitlines()[0][:80] if j.entrypoint else ""
        click.echo(f"{j.submission_id:<40} {j.status.value:<12} {entry}")


@job.command("logs")
@click.argument("submission_id")
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option("-f", "--follow", is_flag=True, help="Stream new output until Ctrl+C.")
def job_logs(
    submission_id: str,
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    follow: bool,
) -> None:
    """Stream logs for a submitted run by its id.

    Dispatches on the cached handle (``~/.cache/dockyard-k8s/runs/<id>.json``):
    exec handles tail the head pod's stdout file; port-forward handles (or any
    job with no cached handle) go through Ray's log tail API.
    """
    del follow  # always follows
    from .submitters import load_handle

    loaded = _load_or_exit(recipe, overrides, infra_path)

    handle = load_handle(submission_id)
    if handle is not None and handle.kind == "exec":
        _follow_handle(handle)
        return
    _tail_job(_cluster_name_for_jobs(loaded), loaded.infra.namespace, submission_id)


@job.command("stop")
@click.argument("submission_id")
@_RECIPE_ARG
@_OVERRIDES_ARG
@_INFRA_OPT
@click.option("--force", is_flag=True, help="Exec mode only: SIGKILL instead of SIGTERM.")
def job_stop(
    submission_id: str,
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    force: bool,
) -> None:
    """Stop a submitted run by id (transport-aware via the cached handle)."""
    from .submitters import load_handle

    loaded = _load_or_exit(recipe, overrides, infra_path)

    handle = load_handle(submission_id)
    if handle is not None and handle.kind == "exec":
        from .submitters.exec_ import ExecSubmitter

        tmp_root = (handle.tmp_dir or "/tmp/dockyard-x").rsplit("/", 1)[0] or "/tmp"
        try:
            ExecSubmitter(exec_tmp_dir=tmp_root).stop(handle, force=force)
        except Exception as exc:  # noqa: BLE001
            _explain_and_exit(exc, context=f"stop {submission_id} failed")
        click.echo(f"stopped {submission_id} (exec)")
        return

    from ray.job_submission import JobSubmissionClient

    from . import submit

    try:
        with submit.dashboard_url(
            _cluster_name_for_jobs(loaded), loaded.infra.namespace
        ) as dash:
            JobSubmissionClient(dash).stop_job(submission_id)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"stop {submission_id} failed")
    click.echo(f"stopped {submission_id}")


# dev group

@main.group()
def dev() -> None:
    """Manage a lightweight CPU dev pod on the cluster."""


@dev.command("connect")
@click.option("--image", default=None, help="Container image for the dev pod (default: ubuntu-swe).")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_connect(image: str | None, namespace: str | None) -> None:
    """Create a dev pod (if needed) and exec into it."""
    import time

    from . import k8s
    from .config import _infer_kube_namespace, get_username
    from .dev import DEFAULT_IMAGE, build_dev_pod_manifest

    img = image or DEFAULT_IMAGE
    user = get_username()
    pod_name = f"{user}-dev-pod"
    ns = namespace or _infer_kube_namespace()

    phase = k8s.get_pod_phase(pod_name, ns)
    if phase is None:
        click.echo(f"creating dev pod {pod_name} in {ns} ...")
        k8s.create_pod(build_dev_pod_manifest(user, ns, img), ns)
        phase = "Pending"
    else:
        running_image = k8s.get_pod_image(pod_name, ns)
        if running_image and running_image != img:
            click.echo(
                f"warning: dev pod is using image {running_image}, not {img} — "
                f"stop and reconnect to switch",
                err=True,
            )

    if phase != "Running":
        click.echo(f"waiting for {pod_name} to be Running ...")
        for _ in range(120):
            time.sleep(2)
            phase = k8s.get_pod_phase(pod_name, ns)
            if phase == "Running":
                break
            if phase in ("Failed", "Succeeded"):
                _cli_error(
                    f"dev pod reached phase {phase} — check "
                    f"`kubectl describe pod {pod_name} -n {ns}`"
                )
        else:
            _cli_error(f"dev pod did not reach Running after 240s (phase={phase})")

    click.echo(f"connecting to {pod_name} ...")
    subprocess.run(["kubectl", "exec", "-it", "-n", ns, pod_name, "--", "bash"])
    click.echo(f"\npod {pod_name} is still running — stop with: dockyard-k8s dev stop")


@dev.command("stop")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_stop(namespace: str | None) -> None:
    """Delete your dev pod."""
    from . import k8s
    from .config import _infer_kube_namespace, get_username

    user = get_username()
    pod_name = f"{user}-dev-pod"
    ns = namespace or _infer_kube_namespace()

    if k8s.get_pod_phase(pod_name, ns) is None:
        click.echo(f"no dev pod {pod_name} found in {ns}")
        return
    click.echo(f"deleting {pod_name} ...")
    k8s.delete_pod(pod_name, ns)
    click.echo(f"{pod_name} deleted.")


_REQUIRED_FIRST_TIME = ("HF_TOKEN", "WANDB_API_KEY")


@dev.command("setup-secrets")
@click.argument("kvs", nargs=-1)
@click.option("--ssh-key", type=click.Path(exists=True), help="Path to an SSH private key.")
@click.option("--add-rclone", is_flag=True, help="Store ~/.config/rclone/rclone.conf in the secret.")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_setup_secrets(
    kvs: tuple[str, ...],
    ssh_key: str | None,
    add_rclone: bool,
    namespace: str | None,
) -> None:
    r"""Create or update your user secrets.

    Pass token values as NAME=VAL positional args and SSH keys via --ssh-key.
    First-time usage requires HF_TOKEN, WANDB_API_KEY, and --ssh-key.
    """
    from . import k8s
    from .config import _infer_kube_namespace, get_username

    user = get_username()
    secret_name = f"{user}-secrets"
    ns = namespace or _infer_kube_namespace()

    data: dict[str, str] = {}
    for kv in kvs:
        if "=" not in kv:
            _cli_error(f"invalid argument {kv!r} — expected NAME=VAL")
        k, v = kv.split("=", 1)
        data[k] = v

    if ssh_key:
        p = Path(ssh_key)
        data["SSH_KEY_NAME"] = p.name
        data["SSH_KEY_CONTENT"] = p.read_text()

    if add_rclone:
        rclone_conf = Path.home() / ".config" / "rclone" / "rclone.conf"
        if not rclone_conf.exists():
            _cli_error(f"rclone config not found at {rclone_conf}")
        data["RCLONE_CONF"] = rclone_conf.read_text()

    is_new = not k8s.secret_exists(secret_name, ns)
    if is_new:
        missing = [k for k in _REQUIRED_FIRST_TIME if k not in data]
        if missing:
            _cli_error(f"first-time setup requires: {', '.join(missing)}")
        if not ssh_key:
            _cli_error("first-time setup requires --ssh-key")

    k8s.create_or_update_secret(secret_name, ns, data)
    action = "created" if is_new else "updated"
    click.echo(f"{action} secret {secret_name} in {ns} (keys: {', '.join(sorted(data))})")


# doctor

@main.command()
@click.option("--namespace", "-n", default=None, help="Namespace (default: kube context).")
def doctor(namespace: str | None) -> None:
    """Check cluster access: kubectl, auth, and KubeRay CRDs."""
    from . import k8s
    from . import submit as submit_mod
    from .config import _infer_kube_namespace

    ns = namespace or _infer_kube_namespace()
    ok = True

    try:
        submit_mod.kubectl_preflight(ns)
        click.echo(f"[ok] kubectl reachable + authorized in namespace {ns}")
    except RuntimeError as exc:
        ok = False
        click.echo(f"[fail] {exc}")

    try:
        k8s.load_kubeconfig()
        from kubernetes import client as kclient  # type: ignore[import-not-found]

        crds: Any = kclient.ApiextensionsV1Api().list_custom_resource_definition()
        names = {c.metadata.name for c in crds.items}
        for crd in ("rayclusters.ray.io", "rayjobs.ray.io"):
            if crd in names:
                click.echo(f"[ok] CRD present: {crd}")
            else:
                ok = False
                click.echo(f"[fail] CRD missing: {crd} (install the KubeRay operator)")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"[warn] could not list CRDs: {exc}")

    sys.exit(0 if ok else 1)


# Mode resolution

def _resolve_mode_defaults(
    *,
    cli_mode: str | None,
    infra_mode: RunMode,
    cli_submitter: str | None,
    cli_code_source: str | None,
    cli_wait: bool | None,
) -> tuple[RunMode, SubmitterMode, CodeSource, bool]:
    mode = RunMode(cli_mode) if cli_mode else infra_mode
    default_submitter, default_code_src, default_no_wait = _MODE_DEFAULTS[mode]
    submitter = SubmitterMode(cli_submitter) if cli_submitter else default_submitter
    code_src = CodeSource(cli_code_source) if cli_code_source else default_code_src
    no_wait = default_no_wait if cli_wait is None else (not cli_wait)
    return mode, submitter, code_src, no_wait


def _apply_mode_overrides(
    loaded: LoadedConfig,
    *,
    submitter: SubmitterMode,
    code_source: CodeSource,
    code_path: str | None,
) -> None:
    """Push the resolved submitter/codeSource into the loaded InfraConfig."""
    infra = loaded.infra
    infra.submit.submitter = submitter
    infra.launch.codeSource = code_source
    if code_path is not None:
        infra.launch.codePath = code_path
    if code_source in (CodeSource.IMAGE, CodeSource.LUSTRE) and not infra.launch.codePath:
        _cli_error(
            f"--code-source {code_source.value} requires --code-path (or infra.launch.codePath)",
            hint="pass --code-path /opt/dockyard_rl or a Lustre mount path",
        )


# check/render helpers

def _load(
    recipe: Path, overrides: tuple[str, ...], infra_path: Path | None
) -> LoadedConfig:
    try:
        return load_recipe_with_infra(recipe, list(overrides), infra_path=infra_path)
    except Exception as exc:  # surface config errors as clean CLI failures
        raise click.ClickException(str(exc)) from exc


def _write_bundle(loaded: LoadedConfig, manifests: list[dict], output: Path) -> None:
    bundle = {
        "infra": loaded.infra.model_dump(mode="json"),
        "recipe": OmegaConf.to_container(loaded.recipe, resolve=True),
        "manifests": manifests,
    }
    if output.suffix == ".json":
        output.write_text(json.dumps(bundle, indent=2))
    else:
        output.write_text(yaml.safe_dump(bundle, sort_keys=False))


def _print_summary(loaded: LoadedConfig, manifests: list[dict]) -> None:
    infra = loaded.infra
    out = click.echo
    out(f"recipe:     {loaded.source_path}")
    out(f"namespace:  {infra.namespace}")
    out(f"image:      {infra.image}")
    out(f"scheduler:  {infra.scheduler.kind.value}"
        + (f" (queue={infra.scheduler.queue})" if infra.scheduler.queue else ""))
    out(f"launch:     mode={infra.launch.mode.value} "
        f"runMode={infra.launch.runMode.value} codeSource={infra.launch.codeSource.value}")
    if infra.kuberay is not None:
        groups = infra.kuberay.spec.get("workerGroupSpecs") or []
        names = ", ".join(g.get("groupName", "?") for g in groups) or "(none)"
        out(f"kuberay:    {infra.kuberay.name}  workerGroups=[{names}]"
            + (f"  segmentSize={infra.kuberay.segmentSize}"
               if infra.kuberay.segmentSize else ""))
    else:
        out("kuberay:    (none — attach/external)")
    if infra.sandbox is not None:
        sb = infra.sandbox
        out(f"sandbox:    {sb.name}  replicas={sb.replicas}  port={sb.port}")
        if sb.injectUrls and infra.launch.mode is not LaunchMode.ATTACH:
            out(f"            DOCKYARD_SANDBOX_URLS -> {sandbox_service_dns(sb, infra)}")
    else:
        out("sandbox:    (none — external; set DOCKYARD_SANDBOX_URLS in launch.env)")
    if infra.launch.entrypoint:
        out("entrypoint:")
        for line in infra.launch.entrypoint.rstrip().splitlines():
            out(f"  {line}")
    kinds = [f"{m['kind']}/{m['metadata']['name']}" for m in manifests]
    out(f"manifests:  {', '.join(kinds) if kinds else '(none)'}")


# Live-command helpers

def _load_or_exit(
    recipe: Path, overrides: tuple[str, ...], infra_path: Path | None = None
) -> LoadedConfig:
    try:
        return load_recipe_with_infra(recipe, list(overrides), infra_path=infra_path)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="failed to load recipe")


def _pick_cluster_or_exit(loaded: LoadedConfig):
    cluster = loaded.infra.kuberay
    if cluster is None:
        _cli_error(
            f"infra.kuberay is not defined in {loaded.source_path}",
            hint="declare a `kuberay` block in the recipe/infra",
        )
    return cluster


def _cluster_name_for_jobs(loaded: LoadedConfig) -> str:
    """Resolve the cluster name job commands talk to (attach target or kuberay)."""
    if loaded.infra.launch.mode is LaunchMode.ATTACH and loaded.infra.launch.attach:
        return loaded.infra.launch.attach
    return _pick_cluster_or_exit(loaded).name


def _first_worker_pod_or_exit(cluster_name: str, namespace: str) -> str:
    from . import inspect as ins

    pods = ins.list_cluster_pods(cluster_name, namespace)
    if not pods.worker_names:
        _cli_error(
            f"no worker pods for {cluster_name} in {namespace}",
            hint="is the RayCluster still scheduling? check `dockyard-k8s status` first.",
        )
    return pods.worker_names[0]


def _check_head_svc_collision(name: str, namespace: str, *, creating: str) -> None:
    """Fail if creating this resource would collide with an existing one's head-svc.

    KubeRay derives the head Service name as ``{name}-head-svc`` for both
    RayJobs and RayClusters. When both exist with the same metadata name the
    second resource can never create its Service and silently hangs.
    """
    from . import k8s

    if creating == "rayjob":
        existing = k8s.get_raycluster(name, namespace)
        other_kind = "raycluster"
    else:
        existing = k8s.get_rayjob(name, namespace)
        other_kind = "rayjob"

    if existing is None:
        return
    _cli_error(
        f"a {other_kind} named '{name}' already exists in namespace {namespace}. "
        f"KubeRay names the head Service '{name}-head-svc' for both resource types; "
        f"creating this {creating} with the same name will collide and hang.",
        hint=f"delete the existing resource:\n"
        f"  kubectl delete {other_kind} {name} -n {namespace}\n"
        f"or use a different name for the {creating}",
    )


def _preflight_or_exit(namespace: str) -> None:
    from . import submit

    try:
        submit.kubectl_preflight(namespace)
    except RuntimeError as exc:
        _cli_error(str(exc), hint="see `dockyard-k8s doctor` for cluster access checks")


def _emit_handle(handle, recipe: Path) -> None:  # type: ignore[no-untyped-def]
    click.echo(f"run id:  {handle.run_id}")
    click.echo(f"kind:    {handle.kind}")
    click.echo(f"cluster: {handle.cluster_name}  (ns={handle.namespace})")
    if handle.kind == "exec":
        click.echo(f"pod:     {handle.pod}")
        click.echo(f"tmp:     {handle.tmp_dir}")
    click.echo(f"follow:  dockyard-k8s job logs {handle.run_id} {recipe} -f")
    click.echo(f"stop:    dockyard-k8s job stop {handle.run_id} {recipe}")


def _follow_handle(handle) -> None:  # type: ignore[no-untyped-def]
    """Stream logs for a handle using whichever transport submitted it."""
    from .submitters import SubmissionHandle  # noqa: F401  (type clarity)

    if handle.kind == "exec":
        from .submitters.exec_ import ExecSubmitter

        tmp_root = (handle.tmp_dir or "/tmp/dockyard-x").rsplit("/", 1)[0] or "/tmp"
        submitter = ExecSubmitter(exec_tmp_dir=tmp_root)
    else:
        from .submitters.portforward import PortForwardSubmitter

        submitter = PortForwardSubmitter()
    try:
        for line in submitter.follow(handle):
            click.echo(line, nl=False)
    except KeyboardInterrupt:
        click.echo("\n(interrupted — run continues)", err=True)


def _tail_job(cluster_name: str, namespace: str, submission_id: str) -> None:
    """Open a dashboard port-forward and tail a Ray Job by submission_id."""
    from . import submit as submit_mod

    try:
        with submit_mod.dashboard_url(cluster_name, namespace) as dash:
            click.echo(f"# tailing {submission_id} via {dash}", err=True)
            for line in submit_mod.tail_job_logs(dash, submission_id):
                click.echo(line, nl=False)
    except KeyboardInterrupt:
        click.echo("\n(interrupted — job continues running)", err=True)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"tailing {submission_id} failed")


def _cli_error(msg: str, *, hint: str | None = None, exit_code: int = 1) -> NoReturn:
    click.echo(f"error: {msg}", err=True)
    if hint:
        click.echo(f"hint: {hint}", err=True)
    sys.exit(exit_code)


def _explain_and_exit(exc: BaseException, *, context: str) -> NoReturn:
    hint: str | None = None
    if isinstance(exc, ApiException):
        if exc.status == 403:
            hint = (
                "missing RBAC for this action; run `dockyard-k8s doctor` or ask an "
                "admin to grant the edit role on the namespace."
            )
        elif exc.status == 401:
            hint = "kubectl credentials rejected; re-authenticate."
        elif exc.status in (500, 502, 503, 504):
            hint = "control-plane 5xx — retry in a few seconds."
    elif isinstance(exc, ConnectionRefusedError):
        hint = (
            "connection refused — kubectl port-forward to the dashboard failed; "
            "is kubectl authenticated?"
        )
    elif isinstance(exc, ValueError) and "launch.entrypoint" in str(exc):
        hint = "set infra.launch.entrypoint in your recipe/infra."
    _cli_error(f"{context}: {exc}", hint=hint)


if __name__ == "__main__":
    main()


__all__ = ["main"]
