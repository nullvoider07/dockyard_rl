"""Bring-up + submit + teardown for a dockyard_rl run.

``dockyard-k8s run`` delegates here for the long-lived paths. The topology is
one GPU RayCluster (trainer + inference worker groups) plus an optional
GPU-free sandbox executor pool (Deployment + Service). There is no Gym,
endpoint-registry ConfigMap, daemon, or DRA in this stack.

Dispatch is on ``infra.launch.mode``:

* ``bringup`` — apply the sandbox pool (if declared), bring up the GPU
  RayCluster, wait for ``state=ready``, then submit the training entrypoint
  via the configured transport. The cluster stays up for re-runs.
* ``attach`` — submit onto an existing RayCluster named ``infra.launch.attach``
  (the sandbox pool is still ensured if declared).

The ephemeral ``rayjob`` mode is handled directly by the CLI (KubeRay owns the
RayCluster lifecycle), not here.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from . import k8s, submit, workdir
from .config import LoadedConfig, get_username
from .manifest import (
    build_compute_domain_manifest,
    build_pdb,
    build_raycluster_manifest,
    build_roce_template_manifest,
    build_sandbox_deployment,
    build_sandbox_network_policy,
    build_sandbox_service,
    dra_resources_for_cluster,
    sandbox_service_dns,
)
from .schema import ClusterSpec, CodeSource, InfraConfig, LaunchMode, SubmitterMode
from .submitters import SubmissionHandle, build_submitter, save_handle

# Staged copy of the resolved recipe, dropped at the working_dir root in
# upload mode so the in-pod entrypoint reads exactly what the CLI resolved.
STAGED_RECIPE_NAME = "dockyard_run.yaml"


@dataclass
class RunResult:
    """Outcome of a training submission."""

    handle: SubmissionHandle
    training_job_id: str = ""


def _fresh_submission_id(base: str) -> str:
    return f"{base}-{int(time.time())}"


# Sandbox executor pool

def ensure_sandbox(loaded: LoadedConfig, *, log: Callable[[str], None]) -> str | None:
    """Apply the sandbox Deployment + Service and wait for it to be ready."""
    infra = loaded.infra
    sandbox = infra.sandbox
    if sandbox is None:
        return None
    namespace = infra.namespace
    name = sandbox.name

    live = k8s.get_deployment(name, namespace)
    if live is not None:
        live_owner = ""
        if getattr(live, "metadata", None) and live.metadata.labels:
            live_owner = live.metadata.labels.get("dockyard-k8s/owner", "")
        me = get_username()
        if live_owner and live_owner != me:
            raise RuntimeError(
                f"Deployment {name} in namespace {namespace} is owned by "
                f"'{live_owner}' (you are '{me}'). Use a different sandbox "
                f"name or ask {live_owner} to tear it down."
            )

    log(f"[sandbox] applying Deployment {name} in namespace {namespace}")
    k8s.apply_deployment(build_sandbox_deployment(sandbox, infra), namespace)
    log(f"[sandbox] applying ClusterIP Service {name}")
    k8s.apply_service(build_sandbox_service(sandbox, infra), namespace)
    if sandbox.networkPolicy.enabled:
        log(f"[sandbox] applying NetworkPolicy {name}-netpol")
        k8s.apply_network_policy(
            build_sandbox_network_policy(sandbox, infra), namespace
        )
    if sandbox.pdb.enabled:
        log(f"[sandbox] applying PodDisruptionBudget {name}-pdb")
        k8s.apply_pod_disruption_budget(
            build_pdb(
                f"{name}-pdb",
                namespace,
                {"app.kubernetes.io/name": name},
                sandbox.pdb,
                infra,
            ),
            namespace,
        )
    log(f"[sandbox] DNS: {sandbox_service_dns(sandbox, infra)}")

    log(f"[sandbox] waiting for Deployment {name} to be ready ...")
    k8s.wait_for_deployment_ready(
        name, namespace, timeout_s=sandbox.healthCheckTimeoutS
    )
    log(f"[sandbox] Deployment {name} is ready.")
    return name


def delete_sandbox(loaded: LoadedConfig, *, log: Callable[[str], None]) -> None:
    """Delete the sandbox Service + Deployment."""
    infra = loaded.infra
    if infra.sandbox is None:
        return
    sandbox = infra.sandbox
    name = sandbox.name
    namespace = infra.namespace
    if sandbox.pdb.enabled:
        log(f"[sandbox] deleting PodDisruptionBudget {name}-pdb")
        k8s.delete_pod_disruption_budget(f"{name}-pdb", namespace)
    if sandbox.networkPolicy.enabled:
        log(f"[sandbox] deleting NetworkPolicy {name}-netpol")
        k8s.delete_network_policy(f"{name}-netpol", namespace)
    log(f"[sandbox] deleting Service {name}")
    k8s.delete_service(name, namespace)
    log(f"[sandbox] deleting Deployment {name}")
    k8s.delete_deployment(name, namespace)


# DRA: ComputeDomain + RoCE ResourceClaimTemplate

def ensure_dra_resources(loaded: LoadedConfig, *, log: Callable[[str], None]) -> None:
    """Auto-create the ComputeDomain / RoCE CRDs the worker claims reference."""
    infra = loaded.infra
    if infra.kuberay is None or not infra.dra.autoCreate:
        return
    namespace = infra.namespace
    for kind, name in dra_resources_for_cluster(infra.kuberay.name, infra.kuberay.spec):
        if kind == "compute-domain":
            log(f"[dra] ensuring ComputeDomain {name}")
            k8s.apply_compute_domain(
                build_compute_domain_manifest(
                    name, namespace, num_nodes=infra.dra.computeDomainNumNodes
                ),
                namespace,
            )
        elif kind == "roce":
            log(f"[dra] ensuring RoCE ResourceClaimTemplate {name}")
            k8s.apply_resource_claim_template(
                build_roce_template_manifest(name, namespace, count=infra.dra.roceCount),
                namespace,
            )


def delete_dra_resources(loaded: LoadedConfig, *, log: Callable[[str], None]) -> None:
    """Delete the auto-created DRA CRDs (reverse of :func:`ensure_dra_resources`)."""
    infra = loaded.infra
    if infra.kuberay is None or not infra.dra.autoCreate:
        return
    namespace = infra.namespace
    resources = dra_resources_for_cluster(infra.kuberay.name, infra.kuberay.spec)
    for kind, name in reversed(resources):
        if kind == "roce":
            log(f"[dra] deleting RoCE ResourceClaimTemplate {name}")
            k8s.delete_resource_claim_template(name, namespace)
        elif kind == "compute-domain":
            log(f"[dra] deleting ComputeDomain {name}")
            k8s.delete_compute_domain(name, namespace)


# GPU RayCluster disruption budget
#
# Selects the cluster's pods by KubeRay's ray.io/cluster=<name> label. Only
# applied for the bringup path where the RayCluster name is known (rayjob mode
# lets KubeRay generate the name).

def ensure_cluster_pdb(loaded: LoadedConfig, *, log: Callable[[str], None]) -> None:
    infra = loaded.infra
    cluster = infra.kuberay
    if cluster is None or not cluster.pdb.enabled:
        return
    name = f"{cluster.name}-pdb"
    log(f"[kuberay] applying PodDisruptionBudget {name}")
    k8s.apply_pod_disruption_budget(
        build_pdb(
            name,
            infra.namespace,
            {"ray.io/cluster": cluster.name},
            cluster.pdb,
            infra,
        ),
        infra.namespace,
    )


def delete_cluster_pdb(loaded: LoadedConfig, *, log: Callable[[str], None]) -> None:
    infra = loaded.infra
    cluster = infra.kuberay
    if cluster is None or not cluster.pdb.enabled:
        return
    name = f"{cluster.name}-pdb"
    log(f"[kuberay] deleting PodDisruptionBudget {name}")
    k8s.delete_pod_disruption_budget(name, infra.namespace)


# GPU RayCluster

def bring_up_cluster(
    loaded: LoadedConfig,
    *,
    log: Callable[[str], None],
    wait_ready: bool = True,
    ready_timeout_s: int = 900,
) -> str:
    """Apply the GPU RayCluster and (optionally) wait for it to be ready."""
    cluster = _require_cluster(loaded.infra)
    manifest = build_raycluster_manifest(cluster, loaded.infra)
    name = cluster.name
    namespace = loaded.infra.namespace

    ensure_dra_resources(loaded, log=log)
    log(f"[kuberay] applying RayCluster {name} in namespace {namespace}")
    k8s.apply_raycluster(manifest, namespace)
    ensure_cluster_pdb(loaded, log=log)

    if wait_ready:
        log(f"[kuberay] waiting for RayCluster {name} to reach state=ready ...")
        k8s.wait_for_raycluster_ready(name, namespace, timeout_s=ready_timeout_s)
        log(f"[kuberay] RayCluster {name} is ready.")
    return name


def ensure_cluster(
    loaded: LoadedConfig,
    *,
    log: Callable[[str], None],
    recreate: bool = False,
    wait_ready: bool = True,
    ready_timeout_s: int = 900,
) -> str:
    """Idempotent cluster up: reuse when live matches rendered, warn on drift.

    Never silently patches a live cluster. If it exists and its spec matches
    the rendered one, just wait for readiness. If it drifted, warn and reuse —
    pass ``recreate=True`` to delete + re-apply instead.
    """
    cluster = _require_cluster(loaded.infra)
    manifest = build_raycluster_manifest(cluster, loaded.infra)
    name = cluster.name
    namespace = loaded.infra.namespace

    live = k8s.get_raycluster(name, namespace)
    if live is not None:
        live_owner = (live.get("metadata", {}).get("labels") or {}).get(
            "dockyard-k8s/owner"
        )
        me = get_username()
        if live_owner and live_owner != me:
            raise RuntimeError(
                f"RayCluster {name} in namespace {namespace} is owned by "
                f"'{live_owner}' (you are '{me}'). Use a different cluster "
                f"name or ask {live_owner} to tear it down."
            )

    ensure_dra_resources(loaded, log=log)
    if live is None:
        log(f"[kuberay] applying RayCluster {name} in namespace {namespace}")
        k8s.apply_raycluster(manifest, namespace)
    elif _spec_drifted(live.get("spec") or {}, manifest["spec"]):
        if recreate:
            log(
                f"[kuberay] --recreate: RayCluster {name} drifted from the rendered "
                f"manifest; deleting and re-applying"
            )
            k8s.delete_raycluster(name, namespace)
            k8s.wait_for_raycluster_gone(name, namespace)
            k8s.apply_raycluster(manifest, namespace)
        else:
            log(
                f"[kuberay] warning: live RayCluster {name} drifted from the rendered "
                f"manifest; reusing as-is (pass --recreate to replace)"
            )
    else:
        log(f"[kuberay] RayCluster {name} already exists and matches — reusing")

    ensure_cluster_pdb(loaded, log=log)

    if wait_ready:
        log(f"[kuberay] waiting for RayCluster {name} to reach state=ready ...")
        k8s.wait_for_raycluster_ready(name, namespace, timeout_s=ready_timeout_s)
        log(f"[kuberay] RayCluster {name} is ready.")
    return name


# Server-managed fields that never appear in a rendered manifest, ignored when
# diffing for drift.
_DRIFT_IGNORE_TOP = ("status",)
_DRIFT_IGNORE_METADATA = (
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
)


def _spec_drifted(live_spec: dict, rendered_spec: dict) -> bool:
    return _strip_server_fields(live_spec) != _strip_server_fields(rendered_spec)


def _strip_server_fields(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_server_fields(v)
            for k, v in obj.items()
            if k not in _DRIFT_IGNORE_TOP and k not in _DRIFT_IGNORE_METADATA
        }
    if isinstance(obj, list):
        return [_strip_server_fields(v) for v in obj]
    return obj


# Training submission

def submit_training(
    loaded: LoadedConfig,
    cluster_name: str,
    *,
    log: Callable[[str], None],
    repo_root: Path,
    replace: bool = False,
    run_id: str | None = None,
) -> RunResult:
    """Submit the training entrypoint against ``cluster_name``.

    Dispatches on ``infra.submit.submitter`` + ``infra.launch.codeSource``:

    * ``portForward`` + ``upload`` — stage a working_dir, open a port-forward,
      submit via the Ray SDK.
    * ``portForward`` + ``image``/``lustre`` — no staging; the Ray job inherits
      the head pod's cwd. The entrypoint is responsible for ``cd`` + ``source``.
    * ``exec`` — ``kubectl exec`` into the head, run the entrypoint under
      ``nohup``. No staging, no port-forward; ``codeSource`` must not be upload.
    """
    infra = loaded.infra
    launch = infra.launch
    if not launch.entrypoint:
        raise ValueError("infra.launch.entrypoint must be set for `dockyard-k8s run`")

    submitter = build_submitter(infra)
    is_exec = infra.submit.submitter is SubmitterMode.EXEC
    upload = launch.codeSource is CodeSource.UPLOAD

    if is_exec and upload:
        raise ValueError(
            "infra.submit.submitter=exec is incompatible with "
            "infra.launch.codeSource=upload — pick image or lustre, or switch "
            "submitter to portForward."
        )

    wd: Path | None = None
    if upload:
        log("[training] staging working_dir ...")
        recipe_yaml = OmegaConf.to_yaml(loaded.recipe)
        wd = workdir.stage_workdir(
            repo_root,
            include_paths=_upload_paths(infra),
            extra_files={STAGED_RECIPE_NAME: recipe_yaml},
        )

    # `--replace`: stop any running job on the cluster so the new one can claim
    # GPUs and worker actors are cleaned up. Always go through the Ray dashboard
    # (the only reliable way to tear down vLLM/actor state on workers); the exec
    # path additionally kills the driver process group on the head pod.
    if replace:
        if is_exec:
            from .submitters.exec_ import ExecSubmitter

            ExecSubmitter(exec_tmp_dir=infra.submit.execTmpDir).stop_all_running(
                cluster_name, infra.namespace, log=log
            )
        _stop_running_jobs(cluster_name, infra.namespace, log=log)

    if is_exec:
        run_id = run_id or default_run_id("training")
        log(f"[training] exec submitter: launching as run_id={run_id} on head pod")
    else:
        log("[training] port-forward submitter: submitting Ray Job")

    # Expose the run id to the entrypoint via both transports so recipe authors
    # can reference $DOCKYARD_K8S_RUN_ID (logger.wandb.name etc.) uniformly.
    env_vars = {**dict(launch.env)}
    if run_id:
        env_vars.setdefault("DOCKYARD_K8S_RUN_ID", run_id)

    entrypoint = _rewrite_entrypoint_recipe(
        launch.entrypoint, loaded, repo_root, upload=upload, log=log
    )

    try:
        handle = submitter.submit(
            cluster_name,
            infra.namespace,
            entrypoint=entrypoint,
            run_id=run_id or "",
            env_vars=env_vars,
            working_dir=wd,
        )
    finally:
        if wd is not None:
            shutil.rmtree(wd, ignore_errors=True)
    save_handle(handle)
    log(f"[training] run handle: kind={handle.kind} id={handle.run_id}")
    return RunResult(handle=handle, training_job_id=handle.run_id)


def run(
    loaded: LoadedConfig,
    *,
    log: Callable[[str], None],
    repo_root: Path,
    replace: bool = False,
    run_id: str | None = None,
    recreate: bool = False,
) -> RunResult:
    """Ensure sandbox + GPU cluster (bringup) or attach, then submit training."""
    infra = loaded.infra
    mode = infra.launch.mode
    if mode is LaunchMode.RAYJOB:
        raise ValueError(
            "orchestrate.run handles only bringup/attach; rayjob mode is applied "
            "directly by the CLI"
        )

    ensure_sandbox(loaded, log=log)

    if mode is LaunchMode.ATTACH:
        cluster_name = infra.launch.attach
        if not cluster_name:
            raise ValueError("infra.launch.attach is required when launch.mode=attach")
        log(f"[kuberay] attaching to existing RayCluster {cluster_name}")
    else:
        cluster_name = ensure_cluster(loaded, log=log, recreate=recreate)

    return submit_training(
        loaded,
        cluster_name,
        log=log,
        repo_root=repo_root,
        replace=replace,
        run_id=run_id,
    )


def default_run_id(role: str = "training") -> str:
    """Human-readable default id when the user didn't supply ``--run-id``."""
    return f"{role}-{int(time.time())}"


# ``--config <path>.yaml`` (optional ``=`` separator and surrounding quotes).
# The mandatory ``\s|=`` after ``--config`` avoids matching Hydra's
# ``--config-name`` / ``--config-dir`` / ``--config-path``.
_CONFIG_FLAG_RE = re.compile(r"(--config(?:\s+|=))(['\"]?)([^\s'\"]+\.ya?ml)\2")


def _recipe_path_in_pod(
    loaded: LoadedConfig, repo_root: Path, *, upload: bool
) -> str | None:
    """Translate the user's recipe path to the path the pod should read.

    * ``upload`` — the staged copy at the working_dir root.
    * ``image`` / ``lustre`` — the recipe path relative to the repo root (the
      entrypoint's ``cd`` into codePath puts it at the working directory).

    Returns ``None`` if the recipe lives outside the repo root in image/lustre
    mode, so the caller leaves the entrypoint alone.
    """
    if upload:
        return STAGED_RECIPE_NAME
    try:
        return loaded.source_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _rewrite_entrypoint_recipe(
    entrypoint: str,
    loaded: LoadedConfig,
    repo_root: Path,
    *,
    upload: bool,
    log: Callable[[str], None],
) -> str:
    """Rewrite ``--config <path>.yaml`` flags to point at the CLI's RECIPE.

    Without this, ``dockyard-k8s run RECIPE`` would run whatever recipe the
    infra entrypoint hardcoded; RECIPE would only affect client-side
    validation + Hydra override merging, never the pod. Substituting the path
    in-place makes the CLI argument authoritative. No-op when the recipe lives
    outside the repo root in image/lustre mode, or when the entrypoint has no
    ``--config <yaml>``.
    """
    new_path = _recipe_path_in_pod(loaded, repo_root, upload=upload)
    if new_path is None:
        return entrypoint

    seen: set[str] = set()

    def _sub(match: re.Match) -> str:
        flag, quote, original = match.group(1), match.group(2), match.group(3)
        if original != new_path and original not in seen:
            log(
                f"[training] rewrote `--config {original}` -> "
                f"`--config {new_path}` from CLI recipe arg"
            )
            seen.add(original)
        return f"{flag}{quote}{new_path}{quote}"

    return _CONFIG_FLAG_RE.sub(_sub, entrypoint)


def _get_cluster(infra: InfraConfig) -> ClusterSpec | None:
    return infra.kuberay


def _require_cluster(infra: InfraConfig) -> ClusterSpec:
    cluster = infra.kuberay
    if cluster is None:
        raise ValueError("infra.kuberay is not defined")
    return cluster


def _upload_paths(infra: InfraConfig) -> list[str]:
    if infra.launch.rayUploadPaths is not None:
        return list(infra.launch.rayUploadPaths)
    return list(workdir.DEFAULT_RAY_UPLOAD_PATHS)


def _stop_running_jobs(
    cluster_name: str, namespace: str, *, log: Callable[[str], None]
) -> None:
    """Stop every RUNNING Ray Job on the cluster via the dashboard."""
    from ray.job_submission import JobStatus, JobSubmissionClient

    with submit.dashboard_url(cluster_name, namespace) as dash:
        client = JobSubmissionClient(dash)
        for job in client.list_jobs():
            if job.status is JobStatus.RUNNING and job.submission_id:
                log(f"[training] --replace: stopping running job {job.submission_id}")
                try:
                    client.stop_job(job.submission_id)
                    _wait_job_stopped(client, job.submission_id, log=log)
                except Exception as exc:  # noqa: BLE001
                    log(f"[training] warning: stop failed: {exc}")


def _wait_job_stopped(
    client: Any,
    submission_id: str,
    *,
    log: Callable[[str], None],
    timeout_s: int = 60,
) -> None:
    from ray.job_submission import JobStatus

    terminal = (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.SUCCEEDED)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status = client.get_job_status(submission_id)
        except Exception:  # noqa: BLE001
            return
        if status in terminal:
            log(f"[training] previous job {submission_id} → {status.value}")
            return
        time.sleep(2)
    log(f"[training] previous job {submission_id} did not stop within {timeout_s}s; continuing")


__all__ = [
    "RunResult",
    "STAGED_RECIPE_NAME",
    "bring_up_cluster",
    "default_run_id",
    "delete_cluster_pdb",
    "delete_dra_resources",
    "delete_sandbox",
    "ensure_cluster",
    "ensure_cluster_pdb",
    "ensure_dra_resources",
    "ensure_sandbox",
    "run",
    "submit_training",
]
