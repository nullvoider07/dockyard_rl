import os
import sys
from typing import Any, NotRequired, TypedDict
from dockyard_rl.environments.interfaces import EnvironmentInterface

# Environment registry entry schema.
class EnvRegistryEntry(TypedDict):
    # Removing total=False makes actor_class_fqn strictly required by default
    actor_class_fqn: str
    default_processor: NotRequired[str]

# Environment registry. Key is the env name, value is a dict with the actor class FQN.
ENV_REGISTRY: dict[str, EnvRegistryEntry] = {
    "code": {
        "actor_class_fqn": "dockyard_rl.environments.code_environment.CodeEnvironment",
    },
    "math": {
        "actor_class_fqn": "dockyard_rl.environments.math_environment.MathEnvironment",
        "default_processor": "math_hf_data_processor",
    },
    "terminal_bench": {
        "actor_class_fqn": "dockyard_rl.environments.terminal_bench_environment.TerminalBenchEnvironment",
        "default_processor": "terminal_bench_data_processor",
    },
    "program_bench": {
        "actor_class_fqn": "dockyard_rl.environments.program_bench_environment.ProgramBenchEnvironment",
        "default_processor": "program_bench_data_processor",
    },
    "hle": {
        "actor_class_fqn": "dockyard_rl.environments.hle_environment.HLEEnvironment",
        "default_processor": "hle_data_processor",
    },
    "gdpval": {
        "actor_class_fqn": "dockyard_rl.environments.gdpval_environment.GDPvalEnvironment",
        "default_processor": "gdpval_data_processor",
    },
    "gdpval_agentic": {
        "actor_class_fqn": "dockyard_rl.environments.gdpval_agentic_environment.GDPvalAgenticEnvironment",
        "default_processor": "gdpval_agentic_data_processor",
    },
    "osworld": {
        "actor_class_fqn": "dockyard_rl.experience.cua.environment.OSWorldEnvironment",
        "default_processor": "osworld_data_processor",
    },
}

def chunk_list_to_workers(to_chunk: list[Any], num_workers: int) -> list[list[Any]]:
    """Chunk a list into a list of lists, where each sublist is assigned to a worker. Keeps ordering of elements.

    If the list is not divisible by the number of workers, the last worker may have fewer elements.
    If there are more workers than elements, the first len(list) workers will have a single element each,
    and the remaining workers will have empty lists.

    Args:
        to_chunk: The list to be chunked.
        num_workers: The number of workers to distribute the list to.

    Returns:
        A list of lists, where each sublist contains elements assigned to a worker.

    Examples:
    ```{doctest}
    >>> from dockyard_rl.environments.utils import chunk_list_to_workers
    >>> chunk_list_to_workers([1, 2, 3, 4, 5], 3)
    [[1, 2], [3, 4], [5]]
    ```
    """
    if not to_chunk:
        return [[] for _ in range(num_workers)]

    # Handle case where we have more workers than elements
    if len(to_chunk) <= num_workers:
        result = [[item] for item in to_chunk]
        result.extend([[] for _ in range(num_workers - len(to_chunk))])
        return result

    # Calculate chunk size (ceiling division to ensure all elements are covered)
    chunk_size = (len(to_chunk) + num_workers - 1) // num_workers

    # Create chunks
    chunks = []
    for i in range(0, len(to_chunk), chunk_size):
        chunks.append(to_chunk[i : i + chunk_size])

    # If we somehow ended up with more chunks than workers (shouldn't happen with ceiling division)
    # merge the last chunks
    if len(chunks) > num_workers:
        chunks[num_workers - 1 :] = [sum(chunks[num_workers - 1 :], [])]

    return chunks

def create_env(env_name: str, env_config: dict) -> EnvironmentInterface:
    """Instantiate a registered environment as a Ray remote actor.

    Uses sys.executable for the actor runtime environment, consistent with
    the dockyard_rl convention of no uv/venv management.

    Args:
        env_name: Key in ENV_REGISTRY.
        env_config: Config dict forwarded to the actor constructor.

    Returns:
        A Ray actor handle implementing EnvironmentInterface.
    """
    assert env_name in ENV_REGISTRY, (
        f"Env name {env_name} is not registered in ENV_REGISTRY. "
        "Call register_env() to register the environment."
    )
    actor_class_fqn = ENV_REGISTRY[env_name]["actor_class_fqn"]

    # Resolve the actor class from its fully-qualified name without Hydra.
    module_path, cls_name = actor_class_fqn.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    actor_class = getattr(mod, cls_name)

    options_kwargs: dict[str, Any] = {
        "runtime_env": {
            "py_executable": sys.executable,
            "env_vars": dict(os.environ),
        }
    }
    # Multi-turn environments need the actor to service concurrent step() calls
    # (a Ray actor serializes by default). Opt in via env_config.max_concurrency.
    max_concurrency = env_config.get("max_concurrency")
    if max_concurrency:
        options_kwargs["max_concurrency"] = int(max_concurrency)

    # Pin the actor to nodes advertising custom Ray resources — e.g.
    # {"kvm": 0.001} lands the OSWorld official-backend actor (which runs a
    # KVM-in-docker DesktopEnv locally) only on KVM-capable sandbox nodes that
    # declare a "kvm" resource (see cluster/bootstrap.py).
    node_resources = env_config.get("node_resources")
    if node_resources:
        options_kwargs["resources"] = {
            str(k): float(v) for k, v in node_resources.items()
        }
    num_cpus = env_config.get("num_cpus")
    if num_cpus is not None:
        options_kwargs["num_cpus"] = float(num_cpus)

    env = actor_class.options(  # type: ignore  # wrapped with ray.remote
        **options_kwargs
    ).remote(env_config)
    return env

def register_env(env_name: str, actor_class_fqn: str) -> None:
    """Register a new environment in the global registry.

    Args:
        env_name: The name to register under.
        actor_class_fqn: Fully-qualified class name of the Ray remote actor.

    Raises:
        ValueError: If the name is already registered.
    """
    if env_name in ENV_REGISTRY:
        raise ValueError(f"Env name {env_name} already registered")

    ENV_REGISTRY[env_name] = {"actor_class_fqn": actor_class_fqn}