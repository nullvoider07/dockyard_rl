import importlib

from dockyard_rl.data import ResponseDatasetConfig
from dockyard_rl.data.datasets.response_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
)
from dockyard_rl.data.datasets.response_datasets.response_dataset import (
    ResponseDataset,
)

# Datasets below are deferred imports — only load what's actually registered
# in DATASET_REGISTRY. Non-Dockyard datasets (math, VLM, and other
# upstream-specific formats) are kept in the registry for compatibility but
# their classes are imported lazily so they don't break if those dependencies
# aren't installed.

DATASET_REGISTRY: dict = {
    # Generic loaders — primary use in Dockyard
    "ResponseDataset": ResponseDataset,
    "openai_format": OpenAIFormatDataset,
}

# Lazily populate the rest of the upstream registry entries. Each import is
# wrapped so a missing optional dependency doesn't break the whole module.
def _try_register(name: str, module_path: str, class_name: str) -> None:
    try:
        import importlib

        mod = importlib.import_module(module_path)
        DATASET_REGISTRY[name] = getattr(mod, class_name)
    except (ImportError, AttributeError):
        pass


_try_register(
    "AIME2024",
    "dockyard_rl.data.datasets.response_datasets.aime24",
    "AIME2024Dataset",
)
_try_register(
    "avqa",
    "dockyard_rl.data.datasets.response_datasets.avqa",
    "AVQADataset",
)
_try_register(
    "clevr-cogent",
    "dockyard_rl.data.datasets.response_datasets.clevr",
    "CLEVRCoGenTDataset",
)
_try_register(
    "daily-omni",
    "dockyard_rl.data.datasets.response_datasets.daily_omni",
    "DailyOmniDataset",
)
_try_register(
    "general-conversation-jsonl",
    "dockyard_rl.data.datasets.response_datasets.general_conversations_dataset",
    "GeneralConversationsJsonlDataset",
)
_try_register(
    "DAPOMath17K",
    "dockyard_rl.data.datasets.response_datasets.dapo_math",
    "DAPOMath17KDataset",
)
_try_register(
    "DAPOMathAIME2024",
    "dockyard_rl.data.datasets.response_datasets.dapo_math",
    "DAPOMathAIME2024Dataset",
)
_try_register(
    "DeepScaler",
    "dockyard_rl.data.datasets.response_datasets.deepscaler",
    "DeepScalerDataset",
)
_try_register(
    "geometry3k",
    "dockyard_rl.data.datasets.response_datasets.geometry3k",
    "Geometry3KDataset",
)
_try_register(
    "HelpSteer3",
    "dockyard_rl.data.datasets.response_datasets.helpsteer3",
    "HelpSteer3Dataset",
)
_try_register(
    "open_assistant",
    "dockyard_rl.data.datasets.response_datasets.oasst",
    "OasstDataset",
)
_try_register(
    "OpenMathInstruct-2",
    "dockyard_rl.data.datasets.response_datasets.openmathinstruct2",
    "OpenMathInstruct2Dataset",
)
_try_register(
    "refcoco",
    "dockyard_rl.data.datasets.response_datasets.refcoco",
    "RefCOCODataset",
)
_try_register(
    "squad",
    "dockyard_rl.data.datasets.response_datasets.squad",
    "SquadDataset",
)
_try_register(
    "tulu3_sft_mixture",
    "dockyard_rl.data.datasets.response_datasets.tulu3",
    "Tulu3SftMixtureDataset",
)
_try_register(
    "gsm8k",
    "dockyard_rl.data.datasets.response_datasets.gsm8k",
    "GSM8KDataset",
)
_try_register(
    "Nemotron-Cascade-2-SFT-Math",
    "dockyard_rl.data.datasets.response_datasets.nemotron_cascade2_sft",
    "NemotronCascade2SFTMathDataset",
)
# SWE-bench style datasets — Dockyard custom
_try_register("swe_bench", "dockyard_rl.data.datasets.swe_bench", "SWEBenchDataset")
_try_register(
    "swe_bench_pro",
    "dockyard_rl.data.datasets.swe_bench_pro",
    "SWEBenchProDataset",
)
_try_register(
    "terminal_bench",
    "dockyard_rl.data.datasets.terminal_bench",
    "TerminalBenchDataset",
)
_try_register(
    "program_bench",
    "dockyard_rl.data.datasets.program_bench",
    "ProgramBenchDataset",
)
_try_register(
    "hle",
    "dockyard_rl.data.datasets.hle",
    "HLEDataset",
)
_try_register(
    "gdpval",
    "dockyard_rl.data.datasets.gdpval",
    "GDPvalDataset",
)
_try_register(
    "gdpval_agentic",
    "dockyard_rl.data.datasets.gdpval",
    "GDPvalAgenticDataset",
)
# OSWorld (CUA / computer-use) — dataset lives in the self-contained
# experience/cua package; importing it also registers osworld_data_processor.
_try_register(
    "osworld",
    "dockyard_rl.experience.cua.datasets.osworld",
    "OSWorldDataset",
)

def resolve_external_dataset_class(dataset_name: str):
    """Resolve a dotted import path (``pkg.module.ClassName``) to a class.

    Lets configs reference a custom dataset class by import path without
    registering it in DATASET_REGISTRY. The module must be importable from
    PYTHONPATH.
    """
    module_path, _, class_name = dataset_name.rpartition(".")
    if not module_path:
        raise ValueError(
            f"dataset_name={dataset_name!r} is not a dotted import path."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Could not import module {module_path!r} for dataset_name="
            f"{dataset_name!r}: {exc}"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_path!r} has no attribute {class_name!r} "
            f"(from dataset_name={dataset_name!r})."
        ) from exc


def load_response_dataset(data_config: ResponseDatasetConfig):
    """Loads response dataset.

    Resolution order for ``data_config["dataset_name"]``:

    1. A key in DATASET_REGISTRY → use the built-in class.
    2. Otherwise, a dotted import path (contains ``.``) → import the class
       dynamically via resolve_external_dataset_class.
    3. Otherwise → raise ValueError.
    """
    dataset_name = data_config.get("dataset_name")

    # load dataset
    if dataset_name in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_name]
    elif isinstance(dataset_name, str) and "." in dataset_name:
        dataset_class = resolve_external_dataset_class(dataset_name)
    else:
        raise ValueError(
            f"Unsupported {dataset_name}. Please set dataset_name to one of: "
            "(1) a built-in dataset name, "
            "(2) 'ResponseDataset' to load from a local JSONL file or HuggingFace, or "
            "(3) an importable dotted path to a dataset class."
        )

    dataset = dataset_class(
        **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
    )

    # bind prompt, system prompt and data processor
    dataset.set_task_spec(data_config)
    # Remove this after the data processor is refactored.
    dataset.set_processor()

    return dataset


__all__ = [
    "OpenAIFormatDataset",
    "ResponseDataset",
    "load_response_dataset",
    "DATASET_REGISTRY",
]