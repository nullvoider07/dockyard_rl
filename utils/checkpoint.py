"""Checkpoint management utilities for the RL algorithm loop.

Handles logic at the algorithm level. Each RL Actor is expected to have its
own checkpoint saving function (called by the algorithm loop).
"""

import glob
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Mapping, NotRequired, Optional, TypedDict, Union, cast
import numpy as np
import torch
import yaml
from pydantic import BaseModel

PathLike = Union[str, "os.PathLike[Any]"]

class PretrainedCheckpointConfig(TypedDict):
    """Configuration for restoring initial weights from a prior dockyard checkpoint.

    Initializes a NEW run from a previous run's weights — e.g. an SFT→DPO→GRPO
    hand-off — by loading a prior dockyard checkpoint directory (the same layout
    ``CheckpointManager`` writes: ``policy/weights`` and, optionally,
    ``policy/optimizer``). Weights load via the same DTensor loader used for
    resume, so no separate backend is required.

    Distinct from resume: training state (the step counter / consumed_samples) is
    NOT restored — the new run starts at step 0. To continue an interrupted run,
    use the run's own latest checkpoint (resume), not ``pretrained_checkpoint``.

    Attributes:
        path: Path to a prior dockyard checkpoint directory (a ``step_N`` dir
            containing ``policy/weights`` and, when available, ``policy/optimizer``).
        restore_optimizer: Whether to also load the optimizer state (default True).
            When False, or when no optimizer state is present, the optimizer is
            freshly initialized.
    """

    path: str
    restore_optimizer: NotRequired[bool]

class CheckpointingConfig(TypedDict):
    """Configuration for checkpoint management.

    Attributes:
        enabled (bool): Whether checkpointing is enabled.
        checkpoint_dir (PathLike): Directory where checkpoints will be saved.
        metric_name (str | None): Name of the metric to use for determining best checkpoints.
            Must be of the form "val:<metric_name>" or "train:<metric_name>" to indicate whether
            the metric should be taken from the validation or training metrics.
        higher_is_better (bool): Whether higher values of the metric indicate better performance.
        keep_top_k (Optional[int]): Number of best checkpoints to keep. If None, all checkpoints are kept.
        model_save_format (str | None): Format for saving model (v2 allowed values: "torch_save" or "safetensors", v1 allowed values: None).
        save_consolidated (bool): Whether to save consolidated checkpoints (for HF compatibility).
        model_cache_dir (str): Directory for model cache (for safetensors format).
        model_repo_id (str): Repository ID for the model (for safetensors format).
        is_peft (bool): Whether the model uses PEFT.
        save_optimizer (bool): Whether to save optimizer state with checkpoints.
    """

    enabled: bool
    checkpoint_dir: PathLike
    metric_name: str | None
    higher_is_better: bool
    save_period: int
    keep_top_k: NotRequired[int]
    checkpoint_must_save_by: NotRequired[str | None]
    pretrained_checkpoint: NotRequired[PretrainedCheckpointConfig]
    save_optimizer: NotRequired[bool]  # Default: True
    model_save_format: NotRequired[str | None]  # Default: "safetensors"
    save_consolidated: NotRequired[bool]  # Default: False
    model_cache_dir: NotRequired[str]  # Default: ""
    model_repo_id: NotRequired[str]  # Default: ""
    is_peft: NotRequired[bool]  # Default: False
    peft_config: NotRequired[Any]  # Default: None
    is_async: NotRequired[bool]  # Default: False

class CheckpointManager:
    """Manages model checkpoints during training.

    This class handles creating checkpoint dirs, saving training info, and
    configurations. It also provides utilities for keeping just the top-k checkpoints.
    The checkpointing structure looks like this:
    ```
    checkpoint_dir/
        step_0/
            training_info.json
            config.yaml
            policy.py (up to the algorithm loop to save here)
            policy_optimizer.py (up to the algorithm loop to save here)
            ...
        step_1/
            ...
    ```

    Attributes: Derived from the CheckpointingConfig.
    """

    def __init__(self, config: CheckpointingConfig):
        """Initialize the checkpoint manager.

        Args:
            config (CheckpointingConfig)
        """
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.metric_name: str | None = config["metric_name"]
        self.higher_is_better = config["higher_is_better"]
        self.keep_top_k = config.get("keep_top_k", None)
        self.save_optimizer = config.get("save_optimizer", True)

        # Store model-format specific config options
        self.model_save_format = config.get("model_save_format", "safetensors")
        self.save_consolidated = config.get("save_consolidated", False)
        self.model_cache_dir = config.get("model_cache_dir", "")
        self.model_repo_id = config.get("model_repo_id", "")
        self.is_peft = config.get("is_peft", False)

    @staticmethod
    def get_resume_paths(
        last_checkpoint_path: Optional[PathLike],
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Get weights and optimizer paths for resuming from a checkpoint.

        Args:
            last_checkpoint_path: Path to the last checkpoint, or None if starting fresh.

        Returns:
            Tuple of (weights_path, optimizer_path). Both are None if no checkpoint.
            optimizer_path is None if checkpoint exists but optimizer state was not saved.
        """
        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"

            if optimizer_path.exists():
                return weights_path, optimizer_path

            warnings.warn(
                f"Optimizer state not found at {optimizer_path}. "
                "Optimizer will be freshly initialized.",
                stacklevel=2,
            )
            return weights_path, None
        return None, None

    @staticmethod
    def get_pretrained_paths(
        pretrained_cfg: "PretrainedCheckpointConfig",
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Resolve (weights_path, optimizer_path) for a pretrained-checkpoint init.

        Points at a prior dockyard checkpoint directory and reuses the same
        DTensor layout as resume (``policy/weights`` + ``policy/optimizer``).
        Honors ``restore_optimizer`` (default True); the optimizer path is None
        when restore is disabled or no optimizer state is present. Training state
        is intentionally not restored — a pretrained init starts a fresh run.
        """
        path = pretrained_cfg["path"]
        weights_path = Path(path) / "policy" / "weights"
        optimizer_path: Optional[Path] = Path(path) / "policy" / "optimizer"
        if not pretrained_cfg.get("restore_optimizer", True):
            optimizer_path = None
        elif not optimizer_path.exists():
            warnings.warn(
                f"Pretrained optimizer state not found at {optimizer_path}; "
                "optimizer will be freshly initialized.",
                stacklevel=2,
            )
            optimizer_path = None
        return weights_path, optimizer_path

    def init_tmp_checkpoint(
        self,
        step: int,
        training_info: Mapping[str, Any],
        run_config: Optional[BaseModel] = None,
    ) -> PathLike:
        """Initialize a temporary checkpoint directory.

        Creates a temporary directory for a new checkpoint and saves training info
        and configuration. The directory is named 'tmp_step_{step}' and will be renamed
        to 'step_{step}' when the checkpoint is completed.
        We do it this way to allow the algorithm loop to save any files it wants to save
        in a safe, temporary directory.

        Args:
            step (int): The training step number.
            training_info (dict[str, Any]): Dictionary containing training metrics and info.
            run_config (Optional[BaseModel]): Optional configuration for the training run.

        Returns:
            PathLike: Path to the temporary checkpoint directory.
        """
        save_dir = self.checkpoint_dir / f"tmp_step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # save training info
        with open(save_dir / "training_info.json", "w") as f:
            # make any numpy items serializable
            serializable_training_info = dict(training_info)
            for k, v in serializable_training_info.items():
                if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                    serializable_training_info[k] = v.item()
            json.dump(serializable_training_info, f)

        # save config
        if run_config is not None:
            with open(save_dir / "config.yaml", "w") as f:
                yaml.safe_dump(run_config.model_dump(), f)

        return Path(os.path.abspath(save_dir))

    def finalize_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Complete a checkpoint by moving it from temporary to permanent location.

        If a checkpoint at the target location already exists (i.e. when resuming training),
        we override the old one. Also triggers cleanup of old checkpoints based on the
        keep_top_k setting.

        Args:
            checkpoint_path (PathLike): Path to the temporary checkpoint directory.
        """
        # rename tmp_step_{step} to step_{step}
        checkpoint_path = Path(checkpoint_path)
        to_checkpoint_path = (
            checkpoint_path.parent / f"step_{checkpoint_path.name.split('_')[2]}"
        )
        if to_checkpoint_path.exists():
            # If step_{step} exists, rename it to old_step_{step}, move tmp_step_{step} to step_{step},
            # then delete. We do this trickery to have a 'pseudo-atomic' checkpoint save.
            old_checkpoint_path = (
                checkpoint_path.parent
                / f"old_step_{checkpoint_path.name.split('_')[2]}"
            )
            os.rename(to_checkpoint_path, old_checkpoint_path)
            os.rename(checkpoint_path, to_checkpoint_path)
            if old_checkpoint_path.exists():
                shutil.rmtree(old_checkpoint_path)
        else:
            os.rename(checkpoint_path, to_checkpoint_path)
        self.remove_old_checkpoints()

    def remove_old_checkpoints(self, exclude_latest: bool = True) -> None:
        """Remove checkpoints that are not in the top-k or latest based on the (optional) metric.

        If keep_top_k is set, this method removes all checkpoints except the top-k
        best ones. The "best" checkpoints are determined by:
        - If a metric is provided: the given metric value and the higher_is_better setting.
          When multiple checkpoints have the same metric value, more recent checkpoints
          (higher step numbers) are preferred.
        - If no metric is provided: the step number. The most recent k checkpoints are kept.

        Args:
            exclude_latest (bool): Whether to exclude the latest checkpoint from deletion.
                (may result in K+1 checkpoints)
        """
        if self.keep_top_k is None:
            return
        checkpoint_history = _load_checkpoint_history(self.checkpoint_dir)
        latest_step = (
            max([step for step, _, _ in checkpoint_history])
            if checkpoint_history
            else None
        )

        if self.metric_name is None:
            checkpoint_history.sort(key=lambda x: x[0], reverse=True)
        else:
            # Cast to str because the 'else' branch guarantees self.metric_name is not None
            metric_key = cast(str, self.metric_name)
            if self.higher_is_better:
                # higher metric values first, then higher step numbers for ties
                checkpoint_history.sort(
                    key=lambda x: (x[2].get(metric_key, -float("inf")), x[0]),
                    reverse=True,
                )
            else:
                # lower metric values first, then higher step numbers for equal values
                checkpoint_history.sort(
                    key=lambda x: (x[2].get(metric_key, float("inf")), -x[0])
                )

        for checkpoint in checkpoint_history[self.keep_top_k :]:
            if exclude_latest and checkpoint[0] == latest_step:
                continue
            print(
                f"Removing checkpoint {checkpoint[1]} due to being outside top-{self.keep_top_k}"
            )
            shutil.rmtree(checkpoint[1])

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get the path to the best checkpoint based on the metric.

        Returns the path to the checkpoint with the best metric value. If no checkpoints
        exist, returns None. If some checkpoints are missing the metric, they are filtered
        out with a warning. If no checkpoints have the metric, returns the latest checkpoint.

        Returns:
            Optional[str]: Path to the best checkpoint, or None if no checkpoints exist.
        """
        checkpoint_history = _load_checkpoint_history(self.checkpoint_dir)
        if len(checkpoint_history) == 0:
            return None

        # Add a direct sentinel guard check first to clear the str | None mapping constraint
        if self.metric_name is None:
            return self.get_latest_checkpoint_path()

        metric_key = cast(str, self.metric_name)
        valid_checkpoints = [c for c in checkpoint_history if metric_key in c[2]]
        ignored_count = len(checkpoint_history) - len(valid_checkpoints)

        if ignored_count > 0:
            ignored_steps = [
                c[0] for c in checkpoint_history if metric_key not in c[2]
            ]
            warnings.warn(
                f"Ignoring {ignored_count} checkpoint(s) at step(s) {ignored_steps} that do not have "
                f"metric '{self.metric_name}'. Consider enabling val_at_end or adjusting val_period "
                f"to align with max_steps."
            )

        if len(valid_checkpoints) == 0:
            warnings.warn(
                f"No checkpoints contain metric '{self.metric_name}'. Returning latest checkpoint. "
                f"Consider enabling val_at_end or adjusting val_period to align with max_steps."
            )
            return self.get_latest_checkpoint_path()

        # FIX: Reference local metric_key instead of self.metric_name inside lambda
        valid_checkpoints.sort(
            key=lambda x: x[2][metric_key], reverse=self.higher_is_better
        )
        return str(valid_checkpoints[0][1])

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns the path to the checkpoint with the highest step number.

        Returns:
            Optional[str]: Path to the latest checkpoint, or None if no checkpoints exist.
        """
        step_dirs = [
            x
            for x in glob.glob(str(self.checkpoint_dir / "step_*"))
            if re.fullmatch(r"step_\d+", Path(x).name)
        ]
        step_dirs.sort(key=lambda x: int(Path(x).name.split("_")[1]))
        if len(step_dirs) == 0:
            return None
        return str(step_dirs[-1])

    def load_training_info(
        self, checkpoint_path: Optional[PathLike] = None
    ) -> Optional[dict[str, Any]]:
        """Load the training info from a checkpoint.

        Args:
            checkpoint_path (Optional[PathLike]): Path to the checkpoint. If None,
                returns None.

        Returns:
            Optional[dict[str, Any]]: Dictionary containing the training info, or None if
                checkpoint_path is None.
        """
        if checkpoint_path is None:
            return None
        with open(Path(checkpoint_path) / "training_info.json", "r") as f:
            return json.load(f)

def _load_checkpoint_history(
    checkpoint_dir: Path,
) -> list[tuple[int, PathLike, dict[str, Any]]]:
    """Load the history of checkpoints and their metrics.

    Args:
        checkpoint_dir (Path): Directory containing the checkpoints.

    Returns:
        list[tuple[int, PathLike, dict[str, Any]]]: List of tuples containing
            (step_number, checkpoint_path, info) for each checkpoint.
    """
    checkpoint_history: list[tuple[int, PathLike, dict[str, Any]]] = []

    step_dirs = [
        x
        for x in glob.glob(str(checkpoint_dir / "step_*"))
        if re.fullmatch(r"step_\d+", Path(x).name)
    ]

    for step_dir in step_dirs:
        info_file = Path(step_dir) / "training_info.json"
        if info_file.exists():
            with open(info_file) as f:
                info: dict[str, Any] = json.load(f)
                step = int(Path(step_dir).name.split("_")[1])
                checkpoint_history.append((step, step_dir, info))

    return checkpoint_history