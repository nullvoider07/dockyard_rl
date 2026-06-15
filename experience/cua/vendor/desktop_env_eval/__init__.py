# Vendored OSWorld evaluator subset + standalone grading dispatch.
#
# Upstream: xlang-ai/OSWorld, pinned 705623ca18e0055dd995fd5a350d6588cff2caf5,
# desktop_env/evaluators/{getters,metrics} (file/general/info/misc getters;
# basic_os/general/table/pdf/utils metrics) — the subset exercised by the
# os + libreoffice_calc task domains. See README.md for coverage and deps.
#
# evaluate(task_config, env, setup_fn=None) -> float grades a finished episode
# without DesktopEnv, against a GradingEnv pointed at the guest server (:5000).

from .dispatch import evaluate
from .grading_env import GradingEnv, GuestServerController

__all__ = ["evaluate", "GradingEnv", "GuestServerController"]
