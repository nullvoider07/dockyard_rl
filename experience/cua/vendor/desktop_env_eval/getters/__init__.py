# Vendored subset of OSWorld desktop_env/evaluators/getters, pinned at
# xlang-ai/OSWorld 705623ca18e0055dd995fd5a350d6588cff2caf5.
#
# Only the file/general/info/misc getter modules (those serving the os +
# libreoffice_calc result/expected types: vm_command_line, vm_terminal_output,
# rule, list_directory, vm_file, cloud_file) are vendored. The upstream
# aggregator's chrome/gimp/impress/replay/vlc/vscode/calc blocks are dropped and
# added per-domain in later slices. Getters take a duck-typed ``env``
# (vm_ip/server_port/cache_dir/controller) — see ..grading_env.GradingEnv.

from .file import (
    get_cloud_file,
    get_vm_file,
    get_cache_file,
    get_content_from_vm_file,
)
from .general import (
    get_vm_command_line,
    get_vm_terminal_output,
    get_vm_command_error,
)
from .info import (
    get_vm_screen_size,
    get_vm_window_size,
    get_vm_wallpaper,
    get_list_directory,
)
from .misc import (
    get_rule,
    get_accessibility_tree,
    get_rule_relativeTime,
    get_time_diff_range,
)
