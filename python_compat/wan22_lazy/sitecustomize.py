"""Run-only compatibility for LightX2V platform attention lazy loading.

The platform attention adapter has no tensor state, but its base template in
the pinned Hot28 image lacks the no-op disk-load method that the common
attention template already provides. Lazy block traversal calls the method on
every child weight object, so add the equivalent no-op before model imports.
"""

import builtins
import sys


_TARGET_MODULE = "lightx2v_platform.ops.attn.template"
_original_import = builtins.__import__


def _patch_if_loaded() -> bool:
    module = sys.modules.get(_TARGET_MODULE)
    if module is None:
        return False
    # During import, Python inserts a partially initialized module into
    # sys.modules before executing its class definitions. Do not inspect the
    # class until the module body has finished.
    template = getattr(module, "AttnWeightTemplate", None)
    if template is None:
        return False
    if not hasattr(template, "load_state_dict_from_disk"):

        def _load_state_dict_from_disk(
            self, block_index, adapter_block_index=None
        ) -> None:
            return None

        template.load_state_dict_from_disk = _load_state_dict_from_disk
    return True


def _import_with_lazy_attention_compat(
    name, globals=None, locals=None, fromlist=(), level=0
):
    result = _original_import(name, globals, locals, fromlist, level)
    if _patch_if_loaded():
        builtins.__import__ = _original_import
    return result


# Do not import LightX2V during Python startup. Install a short-lived import
# hook and remove it immediately after the platform template is naturally
# imported by LightX2V.
if not _patch_if_loaded():
    builtins.__import__ = _import_with_lazy_attention_compat
