"""Run-only compatibility for LightX2V Wan2.2 lazy loading.

The platform attention adapter has no tensor state, but its base template in
the pinned Hot28 image lacks the no-op disk-load method that the common
attention template already provides. Lazy block traversal calls the method on
every child weight object, so add the equivalent no-op before model imports.

The same image also preserves the checkpoint dtype when it creates the lazy
FP8 CUDA bias buffer. Wan2.2 checkpoints store these biases as float32 while
the GEMM output is bfloat16, which Aiter correctly rejects. Match the regular
and pinned-buffer paths by normalizing the CUDA bias buffer after loading.
"""

import builtins
import sys


_ATTENTION_MODULE = "lightx2v_platform.ops.attn.template"
_MM_MODULE = "lightx2v.common.ops.mm.mm_weight"
_original_import = builtins.__import__


def _patch_attention_if_loaded() -> bool:
    module = sys.modules.get(_ATTENTION_MODULE)
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


def _patch_mm_if_loaded() -> bool:
    module = sys.modules.get(_MM_MODULE)
    if module is None:
        return False
    template = getattr(module, "MMWeightQuantTemplate", None)
    if template is None:
        return False
    if getattr(template, "_lightx2v_lazy_bias_dtype_compat", False):
        return True

    original_post_process = template.post_process
    torch_module = module.torch

    def _post_process(self) -> None:
        original_post_process(self)
        bias_buffer = getattr(self, "bias_cuda_buffer", None)
        if not getattr(self, "lazy_load", False) or bias_buffer is None:
            return
        target_dtype = (
            torch_module.float32
            if getattr(self, "bias_force_fp32", False)
            else self.infer_dtype
        )
        if bias_buffer.dtype != target_dtype:
            self.bias_cuda_buffer = bias_buffer.to(target_dtype)

    template.post_process = _post_process
    template._lightx2v_lazy_bias_dtype_compat = True
    return True


def _patch_if_loaded() -> bool:
    attention_ready = _patch_attention_if_loaded()
    mm_ready = _patch_mm_if_loaded()
    return attention_ready and mm_ready


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
