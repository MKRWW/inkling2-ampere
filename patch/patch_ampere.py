"""Ampere-Patch fuer das vLLM-Inkling-2-Mini-Image: oeffnet den Hopper-Pfad fuer SM 8.x.

Mit TRITON_DISABLE_GDC=1 erzeugt Triton keine griddepcontrol-Instruktionen (erst ab SM 9.0)
und startet keine Kernel mit Programmatic Dependent Launch. Auf SM 8.x laufen Kernel
in einem Stream ohnehin strikt nacheinander, die Synchronisation ist dort ueberfluessig.
"""
import sys

SP = "/usr/local/lib/python3.14/dist-packages/"
V = SP + "vllm/models/inkling2/"
T = SP + "triton/"
PATCHES = [
    (V + "quant_config.py",
     "    def get_min_capability(cls) -> int:\n        return 90\n",
     "    def get_min_capability(cls) -> int:\n        return 80\n"),
    (V + "nvidia/ops/linear.py",
     "    if tokens >= TMA_MIN_TOKENS and not sm120:\n",
     "    if tokens >= TMA_MIN_TOKENS and not sm120 and current_platform.has_device_capability(90):\n"),
    (V + "nvidia/vision.py",
     "            version = 2 if current_platform.is_device_capability_family(120) else 3\n",
     "            version = 2 if (current_platform.is_device_capability_family(120) or not current_platform.has_device_capability(90)) else 3\n"),
    (T + "language/extra/cuda/gdc.py",
     "from triton.language import core\n",
     "import os\n\nfrom triton.language import core\n\n_NO_GDC = os.environ.get(\"TRITON_DISABLE_GDC\") == \"1\"\n"),
    (T + "language/extra/cuda/gdc.py",
     "    core.inline_asm_elementwise(\"griddepcontrol.wait; // dummy $0\"",
     "    if _NO_GDC:\n        return\n    core.inline_asm_elementwise(\"griddepcontrol.wait; // dummy $0\""),
    (T + "language/extra/cuda/gdc.py",
     "    core.inline_asm_elementwise(\"griddepcontrol.launch_dependents; // dummy $0\"",
     "    if _NO_GDC:\n        return\n    core.inline_asm_elementwise(\"griddepcontrol.launch_dependents; // dummy $0\""),
    (T + "backends/nvidia/driver.py",
     "        self.launch_pdl = metadata.launch_pdl\n",
     "        self.launch_pdl = metadata.launch_pdl and __import__(\"os\").environ.get(\"TRITON_DISABLE_GDC\") != \"1\"\n"),
]
for path, old, new in PATCHES:
    src = open(path).read()
    n = src.count(old)
    if n != 1:
        sys.exit(f"PATCH FAIL {path}: {n} Treffer")
    open(path, "w").write(src.replace(old, new))
    print("patched", path[len(SP):])

# Ampere hat keinen nativen conv_mix-Kernel: denselben Triton-Weg wie SM12x nehmen.
_p = V + "nvidia/ops/conv_mix.py"
_old = "        if current_platform.is_device_capability_family(120):\n            # Native conv/mix kernels require Hopper or SM10x."
_new = "        if current_platform.is_device_capability_family(120) or not current_platform.has_device_capability(90):\n            # Native conv/mix kernels require Hopper or SM10x."
_s = open(_p).read()
if _s.count(_old) != 1:
    sys.exit(f"PATCH FAIL {_p}: {_s.count(_old)} Treffer")
open(_p, "w").write(_s.replace(_old, _new))
print("patched", _p[len(SP):])

# moe_align_block_size: der _moe_C-Kernel hat in diesem Build kein SM-8.6-Image -> PyTorch-Ersatz.
_p = SP + "vllm/_custom_ops.py"
_s = open(_p).read()
_old = ("    torch.ops._moe_C.moe_align_block_size(\n        topk_ids,\n        num_experts,\n        block_size,\n"
        "        sorted_token_ids,\n        experts_ids,\n        num_tokens_post_pad,\n        expert_map,\n        scatter_idx,\n    )\n")
_new = ("    if not current_platform.has_device_capability(90):\n"
        "        return _moe_align_block_size_torch(topk_ids, num_experts, block_size, sorted_token_ids,\n"
        "                                           experts_ids, num_tokens_post_pad, expert_map, scatter_idx)\n" + _old)
if _s.count(_old) != 1:
    sys.exit(f"PATCH FAIL {_p}: {_s.count(_old)} Treffer")
_s = _s.replace(_old, _new) + open("/tmp/moe_align_torch.py").read()
open(_p, "w").write(_s)
print("patched", _p[len(SP):])


# ---- FP8-KV auf SM 8.x: Cache als uint8 an Triton (fp8e4nv gibt es dort nicht), E4M3 in Software ----
def _patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"PATCH FAIL {path}: {n} Treffer statt {count}")
    open(path, "w").write(s.replace(old, new))
    print("patched", path[len(SP):])

_FP8 = open("/tmp/fp8_sw.py").read()
_A = V + "nvidia/attention.py"
_patch(_A, "        if current_platform.is_device_capability_family(120):\n            supported += (\"float16\", \"fp8\", \"fp8_e4m3\")\n",
           "        if current_platform.is_device_capability_family(120) or not current_platform.has_device_capability(90):\n            supported += (\"float16\", \"fp8\", \"fp8_e4m3\")\n")
_patch(_A, "            kv_cache = kv_cache.view(torch.float8_e4m3fn)\n",
           "            kv_cache = kv_cache.view(torch.float8_e4m3fn if current_platform.has_device_capability(89) else torch.uint8)\n")

_S90 = V + "nvidia/ops/attention_sm90.py"
_s = open(_S90).read()
_i = _s.index("    # Load each E4M3 pair once")
_j = _s.index("    return tl.interleave(lo, hi).to(dtype)\n") + len("    return tl.interleave(lo, hi).to(dtype)\n")
_s = (_s[:_i]
      + "    # Ampere: Bytes einzeln laden, E4M3 in Software nach float (exakt).\n"
      + "    cols = tl.arange(0, D)\n"
      + "    b = tl.load(ptr.to(tl.pointer_type(tl.uint8)) + tl.multiple_of(offsets, D)[:, None] + cols[None, :],\n"
      + "                mask=mask[:, None], other=0)\n"
      + "    return _e4m3_bits_to_f32(b).to(dtype)\n"
      + _s[_j:] + _FP8)
open(_S90, "w").write(_s)
print("patched", _S90[len(SP):])

_SC = V + "nvidia/ops/sconv.py"
_patch(_SC, "    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n",
            "    if cache.dtype.element_ty == tl.uint8:\n"
            "        tl.store(dst, _f32_to_e4m3_bits(y), mask=(kv_slot >= 0)[:, None])\n"
            "    else:\n"
            "        tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n")
open(_SC, "a").write(_FP8)
