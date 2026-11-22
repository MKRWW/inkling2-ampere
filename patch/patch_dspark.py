"""Ampere-Patch fuer den DSpark-Draft (spekulatives Dekodieren) von Inkling 2 Mini.

Laeuft nach patch_ampere.py (braucht dessen _f32_to_e4m3_bits in attention_sm90).
Auf SM < 9.0:

1. _RMSNorm nutzt flashinfer-Norm; weder flashinfer noch vLLM-_C haben hier sm_86-Kernel
   -> RMSNorm in PyTorch (fp32-Rechnung wie die Referenz).
2. _write_kv nutzt reshape_and_cache_flash (vLLM-_C, kein sm_86-Kernel)
   -> eigener Triton-Scatter, ueberspringt Slots < 0, FP8 per Software-E4M3
   (CUDA-Graph-tauglich, keine Host-Synchronisation).
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia/dspark.py"
t = open(p).read()

# --- Teil 1: RMSNorm ---
old1 = """        from flashinfer.norm import fused_add_rmsnorm, rmsnorm

        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
"""
new1 = """        if torch.cuda.get_device_capability(x.device) < (9, 0):
            # Ampere: weder flashinfer noch vLLM-_C haben hier sm_86-Kernel.
            w = self.weight.float()
            if residual is None:
                xf = x.float()
                return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(x.dtype)
            s = x.float() + residual.float()
            residual.copy_(s.to(residual.dtype))
            x.copy_((s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(x.dtype))
            return x
        from flashinfer.norm import fused_add_rmsnorm, rmsnorm

        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
"""
if new1 in t:
    print("Teil 1 schon drin")
else:
    if t.count(old1) != 1:
        sys.exit(f"PATCH FAIL {p}: Teil 1, {t.count(old1)} Treffer")
    t = t.replace(old1, new1)
    print("Teil 1 gepatcht")

# --- Teil 2: KV-Schreiben ---
old2 = """        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
new2 = """        impl = cast(FlashAttentionImpl, self.attn.impl)
        if torch.cuda.get_device_capability(k.device) < (9, 0):
            _ampere_write_kv(self.attn, impl, k, v, slot_mapping)
            return
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
helper = '''

from vllm.triton_utils import tl as _tl, triton as _triton

from .ops.attention_sm90 import _f32_to_e4m3_bits


@_triton.jit
def _ampere_kv_scatter_kernel(
    SRC, DST, SLOTS, SCALE,
    s_t, s_h, d_b, d_n, d_h,
    BS: _tl.constexpr, D: _tl.constexpr, FP8: _tl.constexpr, HAS_SCALE: _tl.constexpr,
):
    t = _tl.program_id(0)
    h = _tl.program_id(1)
    slot = _tl.load(SLOTS + t).to(_tl.int64)
    if slot >= 0:
        blk = slot // BS
        off = slot % BS
        d = _tl.arange(0, D)
        x = _tl.load(SRC + t * s_t + h * s_h + d).to(_tl.float32)
        dst = DST + blk * d_b + off * d_n + h * d_h + d
        if FP8:
            if HAS_SCALE:
                x = x / _tl.load(SCALE).to(_tl.float32)
            _tl.store(dst, _f32_to_e4m3_bits(x))
        else:
            _tl.store(dst, x.to(DST.dtype.element_ty))


def _ampere_write_kv(layer, impl, k, v, slot_mapping):
    """Ampere: Scatter in den Draft-KV-Cache ohne reshape_and_cache_flash (gleiches Layout)."""
    kv_cache = layer.kv_cache
    key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
    n = slot_mapping.shape[0]
    if n == 0:
        return
    for src, cache, scale_name in ((k, key_cache, "_k_scale"), (v, value_cache, "_v_scale")):
        fp8 = cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        dst = cache.view(torch.uint8) if cache.dtype == torch.float8_e4m3fn else cache
        scale = getattr(layer, scale_name, None) if fp8 else None
        assert src.stride(-1) == 1 and dst.stride(-1) == 1
        _ampere_kv_scatter_kernel[(n, src.shape[1])](
            src, dst, slot_mapping, scale if scale is not None else src,
            src.stride(0), src.stride(1), dst.stride(0), dst.stride(1), dst.stride(2),
            BS=dst.shape[1], D=src.shape[2], FP8=fp8, HAS_SCALE=scale is not None,
        )
'''
if "_ampere_write_kv" in t:
    print("Teil 2 schon drin")
else:
    if t.count(old2) != 1:
        sys.exit(f"PATCH FAIL {p}: Teil 2, {t.count(old2)} Treffer")
    t = t.replace(old2, new2)
    anchor = "\nclass _RMSNorm(nn.Module):"
    if t.count(anchor) != 1:
        sys.exit(f"PATCH FAIL {p}: Anker _RMSNorm, {t.count(anchor)} Treffer")
    t = t.replace(anchor, helper + anchor)
    print("Teil 2 gepatcht")

open(p, "w").write(t)
