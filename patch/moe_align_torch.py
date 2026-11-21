
# --- Ampere-Ersatz fuer torch.ops._moe_C.moe_align_block_size (Kernel fehlt fuer SM 8.6) ---
def _moe_align_block_size_torch(topk_ids, num_experts, block_size, sorted_token_ids,
                                experts_ids, num_tokens_post_pad, expert_map=None,
                                scatter_idx=None):
    """Gleiche Ausgabe wie der CUDA-Kernel, nur feste Tensorgroessen (CUDA-Graph-tauglich)."""
    import torch as _t
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(_t.int64)
    n = flat.numel()
    if expert_map is not None:
        flat = _t.where(flat >= 0, expert_map.to(_t.int64)[flat.clamp(min=0)], flat)
    valid = (flat >= 0) & (flat < num_experts)
    e = _t.where(valid, flat, _t.full_like(flat, num_experts))
    counts = _t.zeros(num_experts + 1, dtype=_t.int64, device=dev)
    counts.scatter_add_(0, e, _t.ones_like(e))
    counts = counts[:num_experts]
    padded = (counts + block_size - 1) // block_size * block_size
    pad_end = _t.cumsum(padded, 0)
    pad_start = pad_end - padded
    cnt_start = _t.cumsum(counts, 0) - counts
    order = _t.sort(e, stable=True).indices
    se = e[order]
    pos = _t.arange(n, device=dev, dtype=_t.int64)
    se_c = se.clamp(max=num_experts - 1)
    rank = pos - cnt_start[se_c]
    size = sorted_token_ids.numel()
    dest = _t.where(se < num_experts, pad_start[se_c] + rank, _t.full_like(se, size))
    buf = _t.full((size + 1,), n, dtype=_t.int64, device=dev)
    buf.scatter_(0, dest, order)
    sorted_token_ids.copy_(buf[:size].to(sorted_token_ids.dtype))
    total = pad_end[-1:]
    num_tokens_post_pad.copy_(total.to(num_tokens_post_pad.dtype))
    starts = _t.arange(experts_ids.numel(), device=dev, dtype=_t.int64) * block_size
    blk = _t.searchsorted(pad_end, starts, right=True).clamp(max=num_experts - 1)
    blk = _t.where(starts < total, blk, _t.full_like(blk, -1))
    experts_ids.copy_(blk.to(experts_ids.dtype))
    if scatter_idx is not None:
        scatter_idx.copy_(_t.where(valid, pos, _t.full_like(pos, -1)).view(-1, 1).to(scatter_idx.dtype))
