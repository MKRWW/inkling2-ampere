# Inkling 2 Mini on Ampere (RTX 3090)

Inkling 2 Mini ships with NVFP4/MXFP8 expert weights, and its vLLM image only has kernels for Hopper and
Blackwell. On an Ampere card (SM 8.x, e.g. RTX 3090) the image refuses to start.

This repository holds a small patch set (13 patches, 165 lines) that runs the model on a **single 24 GB RTX 3090**.
It also has a tool that moves the last four MXFP8 expert layers to NVFP4. With that, 256K context fits on the card.
A second, optional patch enables speculative decoding with the bundled DSpark draft head (code roughly 1.8x faster).

Tested on one RTX 3090 (power limit 250 W) with the image `vllm/vllm-openai:inkling-2-mini`, November 2026.

## Results on one RTX 3090

| | Original checkpoint | All-NVFP4 copy |
|---|---|---|
| Decode, single stream | 72 tok/s | 72 tok/s |
| Prefill (128 batched tokens) | ~780 tok/s | ~780 tok/s |
| KV cache (FP8, software) | 3.85 GiB, 181,781 tokens | 5.64 GiB, 297,076 tokens |
| Max context used | 176K | 256K |
| Needle in a haystack, 3 facts | 3/3 at 173.5K | 3/3 at 251.7K |

Tool calls (`--tool-call-parser inkling`) and image input work.

## What the patches do

`patch/patch_ampere.py` edits the installed packages at image build time. If any target line has changed, it
fails loudly.

1. Lowers the minimum compute capability of the Inkling quantization config from 9.0 to 8.0.
2. Uses TMA (Tensor Memory Accelerator) only on SM 9.0 and newer.
3. Switches the vision encoder to FlashAttention 2 on pre-Hopper GPUs.
4. to 7. Triton grid dependency control and programmatic dependent launch (PDL) exist only from SM 9.0. With
   `TRITON_DISABLE_GDC=1`, the `griddepcontrol` instructions and PDL launches are skipped. On SM 8.x, kernels in
   one stream run strictly in order anyway.
8. `conv_mix` takes the Triton path that is meant for SM 12.x consumer Blackwell.
9. `moe_align_block_size`: the compiled `_moe_C` kernel has no SM 8.6 image in this build. It is replaced on
   pre-Hopper GPUs by a pure PyTorch version with fixed tensor sizes, so it stays CUDA-graph safe
   (`patch/moe_align_torch.py`).
10. to 13. **FP8 KV cache in software.** On Ampere, Triton does not even accept `fp8e4nv` as an argument type.
    So the cache is passed as `uint8` and E4M3 is converted with bit arithmetic inside the kernels
    (`patch/fp8_sw.py`). Decoding is exact. Encoding (round to nearest even, saturating at ±448) matches
    PyTorch on 99.95 % of values. Subnormals are flushed.

The MoE runs through the image's existing Humming kernels (the Hopper fallback). They unpack NVFP4/MXFP8 in
software and compute in BF16.

## Optional: speculative decoding (DSpark)

The checkpoint ships a DSpark draft head (`mtp.safetensors`, 0.64 GB). vLLM uses it with
`--speculative-config '{"method":"dspark"}'`. It always drafts 7 tokens (the checkpoint's block size minus one);
vLLM rejects other values.

On Ampere the draft path needs `patch/patch_dspark.py` (included in the `Dockerfile`, it runs after
`patch_ampere.py`). Two parts of the draft have no SM 8.6 kernels in this image:

1. `_RMSNorm` uses flashinfer norms, and the vLLM `_C` rms_norm has no SM 8.6 image either. On pre-Hopper GPUs it
   now runs in PyTorch with fp32 math.
2. The draft KV write uses `reshape_and_cache_flash` from vLLM `_C`. It is replaced by a small Triton scatter
   kernel. It skips padding slots (< 0) and writes FP8 with the same software E4M3 encoding as patches 10 to 13.
   It needs no host synchronization, so CUDA graphs still work.

Measured on one RTX 3090 (250 W) with the all-NVFP4 copy, 2 sequences of 128K, 256 batched tokens, FP8 KV cache,
700 output tokens, thinking off:

| | Without DSpark | With DSpark |
|---|---|---|
| Code, single stream | 74 tok/s | 135 tok/s |
| Code, 2 streams | 61 tok/s each | 100 to 114 tok/s each |
| Prose, single stream | 74 tok/s | 72 tok/s |
| Prose, 2 streams | 61 tok/s each | 57 to 58 tok/s each |
| KV cache | 246,531 tokens | 163,892 tokens |

About 19 % of the drafted tokens are accepted. Code and structured output gain the most, prose stays about the
same. The draft head costs KV cache: with 2 × 128K, both sequences no longer fit at full length at the same time.
Answers to the smoke tests (arithmetic, a fact, a tool call) are unchanged.

A debugging note: on this image a missing kernel often shows up as `no kernel image is available` one launch
later, at the next PyTorch kernel. To find the real culprit, run the operation, call `torch.cuda.synchronize()`,
then launch any small PyTorch kernel.

## Build and run

```bash
docker build -t inkling-2-mini:ampere .
MODEL_DIR=/models/Inkling-2-Mini GPU=0 docker compose up -d
python3 tests/smoke.py                 # chat, tool call, throughput, image
INKLING_URL=http://127.0.0.1:8000 python3 tests/needle.py 150000
```

The first start takes about 2.5 minutes. Kernel autotuning is cached in `./humming-cache`.

## Optional: all experts in NVFP4

The vendor keeps the experts of layers 20 to 23 in MXFP8. `tools/nvfp4_convert.py` dequantizes them and
requantizes them with round-to-nearest. It uses exactly the layout of the other NVFP4 layers: E2M1 values with
the low nibble first, an E4M3 block scale per 16 values, and a global scale `amax / 2688` per expert. It writes
a new checkpoint directory and leaves the source untouched. It runs on the CPU in about two minutes.

```bash
python3 tools/nvfp4_convert.py /models/Inkling-2-Mini /models/Inkling-2-Mini-nvfp4all
MODEL_DIR=/models/Inkling-2-Mini-nvfp4all INK_CTX=262144 docker compose up -d
```

Self-test: requantizing existing NVFP4 experts reproduces them bit for bit. The relative RMS error of the new
weights against MXFP8 is 9.5 %.

Quality against the original checkpoint was measured with `tools/kld_eval.py` on the same patched server.
It uses teacher-forced prompt logprobs (top 20) on 63,800 positions of German text, English text and code.
Two runs of the original were bit-identical, so the noise floor is zero.

| | Original | All-NVFP4 |
|---|---|---|
| KL divergence (mean / p99) | 0 / 0 | 0.0014 / 0.020 |
| Top-1 agreement | 100 % | 98.6 % |
| Perplexity | 5.593 | 5.601 (+0.15 %) |

It frees 1.91 GB of weights. The KV cache grows from 181,781 to 297,076 tokens.

## Coding benchmark (own tasks)

This uses a terminal coding agent with 131K context, 32K output and a 45-minute limit per run. There are hidden
tests, checked against a reference solution beforehand.

| Task | Result |
|---|---|
| E: debugging an existing package, 6 tickets | 3/3 runs fully solved (25/25 hidden, 19/19 regression), 202 to 271 s |
| F: spreadsheet engine from scratch, 210 hidden tests | 208, 209, 1 |

The finished solutions match the best model I have tested locally. The weak spot is knowing when to stop. In
two of the three F runs the model kept refactoring after its own tests were green and ran into the time limit.
Once it was left with a half-finished change (1/210).

## Limits

- With one sequence, a long prefill blocks every other request for minutes. Two sequences of 128K fit; then a
  prefill on one slows the other one down a lot (to about 7 tok/s).
- Only tested on RTX 3090 (SM 8.6). A100 (SM 8.0) should work but is untested.
- Tied to the image `vllm/vllm-openai:inkling-2-mini`. A new image version may move the patched lines.
  The patch script then stops with `PATCH FAIL`.

## License

Apache-2.0, like vLLM.
