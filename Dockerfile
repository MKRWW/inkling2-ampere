# vLLM image for Inkling 2 Mini, patched to run on Ampere GPUs (SM 8.x, e.g. RTX 3090 / A100).
FROM vllm/vllm-openai:inkling-2-mini
COPY patch/patch_ampere.py patch/moe_align_torch.py patch/fp8_sw.py patch/patch_dspark.py /tmp/
RUN python3 /tmp/patch_ampere.py && python3 /tmp/patch_dspark.py
