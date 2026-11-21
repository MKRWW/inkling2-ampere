

# --- Ampere: E4M3 in Software (cvt.*e4m3x2 und Tritons fp8e4nv gibt es erst ab sm_89) ---
@triton.jit
def _f32_to_e4m3_bits(x):
    """float -> E4M3-Bits (uint8), Runden zur naechsten geraden Zahl, Saettigung bei +-448.
    E4M3 entspricht FP16 * 2^8 mit 3 statt 10 Mantissenbits; Subnormale fallen mit ab."""
    x = tl.minimum(tl.maximum(x.to(tl.float32), -448.0), 448.0)
    h = (x * 0.00390625).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    sign = (h >> 8) & 0x80
    mag = h & 0x7FFF
    r = (mag + 0x3F + ((mag >> 7) & 1)) >> 7
    r = tl.minimum(r, 0x7E)
    return (sign | r).to(tl.uint8)


@triton.jit
def _e4m3_bits_to_f32(b):
    """E4M3-Bits (uint8) -> float32, exakt (ohne NaN-Sonderfall)."""
    b = b.to(tl.uint16)
    h = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
    return h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
