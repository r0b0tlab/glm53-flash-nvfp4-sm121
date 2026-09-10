import torch
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(0))
import vllm._custom_ops as co
failures = []

# rms_norm: (result, input, Optional[Tensor] weight, float eps) -> ()
x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
try:
    co.rms_norm(y, x, None, 1e-5)
    torch.cuda.synchronize()
    print("rms_norm OK", float(y.abs().mean()))
except Exception as e:
    failures.append(("rms_norm", type(e).__name__, str(e)[:250]))
    print("rms_norm FAIL", type(e).__name__, str(e)[:250])

# silu_and_mul: not a standalone custom op at this revision (torch.compile
# fusion path; quant variants exist). Verify the fusion family is present.
try:
    has_family = any(
        hasattr(co, n) for n in ("silu_and_mul_per_block_quant",
                                 "silu_and_mul_scaled_fp4_experts_quant",
                                 "silu_and_mul_mxfp4_experts_quant"))
    print("silu_and_mul family present:", has_family)
    if not has_family:
        failures.append(("silu_and_mul_family", "absent", ""))
except Exception as e:
    failures.append(("silu_and_mul_family", type(e).__name__, str(e)[:250]))

# rotary_embedding: (positions[num_tokens], query[num_tokens,heads,head_size],
#                    key[...], head_size, cos_sin_cache[max_pos, head_size], is_neox)
nt = 256
q = torch.randn(nt, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(nt, 2, 128, device="cuda", dtype=torch.bfloat16)
cs = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
pos_flat = torch.arange(nt, device="cuda", dtype=torch.int64)
try:
    co.rotary_embedding(pos_flat, q, k, 128, cs, False)
    torch.cuda.synchronize()
    print("rotary_embedding OK")
except Exception as e:
    failures.append(("rotary_embedding", type(e).__name__, str(e)[:250]))
    print("rotary_embedding FAIL", type(e).__name__, str(e)[:250])

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
c = a @ a.t(); torch.cuda.synchronize()
print("torch GEMM OK", float(c.abs().mean()))
if failures:
    raise SystemExit("MICROTEST FAILURES: %s" % failures)
print("MICROTESTS ALL PASS")
