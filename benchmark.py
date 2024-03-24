import os
import torch
import triton
from attention import RotaryPositionEmbedding, apply_rotary_pos_emb

def run_rope(t, emb, grad_output, fused=False, triton_mode=False):
    output_fused = apply_rotary_pos_emb(t, emb, tensor_format="sbhd", fused=fused, triton_mode=triton_mode)
    output_fused.backward(grad_output)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['cuda', 'triton'],
        line_names=['Cuda', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    ))
def benchmark(size, provider):
    device = torch.device("cuda:0")
    dtype = torch.float16
    seq_length, batch_size, head_num, hidden_size = (1024, 1, 32, 32)
    t = torch.rand((seq_length, batch_size, head_num, hidden_size), dtype=dtype, device=device)
    t.requires_grad = True
    grad_output = torch.ones_like(t)
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, 1.0)
    emb = rotary_pos_emb(seq_length)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_rope(t,emb,grad_output, fused=True), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_rope(t,emb,grad_output, triton_mode=True), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


os.makedirs("reports",exist_ok=True)
benchmark.run(print_data=True, show_plots=True, save_path="reports")