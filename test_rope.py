import pytest
import torch
from attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)

import random
seed = 0
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_tol(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


# batch 2이상 이거나 hidden_size 64 이상이면 Fail

@pytest.mark.parametrize("hidden_size", [8,16,32])
@pytest.mark.parametrize("head_num", [8,16,32,64])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_length", [1204, 512, 256])
def test_rope_kernel(
    seq_length: int,
    hidden_size: int,
    head_num: int,
    batch_size: int,
):
    
    dtype = torch.float16
    
    t = torch.rand(
        (seq_length, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=torch.device("cuda:0"),
    )
    
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size)
    emb = rotary_pos_emb(seq_length)

    # triton kernel
    grad_triton = torch.ones_like(t)
    output_triton = apply_rotary_pos_emb(t, emb, triton_mode=True)
    output_triton.backward(grad_triton)
    grad_triton = t.grad.detach().clone()
    t.grad = None
    
    # cuda kernel
    grad_cuda = torch.ones_like(t)
    output_cuda = apply_rotary_pos_emb(t, emb, fused=True)
    output_cuda.backward(grad_cuda)
    grad_cuda = t.grad.detach().clone()
    t.grad = None
    
    

    torch.testing.assert_close(output_triton, output_cuda, **get_tol(dtype))
    torch.testing.assert_close(grad_triton, grad_cuda, **get_tol(dtype))