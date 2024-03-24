import torch
import triton
import triton.language as tl
from typing import Union, Tuple

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

class TritonRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        
        cos = torch.cos(freqs).to(dtype=t.dtype)
        sin = torch.sin(freqs).to(dtype=t.dtype)
        
        seq_len, batch, n_heads, head_dim = t.shape

        t = t.view(batch*seq_len, n_heads*head_dim)
        output = torch.zeros_like(t)
        output_rot = torch.zeros_like(t)
        
        n_rows, n_cols = t.shape
        BLOCK_SIZE, num_warps = calculate_settings(head_dim)
        _rope_embedding[(n_rows, n_heads,)](
            t,   t.stride(0),
            output,
            output_rot,
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.cos = cos
        ctx.sin = sin
        
        return output.view(seq_len, batch, n_heads, head_dim)

    @staticmethod
    def backward(
        ctx, dY: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        seq_len, batch, n_heads, head_dim = dY.shape
        cos = ctx.cos
        sin = ctx.sin
        
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        output = torch.zeros_like(dY)
        output_rot = torch.zeros_like(dY)
        n_rows, n_cols = dY.shape
        
        _rope_embedding[(n_rows, n_heads,)](
            dY,  dY.stride(0),
            output,
            output_rot,
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        output = output.view(seq_len, batch, n_heads, head_dim)

        return output, None, None, None

@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit()
def _rope_embedding(
    Q,     Q_row_stride,
    output_ptr,
    output_rot_ptr,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position  = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    
    half_head_dim = head_dim // 2
    mask = col_offsets < head_dim
    mask_2 = col_offsets < half_head_dim
    

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        sin1 = -sin1

    #[1] : t * cos_
    Q0 = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + col_offsets, mask=mask, other=0).to(sin1.dtype)
    z1 = Q0*cos1

    # [2] : _rotate_half(t)
    Q1 = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + half_head_dim*0 + col_offsets, mask = mask_2, other = 0).to(sin1.dtype)
    Q2 = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + half_head_dim*1 + col_offsets, mask = mask_2, other = 0).to(sin1.dtype)
    tl.store(output_rot_ptr + row_position*Q_row_stride + head_position*head_dim + half_head_dim*0 + col_offsets, -Q2, mask=mask_2)
    tl.store(output_rot_ptr + row_position*Q_row_stride + head_position*head_dim + half_head_dim*1 + col_offsets, Q1, mask=mask_2)
    
    # [3] : t_rotated = (t * cos_) + (_rotate_half(t) * sin_)
    Q3 = tl.load(output_rot_ptr + row_position*Q_row_stride + head_position*head_dim + col_offsets, mask=mask).to(sin1.dtype)
    z2 = Q3 * sin1
    z3 = z1 + z2
    tl.store(output_ptr + row_position*Q_row_stride + head_position*head_dim + col_offsets, z3, mask=mask)