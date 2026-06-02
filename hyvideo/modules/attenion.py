import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from st_attn import sliding_tile_attention
except ImportError:
    sliding_tile_attention = None

# Optional FlashAttention backend. Used only when attn_mode='flash'.
try:
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    _HAS_FLASH = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
        _HAS_FLASH = True
    except ImportError:
        _flash_attn_varlen_func = None
        pad_input = unpad_input = None
        _HAS_FLASH = False

from hyvideo.parallel.communications import all_gather, all_to_all_4D
from hyvideo.parallel.parallel_states import get_sequence_parallel_state, nccl_info


# Global attention backend, set once by the inference entrypoint via set_attn_mode().
#   "auto"  -> use flash if available, else fall back to torch SDPA (default)
#   "torch" -> F.scaled_dot_product_attention (no external dependency)
#   "flash" -> flash_attn varlen (requires flash-attn)
_ATTN_MODE = "flash" if _HAS_FLASH else "torch"


def set_attn_mode(mode):
    """Select the attention backend. 'auto' picks flash when installed, else torch."""
    global _ATTN_MODE
    assert mode in ("auto", "torch", "flash"), f"Unsupported attn_mode: {mode}"
    if mode == "auto":
        _ATTN_MODE = "flash" if _HAS_FLASH else "torch"
    elif mode == "flash":
        if not _HAS_FLASH:
            raise ImportError(
                "attn_mode='flash' requires flash-attn (flash_attn_varlen_func + "
                "flash_attn.bert_padding). Install flash-attn or use --attn-mode torch/auto."
            )
        _ATTN_MODE = "flash"
    else:
        _ATTN_MODE = "torch"
    print(f"[ICVE] attention backend: {_ATTN_MODE}"
          + ("" if _HAS_FLASH else " (flash-attn not installed)"), flush=True)
    return _ATTN_MODE


def get_attn_mode():
    return _ATTN_MODE


def _flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    """FlashAttention with variable-length unpadding.

    qkv: [B, S, 3, H, D]; key_padding_mask: [B, S] bool (True = keep).
    """
    B, S, three, H, D = qkv.shape
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=H)
    out = _flash_attn_varlen_func(
        x_unpad[:, 0],
        x_unpad[:, 1],
        x_unpad[:, 2],
        cu_seqlens,
        cu_seqlens,
        max_s,
        max_s,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    if isinstance(out, tuple):  # FA3 returns (out, lse)
        out = out[0]
    out_flat = rearrange(out, "nnz h d -> nnz (h d)")
    padded = pad_input(out_flat, indices, B, S)
    return rearrange(padded, "b s (h d) -> b s h d", h=H)


def _sdpa_no_pad(qkv, key_padding_mask, causal=False):
    """PyTorch SDPA equivalent of _flash_attn_no_pad.

    qkv: [B, S, 3, H, D]; key_padding_mask: [B, S] bool (True = keep).
    Padding tokens are masked out as keys via an additive mask: valid queries
    never attend to pad keys, so outputs on all valid positions match flash.
    """
    B, S, three, H, D = qkv.shape
    q, k, v = qkv.unbind(dim=2)  # each [B, S, H, D]
    q = q.transpose(1, 2)        # [B, H, S, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_mask = None
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        neg = torch.finfo(q.dtype).min  # bf16-safe, avoids -inf NaN
        attn_mask = torch.zeros(B, 1, 1, S, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(~key_padding_mask[:, None, None, :], neg)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
    return out.transpose(1, 2)  # [B, S, H, D]


def _attn_compute(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, mode=None):
    """Dispatch the core attention kernel by backend. qkv: [B, S, 3, H, D]."""
    mode = mode or _ATTN_MODE
    if mode == "flash":
        if not _HAS_FLASH:
            raise ImportError(
                "attn_mode='flash' requires flash-attn (flash_attn_varlen_func + "
                "flash_attn.bert_padding). Install flash-attn or use attn_mode='torch'."
            )
        m = key_padding_mask
        if m is not None and m.dtype != torch.bool:
            m = m.bool()
        return _flash_attn_no_pad(qkv, m, causal=causal, dropout_p=dropout_p, softmax_scale=softmax_scale)
    elif mode == "torch":
        return _sdpa_no_pad(qkv, key_padding_mask, causal=causal)
    else:
        raise NotImplementedError(f"Unsupported attn_mode: {mode}")


def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    qkv = torch.stack([q, k, v], dim=2)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = _attn_compute(qkv, attn_mask, causal=causal, dropout_p=drop_rate)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def tile(x, sp_size):
    x = rearrange(x, "b (sp t h w) head d -> b (t sp h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=5,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)


def untile(x, sp_size):
    x = rearrange(x,
                  "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
                  n_t=5,
                  n_h=6,
                  n_w=10,
                  ts_t=6,
                  ts_h=8,
                  ts_w=8)
    return rearrange(x, "b (t sp h w) head d -> b (sp t h w) head d", sp=sp_size, t=30 // sp_size, h=48, w=80)


def parallel_attention(q, k, v, text_mask, mask_strategy=None):
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    if mask_strategy[0] is not None:
        query = torch.cat([tile(query, nccl_info.sp_size), encoder_query], dim=1).transpose(1, 2)
        key = torch.cat([tile(key, nccl_info.sp_size), encoder_key], dim=1).transpose(1, 2)
        value = torch.cat([tile(value, nccl_info.sp_size), encoder_value], dim=1).transpose(1, 2)

        head_num = query.size(1)
        current_rank = nccl_info.rank_within_group
        start_head = current_rank * head_num
        windows = [mask_strategy[head_idx + start_head] for head_idx in range(head_num)]

        hidden_states = sliding_tile_attention(query, key, value, windows, text_length).transpose(1, 2)
    else:
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = _attn_compute(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)

    if mask_strategy[0] is not None:
        hidden_states = untile(hidden_states, nccl_info.sp_size)

    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states, dim=2).contiguous()

    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
