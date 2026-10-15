# -*- coding: utf-8 -*-
"""시험용: B300(torch cu130) cuDNN 어텐션의 버그를 CPU에서 흉내 냄.
전부 가려진 질의 행(왼쪽 패딩 자리)이 있으면 순전파는 그대로 두고, 역전파에서 그 행의 query 기울기를 NaN으로 만든다.
(실제 cuDNN: 그 행의 logsumexp가 -inf → 역전파에서 exp(s - lse) = NaN → dQ·dK·dV로 번짐. LoRA 기울기는 위치를 모두 더하므로 NaN 하나면 전체가 NaN)"""
import torch
import torch.nn.functional as F

_REAL = F.scaled_dot_product_attention
CALLS = {"n": 0, "poisoned": 0}


class _PoisonRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows):
        ctx.save_for_backward(rows)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        (rows,) = ctx.saved_tensors
        return g.masked_fill(rows, float("nan")), None


def buggy_sdpa(query, key, value, attn_mask=None, *args, **kwargs):
    CALLS["n"] += 1
    if attn_mask is not None and torch.is_grad_enabled() and query.requires_grad:
        if attn_mask.dtype == torch.bool:
            dead = ~attn_mask.any(dim=-1, keepdim=True)
        else:
            dead = torch.isneginf(attn_mask).all(dim=-1, keepdim=True)
        if bool(dead.any()):
            CALLS["poisoned"] += 1
            query = _PoisonRows.apply(query, dead)
    return _REAL(query, key, value, attn_mask, *args, **kwargs)


def install():
    F.scaled_dot_product_attention = buggy_sdpa
    torch.nn.functional.scaled_dot_product_attention = buggy_sdpa


def uninstall():
    F.scaled_dot_product_attention = _REAL
    torch.nn.functional.scaled_dot_product_attention = _REAL
