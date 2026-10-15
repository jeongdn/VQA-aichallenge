# === GLM-5.3-Flash(glm5_next) 보정: FP8 가중치를 그대로 두고, 계산할 때 층마다 bf16으로 풀어서 곱하기 ===
# 공식 체크포인트는 FP8(전문가·공유 전문가·dense MLP·DSA 투영)이다. bf16으로 전부 풀면 585GiB라 B300 2장(536GiB)에 안 들어간다.
# transformers의 FP8 층(FP8Linear·FP8Experts)은 추론 전용 커널이라 역전파가 안 되고, 활성값까지 FP8로 양자화한다.
# → 두 층의 forward만 "블록 스케일로 bf16 복원(transformers Fp8Dequantize와 같은 계산) → 보통 행렬곱/grouped_mm"으로 바꾼다.
#   결과는 dequantize=True로 불러온 bf16 모델과 같고(작은 모델로 대조), 역전파가 되며, GPU에는 FP8(약 305GiB)만 남는다.
import types
import torch
import torch.nn.functional as F

FP8_COMPUTE_DTYPE = torch.bfloat16


def fp8_dequant(w, s, dtype, chunk=16):
    """FP8 가중치 w (…, rows, cols) · 블록 스케일 s (…, rows/bm, cols/bn) → dtype. 3차원(전문가)은 chunk개씩 풀어 임시 메모리를 줄임"""
    if w.element_size() > 1:
        return w.to(dtype)
    if s.dim() == 0 or s.numel() == 1:
        return (w.to(torch.float32) * s.to(torch.float32)).to(dtype)

    def one(q, sc):
        rows, cols = q.shape[-2:]
        sr, scn = sc.shape[-2:]
        assert rows % sr == 0 and cols % scn == 0, f"FP8 블록 격자가 맞지 않음: {tuple(q.shape)} / {tuple(sc.shape)}"
        qf = q.to(torch.float32).reshape(-1, sr, rows // sr, scn, cols // scn)
        sf = sc.to(torch.float32).reshape(-1, sr, scn).unsqueeze(-1).unsqueeze(2)
        return (qf * sf).to(dtype).reshape(q.shape)

    if w.dim() == 2:
        return one(w, s)
    out = torch.empty(w.shape, dtype=dtype, device=w.device)
    for e in range(0, w.shape[0], chunk):
        out[e:e + chunk] = one(w[e:e + chunk], s[e:e + chunk])
    return out


def _fp8_linear_forward(self, input):
    w = fp8_dequant(self.weight, self.weight_scale_inv, FP8_COMPUTE_DTYPE)
    return F.linear(input.to(w.dtype), w, self.bias.to(w.dtype) if self.bias is not None else None)


def _fp8_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    from transformers.integrations.moe import grouped_mm_experts_forward
    dt = FP8_COMPUTE_DTYPE
    proxy = types.SimpleNamespace(num_experts=self.num_experts, has_gate=self.has_gate, has_bias=False, is_transposed=False,
                                  _apply_gate=self._apply_gate, act_fn=self.act_fn)
    if self.has_gate:
        proxy.gate_up_proj = fp8_dequant(self.gate_up_proj, self.gate_up_proj_scale_inv, dt)
    else:
        proxy.up_proj = fp8_dequant(self.up_proj, self.up_proj_scale_inv, dt)
    proxy.down_proj = fp8_dequant(self.down_proj, self.down_proj_scale_inv, dt)
    return grouped_mm_experts_forward(proxy, hidden_states.to(dt), top_k_index, top_k_weights).to(hidden_states.dtype)


def install_fp8_train_patch(model, compute_dtype=torch.bfloat16):
    """모델 안의 FP8Linear·FP8Experts 인스턴스 forward를 바꿈. 바꾼 개수 (선형, 전문가)를 돌려줌"""
    global FP8_COMPUTE_DTYPE
    from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
    FP8_COMPUTE_DTYPE = compute_dtype
    n_lin = n_exp = 0
    for m in model.modules():
        if isinstance(m, FP8Experts):
            m.forward = types.MethodType(_fp8_experts_forward, m); n_exp += 1
        elif isinstance(m, FP8Linear) and m.weight.element_size() == 1:
            m.forward = types.MethodType(_fp8_linear_forward, m); n_lin += 1
    return n_lin, n_exp


def reinit_lora_on_fp8(model, seed):
    """PEFT는 LoRA를 기반 층 dtype(FP8)으로 한 번 바꿨다가 float32로 되돌린다 → lora_A 초기값이 FP8 격자로 뭉개짐.
    FP8 층에 붙은 LoRA만 PEFT와 같은 규칙(kaiming_uniform, a=√5)으로 float32에서 다시 초기화 (lora_B는 0 그대로)."""
    import math
    from transformers.integrations.finegrained_fp8 import FP8Linear
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = 0
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and isinstance(getattr(m, "base_layer", None), FP8Linear):
            for key, lin in m.lora_A.items():
                w = torch.empty(lin.weight.shape, dtype=torch.float32)
                bound = 1.0 / math.sqrt(w.shape[1])            # kaiming_uniform_(a=√5)의 한계 = √(6/((1+5)·fan_in)) = 1/√fan_in
                w.uniform_(-bound, bound, generator=g)
                with torch.no_grad():
                    lin.weight.data = w.to(device=lin.weight.device, dtype=torch.float32)
                n += 1
    return n
