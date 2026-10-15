# === 공통 보정: SDPA에 '전부 가려진 행'을 넘기지 않기 (B300 · CUDA 13에서 첫 스텝부터 기울기 NaN이던 원인) ===
# 문항 여러 개를 한 배치로 묶으면 짧은 문항 앞에 패딩이 붙는다(왼쪽 패딩). 패딩 자리의 질의 행은 볼 수 있는 키가 하나도 없다(마스크 행이 전부 False).
# torch cu130의 cuDNN 어텐션은 이런 행의 순전파는 멀쩡히 내지만 역전파에서 NaN을 낸다 → LoRA 기울기 전체가 NaN.
#   Flash-Next DRY_RUN(B300)에서 확인: 2문항 배치 NaN 588/600 · 1문항씩 0 · 아래 처리 후 0 · cuDNN 어텐션을 끄면 0.
# → 학습 중(기울기 계산 중)에만, 전부 가려진 행을 '모든 키를 보게' 풀어서 넘긴다.
#   진짜 토큰의 행은 한 칸도 바뀌지 않아 결과가 같다(같은 배치의 손실이 소수점까지 같음).
#   패딩 자리의 출력은 다음 층에서 0으로 지워지거나(선형 어텐션) 가려져서(어텐션) 손실에 닿지 않는다.
#   평가·예측(기울기 없음)은 아무것도 바꾸지 않는다.
# transformers 4.x는 같은 처리(_unmask_unattended)를 했지만 5.x에서 빠졌다. Flash-Next·GLM-5.3-Flash·122B 모두 이 함수를 거친다.
import torch


def install_sdpa_unmask_patch():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(orig, "_unmask_rows", False):
        return orig

    def sdpa_unmask_rows(module, query, key, value, attention_mask, *args, **kwargs):
        if attention_mask is not None and attention_mask.dim() >= 2 and torch.is_grad_enabled():
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask | ~attention_mask.any(dim=-1, keepdim=True)
            elif attention_mask.is_floating_point():
                dead = (attention_mask <= torch.finfo(attention_mask.dtype).min / 2).all(dim=-1, keepdim=True)
                attention_mask = attention_mask.masked_fill(dead, 0.0)
        return orig(module, query, key, value, attention_mask, *args, **kwargs)

    sdpa_unmask_rows._unmask_rows = True
    sdpa_unmask_rows._orig = orig
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_unmask_rows
    return sdpa_unmask_rows
