# -*- coding: utf-8 -*-
"""모델 단위 점검: AutoModelForImageTextToText 로드 → stage2와 같은 LoRA 정규식 → (가능하면 실제 프로세서 입력으로) 순전파·역전파."""
import sys, re, torch, numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
SP = "/tmp/vqa_scratch/newfam"
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"

def check(name, enc_fn):
    d = f"{SP}/tiny_{name}"
    base = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.float32)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS))
    if hasattr(base, "disable_input_require_grads") and getattr(base, "_require_grads_hooks", None): base.disable_input_require_grads()
    names = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    vis = [n for n in names if any(k in n for k in ("visual", "vision_tower", "vision_model"))]
    kinds = sorted({re.sub(r".*\.(\w+)\.lora_A$", r"\1", n) for n in names})
    print(f"[{name}] {type(base).__name__} | LoRA 층 {len(names)} (비전 {len(vis)}) | 종류 {kinds}")
    assert names and not vis
    model.train()
    enc = enc_fn()
    out = model(**enc, logits_to_keep=1, use_cache=False)
    lg = out.logits[:, -1, :].float()
    loss = torch.nn.functional.cross_entropy(lg, torch.zeros(lg.shape[0], dtype=torch.long))
    loss.backward()
    g = sum(p.grad.abs().sum().item() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None)
    print(f"   logits {tuple(out.logits.shape)} | loss {loss.item():.3f} | LoRA grad 합 {g:.3e}")
    assert g > 0 and out.logits.shape[1] == 1

def enc_real(name, kw):
    def f():
        p = AutoProcessor.from_pretrained(f"{SP}/tiny_{name}")
        msgs = [{"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "질문 (a) 1 (b) 2"}]}]
        t = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ims = [Image.fromarray((np.random.RandomState(i).rand(70, 90, 3) * 255).astype("uint8")) for i in range(2)]
        e = p(text=[t, t + " 더 긴 문장"], images=ims, padding=True, return_tensors="pt", **kw)
        print("   입력 키:", sorted(e.keys()), "| 이미지 토큰 수:", (e["input_ids"] == p.image_token_id).sum(1).tolist())
        return e
    return f

def enc_fake_qwen():
    sys.path.insert(0, "/home/user/vqa/e1")
    from tiny_qwen import FakeQwenProcessor
    p = FakeQwenProcessor(); p.tokenizer.padding_side = "left"
    ims = [Image.fromarray((np.random.RandomState(i).rand(20, 30, 3) * 255).astype("uint8")) for i in range(2)]
    return p(text=["a|<IMG>|질문<GEN>", "b|<IMG>|질문 두번째<GEN>"], images=ims)

def enc_text_only():
    ids = torch.randint(3, 380, (2, 12)); return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

if __name__ == "__main__":
    check("muse", enc_real("muse", {"max_image_tokens": 40}))
    check("exaone45", enc_real("exaone45", {"min_pixels": 28 * 28 * 4, "max_pixels": 28 * 28 * 30}))
    check("qwen35moe", enc_fake_qwen)
    check("gemma4moe", enc_text_only)
    print("모두 통과")
