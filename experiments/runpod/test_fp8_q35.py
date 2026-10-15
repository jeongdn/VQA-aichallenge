# -*- coding: utf-8 -*-
"""glm_patch.py의 FP8 학습 보정을 Qwen3.5 MoE(qwen3_5_moe) 공식 FP8 형식에 대조 (작은 397B-FP8, CPU)
A. FP8 그대로 + 보정 로짓 = dequantize=True로 복원한 모델 로짓 (이미지 포함 왼쪽 패딩 배치)
B. LoRA: 같은 초기값에서 손실·LoRA 기울기가 같음 · FP8 기반 가중치는 FP8로 남음 · FP8 층 위 LoRA에도 기울기
C. 선형 어텐션(GDN)·라우터·비전은 FP8이 아님 (공식 modules_to_not_convert와 같은 규칙)"""
import os, sys, glob, copy, torch
sys.path.insert(0, "/home/user/vqa/runpod")
import glm_patch as GP
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
from peft import LoraConfig, get_peft_model

D = "/tmp/vqa_scratch/runpod/tiny397b_fp8"
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)

ref = AutoModelForImageTextToText.from_pretrained(D, dtype=torch.float32, quantization_config=FineGrainedFP8Config(dequantize=True)).eval()
ok(not any(isinstance(m, (FP8Linear, FP8Experts)) for m in ref.modules()), "기준: dequantize=True → FP8 층 없음")
_ia, _gc = torch.cuda.is_available, torch.cuda.get_device_capability
torch.cuda.is_available = lambda: True; torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
try: f8 = AutoModelForImageTextToText.from_pretrained(D, dtype=torch.float32, device_map={"": "cpu"}).eval()
finally: torch.cuda.is_available, torch.cuda.get_device_capability = _ia, _gc
n_lin, n_exp = GP.install_fp8_train_patch(f8, compute_dtype=torch.float32)
ok((n_lin, n_exp) == (16, 4), f"FP8 보정 설치: 선형 {n_lin}개(전체 어텐션 q/k/v/o 4 + 공유 전문가 12) · 전문가 {n_exp}개(MoE 4층)")
fp8_mods = {n for n, m in f8.named_modules() if isinstance(m, FP8Linear)}
ok(not any(("linear_attn" in n or "visual" in n or n.endswith("mlp.gate") or "shared_expert_gate" in n or "lm_head" in n) for n in fp8_mods),
   "C. 선형 어텐션·라우터·shared_expert_gate·비전·lm_head는 FP8이 아님")

proc = AutoProcessor.from_pretrained(D); proc.tokenizer.padding_side = "left"
imgs = sorted(glob.glob("/tmp/vqa_sample_images/*.jpg"))[:2]
proc.image_processor.size = {"shortest_edge": 1024 * 3, "longest_edge": 1024 * 6}
msgs = [[{"role": "system", "content": [{"type": "text", "text": "You are a helpful visual question answering assistant."}]},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in ["표지판은 어디로 향하는 길을 안내하고 있나요?\n(a) 역삼\n(b) 선릉\n(c) 사상\n(d) 강남", "무엇?"]]
texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
enc = proc(text=texts, images=[Image.open(p).convert("RGB") for p in imgs], return_tensors="pt", padding=True)
ok((enc["attention_mask"] == 0).sum(1).max() > 0, f"왼쪽 패딩 배치 {tuple(enc['input_ids'].shape)}")
with torch.no_grad():
    la = ref(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]
    lb = f8(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]
d = (la - lb).abs().max().item()
ok(d < 1e-4, f"A. FP8 그대로+보정 로짓 = dequantize 복원 로짓 (최대 차이 {d:.1e})")

TGT = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
cfgL = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TGT)
torch.manual_seed(0); pa = get_peft_model(copy.deepcopy(ref), cfgL)
torch.manual_seed(0); pb = get_peft_model(f8, cfgL)
nr = GP.reinit_lora_on_fp8(pb, seed=42)
ok(nr == 16, f"FP8 층 위 LoRA {nr}개 float32로 다시 초기화")
sa, sb = dict(pa.named_parameters()), dict(pb.named_parameters())
names = sorted(n for n in sb if "lora_" in n)
ok(names == sorted(n for n in sa if "lora_" in n) and len(names) > 0, f"LoRA 층 이름·개수 같음 ({len(names) // 2}층)")
with torch.no_grad():
    for n in names: sa[n].data.copy_(sb[n].data.to(sa[n].dtype))
ok(all(p.dtype == torch.float8_e4m3fn for n, p in pb.named_parameters() if "lora_" not in n and p.element_size() == 1), "기반 가중치는 FP8 그대로")
for m in (pa, pb):
    m.train(); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for n in sorted(k for k in dict(m.named_parameters()) if "lora_B" in k):
            dict(m.named_parameters())[n].data = torch.randn(dict(m.named_parameters())[n].shape, generator=g) * 0.01
tgt = torch.tensor([5, 7]); losses, grads = [], []
for m in (pa, pb):
    out = m(**enc, logits_to_keep=1, use_cache=False)
    loss = torch.nn.functional.cross_entropy(out.logits[:, -1].float(), tgt); loss.backward(); losses.append(loss.item())
    grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters() if "lora_" in n and p.grad is not None})
ok(abs(losses[0] - losses[1]) < 1e-5, f"B. 손실 같음 ({losses[0]:.6f} vs {losses[1]:.6f})")
common = sorted(set(grads[0]) & set(grads[1]))
gd = max((grads[0][n] - grads[1][n]).abs().max().item() for n in common); gm = max(grads[0][n].abs().max().item() for n in common)
ok(len(common) == len(names) and gd < 1e-5 * max(1.0, gm), f"B. LoRA 기울기 {len(common)}개 모두 같음 (최대 차이 {gd:.1e}, 크기 {gm:.1e})")
fp8_names = [n for n, mod in pb.named_modules() if hasattr(mod, "base_layer") and isinstance(mod.base_layer, FP8Linear)]
ok(len(fp8_names) == 16 and all(grads[1].get(n + ".lora_A.default.weight") is not None for n in fp8_names), f"B. FP8 층 위 LoRA({len(fp8_names)}개)에도 기울기")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
