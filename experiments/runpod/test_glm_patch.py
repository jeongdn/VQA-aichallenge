# -*- coding: utf-8 -*-
"""glm_patch.py 대조 시험 (작은 glm5_next, CPU)
A. FP8 그대로 불러온 모델 + 보정  vs  dequantize=True로 bf16(여기선 float32) 복원한 모델 → 로짓이 같은지 (이미지 포함, 왼쪽 패딩 배치)
B. LoRA 학습: 같은 LoRA 초기값에서 손실·LoRA 기울기가 같은지, FP8 가중치는 FP8로 남는지
C. 이미지 처리: 28 배수로 미리 맞춘 이미지는 프로세서가 크기를 바꾸지 않는지 (토큰 = h·w/784)
D. 템플릿·토크나이저: [gMASK]<sop> 중복 감지, 생성 프롬프트 끝, 글자 토큰"""
import os, sys, math, glob, copy, re, torch
sys.path.insert(0, "/home/user/vqa/runpod")
import glm_patch as GP
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
from peft import LoraConfig, get_peft_model

D = "/tmp/vqa_scratch/runpod/tinyglm"
F8, BF = f"{D}/fp8", f"{D}/bf16"
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)

# ---------- 불러오기 ----------
ref = AutoModelForImageTextToText.from_pretrained(F8, dtype=torch.float32, quantization_config=FineGrainedFP8Config(dequantize=True)).eval()
ok(not any(isinstance(m, (FP8Linear, FP8Experts)) for m in ref.modules()), "기준: dequantize=True → FP8 층 없음 (float32 복원)")
_ia, _gc = torch.cuda.is_available, torch.cuda.get_device_capability
torch.cuda.is_available = lambda: True; torch.cuda.get_device_capability = lambda *a, **k: (9, 0)     # CPU에서 FP8 그대로 불러오기 위한 시험용 속임
try:
    f8 = AutoModelForImageTextToText.from_pretrained(F8, dtype=torch.float32, device_map={"": "cpu"}).eval()
finally:
    torch.cuda.is_available, torch.cuda.get_device_capability = _ia, _gc
nl = sum(isinstance(m, FP8Linear) for m in f8.modules()); ne = sum(isinstance(m, FP8Experts) for m in f8.modules())
ok(nl > 0 and ne == 3, f"FP8 그대로: FP8Linear {nl}개 · FP8Experts {ne}개 (MoE 3층)")
fp8_bytes = sum(p.numel() for n, p in f8.named_parameters() if p.dtype == torch.float8_e4m3fn)
n_lin, n_exp = GP.install_fp8_train_patch(f8, compute_dtype=torch.float32)
nl8 = sum(isinstance(m, FP8Linear) and m.weight.element_size() == 1 for m in f8.modules())
ok((n_lin, n_exp) == (nl8, ne), f"보정 설치: FP8 가중치 선형 {n_lin}/{nl8} · 전문가 {n_exp} (FP8Linear {nl - nl8}개는 체크포인트가 bf16이라 원래도 보통 행렬곱: KDA forget_gate — 이름 정규화 차이, 무해)")

# ---------- 입력 (실제 프로세서, 이미지 2장, 왼쪽 패딩) ----------
proc = AutoProcessor.from_pretrained(F8); proc.tokenizer.padding_side = "left"; tok = proc.tokenizer
imgs = sorted(glob.glob("/tmp/vqa_sample_images/*.jpg"))[:2]
def fit28(im, mn, mx, f=28):
    w, h = im.size; hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
    elif hb * wb < mn: b = math.sqrt(mn / (h * w)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
    return im.resize((wb, hb), Image.BICUBIC)
ims = [fit28(Image.open(p).convert("RGB"), 1024 * 12, 1024 * 16) for p in imgs]
msgs = [[{"role": "system", "content": [{"type": "text", "text": "You are a helpful visual question answering assistant."}]},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in ["표지판은 어디로 향하는 길을 안내하고 있나요?\n(a) 역삼\n(b) 선릉\n(c) 사상\n(d) 강남", "무엇?"]]
texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "</think>" for m in msgs]
enc = proc(text=texts, images=ims, return_tensors="pt", padding=True, add_special_tokens=False, min_image_tokens=1, max_image_tokens=16 * 1024 // 784 + 64)
n_img = (enc["input_ids"] == proc.image_token_id).sum(1).tolist()
ok(n_img == [im.size[0] * im.size[1] // 784 for im in ims], f"C. 28 배수 이미지는 그대로: 이미지 토큰 {n_img} = h·w/784 {[im.size[0] * im.size[1] // 784 for im in ims]} (크기 {[im.size for im in ims]})")
ok(texts[0].startswith("[gMASK]<sop><|system|>Reasoning Effort: Low") and texts[0].endswith("<|assistant|><think></think>"), "D. 템플릿: [gMASK]<sop> 시작 · reasoning_effort=low · 답 자리 앞 '<|assistant|><think></think>'")
dup = tok(texts[0], add_special_tokens=True).input_ids[:4]; nodup = tok(texts[0], add_special_tokens=False).input_ids[:2]
ok(dup[:2] == dup[2:4] == nodup, "D. add_special_tokens=True면 [gMASK]<sop>가 두 번 → 노트북은 add_special_tokens=False로 (중복 방지)")

# ---------- A. 로짓 ----------
with torch.no_grad():
    la = ref(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]
    lb = f8(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]
d = (la - lb).abs().max().item()
ok(d < 1e-4, f"A. FP8 그대로+보정 로짓 = dequantize 복원 모델 로짓 (최대 차이 {d:.1e})")

# ---------- B. LoRA 학습 ----------
TGT = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
cfgL = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TGT)
torch.manual_seed(0); pa = get_peft_model(copy.deepcopy(ref), cfgL)
torch.manual_seed(0); pb = get_peft_model(f8, cfgL)
nr = GP.reinit_lora_on_fp8(pb, seed=42)
ok(nr > 0, f"FP8 층에 붙은 LoRA {nr}개 다시 초기화 (float32)")
la_names = sorted(n for n, _ in pa.named_parameters() if "lora_" in n); lb_names = sorted(n for n, _ in pb.named_parameters() if "lora_" in n)
ok(la_names == lb_names and len(la_names) > 0, f"LoRA 층 이름·개수 같음 ({len(la_names) // 2}층, 비전 {sum('visual' in n for n in la_names)}개)")
sa, sb = dict(pa.named_parameters()), dict(pb.named_parameters())
with torch.no_grad():
    for n in la_names: sa[n].data.copy_(sb[n].data.to(sa[n].dtype))            # 같은 초기값에서 비교 (B는 0이라 한 번 스텝 후 비교)
ok(all(sb[n].dtype == torch.float32 for n in lb_names), "LoRA 가중치는 float32 (FP8로 뭉개지지 않음)")
ok(all(p.dtype == torch.float8_e4m3fn for n, p in pb.named_parameters() if n.endswith("weight") and "lora_" not in n and p.element_size() == 1), "기반 가중치는 FP8 그대로")
for m in (pa, pb):
    m.train(); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.no_grad():                                                        # lora_B를 0이 아닌 같은 값으로 → A 기울기도 살아 있게
        g = torch.Generator().manual_seed(7)
        for n in sorted(k for k in dict(m.named_parameters()) if "lora_B" in k):
            dict(m.named_parameters())[n].data = torch.randn(dict(m.named_parameters())[n].shape, generator=g) * 0.01
tgt = torch.tensor([5, 7])
losses, grads = [], []
for m in (pa, pb):
    out = m(**enc, logits_to_keep=1, use_cache=False)
    loss = torch.nn.functional.cross_entropy(out.logits[:, -1].float(), tgt); loss.backward(); losses.append(loss.item())
    grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters() if "lora_" in n and p.grad is not None})
ok(abs(losses[0] - losses[1]) < 1e-5, f"B. 손실 같음 ({losses[0]:.6f} vs {losses[1]:.6f})")
common = sorted(set(grads[0]) & set(grads[1]))
gd = max((grads[0][n] - grads[1][n]).abs().max().item() for n in common)
gmag = max(grads[0][n].abs().max().item() for n in common)
ok(len(common) == len(la_names) and gd < 1e-5 * max(1.0, gmag), f"B. LoRA 기울기 {len(common)}개 모두 같음 (최대 차이 {gd:.1e}, 크기 {gmag:.1e})")
fp8_names = [n for n, mod in pb.named_modules() if hasattr(mod, "base_layer") and isinstance(mod.base_layer, FP8Linear)]
ok(len(fp8_names) > 0 and all(grads[1].get(n + ".lora_A.default.weight") is not None for n in fp8_names), f"B. FP8 층 위 LoRA({len(fp8_names)}개)에도 기울기가 흐름")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
