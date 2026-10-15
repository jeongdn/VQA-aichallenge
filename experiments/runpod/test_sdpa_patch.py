# -*- coding: utf-8 -*-
"""sdpa_patch.py 시험 (CPU, 작은 Flash-Next · GLM-5.3-Flash(FP8) · 122B)
0. 마스크 단위: 볼 키가 하나라도 있는 행은 그대로 · 전부 가려진 행만 풀림 (bool·float) · 기울기 없으면(평가) 손대지 않음 · 두 번 설치해도 한 번
1. 모델마다 왼쪽 패딩 배치(이미지 포함) + LoRA, 체크포인팅 켜고:
   a. 기준(보통 SDPA): 손실·LoRA 기울기 유한
   b. B300 cuDNN 버그 흉내 + 보정 없음 → LoRA 기울기 NaN (실제 증상 재현: 1문항씩이면 NaN 없음)
   c. 버그 흉내 + 보정 → 손실 = 기준, LoRA 기울기 = 기준 (유한)
   d. 평가(기울기 없음): 보정이 있어도 로짓이 비트 단위로 같음"""
import os, sys, glob, math, re, copy, torch
sys.path.insert(0, "/home/user/vqa/runpod")
import sdpa_patch as SP_, cudnn_nan_emu as EMU, flash_patch as FP, glm_patch as GP
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from peft import LoraConfig, get_peft_model

SPD = "/tmp/vqa_scratch/runpod"
IMGS = sorted(glob.glob("/tmp/vqa_sample_images/*.jpg"))[:2]
assert len(IMGS) == 2, "시험 이미지 없음"
ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
def set_patch(on):
    if on: SP_.install_sdpa_unmask_patch()
    else: ALL_ATTENTION_FUNCTIONS["sdpa"] = ORIG_SDPA

# ───────── 0. 마스크 단위 ─────────
seen = {}
def spy(module, q, k, v, m, *a, **kw):
    seen["m"] = m; return q.new_zeros(q.shape[0], q.shape[2], q.shape[1], v.shape[-1]), None
ALL_ATTENTION_FUNCTIONS["sdpa"] = spy
w = SP_.install_sdpa_unmask_patch(); w2 = SP_.install_sdpa_unmask_patch()
ok(w is w2 and w._orig is spy and ALL_ATTENTION_FUNCTIONS["sdpa"] is w, "두 번 설치해도 한 번만 감쌈")
g = torch.Generator().manual_seed(0)
mb = torch.rand(3, 1, 9, 9, generator=g) > 0.6
mb[0, 0, :4] = False; mb[2, 0, 7] = False                                   # 전부 가려진 행 5개
q = torch.randn(3, 2, 9, 4, requires_grad=True)
w(None, q, q, q, mb); out = seen["m"]
alive = mb.any(-1)
ok(torch.equal(out[alive], mb[alive]) and bool(out[~alive].all()) and int((~alive).sum()) == 5,
   f"bool: 볼 키가 있는 행 {int(alive.sum())}개는 그대로 · 전부 가려진 행 {int((~alive).sum())}개만 모두 True")
mn = torch.finfo(torch.float32).min
mf = torch.where(mb, torch.zeros(()), torch.tensor(mn)); mf[1, 0, 3] = float("-inf")
w(None, q, q, q, mf); outf = seen["m"]
alive_f = (mf > mn / 2).any(-1)
ok(torch.equal(outf[alive_f], mf[alive_f]) and bool((outf[~alive_f] == 0).all()), "float(min·-inf): 전부 가려진 행만 0으로")
with torch.no_grad(): w(None, q, q, q, mb)
ok(seen["m"] is mb, "기울기 없음(평가·예측): 마스크를 그대로 넘김 (같은 객체)")
w(None, q, q, q, None); ok(seen["m"] is None, "마스크 None(패딩 없음)은 그대로")
ALL_ATTENTION_FUNCTIONS["sdpa"] = ORIG_SDPA

# ───────── 1. 모델 ─────────
def fit28(im, mn, mx, f=28):
    w_, h = im.size; hb, wb = max(f, round(h / f) * f), max(f, round(w_ / f) * f)
    if hb * wb > mx: b = math.sqrt(h * w_ / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w_ / b / f) * f)
    elif hb * wb < mn: b = math.sqrt(mn / (h * w_)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w_ * b / f) * f
    return im.resize((wb, hb), Image.BICUBIC)

QS = ["표지판은 어디로 향하는 길을 안내하고 있나요? 사진 속 글자를 잘 읽어 보세요.\n(a) 역삼\n(b) 선릉\n(c) 사상\n(d) 강남", "무엇?\n(a) 1\n(b) 2\n(c) 3\n(d) 4"]
TGT = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"

def load(var):
    if var == "glm":
        d = f"{SPD}/tinyglm/fp8"
        _ia, _gc = torch.cuda.is_available, torch.cuda.get_device_capability
        torch.cuda.is_available = lambda: True; torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
        try: m = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.float32, device_map={"": "cpu"}, attn_implementation="sdpa")
        finally: torch.cuda.is_available, torch.cuda.get_device_capability = _ia, _gc
        GP.install_fp8_train_patch(m, compute_dtype=torch.float32)
    else:
        d = f"{SPD}/tinyflash" if var == "flash" else f"{SPD}/tiny122b"
        m = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa")
        if var == "flash": FP.install_qsa_patch()
    proc = AutoProcessor.from_pretrained(d); proc.tokenizer.padding_side = "left"
    msgs = [[{"role": "system", "content": [{"type": "text", "text": "You are a helpful visual question answering assistant."}]},
             {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in QS]
    if var == "glm":
        texts = [proc.apply_chat_template(x, tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "</think>" for x in msgs]
        ims = [fit28(Image.open(p).convert("RGB"), 1024 * 3, 1024 * 6) for p in IMGS]
        enc = proc(text=texts, images=ims, return_tensors="pt", padding=True, add_special_tokens=False, min_image_tokens=1, max_image_tokens=6 * 1024 // 784 + 64)
    else:
        proc.image_processor.size = {"shortest_edge": 1024 * 3, "longest_edge": 1024 * 6}
        texts = [proc.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in msgs]
        enc = proc(text=texts, images=[Image.open(p).convert("RGB") for p in IMGS], return_tensors="pt", padding=True)
    torch.manual_seed(0)
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TGT))
    if var == "glm": GP.reinit_lora_on_fp8(pm, seed=1)
    gg = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n: p.copy_(torch.randn(p.shape, generator=gg) * 0.02)     # B≠0 → A 기울기도 살아 있게
    pm.train(); pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return pm, enc

def step(pm, enc, rows=None):
    e = {k: (v[rows] if rows is not None and torch.is_tensor(v) and v.shape[0] == 2 else v) for k, v in enc.items()}
    if rows is not None:                                     # 1문항: 패딩 열 제거 + 그 문항 이미지만
        keep = e["attention_mask"][0].bool(); e["input_ids"] = e["input_ids"][:, keep]; e["attention_mask"] = e["attention_mask"][:, keep]
        if "mm_token_type_ids" in e: e["mm_token_type_ids"] = e["mm_token_type_ids"][:, keep]
        if "image_grid_thw" in enc:
            thw = enc["image_grid_thw"]; npat = (thw[:, 0] * thw[:, 1] * thw[:, 2]).tolist(); r = rows.start
            e["image_grid_thw"] = thw[r:r + 1]; e["pixel_values"] = enc["pixel_values"][sum(npat[:r]):sum(npat[:r + 1])]
    pm.zero_grad(set_to_none=True)
    out = pm(**e, logits_to_keep=1, use_cache=False)
    tgt = torch.tensor([5, 7])[: out.logits.shape[0]]
    loss = torch.nn.functional.cross_entropy(out.logits[:, -1].float(), tgt); loss.backward()
    gr = {n: p.grad.detach().clone() for n, p in pm.named_parameters() if "lora_" in n and p.grad is not None}
    return loss.item(), gr

for var in ("flash", "glm", "122b"):
    print(f"\n━━ {var} ━━", flush=True)
    set_patch(False); EMU.uninstall()
    pm, enc = load(var)
    npad = (enc["attention_mask"] == 0).sum(1).tolist()
    ok(max(npad) > 0 and min(npad) == 0, f"{var}: 왼쪽 패딩 배치 {tuple(enc['input_ids'].shape)} · 패딩 {npad}")
    la, ga = step(pm, enc)
    bad_a = sum(not torch.isfinite(v).all() for v in ga.values())
    ok(len(ga) > 0 and bad_a == 0, f"a. 기준(보통 SDPA): 손실 {la:.6f} · LoRA 기울기 {len(ga)}개 모두 유한")
    EMU.install(); EMU.CALLS.update(n=0, poisoned=0)
    lb, gb = step(pm, enc)
    bad_b = sum(not torch.isfinite(v).all() for v in gb.values())
    ok(bad_b > 0 and EMU.CALLS["poisoned"] > 0, f"b. 버그 흉내 + 보정 없음: LoRA 기울기 NaN {bad_b}/{len(gb)} (전부 가려진 행이 있는 SDPA 호출 {EMU.CALLS['poisoned']}/{EMU.CALLS['n']}) — 실제 B300 증상")
    l1, g1 = step(pm, enc, rows=slice(1, 2))
    ok(all(torch.isfinite(v).all() for v in g1.values()), "b'. 버그 흉내 + 1문항(패딩 없음): NaN 없음 — 진단 ②와 같음")
    set_patch(True); EMU.CALLS.update(n=0, poisoned=0)
    lc, gc_ = step(pm, enc)
    bad_c = sum(not torch.isfinite(v).all() for v in gc_.values())
    dg = max((ga[n] - gc_[n]).abs().max().item() for n in ga); mg = max(v.abs().max().item() for v in ga.values())
    ok(bad_c == 0 and EMU.CALLS["poisoned"] == 0 and EMU.CALLS["n"] > 0 and abs(la - lc) < 1e-6 and dg <= 1e-5 * max(1.0, mg),
       f"c. 버그 흉내 + 보정: NaN 0 · 손실 {lc:.6f} = 기준 · LoRA 기울기 최대 차이 {dg:.1e} (크기 {mg:.1e})")
    EMU.uninstall()
    pm.eval()
    with torch.no_grad():
        set_patch(False); lo0 = pm(**enc, logits_to_keep=1, use_cache=False).logits
        set_patch(True); lo1 = pm(**enc, logits_to_keep=1, use_cache=False).logits
    ok(torch.equal(lo0, lo1), "d. 평가(기울기 없음): 보정 있어도 로짓 비트 단위로 같음")
    set_patch(False)
    del pm

print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
