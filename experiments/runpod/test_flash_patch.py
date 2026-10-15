# -*- coding: utf-8 -*-
"""flash_patch.py 대조 시험 (작은 qwen4_exp, CPU)
A. QSA 인덱서: 원래 구현(질의마다 루프) vs 벡터화 — 선택 마스크가 같은지 (왼쪽 패딩·배치·여러 길이·bool/float 마스크)
B. 모델 전체: 패치 전/후 로짓이 같은지 + 역전파가 도는지 (이미지 포함 실제 프로세서 입력)
C. 불러오기: n-gram 조각을 빼고 불러온 뒤 채운 표 = transformers 기본 불러오기 결과 (값·로짓 일치), device_map에서 표만 빠짐
D. 속도: 원래 vs 벡터화 (S=512)"""
import os, sys, time, glob, copy, torch
sys.path.insert(0, "/home/user/vqa/runpod")
import flash_patch as FP
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as Q

D = "/tmp/vqa_scratch/runpod/tinyflash"
torch.manual_seed(0)
ORIG = Q.Qwen4ExpTextQSAIndexer.forward
real_relu = torch.relu
def no_tie_relu(x): return x.abs() + 1e-3          # 동점(0)이 없게: 두 구현이 같은 점수 함수를 쓰므로 선택 논리 대조가 정확해짐

ref = AutoModelForImageTextToText.from_pretrained(D, dtype=torch.float32, attn_implementation="sdpa").eval()
proc = AutoProcessor.from_pretrained(D); proc.tokenizer.padding_side = "left"

# ---------- A. 인덱서 입력 수집 ----------
cap = []
def spy(self, h, pe, am, pkv):
    cap.append((self, h.detach().clone(), tuple(x.clone() for x in pe), am.clone())); return ORIG(self, h, pe, am, pkv)
Q.Qwen4ExpTextQSAIndexer.forward = spy
for S, pads in [(40, [0, 13, 5]), (97, [0, 30]), (130, [7, 0, 64]), (19, [0, 3])]:
    ids = torch.randint(10, 1400, (len(pads), S)); att = torch.ones_like(ids)
    for b, p in enumerate(pads): att[b, :p] = 0; ids[b, :p] = proc.tokenizer.pad_token_id
    with torch.no_grad(): ref(input_ids=ids, attention_mask=att)
Q.Qwen4ExpTextQSAIndexer.forward = ORIG
FP.install_qsa_patch(); FAST = Q.Qwen4ExpTextQSAIndexer.forward
assert Q.Qwen4ExpTextQSAIndexer._orig_forward is ORIG
n_rows = n_bad_tie = 0
for mod, h, pe, am in cap:
    for relu, exact in ((no_tie_relu, True), (real_relu, False)):
        torch.relu = relu
        try:
            with torch.no_grad(): a, b = ORIG(mod, h, pe, am, None), FAST(mod, h, pe, am, None)
        finally: torch.relu = real_relu
        assert a.shape == b.shape and a.dtype == b.dtype
        if exact: assert torch.equal(a, b), f"마스크 불일치 S={h.shape[1]} 행 {(a != b).any(-1).nonzero()[:5].tolist()}"
        else:
            n_rows += a.shape[0] * a.shape[2]; n_bad_tie += int((a != b).any(-1).sum())
            assert torch.equal(a.sum(-1), b.sum(-1)), "실제 relu: 선택된 토큰 수가 달라짐"
    # float(eager) 마스크 경로
    amf = torch.where(am, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    torch.relu = no_tie_relu
    try:
        with torch.no_grad(): a, b = ORIG(mod, h, pe, amf, None), FAST(mod, h, pe, amf, None)
    finally: torch.relu = real_relu
    assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
print(f"✅ A. 인덱서 {len(cap)}번 호출(S=19~130, 왼쪽 패딩 포함): 원래 구현과 선택 마스크 완전히 같음 (bool·float 마스크)")
print(f"   실제 relu(점수 0 동점 있음): 행 {n_rows}개 중 {n_bad_tie}개만 동점 처리 순서가 달라 다른 블록을 고름 — 고르는 토큰 수는 모두 같음")

# ---------- B. 모델 전체 (이미지 포함) ----------
imgs = sorted(glob.glob("/tmp/vqa_sample_images/*.jpg"))[:2]
assert imgs, "시험 이미지 없음"
proc.image_processor.size = {"shortest_edge": 1024 * 12, "longest_edge": 1024 * 16}
msgs = [[{"role": "system", "content": [{"type": "text", "text": "You are a helpful visual question answering assistant."}]},
         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in ["표지판은 어디로 향하는 길을 안내하고 있나요?\n(a) 역삼\n(b) 선릉\n(c) 사상\n(d) 강남", "무엇?"]]
texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
enc = proc(text=texts, images=[Image.open(p).convert("RGB") for p in imgs], return_tensors="pt", padding=True)
print("   입력 길이", enc["input_ids"].shape, "| 패딩", (enc["attention_mask"] == 0).sum(1).tolist())
assert enc["input_ids"].shape[1] > 16 + 3, "QSA 희소 선택이 일어나려면 budget(16)보다 길어야 함"
fast_m = copy.deepcopy(ref)
Q.Qwen4ExpTextQSAIndexer.forward = ORIG
torch.relu = no_tie_relu
with torch.no_grad(): lo_ref = ref(**enc).logits
Q.Qwen4ExpTextQSAIndexer.forward = FAST
with torch.no_grad(): lo_fast = fast_m(**enc).logits
torch.relu = real_relu
d = (lo_ref[:, -1] - lo_fast[:, -1]).abs().max().item()
assert d < 1e-5, d
print(f"✅ B. 모델 전체 로짓(마지막 위치) 차이 최대 {d:.1e} — 패치 전/후 같음")
fast_m.train(); fast_m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
out = fast_m(**enc); loss = torch.nn.functional.cross_entropy(out.logits[:, -1], torch.tensor([5, 7])); loss.backward()
g = fast_m.model.language_model.layers[3].self_attn.q_proj.weight.grad
assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
assert fast_m.model.language_model.layers[3].self_attn.indexer.index_qk_proj.weight.grad is None   # 인덱서는 선택만 → 기울기 없음(원래도 0)
print("✅ B. 체크포인팅 켠 학습 역전파 OK (q_proj 기울기 유한, 인덱서는 기울기 없음)")

# ---------- C. 불러오기 ----------
cfg = AutoConfig.from_pretrained(D)
dm, skip = FP.flash_device_map_1gpu(cfg, gpu="cpu")
assert skip == ["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"], skip
assert set(dm.values()) == {"cpu"} and not any("ngram_embedding" in k for k in dm) and "" not in dm
assert "model.language_model.layers.1.ple.ple_embedding.layer_multipliers" in dm and "model.language_model.layers.0" in dm and "lm_head" in dm
from transformers.integrations.accelerate import expand_device_map
names = [n for n, _ in ref.named_parameters()] + [n for n, _ in ref.named_buffers()]
ex = expand_device_map(dm, names)
not_covered = [n for n in names if not any(n == k or n.startswith(k + ".") for k in dm)]
assert not_covered == skip, not_covered
print(f"✅ C. 1장용 device_map: 키 {len(dm)}개, n-gram 표만 빠짐 → 나머지 {len(names) - 1}개 텐서는 모두 지정 장치")
shards = FP.ngram_shards(D)
assert [s[2] for s in shards] == [0, 1, 2, 3] and all(s[3].endswith("ple_embedding.ngram_embedding") for s in shards)
logs = []
for mode, kw in (("cpu", dict(dtype=torch.float32, attn_implementation="sdpa")), ("gpu", dict(dtype=torch.float32, attn_implementation="sdpa"))):
    m2 = FP.load_flash_model(D, kw, mode, log=logs.append, gpu="cpu")
    W1 = ref.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight
    W2 = m2.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight
    assert torch.equal(W1, W2), "채운 n-gram 표가 기본 불러오기와 다름"
    sd1, sd2 = ref.state_dict(), m2.state_dict()
    assert sd1.keys() == sd2.keys() and all(torch.equal(sd1[k], sd2[k]) for k in sd1), "다른 가중치가 달라짐"
    m2.eval()
    with torch.no_grad(): lo2 = m2(**enc).logits
    with torch.no_grad(): lo1 = fast_m.eval()(**enc).logits
    assert torch.allclose(lo1, lo2, atol=1e-6)
print("✅ C. n-gram 조각을 빼고 불러와 채운 모델 = 기본 불러오기 (모든 가중치·로짓 일치, cpu/gpu 두 방식)")
print("   " + " | ".join(logs))

# ---------- D. 속도 ----------
mod, h, pe, am = cap[0]
cfgt = ref.config.text_config
S = 512
h2 = torch.randn(1, S, cfgt.hidden_size)
pos = torch.arange(S).view(1, 1, S).expand(3, 1, S)
pe2 = ref.model.language_model.rotary_emb(h2, pos)
am2 = torch.tril(torch.ones(S, S, dtype=torch.bool))[None, None]
with torch.no_grad():
    t = time.time(); a = ORIG(mod, h2, pe2, am2, None); t_orig = time.time() - t
    t = time.time(); b = FAST(mod, h2, pe2, am2, None); t_fast = time.time() - t
assert a.sum() == b.sum()
print(f"✅ D. S={S} 인덱서 한 번: 원래 {t_orig * 1000:.0f}ms → 벡터화 {t_fast * 1000:.1f}ms ({t_orig / t_fast:.0f}배)")
