# Muse 형식 진단 — Muse를 올린 세션에 새 셀로 붙여 넣어 실행 (모델 다시 로드 안 함, 몇 분)
import torch, transformers
transformers.logging.set_verbosity_error()                     # max_length 경고 반복 끄기
r = val_df.iloc[0].to_dict()
def _gen(enc, n):
    enc = enc.to(model.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
        o = model.generate(**enc, max_new_tokens=n, do_sample=False, use_cache=True)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)

# ① 우리 경로의 토큰 구조 (특수 토큰이 한 토큰으로 들어갔는지, BOS가 몇 개인지)
e0 = encode_batch([r], [IDENTITY]); ids = e0["input_ids"][0].tolist(); toks = tok.convert_ids_to_tokens(ids)
print("① 앞 토큰:", toks[:6], "| BOS", tok.bos_token, "앞 3개 중 BOS 수", sum(t == tok.bos_token_id for t in ids[:3]))
print("   끝 토큰:", toks[-10:])
print("   특수 토큰 id:", {t: tok.convert_tokens_to_ids(t) for t in ["<|start|>", "<|message|>", "<|eot|>", "<|patch|>"]}, "| 이미지 토큰 수", sum(t == IMAGE_TOKEN_ID for t in ids))

# ② 텍스트만 (공식 경로: apply_chat_template이 토큰화까지)
q = [{"role": "user", "content": [{"type": "text", "text": "한국의 수도는 어디인가요? 한 단어로 답하세요."}]}]
e = processor.apply_chat_template(q, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **TEMPLATE_KW)
print("② 텍스트만 · 공식 경로:", repr(_gen(e, 32)))

# ③ 이미지+질문 · 공식 경로 (system 없이, 모델 카드와 같은 방식)
im = load_image(r["path"]); mc = build_mc_prompt(str(r["question"]), [str(r[k]) for k in "abcd"])
m3 = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": mc}]}]
e = processor.apply_chat_template(m3, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **TEMPLATE_KW)
print("③ 이미지+질문 · 공식 경로 (정답", r["answer"] + "):", repr(_gen(e, 48)))

# ④ ③과 같지만 우리 system 지시문 포함
m4 = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]}] + m3
e = processor.apply_chat_template(m4, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **TEMPLATE_KW)
print("④ + system 지시문 · 공식 경로:", repr(_gen(e, 48)))

# ⑤ 추론 강도 기본값(high) · 답 채널 강제 없이 → 모델이 스스로 추론 후 답하는지
e = processor.apply_chat_template(m4, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                  **{k: v for k, v in TEMPLATE_KW.items() if k != "reasoning_strength"})
print("⑤ 추론 강도 기본 · 강제 없음:", repr(_gen(e, 160)))
# ⑥ 우리 경로 그대로(우리 system·프롬프트·답 채널 강제)인데 해상도만 native
set_resolution("native"); print("⑥ 우리 경로 · native:", repr(_gen(encode_batch([r], [IDENTITY]), 32)))
# ⑦ 우리 경로 · up2.0 · 답 채널 강제(<|message|>) 없이
_gs = GEN_SUFFIX; GEN_SUFFIX = ""; set_resolution("up2.0")
print("⑦ 우리 경로 · up2.0 · 강제 없음:", repr(_gen(encode_batch([r], [IDENTITY]), 96)))
GEN_SUFFIX = _gs; set_resolution(TRAIN_PRESET)

# ⑧ 답 자리 앞 문자열 후보별: 검증 앞 16문항의 letter_mass · 정확도 (up2.0)
def _mass_acc(suffix, n=16, preset="up2.0"):
    global GEN_SUFFIX
    _old = GEN_SUFFIX; GEN_SUFFIX = suffix; set_resolution(preset)
    rows = val_df.head(n).to_dict("records"); P, M = [], []
    for s in range(0, n, BS):
        p, m, _ = score_rows(rows[s:s + BS]); P.append(p); M.append(m)
    GEN_SUFFIX = _old; set_resolution(TRAIN_PRESET)
    P, M = np.concatenate(P), np.concatenate(M)
    return float(M.mean()), float(np.mean([LETTERS[int(k)] == r["answer"] for k, r in zip(P.argmax(1), rows)]))
for name, sfx in [("지금 (<|message|>)", "<|message|>"),
                  ("빈 추론 뒤 답 채널", " to=self<|message|><|eom|><|start|>assistant<|message|>"),
                  ("답 채널 + '정답: ('", "<|message|>정답: ("),
                  ("빈 추론 + '정답: ('", " to=self<|message|><|eom|><|start|>assistant<|message|>정답: (")]:
    m_, a_ = _mass_acc(sfx); print(f"⑧ {name:<16} letter_mass {m_:.3f} | 정확도 {a_ * 100:.0f}% (16문항) | {sfx!r}")
transformers.logging.set_verbosity_warning()
