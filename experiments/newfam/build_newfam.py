# -*- coding: utf-8 -*-
"""src_stage2.ipynb(팀의 stage2_qwen35_27b_bf16.ipynb와 동일) → stage2_newfam.ipynb
바꾼 것: ① Muse Glimmer 계열 지원(해상도·답 채널) ② 학습 전 zero-shot 점검(ZERO_SHOT_N / ZERO_SHOT_ONLY) ③ 후보 목록 주석.
나머지(분할·프롬프트·학습·평가·제출·이어하기)는 한 글자도 바꾸지 않음."""
import json, copy, sys

SRC = "/home/user/vqa/src_stage2.ipynb"
DST = "/home/user/vqa/newfam/out/stage2_newfam.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))
C = nb["cells"]

def get(i): return "".join(C[i]["source"])
def put(i, s):
    lines = s.split("\n")
    C[i]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if C[i]["cell_type"] == "code": C[i]["outputs"] = []; C[i]["execution_count"] = None
def rep(i, old, new, count=1):
    s = get(i); n = s.count(old)
    assert n == count, f"cell {i}: '{old[:60]}' 가 {n}번 나옴 (기대 {count})"
    put(i, s.replace(old, new))

# ── 0. 제목 ──
rep(0, "# Stage 2 · LoRA 파인튜닝 → 검증 → 제출\n",
    "# Stage 2 (새 계열) · LoRA 파인튜닝 → 검증 → 제출\n\n"
    "기존 stage2와 학습·평가·제출 방식은 완전히 같고, 다음 두 가지만 더했습니다.\n"
    "- **Muse Glimmer(메타) 지원**: 이 모델의 프로세서는 이미지를 확대하지 않으므로, Qwen과 같은 픽셀 범위(`RES_PRESETS`)로 직접 맞춘 뒤 넘깁니다(최대 4,096토큰). "
    "생성 프롬프트가 `<|start|>assistant`로 끝나 다음 토큰이 '추론 채널/답 채널'을 고르는 구조라, 답 채널 `<|message|>`를 붙인 뒤의 글자를 채점합니다.\n"
    "- **학습 전 zero-shot 점검**: 학습 전에 기본 모델로 검증 앞 `ZERO_SHOT_N`문항을 채점해 한국어 읽기 실력과 답 위치(letter_mass)를 봅니다. "
    "`ZERO_SHOT_ONLY = True`면 여기서 멈춥니다 → 결과를 보고 학습할지 정하세요.\n\n"
    "**새 모델 실행 순서**: ① `ZERO_SHOT_ONLY = True`로 전체 실행 (모델 다운로드 + 300문항) → ② 괜찮으면 `ZERO_SHOT_ONLY = False`, `DRY_RUN = True`로 전체 실행 (3스텝, 메모리·예상 시간 확인) → ③ `DRY_RUN = False`로 본 학습 (①의 zero-shot 결과는 재사용됨).\n\n---\n")

# ── 2. 설정 ──
rep(2, 'RUN_NAME   = "q35_27b_r16_native_bf16"        # 실험 이름 = 저장 폴더 이름. 설정을 바꾸면 이름도 바꾸세요\nMODEL_ID   = "Qwen/Qwen3.5-27B"\n',
    'RUN_NAME   = "muse30b_r16_native_bf16"        # 실험 이름 = 저장 폴더 이름. 설정을 바꾸면 이름도 바꾸세요\n'
    'MODEL_ID   = "meta-models/Muse-Glimmer-30B"\n'
    '# ── 새 계열 후보 (2026-09-27 조사) — MODEL_ID와 RUN_NAME만 바꾸면 됩니다 ──\n'
    '#   meta-models/Muse-Glimmer-30B → "muse30b_r16_native_bf16"      완전히 새 계열(메타). bf16 약 60GB → A100 80GB·G4 모두 가능\n'
    '#   Qwen/Qwen3-VL-32B-Instruct   → "q3vl_32b_r16_native_bf16"     지금은 _full만 있음 → 검증 확률을 얻기 위한 학습\n'
    '#   Qwen/Qwen3.6-35B-A3B         → "q36_35ba3b_r16_native_bf16"   MoE, bf16 약 70GB → G4(96GB) 전용\n'
    '#   LGAI-EXAONE/EXAONE-4.5-33B   → "exa45_33b_r16_native_bf16"    라이선스 NC(비상업)\n'
    '#   google/gemma-4-26B-A4B-it    → "gemma4_26ba4b_r16_up20_bf16"  G와 같은 계열 → TRAIN_PRESET = "up2.0"(G와 같게)\n'
    '#   Qwen/Qwen3.6-27B             → "q36_27b_r16_native_bf16"\n')
rep(2, 'DRY_RUN    = False          # True: 3스텝만 학습해 전체 흐름 점검 (본 학습 전 필수) / False: 본 학습\n',
    'DRY_RUN    = False          # True: 3스텝만 학습해 전체 흐름 점검 (본 학습 전 필수) / False: 본 학습\n'
    'ZERO_SHOT_N      = 300      # 학습 전에 기본 모델로 검증 앞 N문항 채점 (한국어 읽기 실력·답 위치 점검). 0이면 생략\n'
    'ZERO_SHOT_ONLY   = True     # True: zero-shot 점검만 하고 멈춤 (새 모델은 이것부터) / False: 이어서 학습\n'
    'ZERO_SHOT_PRESET = "up2.0"  # zero-shot 채점 해상도 (기존 기록 Q35 제로샷 96.1%가 up2.0)\n')
rep(2, 'HF_TOKEN   = ""',
    'MUSE_REASONING = "low"     # Muse Glimmer 전용: 시스템 프롬프트의 추론 강도 (기본 high). 답 한 글자만 보므로 low\n'
    'TEMPLATE_DATE  = "2026-09-27"   # 템플릿이 \'오늘 날짜\'를 프롬프트에 넣는 모델(Muse)은 날짜를 고정 → 학습·검증·test 입력이 날마다 달라지지 않음(재현성)\n'
    'MUSE_ANSWER_PREFIX = "정답: ("  # Muse 전용: 답 채널(<|message|>) 뒤에 붙여 다음 토큰이 곧 답 글자가 되게 함 (진단: letter_mass 0.000 → 0.999, 16문항 16개 정답)\n'
    'GEN_SUFFIX_OVERRIDE = None     # 답 글자 바로 앞에 붙일 문자열을 직접 지정 (None: 자동). 학습·검증·test 모두 같은 값이 쓰임\n'
    'HF_TOKEN   = ""')

# ── 7. 모델 로드 ──
rep(13, "## 7. 모델 로드 + LoRA 부착\n",
    "## 7. 모델 로드 + LoRA 부착\n출력의 **'답 바로 앞 프롬프트 끝'**이 모델의 답 자리인지 확인하세요 (Qwen: `</think>` 뒤, Muse: `<|start|>assistant<|message|>`).\n")
rep(14, '".*multi_modal_projector.*", ".*embed_vision.*"', '".*multi_modal_projector.*", ".*vision_adapter.*", ".*embed_vision.*"')
rep(14, 'FAMILY = "gemma4" if type(processor).__name__.startswith("Gemma4") else "qwen"     # 프로세서 계열 (해상도 인자·이미지 전달 방식이 다름)\n',
    '_PN = type(processor).__name__\n'
    'FAMILY = "gemma4" if _PN.startswith("Gemma4") else "muse" if _PN.startswith("MuseGlimmer") else "qwen"   # 프로세서 계열 (해상도 인자·이미지 전달 방식이 다름). EXAONE 4.5는 Qwen2-VL식 → "qwen"\n')
rep(14, 'TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}\n',
    'TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}\n'
    'if "reasoning_strength" in (_tmpl or ""): TEMPLATE_KW["reasoning_strength"] = MUSE_REASONING   # Muse Glimmer: 템플릿 기본값 high\n'
    'if "strftime_now" in (_tmpl or ""):                  # 오늘 날짜를 넣는 템플릿 → 날짜 고정\n'
    '    if "current_date" in _tmpl: TEMPLATE_KW["current_date"] = TEMPLATE_DATE\n'
    '    else: print("⚠️ 이 템플릿은 오늘 날짜를 프롬프트에 넣습니다 → 날짜가 바뀌면 입력이 달라집니다. 위 출력을 확인하세요")\n')
rep(14, 'print(f"계열: {FAMILY} | 이미지 토큰 id: {IMAGE_TOKEN_ID} | chat template 추가 인자: {TEMPLATE_KW} | system 역할: {USE_SYSTEM_ROLE}")\n',
    'print(f"계열: {FAMILY} | 이미지 토큰 id: {IMAGE_TOKEN_ID} | chat template 추가 인자: {TEMPLATE_KW} | system 역할: {USE_SYSTEM_ROLE}")\n'
    '_probe = processor.apply_chat_template(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)\n'
    '# Muse Glimmer: 생성 프롬프트가 \'<|start|>assistant\'로 끝나고 다음 토큰에서 추론 채널(\' to=self\')과 답 채널(\'<|message|>\')이 갈림 → 답 채널을 열어 두고 그다음 글자를 채점\n'
    'GEN_SUFFIX = ("<|message|>" + MUSE_ANSWER_PREFIX) if (FAMILY == "muse" and _probe.endswith("<|start|>assistant")) else ""\n'
    'if GEN_SUFFIX_OVERRIDE is not None: GEN_SUFFIX = GEN_SUFFIX_OVERRIDE; print(f"답 자리 앞 문자열을 직접 지정: {GEN_SUFFIX!r}")\n'
    'if FAMILY == "muse": assert GEN_SUFFIX or _probe.endswith("<|message|>"), f"Muse 템플릿의 생성 프롬프트 끝이 예상과 다릅니다: {_probe[-80:]!r} → 위 출력을 확인하세요"\n'
    'MUSE_MAX_TOKENS = int(getattr(processor.image_processor, "max_image_tokens", None) or 4096) if FAMILY == "muse" else None   # 모델 한도 (이미지당 병합 토큰)\n'
    'print("답 바로 앞 프롬프트 끝:", repr((_probe + GEN_SUFFIX)[-80:]))\n'
    '# 템플릿이 이미 BOS로 시작하면 토크나이저가 또 붙이지 않게 (Muse: 진단에서 BOS 2개 확인. 공식 apply_chat_template도 이렇게 함). 기존 계열은 학습 때와 같게 그대로 둠\n'
    'NO_DOUBLE_BOS = FAMILY == "muse" and bool(tok.bos_token) and _probe.startswith(tok.bos_token)\n'
    'import hashlib as _hl\n'
    'FORMAT_TAG = _hl.md5((_probe + GEN_SUFFIX + str(NO_DOUBLE_BOS)).encode()).hexdigest()[:6]   # 프롬프트 형식이 바뀌면 zero-shot 결과를 새로 잼\n'
    'ZS_TAG = "zeroshot" if (FAMILY != "muse" and GEN_SUFFIX_OVERRIDE is None) else f"zeroshot_{FORMAT_TAG}"\n'
    'print(f"BOS 중복 방지: {NO_DOUBLE_BOS} | 형식 태그 {FORMAT_TAG} | zero-shot 파일 태그 {ZS_TAG}")\n')

# ── 8. 채점 함수 ──
rep(16, '''def set_resolution(preset):
    global RES_PRESET, PIXEL_KW
    RES_PRESET = preset
    if FAMILY == "gemma4":
        PIXEL_KW = {"max_soft_tokens": int(RES_PRESETS_GEMMA4[preset])}
    else:
''', '''def set_resolution(preset):
    global RES_PRESET, PIXEL_KW, MUSE_PIX
    RES_PRESET = preset
    if FAMILY == "gemma4":
        PIXEL_KW = {"max_soft_tokens": int(RES_PRESETS_GEMMA4[preset])}
    elif FAMILY == "muse":                       # 프로세서는 확대하지 않고 토큰 상한만 둠 → fit_pixels로 Qwen과 같은 픽셀 범위에 맞춰 넘김
        mn, mx = RES_PRESETS[preset]
        mx = min(mx, MUSE_MAX_TOKENS * 28 * 28)  # 모델 한도 4,096토큰 ≈ 321만 픽셀 (up2.0의 상한 340만보다 조금 작음)
        MUSE_PIX = (int(min(mn, mx)), int(mx))
        PIXEL_KW = {"max_image_tokens": int(MUSE_MAX_TOKENS)}
    else:
''')
rep(16, '''def render(row, perm=IDENTITY):
    return processor.apply_chat_template(build_messages(row, perm), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
''', '''def render(row, perm=IDENTITY):
    return processor.apply_chat_template(build_messages(row, perm), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) + GEN_SUFFIX

# 생성으로 보는 zero-shot: 답 자리 첫 토큰이 글자가 아닌 모델(letter_mass가 낮음)도 실력을 잴 수 있게, 그리디로 몇 토큰 생성해 첫 a~d를 읽음
_ANS_RE1 = re.compile(r"^[\\s\\*\\(\\[\\x22'「『“]*(?:정?답\\s*(?:은|는)?\\s*[:：]?\\s*|(?:the\\s+)?(?:correct\\s+)?answer\\s*(?:is)?\\s*[:：]?\\s*)?[\\(\\[\\*]*([abcd])(?![a-z])", re.I)
_ANS_RE2 = re.compile(r"[\\(\\[]([abcd])[\\)\\]]", re.I)
def parse_answer_letter(s):
    s = str(s).strip()
    m = _ANS_RE1.match(s) or _ANS_RE2.search(s)
    return m.group(1).lower() if m else None

@torch.inference_mode()
def zero_shot_generate(df, max_new_tokens=16):
    was = model.training; model.eval(); outs = []; rows = df.to_dict("records")
    for s in range(0, len(rows), BS):
        b = rows[s: s + BS]
        enc = encode_batch(b, [IDENTITY] * len(b)).to(model.device)
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        outs += [tok.decode(q[enc["input_ids"].shape[1]:], skip_special_tokens=True) for q in out]
    if was: model.train()
    res = df[["id", "answer"]].copy().reset_index(drop=True); res["gen"] = outs; res["pred"] = res.gen.map(parse_answer_letter)
    return res

def fit_pixels(im, min_pixels, max_pixels, factor=28):
    # Qwen2-VL의 smart_resize와 같은 규칙: 가로·세로를 factor 배수로, 전체 픽셀을 [min, max] 안으로 (비율 유지)
    w, h = im.size
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
        if hb * wb > max_pixels:                 # 범위가 아주 좁을 때 올림 때문에 상한을 넘으면 상한 기준으로 내림 (실제 프리셋에서는 생기지 않음)
            beta = math.sqrt(h * w / max_pixels)
            hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    return im if (wb, hb) == (w, h) else im.resize((wb, hb), Image.LANCZOS)
''')
rep(16, '''    if FAMILY == "gemma4": images = [[im] for im in images]          # Gemma 프로세서는 샘플별 이미지 리스트를 받음
''', '''    if FAMILY == "gemma4": images = [[im] for im in images]          # Gemma 프로세서는 샘플별 이미지 리스트를 받음
    if FAMILY == "muse": images = [fit_pixels(im, *MUSE_PIX) for im in images]   # 28 배수 크기로 맞춰 두면 프로세서가 크기를 더 바꾸지 않음
''')
rep(16, '''    return processor(text=texts, images=images, padding=True, return_tensors="pt", **PIXEL_KW)''',
    '''    return processor(text=texts, images=images, padding=True, return_tensors="pt", **PIXEL_KW, **({"add_special_tokens": False} if NO_DOUBLE_BOS else {}))''')

# ── 10. 학습 루프: zero-shot 점검 ──
rep(19, "## 10. 학습 루프\n",
    "## 10. 학습 루프 (+ 학습 전 zero-shot 점검)\n"
    "zero-shot 점검은 LoRA가 아직 0인 상태(= 기본 모델)로 채점합니다. Drive에 다른 실행의 zero-shot 예측(`preds_zeroshot*.csv`, 예: E5 폴더의 Q35)이 있으면 같은 문항 기준 정확도를 같이 보여 줍니다.\n")
rep(20, '''if SKIP_MISSING_IMAGES: quick_df = quick_df[quick_df.path.map(lambda p: os.path.exists(img_path(p)))]
''', '''if SKIP_MISSING_IMAGES: quick_df = quick_df[quick_df.path.map(lambda p: os.path.exists(img_path(p)))]
if ZERO_SHOT_N and state["global_step"] == 0 and not RESUMING:
    set_resolution(ZERO_SHOT_PRESET)
    zs_df = val_df.head(int(ZERO_SHOT_N) if not DRY_RUN else 3 * BS)
    if SKIP_MISSING_IMAGES: zs_df = zs_df[zs_df.path.map(lambda p: os.path.exists(img_path(p)))]
    print(f"학습 전 zero-shot 점검: 검증 앞 {len(zs_df)}문항 | 해상도 {ZERO_SHOT_PRESET} | LoRA가 아직 0이라 기본 모델과 같음")
    zs_acc, zs_m = evaluate(zs_df, ZS_TAG, resume=True, verbose=False, by_type=False)
    zs_mass = float(zs_m.letter_mass.mean())
    state["history"].append({"step": 0, "zero_shot_acc": zs_acc, "zero_shot_n": len(zs_m), "zero_shot_mass": zs_mass})
    import glob as _glob
    for _f in sorted(_glob.glob(os.path.join(OUT_DIR, "*", "preds_zeroshot*.csv"))):      # 같은 문항에서 다른 실행의 zero-shot 기록과 비교
        if os.path.abspath(os.path.dirname(_f)) == os.path.abspath(RUN_DIR): continue
        try: _r = zs_m[["id", "answer"]].merge(pd.read_csv(_f, usecols=["id", "pred"]).drop_duplicates("id", keep="last"), on="id")
        except Exception: continue
        if len(_r) == len(zs_m): print(f"    같은 {len(zs_m)}문항 기준 {os.path.basename(os.path.dirname(_f))}/{os.path.basename(_f)}: {(_r.pred == _r.answer).mean() * 100:.2f}%")
    print("    참고 기록: Q35 제로샷 96.1% → 학습 후 97.2% (검증 1,000문항, up2.0) · Gemma4-31B 제로샷 97.67% → 학습 후 98.33% (검증 앞 300문항)")
    print("    대략의 기준: 같은 문항에서 Q35 제로샷보다 3%p 넘게 낮으면 학습 후에도 96%대 이하일 가능성이 큼 (Q35·Gemma의 학습 이득은 +0.7~1.1%p였음)")
    if zs_mass < 0.9:
        print(f"    ⚠️ letter_mass {zs_mass:.3f}: 답 자리에서 a~d가 아닌 토큰을 내려는 확률이 큼 → 위 정확도는 글자 확률이 거의 0인 상태의 잡음이라 실력이 아닙니다. "
              "학습하면 이 자리에서 바로 글자를 내도록 배우므로 학습 자체에는 문제없음")
    if zs_mass < 0.5:
        with torch.inference_mode():
            _pr = torch.softmax(_last_logits(encode_batch(zs_df.head(2).to_dict("records"), [IDENTITY] * 2)), dim=-1)
        for _i, _r in enumerate(zs_df.head(2).to_dict("records")):
            _tp = torch.topk(_pr[_i], 8)
            print(f"    답 자리 상위 토큰 ({_r['id']}, 정답 {_r['answer']}): " + ", ".join(f"{tok.decode([int(t)])!r} {float(p):.3f}" for p, t in zip(_tp.values, _tp.indices)))
        _gfp = os.path.join(RUN_DIR, f"preds_{ZS_TAG}_gen__{ZERO_SHOT_PRESET}__{len(zs_df)}.csv")
        if os.path.exists(_gfp) and list(pd.read_csv(_gfp).id) == list(zs_df.id):
            zs_gen = pd.read_csv(_gfp, keep_default_na=False); zs_gen["pred"] = zs_gen.gen.map(parse_answer_letter)
        else:
            print(f"    → 생성으로 다시 잽니다: 그리디 16토큰 → 첫 a~d 글자 ({len(zs_df)}문항)")
            zs_gen = zero_shot_generate(zs_df); zs_gen.to_csv(_gfp, index=False)
        for _r in zs_gen.head(6).itertuples(): print(f"      {_r.id} 정답 {_r.answer} → 생성 {_r.gen!r} → {_r.pred}")
        zs_gen_acc = float((zs_gen.pred == zs_gen.answer).mean()); _parsed = float(zs_gen.pred.notna().mean())
        state["history"][-1].update({"zero_shot_gen_acc": zs_gen_acc, "zero_shot_gen_parsed": _parsed})
        print(f"    ▶ 생성 기반 zero-shot: {zs_gen_acc * 100:.2f}% (n={len(zs_gen)}, 글자를 읽어 낸 비율 {_parsed * 100:.0f}%) ← 위 다른 실행들의 zero-shot과 비교할 값")
    set_resolution(TRAIN_PRESET)
    if ZERO_SHOT_ONLY:
        raise SystemExit("ZERO_SHOT_ONLY=True → 학습하지 않고 여기서 멈춥니다. 학습하려면 1번 셀에서 False로 바꾸고 처음부터 다시 실행하세요 (이 zero-shot 결과는 재사용됩니다).")
''')

# ── 13. 요약 ──
rep(26, '''print("RUN:", RUN_NAME, "| epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})
''', '''print("RUN:", RUN_NAME, "| MODEL:", MODEL_ID, "| epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})
_zs = [x for x in state["history"] if "zero_shot_acc" in x]
if _zs: print(f"학습 전 zero-shot: {_zs[-1]['zero_shot_acc'] * 100:.2f}% (n={_zs[-1]['zero_shot_n']}, letter_mass {_zs[-1]['zero_shot_mass']:.3f})"
              + (f" | 생성 기반 {_zs[-1]['zero_shot_gen_acc'] * 100:.2f}% (글자 읽음 {_zs[-1]['zero_shot_gen_parsed'] * 100:.0f}%)" if "zero_shot_gen_acc" in _zs[-1] else ""))
''')

nb["metadata"].setdefault("colab", {})["gpuType"] = "A100"
json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", DST, "| 셀", len(C))
