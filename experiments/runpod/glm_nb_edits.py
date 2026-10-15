# -*- coding: utf-8 -*-
"""build_runpod.py가 stage2의 7번(모델)·8번(채점) 셀에 GLM-5.3-Flash 처리를 넣을 때 쓰는 치환 목록.
(따옴표가 많은 코드 조각이라 빌더 본문과 분리)"""


def model_cell(s, rep):
    s = rep(s, 'FAMILY = "gemma4" if _PN.startswith("Gemma4") else "muse" if _PN.startswith("MuseGlimmer") else "qwen"',
            'FAMILY = "gemma4" if _PN.startswith("Gemma4") else "muse" if _PN.startswith("MuseGlimmer") else "glm5" if _PN.startswith("Glm5Next") else "qwen"')
    s = rep(s, 'if "reasoning_strength" in (_tmpl or ""): TEMPLATE_KW["reasoning_strength"] = MUSE_REASONING   # Muse Glimmer: 템플릿 기본값 high\n',
            'if "reasoning_strength" in (_tmpl or ""): TEMPLATE_KW["reasoning_strength"] = MUSE_REASONING   # Muse Glimmer: 템플릿 기본값 high\n'
            'if FAMILY == "glm5" and "reasoning_effort" in (_tmpl or ""): TEMPLATE_KW["reasoning_effort"] = "low"   # GLM-5.3-Flash: 기본 max, 생각 끄기 옵션은 없음 → 가장 약한 low\n')
    s = rep(s, 'GEN_SUFFIX = ("<|message|>" + MUSE_ANSWER_PREFIX) if (FAMILY == "muse" and _probe.endswith("<|start|>assistant")) else ""\n',
            'GEN_SUFFIX = ("<|message|>" + MUSE_ANSWER_PREFIX) if (FAMILY == "muse" and _probe.endswith("<|start|>assistant")) else ""\n'
            'GEN_STRIP = ""                                    # 생성 프롬프트 끝에서 떼어 낼 문자열 (GLM 답 자리 후보용)\n'
            'if FAMILY == "glm5" and _probe.endswith("<think>"): GEN_SUFFIX = "</think>"   # GLM: 생각을 바로 닫음 (8번 셀이 후보 3가지 중 다시 고름)\n')
    s = rep(s, 'NO_DOUBLE_BOS = FAMILY == "muse" and bool(tok.bos_token) and _probe.startswith(tok.bos_token)\n',
            'def _dup_prefix(text):                            # 템플릿이 이미 넣은 특수 접두어([gMASK]<sop> 등)를 토크나이저가 또 붙이는지\n'
            '    a = tok(text, add_special_tokens=True).input_ids; b = tok(text, add_special_tokens=False).input_ids; k = len(a) - len(b)\n'
            '    return k > 0 and a[k:] == b and a[:k] == b[:k]\n'
            'NO_DOUBLE_BOS = (FAMILY == "muse" and bool(tok.bos_token) and _probe.startswith(tok.bos_token)) or (FAMILY == "glm5" and _dup_prefix(_probe))\n')
    s = rep(s, 'ZS_TAG = "zeroshot" if (FAMILY != "muse" and GEN_SUFFIX_OVERRIDE is None) else f"zeroshot_{FORMAT_TAG}"',
            'ZS_TAG = "zeroshot" if (FAMILY not in ("muse", "glm5") and GEN_SUFFIX_OVERRIDE is None) else f"zeroshot_{FORMAT_TAG}"')
    s = rep(s, 'LORA_TARGETS = r".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"\n',
            'LORA_TARGETS = r".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"\n'
            'if IS_GLM: LORA_TARGETS = LORA_TARGETS[:-1] + "|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"   # GLM DSA 층의 MLA 투영도 (Qwen의 어텐션 q/k/v/o에 해당)\n')
    s = rep(s, '                                            target_modules=LORA_TARGETS))\n',
            '                                            target_modules=LORA_TARGETS))\n'
            '    if IS_FP8:\n'
            '        _nr = reinit_lora_on_fp8(model, TRAIN_SEED)\n'
            '        if _nr: print(f"FP8 층 위 LoRA {_nr}개를 float32로 다시 초기화 (PEFT가 FP8로 바꿨다 되돌린 초기값 대신)")\n')
    s = rep(s, '    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kw)\n',
            '    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kw)\n'
            '    if IS_FP8:                                   # 공식 FP8 그대로(GLM-5.3-Flash · Qwen3.5-397B-FP8) → 계산할 때 층마다 bf16으로 풀기 (7-0번 셀)\n'
            '        _nl, _ne = install_fp8_train_patch(base, compute_dtype=DTYPE)\n'
            '        print(f"{\'GLM \' if IS_GLM else \'\'}FP8 보정: FP8 선형 {_nl}개 · 전문가 {_ne}개 → 계산할 때 {str(DTYPE).replace(\'torch.\', \'\')}로 풀어 곱함 (GPU에는 FP8 그대로)")\n'
            '        if DEVICE == "cuda": assert _ne > 0, "FP8 전문가 층이 없습니다 → 체크포인트가 FP8인지 확인하고 위 출력을 확인하세요"\n')
    s = rep(s, 'if any(k in str(getattr(base.config, "model_type", "")) for k in ("qwen3_5", "qwen4_exp")):',
            'if any(k in str(getattr(base.config, "model_type", "")) for k in ("qwen3_5", "qwen4_exp", "glm5_next")):')
    return s


def scoring_cell(s, rep):
    s = rep(s, '    global RES_PRESET, PIXEL_KW, MUSE_PIX\n', '    global RES_PRESET, PIXEL_KW, MUSE_PIX, GLM_PIX\n')
    s = rep(s, '    else:\n        mn, mx = RES_PRESETS[preset]\n        PIXEL_KW = {"min_pixels": int(mn), "max_pixels": int(mx)}\n',
            '    elif FAMILY == "glm5":                       # GLM 프로세서는 키우지 않고 패딩만 함 → fit_pixels로 28 배수 크기에 맞춰 넘기고, 토큰 범위는 넓게\n'
            '        mn, mx = RES_PRESETS[preset]\n'
            '        GLM_PIX = (int(mn), int(mx))\n'
            '        PIXEL_KW = {"min_image_tokens": 1, "max_image_tokens": int(mx // 784) + 64}\n'
            '    else:\n        mn, mx = RES_PRESETS[preset]\n        PIXEL_KW = {"min_pixels": int(mn), "max_pixels": int(mx)}\n')
    s = rep(s, 'def render(row, perm=IDENTITY):\n    return processor.apply_chat_template(build_messages(row, perm), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) + GEN_SUFFIX\n',
            'def render(row, perm=IDENTITY):\n'
            '    t = processor.apply_chat_template(build_messages(row, perm), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)\n'
            '    if GEN_STRIP and t.endswith(GEN_STRIP): t = t[:-len(GEN_STRIP)]\n'
            '    return t + GEN_SUFFIX\n')
    s = rep(s, '    if FAMILY == "muse": images = [fit_pixels(im, *MUSE_PIX) for im in images]',
            '    if FAMILY == "glm5": images = [fit_pixels(im, *GLM_PIX) for im in images]   # 28 배수 = patch 14 × merge 2\n'
            '    if FAMILY == "muse": images = [fit_pixels(im, *MUSE_PIX) for im in images]')
    s = s.rstrip("\n") + "\n" + GLM_PROBE
    return s


GLM_PROBE = r'''
if FAMILY == "glm5" and GEN_SUFFIX_OVERRIDE is None:
    # GLM-5.3-Flash: 생성 프롬프트가 '<|assistant|><think>'로 끝남(생각 끄기 옵션 없음) → 생각을 닫는 방법 3가지 중
    # 검증 앞 16문항에서 답 글자 확률(letter_mass)이 가장 큰 것을 고름 (학습·검증·test 모두 같은 형식)
    _cands = [("", "</think>"), ("", "</think>\n"), ("<think>", "</think>")]
    _rows16 = val_df.head(16).to_dict("records"); _gold = np.array([LETTERS.index(r["answer"]) for r in _rows16])
    set_resolution(ZERO_SHOT_PRESET); _res = []; _was_tr = model.training; model.eval()
    for _st, _sf in _cands:
        GEN_STRIP, GEN_SUFFIX = _st, _sf
        _p, _m, _ = score_rows(_rows16)
        _res.append((float(np.mean(_m)), float((_p.argmax(1) == _gold).mean()), _st, _sf))
        print(f"  답 자리 후보 {'(끝의 <think> 떼고) ' if _st else ''}{_sf!r}: letter_mass {_res[-1][0]:.3f} · 정확도 {_res[-1][1] * 100:.0f}% (16문항)")
    _best = max(_res, key=lambda t: (t[0], t[1]))
    GEN_STRIP, GEN_SUFFIX = _best[2], _best[3]
    _pp = _probe[:-len(GEN_STRIP)] if (GEN_STRIP and _probe.endswith(GEN_STRIP)) else _probe
    FORMAT_TAG = _hl.md5((_pp + GEN_SUFFIX + str(NO_DOUBLE_BOS)).encode()).hexdigest()[:6]; ZS_TAG = f"zeroshot_{FORMAT_TAG}"
    set_resolution(TRAIN_PRESET); model.train(_was_tr)
    print(f"✅ GLM 답 자리: {'<think> 떼고 ' if GEN_STRIP else ''}{GEN_SUFFIX!r} (letter_mass {_best[0]:.3f}) | 형식 태그 {FORMAT_TAG}")
    if _best[0] < 0.5: print("⚠️ 답 글자 확률이 낮습니다 → zero-shot은 낮게 나올 수 있지만 학습으로 맞춰집니다. 10번 셀 zero-shot 결과를 확인하세요")
'''
