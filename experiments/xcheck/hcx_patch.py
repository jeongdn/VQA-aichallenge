# -*- coding: utf-8 -*-
# build_runpod_xcheck.py가 exec으로 불러 쓰는 HyperCLOVA X 대응 패치 (0928)
# 저장소 커밋 고정 · 생각 블록 닫기 · system 형식 판별 · 이미지 자리 점검 · 기록에 커밋 남기기
# 근거: vLLM v0.30.0 tests/models/registry.py 의 HCXVisionV2ForCausalLM 항목 = trust_remote_code=True, revision="a6cdfd3464d1b767259cad23e164eaf39d3e3960"
#       (HF main은 transformers 네이티브 형식 hyperclovax_vision_v2로 바뀌어 transformers 5.17·vLLM 0.30이 모름.
#        첫 공개 커밋 f5e4236은 원격 코드가 AutoConfig.register("hyperclovax")를 exist_ok 없이 불러, 'hyperclovax'가 이미 있는 5.17에서 오류)
HCX_REV = "a6cdfd3464d1b767259cad23e164eaf39d3e3960"

CFG_X = rep(CFG_X, "TRUST_REMOTE_CODE = False", "TRUST_REMOTE_CODE = False\nREVISION      = None            # HF 저장소 커밋 고정 (None = main). HyperCLOVA X는 MODEL_PRESETS가 고정")
CFG_X = rep(CFG_X, '"naver-hyperclovax/HyperCLOVAX-SEED-Think-32B": {"TRUST_REMOTE_CODE": True},   # 판정 3순위: 네이버·한국어 (95.67%). vLLM 지원이 불확실 → 로드가 안 되면 건너뜀',
            '"naver-hyperclovax/HyperCLOVAX-SEED-Think-32B": {"TRUST_REMOTE_CODE": True, "REVISION": "' + HCX_REV + '"},   '
            '# 판정 3순위: 네이버·한국어 (95.67%). HF main은 transformers 5.18+ 전용 형식 → vLLM 0.30이 시험한 원격 코드판 커밋으로 고정 · 생각 블록은 5번 셀이 닫음')

_P_LOAD = '''HCX_REMOTE_REV = "''' + HCX_REV + '''"   # HyperCLOVAX-SEED-Think-32B 원격 코드판 (vLLM 0.30 시험 기준 커밋)
_IS_HCX = "hyperclova" in MODEL_ID.lower()
def _load_processor(rev):
    p = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE, revision=rev)
    if _IS_HCX: assert hasattr(p, "image_processor"), f"이미지 처리기가 없는 프로세서({type(p).__name__})가 불러와짐"
    return p
try:
    processor = _load_processor(REVISION)
except Exception as _e:
    if not (_IS_HCX and REVISION is None): raise
    # MODEL_PRESETS의 고정을 지운 경우: main은 transformers 네이티브 형식이라 실패 → 원격 코드판 커밋으로 고정 (8·9번 셀도 같은 커밋)
    print(f"⚠️ {MODEL_ID} main 프로세서 로드 실패 ({type(_e).__name__}: {str(_e)[:160]}) → 원격 코드판 커밋 {HCX_REMOTE_REV[:7]}로 고정해 다시 시도")
    REVISION = HCX_REMOTE_REV
    processor = _load_processor(REVISION)
'''
PROMPT_X = rep(PROMPT_X, 'processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)\n', _P_LOAD)

_P_KW = r'''def _think_kw(on):
    # 생각 모드를 켜고 끄는 템플릿 변수: Qwen·Gemma는 enable_thinking, HyperCLOVA X는 skip_reasoning (또는 thinking). 템플릿에 있는 것만 넘김
    if "enable_thinking" in (_tmpl or ""): return {"enable_thinking": bool(on)}
    kw = {}
    if re.search(r"\bthinking\b", _tmpl or ""): kw["thinking"] = bool(on)
    if "skip_reasoning" in (_tmpl or ""): kw["skip_reasoning"] = not on
    return kw
def _close_think(t, on):
    # 생각을 끄는데도 프롬프트가 '<think>'로 열린 채 끝나면 빈 생각 블록을 닫음 → 모델이 바로 답(풀이 + 정답 줄)을 씀
    if on: return t
    if t.endswith("<think>\n"): return t + "\n</think>\n\n"
    if t.endswith("<think>"): return t + "\n\n</think>\n\n"
    return t
TEMPLATE_KW = _think_kw(False)
'''
PROMPT_X = rep(PROMPT_X, 'TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}\n', _P_KW)
PROMPT_X = rep(PROMPT_X, '        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},\n',
               '        return [{"role": "system", "content": SYSTEM_INSTRUCT if globals().get("SYSTEM_AS_STRING") else [{"type": "text", "text": SYSTEM_INSTRUCT}]},\n')
_P_SYS = '''    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")
_raw_list = lambda t: "'type': 'text'" in t or '"type": "text"' in t      # 목록형 content가 풀리지 않고 글자 그대로 찍혔는지
SYSTEM_AS_STRING = False
if USE_SYSTEM_ROLE and _raw_list(_apply(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)):
    SYSTEM_AS_STRING = True; print("템플릿이 목록형 system 내용을 풀지 못함 → system 지시문을 문자열로 넣습니다")
'''
PROMPT_X = rep(PROMPT_X, '    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")\n', _P_SYS)
PROMPT_X = rep(PROMPT_X, '    return _apply(build_messages(row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) + ANSWER_PREFIX',
               '    return _close_think(_apply(build_messages(row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW), False) + ANSWER_PREFIX')
PROMPT_X = rep(PROMPT_X, 'XC_TEMPLATE_KW = {"enable_thinking": bool(THINKING)} if "enable_thinking" in (_tmpl or "") else {}\n',
               'XC_TEMPLATE_KW = _think_kw(THINKING)\n')
PROMPT_X = rep(PROMPT_X, '        return [{"role": "system", "content": [{"type": "text", "text": XC_SYSTEM}]},',
               '        return [{"role": "system", "content": XC_SYSTEM if SYSTEM_AS_STRING else [{"type": "text", "text": XC_SYSTEM}]},')
PROMPT_X = rep(PROMPT_X, '    return _apply(xc_messages(row, A, B), tokenize=False, add_generation_prompt=True, **XC_TEMPLATE_KW)',
               '    return _close_think(_apply(xc_messages(row, A, B), tokenize=False, add_generation_prompt=True, **XC_TEMPLATE_KW), THINKING)')
_P_CHK = r'''print(f"판정 템플릿 추가 인자 {XC_TEMPLATE_KW} | system 문자열 {SYSTEM_AS_STRING} | 저장소 커밋 {REVISION or 'main'} | 예시 끝부분:"); print(_px[-600:])
assert not _raw_list(_px), f"판정 프롬프트에 목록형 content가 글자 그대로 찍힘 → 템플릿이 예상과 다름: {_px[-400:]!r}"
if _IS_HCX:                                  # vLLM 전용 구현은 <|IMAGE_PAD|> 한 개를 이미지 토큰들로 바꿈 · 생각 블록이 닫혀 있어야 바로 풀이+정답을 씀
    assert _px.count("<|IMAGE_PAD|>") == 1, f"HyperCLOVA X 판정 프롬프트에 이미지 자리(<|IMAGE_PAD|>)가 {_px.count('<|IMAGE_PAD|>')}번 → 템플릿이 예상과 다름: {_px[-400:]!r}"
    assert THINKING or _px.endswith("</think>\n\n"), f"HyperCLOVA X 생각 블록이 닫히지 않음: {_px[-120:]!r}"
    print(f"✅ HyperCLOVA X: 커밋 {(REVISION or 'main')[:7]} · 이미지 자리 1개 · 생각 블록 닫힘 (바로 풀이 + '정답: (x)')")'''
PROMPT_X = rep(PROMPT_X, '''print(f"판정 템플릿 추가 인자 {XC_TEMPLATE_KW} | 예시 끝부분:"); print(_px[-600:])''', _P_CHK)
DL_C = rep(DL_C, "model_info(MODEL_ID, files_metadata=True)", "model_info(MODEL_ID, files_metadata=True, revision=REVISION)")
DL_C = rep(DL_C, "MODEL_DIR = snapshot_download(MODEL_ID, token=HF_TOKEN or None, max_workers=16, ",
           "MODEL_DIR = snapshot_download(MODEL_ID, revision=REVISION, token=HF_TOKEN or None, max_workers=16, ")
LLM_C = rep(LLM_C, "llm = LLM(model=MODEL_ID, quantization=QUANTIZATION,",
            "llm = LLM(model=MODEL_ID, revision=REVISION, tokenizer_revision=REVISION, code_revision=REVISION, quantization=QUANTIZATION,")
SAVE_X = rep(SAVE_X, '"THINKING": THINKING, "TEMPLATE_KW": XC_TEMPLATE_KW,',
             '"THINKING": THINKING, "TEMPLATE_KW": XC_TEMPLATE_KW, "REVISION": REVISION, "SYSTEM_AS_STRING": SYSTEM_AS_STRING,')
