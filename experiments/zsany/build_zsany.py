# -*- coding: utf-8 -*-
"""아무 image-text-to-text 모델이나 MODEL_ID만 바꿔 zero-shot 채점하는 vLLM 노트북 빌더 (학습 없음).
- 검증 = train 전체(최대 6,714). 앞 1,000문항은 팀 공통 검증(md5 df05e4bc…) → 기존 멤버·ens_search·E14와 비교 가능
- 설치·데이터·분할·모델 받기·vLLM 로드·확률 계산은 zs122b_vllm.ipynb(G4에서 실제로 돌아간 것)를 그대로 쓰고,
  모델별로 다른 부분(이미지 해상도 인자, 양자화, 이미지 자리)만 일반화
- 결과: vqa_runs/<RUN_NAME>/ — 이어하기용 작업 파일 + E14 형식(검증 1,000 · test 6,714 · train_log ZERO_SHOT) + 단독 제출 파일
사용: python3 build_zsany.py [출력 경로]"""
import sys, json, uuid, io, contextlib, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/zsany/out/zs_any_vllm.ipynb"
SPD = "/tmp/vqa_scratch"
_z = {}
_argv = sys.argv; sys.argv = ["build_zs122b.py", f"{SPD}/_zs_tmp_any.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/zs122b/build_zs122b.py", encoding="utf-8").read(), _z)
sys.argv = _argv
ZC = _z["cells"]
def src(i): return ZC[i]["source"]
def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:90]); return text.replace(old, new)
PROMPT_BLOCK, LETTER_BLOCK, RES_BLOCK = _z["PROMPT_BLOCK"], _z["LETTER_BLOCK"], _z["RES_BLOCK"]
ENV_TAIL = src(2)[src(2).index("import os, sys"):]                   # torchaudio 차단 · 캐시 · vLLM 환경 변수 (zs122b와 같음)
INSTALL, DATA_C, CSV_C, LLM_C = src(4), src(6), src(8), src(16)
LLM_C = rep(LLM_C, 'llm = LLM(model=MODEL_ID, quantization=QUANTIZATION, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,',
            'llm = LLM(model=MODEL_ID, quantization=QUANTIZATION, dtype=DTYPE, max_model_len=MAX_MODEL_LEN, trust_remote_code=TRUST_REMOTE_CODE,')
LLM_C = rep(LLM_C, 'limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs=VLLM_MM_KW, mm_processor_cache_gb=0,',
            'limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs=MM_KW, mm_processor_cache_gb=0,')
assert "PRE_RESIZE" not in LLM_C and "VLLM_MM_KW" not in LLM_C

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# 아무 모델 zero-shot (vLLM) — `MODEL_ID`만 바꿔서 실행 · 학습 없음 · train 6,714문항 전체로 검증

**하는 일**
- 1번 셀의 `MODEL_ID` 모델을 학습 없이 그대로 불러와 답합니다.
  - 입력: 이미지 + 질문 + 보기(a~d)
  - 출력: a~d 한 글자(답 자리의 a/b/c/d 확률을 모읍니다). 생각 모드는 끕니다.
- 학습을 안 하므로 train 6,714문항이 모두 검증입니다.
  - `VAL_N`으로 몇 문항을 쓸지 정합니다.
  - 앞 1,000문항은 팀 공통 검증과 같습니다. 그래서 기존 멤버·ens_search·E14와 같은 문항으로 비교할 수 있습니다.
- 끝나면 `vqa_runs/<RUN_NAME>/`에 세 가지가 남습니다.
  - 검증·test 확률 (이어하기용 작업 파일)
  - E14 형식 파일 (ens_search·E14가 자동으로 멤버로 잡음)
  - 이 모델 단독 제출 파일
- 앙상블은 `zs_ensemble.ipynb`(CPU)에서 모든 zero-shot 실행을 모아 6,714문항으로 고릅니다.

**모델 바꾸기**: `MODEL_ID`만 바꾸고 **세션 다시 시작 → 1번 셀부터 전체 실행**하세요.
- vLLM 엔진은 한 세션에 모델 하나만 올릴 수 있습니다.
- `MODEL_PRESETS`에 있는 모델은 알맞은 설정(양자화 등)이 자동으로 들어갑니다. 없는 모델도 기본값으로 돌아갑니다.

**런타임**: G4(96GB) 새 런타임. 설치는 zs122b와 같습니다(vLLM 0.30.0 · transformers 5.17.0). 설치가 끝난 런타임에서는 설치를 건너뜁니다.
""")

md("## 1. 설정 — `MODEL_ID`만 바꾸면 됩니다")
code(r'''
MODEL_ID  = "google/gemma-4-31B-it"   # ← 여기만 바꾸세요. 예시는 아래 MODEL_PRESETS
PRESET    = "up2.0"                   # 해상도: "up2.0"(기존 멤버와 같음, 작은 글씨에 유리) / "native"(원본 크기, 약 4배 빠름) / "default"(모델 기본 처리)
VAL_N     = 6714                      # 검증에 쓸 train 문항 수 (앞 1,000 = 팀 공통 검증). 빨리 보려면 1000 → 나중에 늘리면 이어서 채점
DO_TEST   = True                      # test 6,714문항도 채점 (앙상블·제출에 필요)
RUN_NAME  = "auto"                    # "auto" → zs_<모델 이름>_<해상도>
IMPORT_FROM = "auto"                  # 같은 모델·해상도·프롬프트의 예전 zero-shot 결과(예: q35_122b_a10b_gptq_zs)를 찾아 그 문항은 다시 안 함 / [] = 안 가져옴
PROMPT_VERSION_USE = "baseline_v1"    # "baseline_v1" (학습 멤버와 같은 입력) / "ko_v1" (system 한 줄도 한국어)
ANSWER_PREFIX = ""                    # 답 자리 바로 앞에 붙일 글자. 첫 묶음 점검에서 letter_mass가 낮으면 안내대로 바꿈
MAKE_SUBMISSION = True

# ── 모델별 설정 (여기 없는 모델은 아래 공통 기본값) ──
MODEL_PRESETS = {
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4": {"QUANTIZATION": "moe_wna16", "MAX_NUM_SEQS": 16},   # zs122b 결과(검증 1,000 · test)를 자동으로 가져옴
    "google/gemma-4-31B-it":           {},                          # zero-shot 최고 (앞 300문항 97.67%). 라이선스 동의 + HF 토큰 필요
    "google/gemma-4-26B-A4B-it":       {},                          # Gemma 4 MoE (활성 3.8B, 빠름)
    "Qwen/Qwen3.5-27B":                {},                          # 앞 300문항 96.33%
    "Qwen/Qwen3.6-35B-A3B":            {},                          # 앞 300문항 96.33% (MoE, 빠름)
    "Qwen/Qwen3.6-27B":                {},
    "Qwen/Qwen3.8-27B":                {},
    "Qwen/Qwen3-VL-32B-Instruct":      {},
    "Qwen/Qwen3-VL-8B-Instruct":       {},
    "Qwen/Qwen3.5-9B":                 {},
    "OpenGVLab/InternVL3_5-38B-HF":    {},                          # 앞 300문항 94.33%
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B": {"TRUST_REMOTE_CODE": True},   # 앞 300문항 95.67% (vLLM 지원이 불확실 → 로드 실패하면 건너뜀)
}
# ── 공통 기본값 (vLLM) ──
DTYPE         = "auto"          # 체크포인트 dtype 그대로 (bf16)
QUANTIZATION  = None            # 양자화 판은 MODEL_PRESETS에서 정함
TRUST_REMOTE_CODE = False
MOE_BACKEND   = "auto"
GPU_MEM_UTIL  = 0.92
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS  = 32
MAX_BATCHED_TOKENS = 32768
TOPK_LOGPROBS = 64
CHUNK         = 128             # 한 번에 넘기는 문항 수 = 저장 단위
IMG_WORKERS   = 8
GDN_PREFILL_BACKEND = "triton"  # Qwen3.5 계열 선형 어텐션 커널 (다른 모델에는 영향 없음)
ENGINE_KW     = {}
for _k, _v in MODEL_PRESETS.get(MODEL_ID, {}).items(): globals()[_k] = _v

import re as _re
if RUN_NAME == "auto":
    RUN_NAME = "zs_" + _re.sub(r"[^a-z0-9]+", "_", MODEL_ID.split("/")[-1].lower()).strip("_") + "_" + PRESET.replace(".", "") + ("" if PROMPT_VERSION_USE == "baseline_v1" else "_" + PROMPT_VERSION_USE)
VAL_ORDER_SEED = 2026           # 앞 1,000(팀 공통 검증) 뒤의 train 5,714문항 순서 (고정)

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
HF_CACHE     = "/content/hf_cache"
SEED, VAL_SIZE = 42, 1000
HF_TOKEN   = ""                 # Gemma처럼 라이선스 동의가 필요한 모델: 비워 두면 Colab 보안 비밀 'HF_TOKEN'을 씀

''' + RES_BLOCK.rstrip() + r'''
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}   # stage2와 같음 (Gemma 4는 이미지당 토큰 수)
print(f"MODEL_ID = {MODEL_ID} | RUN_NAME = {RUN_NAME} | PRESET = {PRESET} | VAL_N = {VAL_N} | 설정 덮어쓰기 {MODEL_PRESETS.get(MODEL_ID, {}) or '없음 (기본값)'}")

''' + ENV_TAIL.rstrip())

md("## 2. 설치 (zs122b와 같음 · 이미 설치된 런타임은 건너뜀)")
code(INSTALL)

md("## 3. 데이터 준비 (stage2와 같음 — 이미지 포함 압축 해제 · HF 로그인)")
code(DATA_C)

md("""
## 4. CSV 로드 · 고정 검증 분할 · 검증 순서
- 검증 순서: 팀 공통 검증 1,000문항을 먼저 두고, 나머지 train 5,714문항은 고정 시드로 섞어 뒤에 둡니다.
- `VAL_N`만큼 앞에서부터 씁니다.
""")
code(CSV_C.rstrip() + r'''

_rest = tr_df.sample(frac=1.0, random_state=VAL_ORDER_SEED)
VAL_ALL = pd.concat([val_df, _rest], ignore_index=True)
assert VAL_ALL.id.is_unique and len(VAL_ALL) == len(train_df)
VAL_USE = VAL_ALL.head(int(VAL_N)).copy()
FIXED_VAL_IDS = set(val_df.id)
print(f"검증 {len(VAL_USE):,}문항 사용 (팀 공통 {int(VAL_USE.id.isin(FIXED_VAL_IDS).sum()):,} + 나머지 {int((~VAL_USE.id.isin(FIXED_VAL_IDS)).sum()):,}) | test {len(test_df):,}")
''')

md("""
## 5. 프롬프트 · 프로세서 (모델 저장소의 채팅 템플릿)
- 입력 형식은 stage2 `baseline_v1`과 같습니다: 시스템 지시 + 이미지 + 질문·보기 + "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
- 채팅 템플릿에 생각 모드 인자가 있으면 끕니다.
- 템플릿이 system 역할을 받지 않으면 지시문을 user 앞에 붙입니다.
""")
code(PROMPT_BLOCK.rstrip() + r'''
if PROMPT_VERSION_USE == "ko_v1":
    PROMPT_VERSION = "ko_v1"
    SYSTEM_INSTRUCT = "당신은 이미지를 보고 객관식 질문에 답하는 도우미입니다. a, b, c, d 중 정확히 한 글자로만 답하고 설명하지 마세요."
assert PROMPT_VERSION == PROMPT_VERSION_USE, (PROMPT_VERSION, PROMPT_VERSION_USE)

from transformers import AutoProcessor
import transformers
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
tok = getattr(processor, "tokenizer", processor)
_tmpl = getattr(processor, "chat_template", None) or getattr(tok, "chat_template", None)
_tmpl = _tmpl if isinstance(_tmpl, str) else json.dumps(_tmpl)
TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
_apply = processor.apply_chat_template if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None) else tok.apply_chat_template
try:
    _apply(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
except Exception as _e:
    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")
IMAGE_TOKEN = getattr(processor, "image_token", None)
IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
if IMAGE_TOKEN_ID is None and IMAGE_TOKEN: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(IMAGE_TOKEN)
PROC_NAME = type(processor).__name__
FAMILY = "gemma4" if PROC_NAME.startswith("Gemma4") else ("qwen" if "Qwen" in PROC_NAME or hasattr(getattr(processor, "image_processor", None), "merge_size") else "other")

def render(row):
    return _apply(build_messages(row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) + ANSWER_PREFIX

_p = render(VAL_USE.iloc[0].to_dict())
if IMAGE_TOKEN and _p.count(IMAGE_TOKEN) != 1:
    print(f"ℹ️ 렌더링한 프롬프트에 이미지 토큰 {IMAGE_TOKEN!r}가 {_p.count(IMAGE_TOKEN)}번 보입니다 (템플릿이 다른 자리표시를 씀 → vLLM이 처리. 첫 묶음 점검에서 확인)")
print(f"transformers {transformers.__version__} | {PROC_NAME} (계열 {FAMILY}) | 템플릿 추가 인자 {TEMPLATE_KW} | system 역할 {USE_SYSTEM_ROLE} | 이미지 토큰 {IMAGE_TOKEN}={IMAGE_TOKEN_ID}")
print("프롬프트 끝부분:", repr(_p[-160:]))

''' + LETTER_BLOCK.rstrip() + r'''
''')

md("""
## 6. 이미지 해상도 인자 (모델 계열별) + 점검
- 계열마다 해상도 인자가 다릅니다.
  - Qwen 계열(Qwen2.5-VL·Qwen3-VL·Qwen3.5/3.6/3.8): 픽셀 범위 `min_pixels`·`max_pixels` (stage2와 같은 값)
  - Gemma 4: 이미지당 토큰 수 `max_soft_tokens` (up2.0 = 1120)
  - 그 밖의 모델, 또는 `PRESET="default"`: 모델 기본 처리
- HF 프로세서로 검증 2장을 처리해, 인자가 먹혔는지(이미지 토큰 수) 보여 줍니다. vLLM도 같은 프로세서를 씁니다.
""")
code(r'''
if PRESET == "default" or FAMILY == "other":
    MM_KW = {}
elif FAMILY == "gemma4":
    MM_KW = {"max_soft_tokens": int(RES_PRESETS_GEMMA4[PRESET])}
else:
    MM_KW = {"min_pixels": int(RES_PRESETS[PRESET][0]), "max_pixels": int(RES_PRESETS[PRESET][1])}
if FAMILY == "other" and PRESET != "default":
    print(f"ℹ️ {PROC_NAME}은 해상도 인자를 모르는 계열이라 모델 기본 처리로 둡니다 (PRESET={PRESET!r}는 이름에만 남음)")
for _r in VAL_USE.head(2).to_dict("records"):
    _im = load_image(_r["path"])
    try:
        _enc = processor(text=[render(_r)], images=[[_im]] if FAMILY == "gemma4" else [_im], return_tensors="pt", **MM_KW)
        _n = int((_enc["input_ids"] == IMAGE_TOKEN_ID).sum()) if IMAGE_TOKEN_ID is not None else -1
        print(f"  {_r['id']}: 원본 {_im.size} → 프롬프트 {int(_enc['input_ids'].shape[1])}토큰 (이미지 토큰 {_n}) | 해상도 인자 {MM_KW or '기본'}")
    except Exception as _e:
        print(f"  ⚠️ HF 프로세서 점검 실패 ({type(_e).__name__}: {str(_e)[:160]}) → vLLM 쪽 처리로 진행 (첫 묶음 점검에서 확인)")
''')

md("## 7-0. 모델 파일 받기 (진행 막대 표시 · 끊기면 다시 실행하면 이어받기)")
code(r'''
import shutil
from huggingface_hub import snapshot_download
t0 = time.time()
MODEL_DIR = snapshot_download(MODEL_ID, ignore_patterns=["*.pth", "*.pt", "*.gguf", "*.msgpack", "*.h5", "*.onnx", "original/*", "consolidated*"])
_sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(MODEL_DIR) for f in fs) / 1e9
_nsh = len([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
print(f"모델 파일 {_sz:.1f}GB · 가중치 조각 {_nsh}개 | {time.time() - t0:.0f}s | {MODEL_DIR}")
_free = shutil.disk_usage(HF_CACHE).free / 1e9 if os.path.exists(HF_CACHE) else float("nan")
print(f"로컬 디스크 여유 {_free:.0f}GB")
''')

md("""
## 7. vLLM으로 모델 로드 (zs122b와 같음)
- 오류가 나면 1번 셀에서 설정을 바꾼 뒤 **세션 다시 시작 → 1번 셀부터** 실행하세요.

| 증상 | 바꿀 설정 |
|---|---|
| 시작할 때 메모리 부족 | `MAX_NUM_SEQS = 8`, `MAX_BATCHED_TOKENS = 8192`, `MAX_MODEL_LEN = 6144` |
| 양자화 관련 오류 | `QUANTIZATION = None` |
| 전문가 층 커널 오류 | `MOE_BACKEND = "triton"` |
| 어텐션 백엔드 오류 | `ENGINE_KW = {"attention_backend": "TRITON_ATTN"}` |
| 모델 구조를 모름(`not supported`) | vLLM 0.30이 지원하지 않는 모델 → 다른 모델로 |
| 커스텀 코드 필요(`trust_remote_code`) | `TRUST_REMOTE_CODE = True` |
""")
code(LLM_C)

md("""
## 8. 채점 함수 (끊겨도 이어서) · 예전 결과 가져오기
- 문항마다 답 자리 상위 64개 토큰에서 a/b/c/d 변형(소문자·대문자·앞 공백) 확률을 합칩니다. 그 뒤 네 글자끼리 정규화합니다(stage2·zs122b와 같은 계산).
- 첫 묶음에서 letter_mass(a~d 확률 합)가 낮으면 멈추고 답 자리 최상위 토큰을 보여 줍니다. 모델이 "("나 "정답" 같은 토큰으로 시작하려는 경우입니다.
- `IMPORT_FROM`: 같은 모델·해상도·프롬프트로 이미 채점한 실행이 있으면 그 문항을 가져와 건너뜁니다. 예: zs122b의 122B 검증 1,000·test 6,714.
""")
code(r'''
import shutil
from concurrent.futures import ThreadPoolExecutor
IMG_POOL, PREFETCH = ThreadPoolExecutor(max_workers=IMG_WORKERS), ThreadPoolExecutor(max_workers=1)
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
VAL_WORK, TEST_WORK = os.path.join(RUN_DIR, f"zs_trainval__{PRESET}.csv"), os.path.join(RUN_DIR, f"zs_test__{PRESET}.csv")
COLS = ["id", "pred", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens", "n_missing", "top_token", "prompt_len", "src"]

def _make_req(r):
    return {"prompt": render(r), "multi_modal_data": {"image": load_image(r["path"])}}
def _prep_chunk(rows): return list(IMG_POOL.map(_make_req, rows))
def _parse(r, o):
    lp = o.outputs[0].logprobs[0]
    slot = np.array([np.logaddexp.reduce([lp[i].logprob for i in ids if i in lp]) if any(i in lp for i in ids) else -np.inf for ids in LETTER_IDS], dtype=np.float64)
    mass = float(np.exp(slot).sum())
    p = np.exp(slot - slot.max()) if np.isfinite(slot).any() else np.ones(4)
    p = p / p.sum()
    top = max(lp.items(), key=lambda kv: kv[1].logprob)[0]
    n_img = sum(1 for t in o.prompt_token_ids if t == IMAGE_TOKEN_ID) if IMAGE_TOKEN_ID is not None else -1
    return {"id": r["id"], "pred": LETTERS[int(p.argmax())], "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": mass,
            "img_tokens": int(n_img), "n_missing": int((~np.isfinite(slot)).sum()), "top_token": int(top), "prompt_len": len(o.prompt_token_ids), "src": "vllm"}
def _check_first(recs, ch):
    d = pd.DataFrame(recs)
    print(f"  첫 묶음 점검: letter_mass 평균 {d.letter_mass.mean():.3f} (최소 {d.letter_mass.min():.3f}) | 글자 누락 문항 {int((d.n_missing > 0).sum())} | "
          f"이미지 토큰 {d.img_tokens.min()}~{d.img_tokens.max()} | 프롬프트 길이 최대 {d.prompt_len.max()}")
    if d.letter_mass.mean() < 0.5:
        for r, x in list(zip(ch, recs))[:3]: print(f"    {r['id']} 답 자리 최상위 토큰: {tok.decode([x['top_token']])!r}")
        raise RuntimeError("letter_mass가 너무 낮음 → 답 자리에서 a~d가 아닌 토큰을 냄. 위 최상위 토큰을 보고 1번 셀 ANSWER_PREFIX를 정하세요 "
                           "(예: 최상위 토큰이 '('이면 ANSWER_PREFIX = \"(\", '정답'이면 \"정답: \")")
    if d.img_tokens.max() == 0:
        print("  ⚠️ 프롬프트에서 이미지 토큰을 찾지 못함 (이 모델의 이미지 토큰 이름이 다를 수 있음 — 정확도가 25% 근처면 이미지가 안 들어간 것)")
def _fix_tail(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n": f.write(b"\n")
def read_preds(path):
    if not os.path.exists(path): return pd.DataFrame(columns=COLS)
    d = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["id", "p_a", "p_b", "p_c", "p_d", "src"]).drop_duplicates("id", keep="last")
    return d
def predict(df, out_path, label):
    _fix_tail(out_path)
    done = set(read_preds(out_path).id)
    rows = df[~df.id.isin(done)].to_dict("records")
    print(f"[{label}] {os.path.basename(out_path)} | 전체 {len(df):,} | 완료 {len(df) - len(rows):,} | 남음 {len(rows):,}", flush=True)
    if not rows: return
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    fut, t0, n = PREFETCH.submit(_prep_chunk, chunks[0]), time.time(), 0
    for k, ch in enumerate(chunks):
        reqs = fut.result()
        if k + 1 < len(chunks): fut = PREFETCH.submit(_prep_chunk, chunks[k + 1])
        outs = llm.generate(reqs, SP, use_tqdm=False)
        if len(outs) > len(ch):
            print(f"  (엔진에 남아 있던 이전 요청 {len(outs) - len(ch)}개의 결과는 버림)"); outs = outs[-len(ch):]
        assert len(outs) == len(ch) and all(getattr(o, "prompt", None) in (None, q["prompt"]) for o, q in zip(outs, reqs)), "요청과 결과의 순서가 맞지 않음"
        recs = [_parse(r, o) for r, o in zip(ch, outs)]
        pd.DataFrame(recs, columns=COLS).to_csv(out_path, mode="a", header=not os.path.exists(out_path) or os.path.getsize(out_path) == 0, index=False)
        n += len(ch)
        if k == 0 and not done: _check_first(recs, ch)
        if k == 0 or (k + 1) % 5 == 0 or k + 1 == len(chunks):
            el = time.time() - t0
            print(f"  {n:>5}/{len(rows)} | {el / n:.3f}s/문항 | 남은 시간 ~{(len(rows) - n) * el / n / 60:.0f}분", flush=True)

# ── 예전 zero-shot 결과 가져오기 (같은 모델·해상도·프롬프트만) ──
def _import_runs():
    if IMPORT_FROM == "auto":
        cands = []
        for d in sorted(os.listdir(OUT_DIR)):
            lp = os.path.join(OUT_DIR, d, "train_log.json")
            if d == RUN_NAME or not os.path.exists(lp): continue
            try: c = (json.load(open(lp, encoding="utf-8")).get("config") or {})
            except Exception: continue
            if c.get("ZERO_SHOT") and c.get("MODEL_ID") == MODEL_ID and c.get("EVAL_PRESET", c.get("PRESET")) == PRESET and c.get("PROMPT_VERSION", "baseline_v1") == PROMPT_VERSION \
               and not c.get("ANSWER_PREFIX"): cands.append(d)
        return cands
    return list(IMPORT_FROM or [])
_imp = _import_runs() if not ANSWER_PREFIX else []
for d in _imp:
    for pat, work, ids in [(r"^preds_final_ep\d+__.+__\d+\.csv$|^preds_zeroshot__.+\.csv$|^zs_trainval__.+\.csv$", VAL_WORK, set(VAL_ALL.id)),
                           (r"^preds_final_ep\d+_test__.+\.csv$|^preds_zeroshot_test__.+\.csv$|^zs_test__.+\.csv$", TEST_WORK, set(test_df.id))]:
        have = set(read_preds(work).id)
        for f in sorted(os.listdir(os.path.join(OUT_DIR, d))):
            if not _re.match(pat, f): continue
            try: x = pd.read_csv(os.path.join(OUT_DIR, d, f), on_bad_lines="skip").dropna(subset=["id", "p_a", "p_b", "p_c", "p_d"]).drop_duplicates("id", keep="last")
            except Exception: continue
            x = x[x.id.isin(ids) & ~x.id.isin(have)].copy()
            if not len(x): continue
            for c in COLS:
                if c not in x.columns: x[c] = -1
            x["src"] = f"import:{d}/{f}"
            _fix_tail(work)
            x[COLS].to_csv(work, mode="a", header=not os.path.exists(work) or os.path.getsize(work) == 0, index=False)
            have |= set(x.id)
            print(f"  가져옴: {d}/{f} → {os.path.basename(work)} {len(x):,}문항")
print(f"✅ 채점 함수 준비 | 결과 폴더 {RUN_DIR}" + (f" | 예전 결과 가져온 실행 {_imp}" if _imp else ""))
''')

md("""
## 9. 검증(train) 채점 → 정확도
- 팀 공통 검증 1,000문항 정확도는 기존 zero-shot 기록과 비교할 수 있습니다.
  - 앞 300문항 기록: Gemma4-31B 97.67 · Q35-27B 96.33 · Qwen3.6-35B-A3B 96.33 · HCX 95.67 · InternVL 94.33
  - 1,000문항 기록: 122B GPTQ 96.0
- 전체 `VAL_N`문항 정확도는 앙상블을 고를 때 씁니다.
""")
code(_z["CATS_BLOCK"].rstrip() + r'''

predict(VAL_USE, VAL_WORK, f"검증 {len(VAL_USE):,}")
PV_ = read_preds(VAL_WORK).set_index("id")
m = VAL_USE[["id", "question", "answer"]].merge(PV_.reset_index(), on="id", how="inner"); m["correct"] = m.pred == m.answer
fx = m[m.id.isin(FIXED_VAL_IDS)]
acc_all, acc_fix = float(m.correct.mean()), float(fx.correct.mean()) if len(fx) else float("nan")
se = math.sqrt(max(acc_all * (1 - acc_all), 1e-9) / max(len(m), 1))
SUMMARY = {"model": MODEL_ID, "run": RUN_NAME, "preset": PRESET, "prompt": PROMPT_VERSION, "val_n": len(m), "val_acc": round(acc_all * 100, 2),
           "val_ci95": round(1.96 * se * 100, 2), "fixed1000_acc": round(acc_fix * 100, 2), "fixed1000_n": len(fx), "letter_mass": round(float(m.letter_mass.mean()), 4)}
print(f"▶ {MODEL_ID} zero-shot | {PRESET} | 검증 {len(m):,}문항 {acc_all * 100:.2f}% (±{1.96 * se * 100:.2f}%p) | 팀 공통 1,000문항 {acc_fix * 100:.2f}% ({len(fx)}문항) | letter_mass {m.letter_mass.mean():.3f}")
for k, pat in CATS.items():
    sel = m.question.str.contains(pat)
    if sel.sum() >= 30: print(f"    {k:<12} {m.correct[sel].mean() * 100:5.1f}%  (n={int(sel.sum())})")
''')

md("""
## 10. test 채점 → E14 형식 저장 · 단독 제출 파일
- E14 형식 파일: `preds_final_ep1__<해상도>__1000.csv`(팀 공통 검증) · `preds_final_ep1_test__<해상도>__6714.csv` · `train_log.json`(`ZERO_SHOT: true`)
  - ens_search·E14·E20이 이 실행을 멤버로 자동으로 잡습니다.
""")
code(r'''
if DO_TEST:
    predict(test_df, TEST_WORK, "test")
    PT_ = read_preds(TEST_WORK).set_index("id")
    miss_t = [i for i in test_df.id if i not in PT_.index]
    assert not miss_t, f"test 예측 누락 {len(miss_t)} → 이 셀을 다시 실행하면 남은 것만 채점합니다"
    VAL_FINAL, TEST_FINAL = os.path.join(RUN_DIR, f"preds_final_ep1__{PRESET}__{len(val_df)}.csv"), os.path.join(RUN_DIR, f"preds_final_ep1_test__{PRESET}__{len(test_df)}.csv")
    if set(val_df.id) <= set(PV_.index):
        PV_.loc[val_df.id].reset_index().to_csv(VAL_FINAL, index=False)
    PT_.loc[test_df.id].reset_index().to_csv(TEST_FINAL, index=False)
    json.dump({"epoch_acc": {"1": acc_fix}, "history": [{"step": 0, "zero_shot_acc": acc_fix, "zero_shot_n": len(fx), "zero_shot_acc_all": acc_all, "zero_shot_n_all": len(m)}],
               "config": {"MODEL_ID": MODEL_ID, "ZERO_SHOT": True, "TRAIN_ON_FULL": False, "SEED": SEED, "EVAL_PRESET": PRESET, "PROMPT_VERSION": PROMPT_VERSION,
                          "TEMPLATE_KW": TEMPLATE_KW, "ANSWER_PREFIX": ANSWER_PREFIX, "MM_KW": MM_KW, "ENGINE": f"vllm {vllm.__version__}", "QUANTIZATION": QUANTIZATION,
                          "VAL_ORDER_SEED": VAL_ORDER_SEED, "SPLIT_MD5": split_md5}},
              open(os.path.join(RUN_DIR, "train_log.json"), "w"), indent=1, default=str)
    print(f"✅ E14 형식 저장: {os.path.basename(VAL_FINAL) if os.path.exists(VAL_FINAL) else '(검증 1,000 미만 → 없음)'} · {os.path.basename(TEST_FINAL)} · train_log.json → {RUN_DIR}")
    if MAKE_SUBMISSION:
        sub = sub_df[["id"]].merge(PT_["pred"].rename("answer").reset_index(), on="id", how="left")
        assert len(sub) == len(sub_df) and sub.answer.notna().all() and set(sub.answer) <= set(LETTERS)
        sp_ = os.path.join(RUN_DIR, f"submission_{RUN_NAME}.csv"); sub.to_csv(sp_, index=False)
        print("단독 제출 파일:", sp_, "| 예측 분포:", sub.answer.value_counts().sort_index().to_dict())
    SUMMARY["test_n"] = len(PT_); SUMMARY["test_letter_mass"] = round(float(PT_.letter_mass.mean()), 4)
else:
    print("test 생략 (DO_TEST=False)")
''')

md("## 11. 결과 요약 (다음 모델은 MODEL_ID를 바꾸고 세션 다시 시작)")
code(r'''
print("RUN:", RUN_NAME, "| vllm", vllm.__version__, "| GPU:", GPU_NAME)
for k, v in SUMMARY.items(): print(f"  {k}: {v}")
print("  가져온 예전 결과:", read_preds(VAL_WORK).src.str.startswith("import").sum(), "(검증) ·", read_preds(TEST_WORK).src.str.startswith("import").sum() if os.path.exists(TEST_WORK) else 0, "(test)")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "G4", "machine_shape": "hm"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
