# -*- coding: utf-8 -*-
"""E7 노트북 빌더: 불확실 문항만 보기별 Yes/No 제로샷 검증 → 기준 앙상블과 섞기 → 검증 판정 → test 후보.

설치·import·프롬프트·해상도·로짓 함수는 stage2 노트북 셀을 그대로 가져옵니다.
"""
import json, sys
import nbformat

SRC_NB = "/home/user/vqa/src_stage2.ipynb"
OUT_NB = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e7/e7_yesno_verify.ipynb"
src_nb = json.load(open(SRC_NB, encoding="utf-8"))
def S(i):
    s = src_nb["cells"][i]["source"]; return "".join(s) if isinstance(s, list) else s
def between(text, a, b):
    i = text.index(a); j = text.index(b, i); return text[i:j]
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80]); return text.replace(old, new)

S16 = S(16)
SET_RES = between(S16, "def set_resolution", "set_resolution(TRAIN_PRESET)").rstrip() + "\n"
LAST_LOGITS = between(S16, "def _last_logits", "def unpermute").rstrip() + "\n"

# ───────────────────────────── 1. 설정 ─────────────────────────────
CFG = r'''
import re
MODEL_ID  = "google/gemma-4-31B-it"   # 보기별 Yes/No 검증에 쓸 모델. 어댑터 없이 제로샷으로 씁니다. 예) "Qwen/Qwen3.8-27B"
RUN_NAME  = "e7_yn_" + re.sub(r"[^a-z0-9]+", "_", MODEL_ID.split("/")[-1].lower()).strip("_")   # 결과 폴더 = vqa_runs/<RUN_NAME>
PRESET    = "up2.0"                   # 채점 해상도 (팀 확정 운용)
USE_4BIT  = "auto"                    # "auto": bf16이 GPU에 안 들어가면 4bit(NF4) / True / False. E1에서 NF4와 bf16 차이가 없음을 확인
RUN_TEST  = "auto"                    # "auto": 1차 가설이 채택될 때만 test 불확실 문항 채점 / True: 항상 / False: 안 함
EXTRA_IDS_CSV = "e7_on_best/e7_missing_ids.csv"   # (선택) E7-b가 남긴 '다른 조합에서 불확실한데 점수가 없는 문항' 목록. 있으면 그 문항도 채점 (OUT_DIR 기준, 파일이 없으면 무시)

# 기준 앙상블: 불확실 문항을 고르고 검증 점수와 섞는 대상 (검증 97.7로 재현되는지 먼저 확인)
Q35, Q38 = ("q35_27b_r16_native", "up2.0"), ("q38_27b_r16_native", "up2.0")
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", "up2.0"), ("gemma4_31b_r16_up20", "up2.0"), ("hcx32b_r16_native", "up2.0")
BASE = dict(name="clean4_qg_hcx", members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2],
            expected_val_acc=97.7, file="submission_clean4_qg_hcx.csv")
VOTE_FILE = "submission_majority_vote.csv"   # 공개 최고(0.97676, 다수결). E7이 바꾼 문항만 이 제출에도 반영한 후보를 만듭니다 (없으면 건너뜀)
ENSEMBLE_SUBDIR = "ensembles"                # 제출 파일 폴더 (vqa_runs 아래)

# ── 사전 기준 (결과를 보기 전에 고정) ──
TAU, ALPHA = 0.5, 0.25                       # 1차 가설: 기준 앙상블 1·2위 확률 차 < 0.5인 문항만, (1−α)·p앙상블 + α·q검증
TAUS, ALPHAS = [0.3, 0.5, 0.8], [0.25, 0.5, 1.0]   # 탐색 표 (참고용. 표에서 가장 좋은 칸을 골라 채택하지 않음)
EXCLUDE_NEG = True                           # 부정형 질문("~가 아닌 것" 등)은 섞지 않음 (Yes/No 방향 혼동 위험)
NET_MIN = 3                                  # 순이득 ≥ +3 채택, +1~+2 공개 1회 확인 후보, 0 이하 불채택

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
BATCH_SIZE   = "auto"          # 한 번에 채점하는 (문항, 보기) 수. OOM이면 자동으로 절반
SEED, VAL_SIZE = 42, 1000      # 검증 분할용 — 팀 공통값이므로 절대 바꾸지 마세요
HF_TOKEN     = ""              # Gemma처럼 라이선스 승인이 필요한 모델일 때 (비워 두면 Colab 보안 비밀 'HF_TOKEN'을 찾아봄)

RES_PRESETS = {
    "baseline384": (384 * 384, 384 * 384),
    "mid":         (256 * 28 * 28, 512 * 28 * 28),
    "native":      (256 * 28 * 28, 1280 * 28 * 28),
    "up1.5":       (1_550_000, 2_100_000),
    "up2.0":       (2_700_000, 3_400_000),
}
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}
RES_PRESET = PRESET
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # GPU 메모리 단편화 완화 (torch import 전에 설정)
'''

DATA = r'''
import os, subprocess, json
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
for (r, p) in BASE["members"]:
    assert os.path.isdir(os.path.join(OUT_DIR, r)), f"기준 앙상블 멤버 폴더가 없습니다: {r} → 1번 셀 BASE를 확인하세요"
if IN_COLAB and not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    print("압축 해제 중... (몇 분 걸립니다)")
    r = subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP], capture_output=True, text=True)
    print(r.stdout[-400:]); assert r.returncode == 0, r.stderr[-800:]
if not HF_TOKEN and IN_COLAB:
    try:
        from google.colab import userdata; HF_TOKEN = userdata.get("HF_TOKEN") or ""
    except Exception: pass
if HF_TOKEN:
    from huggingface_hub import login; login(token=HF_TOKEN, add_to_git_credential=False); print("Hugging Face 로그인 완료")
for f in ["train.csv", "test.csv", "sample_submission.csv"]:
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"{f} 가 {DATA_DIR} 에 없습니다. DATA_DIR / 압축 비밀번호를 확인하세요."
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
INFO_PATH = os.path.join(RUN_DIR, "e7_info.json")
E7_INFO = json.load(open(INFO_PATH, encoding="utf-8")) if os.path.exists(INFO_PATH) else {}
print("데이터 OK:", DATA_DIR, "| 결과 폴더(새로 씀):", RUN_DIR)
'''

SPLIT = rep(S(10),
    'if split_md5 == EXPECTED_SPLIT_MD5:\n    print("✅ 팀 공통 분할과 일치")\nelse:\n'
    '    print("⚠️⚠️ 검증 분할이 기준과 다릅니다 → 팀원 간 점수 비교 불가. train.csv 원본/SEED/VAL_SIZE를 확인하세요.")\n'
    'val_df[["id"]].to_csv(os.path.join(OUT_DIR, "valid_ids.csv"), index=False)',
    'assert split_md5 == EXPECTED_SPLIT_MD5, "⚠️ 검증 분할이 기준과 다릅니다 → 기존 멤버 예측과 비교할 수 없습니다. train.csv 원본/SEED/VAL_SIZE를 확인하세요."\n'
    'print("✅ 팀 공통 분할과 일치")')

PROMPT_YN = S(12).rstrip() + "\n" + r'''

# ── E7: 보기 하나씩 따로 묻는 Yes/No 프롬프트 (사전에 고정) ──
YN_SYSTEM = ("You are a careful visual question answering assistant. Judge whether the candidate answer is correct for the question, "
             "based only on the image. Answer with exactly one word: Yes or No.")
def yn_user_text(question, option):
    return f"질문: {question}\n후보 답: {option}\n\n이미지에 근거할 때 이 후보 답이 질문의 정답입니까? Yes 또는 No로만 답하세요."
def yn_messages(row, k):
    user_text = yn_user_text(str(row["question"]), str(row[LETTERS[k]]))
    if globals().get("USE_SYSTEM_ROLE", True):
        return [{"role": "system", "content": [{"type": "text", "text": YN_SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": YN_SYSTEM + "\n\n" + user_text}]}]
YES_WORDS = ["Yes", "yes", "YES", " Yes", " yes", "예", "네"]          # 첫 토큰 확률을 합쳐서 씀 (공백·대소문자·한국어 변형)
NO_WORDS  = ["No", "no", "NO", " No", " no", "아니오", "아니요", "아니"]
NEG_PAT = r"아닌|않는|없는|틀린|잘못"                                  # stage2의 '부정형' 분류와 같은 규칙
'''

BASECELL = r'''
# ── 기준 앙상블의 검증·test 확률을 멤버 예측 파일로 다시 만들고, 불확실 문항을 고릅니다 (GPU 불필요) ──
def _probs(d): return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
def _load_preds(path, ids):
    d = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in ids if i not in d.index]
    assert not miss, f"{os.path.basename(path)}에 없는 문항 {len(miss)}개 (예: {miss[:3]})"
    return d.loc[ids]
def find_val_file(run, preset):
    d = os.path.join(OUT_DIR, run)
    if not os.path.isdir(d): return None, "폴더 없음"
    rx = re.compile(rf"^preds_final_ep(\d+)__{re.escape(preset)}__1000\.csv$")
    c = {int(m.group(1)): f for f in os.listdir(d) if (m := rx.match(f))}
    if not c: return None, "검증 파일 없음"
    if len(c) == 1: return os.path.join(d, next(iter(c.values()))), ""
    lp = os.path.join(d, "train_log.json")
    ea = {int(k): v for k, v in (json.load(open(lp, encoding="utf-8")).get("epoch_acc") or {}).items()} if os.path.exists(lp) else {}
    ep = max(ea, key=lambda e: (ea[e], e)) if ea else None
    ep = ep if ep in c else max(c)
    return os.path.join(d, c[ep]), f"ep{ep} 사용(여러 개 중)"
def find_member_file(run, preset, split):
    f, why = find_val_file(run, preset)
    if f is None or split == "val": return f, why
    ep = re.match(r"^preds_final_ep(\d+)__", os.path.basename(f)).group(1)
    d = os.path.join(OUT_DIR, run)
    t = sorted(x for x in os.listdir(d) if re.match(rf"^preds_final_ep{ep}_test__{re.escape(preset)}__\d+\.csv$", x))
    return (os.path.join(d, t[0]), why) if t else (None, f"ep{ep} test 파일 없음")
def file_answers(fname):
    p = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR, fname)
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
def margin(P):
    s = np.sort(P, axis=1); return s[:, -1] - s[:, -2]
def _paired(a_old, a_new, y):
    c0, c1 = a_old == y, a_new == y
    fixed, broken = int((~c0 & c1).sum()), int((c0 & ~c1).sum())
    return fixed, broken, fixed - broken, int((a_old != a_new).sum())

VAL_IDS, Y_VAL, TEST_IDS = val_df.id.tolist(), val_df.answer.map(LETTERS.index).to_numpy(), test_df.id.tolist()
_used = []
def base_probs(split):
    Ps = []
    for run, pre in BASE["members"]:
        f, why = find_member_file(run, pre, split)
        if f is None: raise FileNotFoundError(f"{run}@{pre} {split}: {why}")
        _used.append(f"{run}:{os.path.basename(f)}{('(' + why + ')') if why else ''}")
        Ps.append(_probs(_load_preds(f, VAL_IDS if split == "val" else TEST_IDS)))
    w = np.asarray(BASE["weights"], dtype=float); assert len(w) == len(Ps), "BASE의 members와 weights 개수가 다릅니다"
    return sum(wi * Pi for wi, Pi in zip(w, Ps)) / w.sum()
PB_VAL, PB_TEST = base_probs("val"), base_probs("test")
A_VAL, A_TEST = PB_VAL.argmax(1), PB_TEST.argmax(1)
acc_base = (A_VAL == Y_VAL).mean() * 100
print(f"기준 앙상블 {BASE['name']}: 검증 {acc_base:.2f}%" + (f" (기대 {BASE['expected_val_acc']})" if BASE.get("expected_val_acc") is not None else ""))
print("  사용한 멤버 파일:", sorted(set(_used)))
if BASE.get("expected_val_acc") is not None:
    assert abs(acc_base - BASE["expected_val_acc"]) < 0.05, "기준 앙상블이 기록한 검증 정확도로 재현되지 않습니다 → 1번 셀 BASE(멤버·가중치)를 확인하세요"
_fb = file_answers(BASE["file"]) if BASE.get("file") else None
print("  test 재현(제출 파일과 같은 답):", "제출 파일 없음" if _fb is None else f"{int((_fb == A_TEST).sum()):,}/{len(A_TEST):,}" + ("" if (_fb == A_TEST).all() else " ⚠️ 후보는 바뀐 문항만 제출 파일에 반영하므로 계속 진행합니다"))

M_VAL, M_TEST = margin(PB_VAL), margin(PB_TEST)
NEG_VAL = val_df.question.astype(str).str.contains(NEG_PAT).to_numpy()
NEG_TEST = test_df.question.astype(str).str.contains(NEG_PAT).to_numpy()
TAU_SCORE = max(TAUS + [TAU])                      # 이 값보다 차이가 작은 문항은 모두 채점해 두고, τ는 나중에 골라 씀
VAL_GATED, TEST_GATED = val_df[M_VAL < TAU_SCORE], test_df[M_TEST < TAU_SCORE]
EXTRA_VAL, EXTRA_TEST = set(), set()
_xp = os.path.join(OUT_DIR, EXTRA_IDS_CSV) if EXTRA_IDS_CSV else ""
if _xp and os.path.exists(_xp):
    _x = pd.read_csv(_xp)
    EXTRA_VAL = set(_x.id[_x.split == "val"]) - set(VAL_GATED.id); EXTRA_TEST = set(_x.id[_x.split == "test"]) - set(TEST_GATED.id)
    VAL_GATED = pd.concat([VAL_GATED, val_df[val_df.id.isin(EXTRA_VAL)]]); TEST_GATED = pd.concat([TEST_GATED, test_df[test_df.id.isin(EXTRA_TEST)]])
    print(f"추가 채점 목록 {EXTRA_IDS_CSV}: 검증 {len(EXTRA_VAL)} · test {len(EXTRA_TEST)}문항을 더 채점합니다 (기준 조합에서는 확신 문항이라 이 노트북의 판정·후보는 그대로)")
print("\n불확실 문항(1·2위 확률 차 < τ) — 검증 1,000문항 기준")
for t in sorted(set(TAUS + [TAU])):
    g = M_VAL < t
    print(f"  τ={t}: {int(g.sum())}문항 (그중 기준 오답 {int((g & (A_VAL != Y_VAL)).sum())}개, 부정형 {int((g & NEG_VAL).sum())}개) | test {int((M_TEST < t).sum())}문항")
print(f"  기준 오답 {int((A_VAL != Y_VAL).sum())}개 중 τ={TAU} 밖(확신 오답) {int(((M_VAL >= TAU) & (A_VAL != Y_VAL)).sum())}개는 이 방법으로 못 고칩니다")
print(f"→ 채점 대상: 검증 {len(VAL_GATED)}문항 × 보기 4 = {4 * len(VAL_GATED)}건, test {len(TEST_GATED)}문항 × 4 = {4 * len(TEST_GATED)}건 (τ < {TAU_SCORE})")
'''

MODEL = r'''
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import transformers, gc
print("transformers", transformers.__version__, "| torch", torch.__version__)
for _v in ("model", "enc", "logits", "out"):
    if _v in globals(): del globals()[_v]
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

assert torch.cuda.is_available(), "GPU 런타임이 아닙니다 (런타임 → 런타임 유형 변경)"
DEVICE = "cuda"
GPU_NAME = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
assert cap[0] >= 8, f"{GPU_NAME}은 bf16 연산을 지원하지 않습니다 → G4 또는 A100을 쓰세요"
DTYPE = torch.bfloat16
_sz = re.search(r"(?<![\d.])(\d+)B", MODEL_ID, flags=re.I)
NEED_GB = int(_sz.group(1)) * 2.05 + 6 if _sz else 0.0          # bf16 가중치 + 채점 여유
use_4bit = (USE_4BIT is True) or (USE_4BIT == "auto" and VRAM_GB < NEED_GB)
if use_4bit and VRAM_GB < NEED_GB * 0.4:
    raise RuntimeError(f"{MODEL_ID}는 4bit로도 {GPU_NAME}({VRAM_GB:.0f}GB)에 올라가지 않습니다")
BS = (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2) if BATCH_SIZE == "auto" else int(BATCH_SIZE)
load_kw = dict(dtype=DTYPE, attn_implementation="sdpa", device_map={"": 0})
if use_4bit:
    load_kw["quantization_config"] = BitsAndBytesConfig(       # stage2와 같은 NF4 설정
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE,
        llm_int8_skip_modules=["lm_head", ".*visual.*", ".*vision_tower.*", ".*vision_model.*", ".*multi_modal_projector.*", ".*embed_vision.*", ".*audio_tower.*", ".*embed_audio.*"])
print(f"GPU: {GPU_NAME} ({VRAM_GB:.0f}GB) | {'4bit(NF4)' if use_4bit else 'bf16'} | 배치={BS} | 제로샷(어댑터 없음)")

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"
tok = processor.tokenizer
FAMILY = "gemma4" if type(processor).__name__.startswith("Gemma4") else "qwen"
IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
if IMAGE_TOKEN_ID is None: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
_tmpl = processor.chat_template if isinstance(processor.chat_template, str) else json.dumps(processor.chat_template)
TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
try:
    processor.apply_chat_template(yn_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}, 0), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
except Exception:
    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")
print(f"계열: {FAMILY} | 이미지 토큰 id: {IMAGE_TOKEN_ID} | chat template 추가 인자: {TEMPLATE_KW} | system 역할: {USE_SYSTEM_ROLE}")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kw).eval()
print(f"모델 로드 {time.time() - t0:.0f}s | {type(model).__name__}")
if "qwen3_5" in str(getattr(model.config, "model_type", "")):
    import importlib.util
    print("✅ flash-linear-attention 커널 사용" if importlib.util.find_spec("fla") else "⚠️⚠️ flash-linear-attention이 없습니다 → 2번 셀 실행 후 '세션 다시 시작'")

def _first_ids(words):
    out = []
    for wd in words:
        t = tok.encode(wd, add_special_tokens=False)
        if t and t[0] not in out: out.append(t[0])
    return out
YES_IDS, NO_IDS = _first_ids(YES_WORDS), _first_ids(NO_WORDS)
_both = set(YES_IDS) & set(NO_IDS)                        # 양쪽에 다 나오는 첫 토큰(예: 공백)은 빼기
YES_IDS, NO_IDS = [i for i in YES_IDS if i not in _both], [i for i in NO_IDS if i not in _both]
assert YES_IDS and NO_IDS, "Yes/No 토큰을 찾지 못했습니다"
print("Yes 토큰:", [tok.decode([i]) for i in YES_IDS], "| No 토큰:", [tok.decode([i]) for i in NO_IDS])
'''

SCORING = SET_RES + "set_resolution(PRESET)\n\n" + LAST_LOGITS + r'''
PREFILL = ""          # 생성 프롬프트 뒤에 붙일 문자열. 10번 셀에서 Yes/No 확률이 가장 높게 나오는 쪽으로 자동 선택 (Gemma의 빈 생각 블록 대비)
def render_yn(row, k):
    return processor.apply_chat_template(yn_messages(row, k), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) + PREFILL

@torch.inference_mode()
def yn_scores(pairs):
    # pairs: [(문항 row, 보기 번호 0~3)] → s = log P(Yes) − log P(No), P(Yes), P(No), 이미지 토큰 수
    global BS
    enc = logits = None; oom = False
    try:
        texts, images, _cache = [], [], {}
        for r, k in pairs:
            texts.append(render_yn(r, k))
            if r["id"] not in _cache: _cache[r["id"]] = load_image(r["path"])
            images.append(_cache[r["id"]])
        if FAMILY == "gemma4": images = [[im] for im in images]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt", **PIXEL_KW)
        ntok = (enc["input_ids"] == IMAGE_TOKEN_ID).sum(dim=1).tolist()
        logits = _last_logits(enc)
        if not torch.isfinite(logits).all(): raise FloatingPointError("로짓에 NaN/Inf 발생")
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:
        del enc, logits; torch.cuda.empty_cache()
        if len(pairs) == 1: raise RuntimeError("배치 1에서도 GPU 메모리 부족")
        h = len(pairs) // 2; BS = max(1, min(BS, h)); print(f"  ⚠️ OOM → 배치를 {BS}(으)로 줄입니다")
        a, b = yn_scores(pairs[:h]), yn_scores(pairs[h:])
        return tuple(np.concatenate([x, z]) if isinstance(x, np.ndarray) else x + z for x, z in zip(a, b))
    logp = torch.log_softmax(logits, dim=-1)
    ly, ln = torch.logsumexp(logp[:, YES_IDS], dim=1), torch.logsumexp(logp[:, NO_IDS], dim=1)
    f = lambda t: t.cpu().numpy().astype(np.float64)
    return f(ly - ln), f(ly.exp()), f(ln.exp()), ntok

def predict_yn(df, tag, verbose=True, save_every=10):
    # df의 문항마다 보기 4개를 채점해 RUN_DIR/yn_<tag>__<PRESET>.csv에 이어 씀 (끊겨도 다시 부르면 남은 것만)
    out_path = os.path.join(RUN_DIR, f"yn_{tag}__{PRESET}.csv")
    done = set()
    if os.path.exists(out_path):
        _d = pd.read_csv(out_path); done = set(zip(_d.id, _d.opt.astype(int)))
    pairs = [(r, k) for r in df.to_dict("records") for k in range(4) if (r["id"], k) not in done]
    if verbose: print(f"[{os.path.basename(out_path)}] 전체 {4 * len(df):,}건 | 완료 {len(done):,} | 남음 {len(pairs):,}")
    recs, t0, n, nb = [], time.time(), 0, 0
    while n < len(pairs):
        batch = pairs[n: n + BS]
        s, py, pn, nt = yn_scores(batch)
        for (r, k), a, b, c, t in zip(batch, s, py, pn, nt):
            recs.append({"id": r["id"], "opt": k, "s": a, "p_yes": b, "p_no": c, "yn_mass": b + c, "img_tokens": int(t)})
        n += len(batch); nb += 1
        if nb % save_every == 0 or n >= len(pairs):
            pd.DataFrame(recs).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False); recs = []
            if verbose:
                el = time.time() - t0
                print(f"  {n:>5}/{len(pairs)} | {el / n:.2f}s/건 | 남은 시간 ~{(len(pairs) - n) * el / n / 60:.0f}분", flush=True)
    if not os.path.exists(out_path): return pd.DataFrame(columns=["id", "opt", "s", "p_yes", "p_no", "yn_mass", "img_tokens"])
    return pd.read_csv(out_path).drop_duplicates(["id", "opt"], keep="last")

def q_matrix(yn, n_rows, idx_of):
    # 채점 결과 → (문항 수, 4) 검증 확률 q = softmax(s). 채점 안 한 문항은 NaN
    Q = np.full((n_rows, 4), np.nan); Sm = np.full((n_rows, 4), np.nan)
    for (i, k), s in zip(zip(yn.id.map(idx_of), yn.opt.astype(int)), yn.s):
        Sm[i, k] = s
    ok = ~np.isnan(Sm).any(1)
    e = np.exp(Sm[ok] - Sm[ok].max(1, keepdims=True)); Q[ok] = e / e.sum(1, keepdims=True)
    return Q, Sm

def mix(P, Q, m, neg, tau, alpha):
    # 불확실(m < τ)이고 채점된 문항만 (1−α)·P + α·Q로 섞은 뒤 답을 고름. EXCLUDE_NEG면 부정형은 그대로
    sel = (m < tau) & ~np.isnan(Q).any(1)
    if EXCLUDE_NEG: sel &= ~neg
    Pn = P.copy(); Pn[sel] = (1 - alpha) * P[sel] + alpha * Q[sel]
    return Pn.argmax(1), sel
'''

PRECHECK = r'''
# 채점 전 점검: 불확실 문항 6개 × 보기 4. 생성 프롬프트 뒤에 무엇을 붙여야 다음 토큰이 Yes/No인지(확률 합이 가장 큰 쪽) 자동으로 고릅니다
set_resolution(PRESET)
_chk = (VAL_GATED if len(VAL_GATED) else val_df).head(6)
_pairs = [(r, k) for r in _chk.to_dict("records") for k in range(4)]
_cands = [""] + (["<|channel>thought\n<channel|>"] if FAMILY == "gemma4" else [])
_res = []
for _pf in _cands:
    PREFILL = _pf
    _s, _py, _pn, _nt = yn_scores(_pairs)
    _mass = float((_py + _pn).mean())
    _pick = _s.reshape(-1, 4).argmax(1)
    _gold = _chk.answer.map(LETTERS.index).to_numpy()
    print(f"붙이는 문자열 {_pf!r}: Yes/No 확률 합 평균 {_mass:.3f} | 6문항 중 검증 모델이 고른 답이 정답 {int((_pick == _gold).sum())}개 (참고)")
    _res.append((_mass, _pf))
_mass, PREFILL = max(_res)
_prec = "4bit(NF4)" if use_4bit else "bf16"
if E7_INFO.get("precision") and E7_INFO["precision"] != _prec:
    print(f"⚠️ 이전 채점은 {E7_INFO['precision']}, 지금은 {_prec}입니다. 차이는 작지만(E16: NF4↔bf16 결과 같음) 같은 GPU 종류로 이어 채점하는 것이 좋습니다")
if E7_INFO.get("prefill") is not None and E7_INFO["prefill"] != PREFILL:
    raise RuntimeError(f"이전 채점은 PREFILL={E7_INFO['prefill']!r}로 했는데 지금은 {PREFILL!r}입니다 → 섞이지 않도록 RUN_NAME을 바꾸세요")
assert _mass > 0.5, "Yes/No 확률 합이 낮습니다 → 모델이 다른 말을 하려 합니다. 위 출력을 확인하세요"
E7_INFO.update({"model_id": MODEL_ID, "precision": "4bit(NF4)" if use_4bit else "bf16", "prefill": PREFILL, "preset": PRESET, "base": BASE["name"],
                "tau": TAU, "alpha": ALPHA, "exclude_neg": EXCLUDE_NEG, "yes_tokens": [tok.decode([i]) for i in YES_IDS], "no_tokens": [tok.decode([i]) for i in NO_IDS]})
json.dump(E7_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"✅ 사용: PREFILL={PREFILL!r} (Yes/No 확률 합 {_mass:.3f}) | 이미지 토큰 {_nt[:4]}")
print("프롬프트 끝부분 예:", repr(render_yn(_pairs[0][0], 0)[-160:]))
'''

VALRUN = r'''
set_resolution(PRESET)
YN_VAL = predict_yn(VAL_GATED, "val")
print(f"검증 불확실 {len(VAL_GATED)}문항 × 4 = {len(YN_VAL)}건 | Yes/No 확률 합 평균 {YN_VAL.yn_mass.mean():.3f} (최소 {YN_VAL.yn_mass.min():.3f})")
'''

EVAL = r'''
_vi = {i: n for n, i in enumerate(VAL_IDS)}
Q_VAL, S_VAL = q_matrix(YN_VAL, len(VAL_IDS), _vi)
V = {"lines": []}
def say(s): print(s); V["lines"].append(s)

# (1) 검증 모델 단독 실력: 불확실 문항에서 기준 앙상블과 비교
g = (M_VAL < TAU) & ~np.isnan(Q_VAL).any(1)
qa = np.where(g, np.nan_to_num(Q_VAL).argmax(1), -1)
say(f"[검증 모델 단독] τ={TAU} 불확실 {int(g.sum())}문항: 기준 앙상블 정답 {int((A_VAL[g] == Y_VAL[g]).sum())} / 검증 모델 정답 {int((qa[g] == Y_VAL[g]).sum())}"
    f" | 기준이 틀린 {int((g & (A_VAL != Y_VAL)).sum())}문항 중 검증 모델이 맞힘 {int((g & (A_VAL != Y_VAL) & (qa == Y_VAL)).sum())}"
    f" | 기준이 맞힌 {int((g & (A_VAL == Y_VAL)).sum())}문항 중 검증 모델이 다르게 고름 {int((g & (A_VAL == Y_VAL) & (qa != Y_VAL)).sum())}")
gn = g & NEG_VAL
if gn.any(): say(f"  부정형 {int(gn.sum())}문항: 기준 정답 {int((A_VAL[gn] == Y_VAL[gn]).sum())} / 검증 모델 정답 {int((qa[gn] == Y_VAL[gn]).sum())}" + (" (섞지 않음)" if EXCLUDE_NEG else ""))
_m = float(YN_VAL.yn_mass.mean())
health = [] if _m >= 0.8 else [f"Yes/No 확률 합 평균 {_m:.3f} < 0.8"]

# (2) 탐색 표 + 1차 가설
rows = []
for t in TAUS:
    for a_ in ALPHAS:
        a1, sel = mix(PB_VAL, Q_VAL, M_VAL, NEG_VAL, t, a_)
        f, b, n, D = _paired(A_VAL, a1, Y_VAL)
        rows.append({"τ": t, "α": a_, "섞은 문항": int(sel.sum()), "고침": f, "새로 틀림": b, "순이득": n, "바뀐 답": D, "정확도(%)": round((a1 == Y_VAL).mean() * 100, 2),
                     "1차": "◀" if (t == TAU and a_ == ALPHA) else ""})
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(RUN_DIR, f"e7_grid_{PRESET}.csv"), index=False)
print("\n탐색 표 (참고용 — 가장 좋은 칸을 고르면 검증셋에 맞춘 것이 됩니다):"); print(tab.to_string(index=False))
a1, sel = mix(PB_VAL, Q_VAL, M_VAL, NEG_VAL, TAU, ALPHA)
f1, b1, n1, D1 = _paired(A_VAL, a1, Y_VAL)
say(f"[1차 가설] τ={TAU}, α={ALPHA}, 부정형 {'제외' if EXCLUDE_NEG else '포함'}: {acc_base:.2f}% → {(a1 == Y_VAL).mean() * 100:.2f}% | 고침 {f1} | 새로 틀림 {b1} | 순이득 {n1:+d} | 바뀐 답 {D1}")
_chg = val_df.assign(기준=[LETTERS[i] for i in A_VAL], E7=[LETTERS[i] for i in a1], p_gold_기준=PB_VAL[np.arange(len(Y_VAL)), Y_VAL].round(3),
                     q_gold=np.round(np.nan_to_num(Q_VAL)[np.arange(len(Y_VAL)), Y_VAL], 3), 차이=np.round(M_VAL, 3))
_chg = _chg[_chg.기준 != _chg.E7].copy()
_chg["결과"] = np.where(_chg.E7 == _chg.answer, "고침", np.where(_chg.기준 == _chg.answer, "새로 틀림", "둘 다 틀림"))
_chg[["id", "결과", "answer", "기준", "E7", "p_gold_기준", "q_gold", "차이", "question", "a", "b", "c", "d"]].to_csv(
    os.path.join(RUN_DIR, f"e7_changed_{PRESET}.csv"), index=False, encoding="utf-8-sig")
if len(_chg): print("\n1차 가설에서 답이 바뀐 문항:"); print(_chg[["id", "결과", "answer", "기준", "E7", "p_gold_기준", "q_gold", "차이", "question"]].to_string(index=False))

# (3) 판정
if health:
    V["verdict"] = f"확인 필요 — {'; '.join(health)} → 판정하지 않습니다. 위 출력을 확인하세요"
elif n1 >= NET_MIN:
    V["verdict"] = f"채택 — 1차 가설 순이득 {n1:+d} ≥ +{NET_MIN}" + ("" if n1 >= 2 * math.sqrt(D1) else " / 단, 2√D 미만이라 공개 제출로 한 번 더 확인 권장")
elif n1 >= 1:
    V["verdict"] = f"보류 — 1차 가설 순이득 {n1:+d} → 공개 1회 확인 후보 (test를 돌리려면 13번 셀에서 직접 켜기)"
else:
    V["verdict"] = f"불채택 — 1차 가설 순이득 {n1:+d}"
_expl = tab[(tab.순이득 >= NET_MIN) & (tab["1차"] == "")]
if len(_expl) and n1 < NET_MIN: say("  (탐색 표에서만 +3 이상: " + ", ".join(f"τ={r.τ}/α={r.α} {r.순이득:+d}" for r in _expl.itertuples()) + " → 사전 기준이 아니므로 자동 채택하지 않습니다)")
AUTO_TEST_OK = V["verdict"].startswith("채택")
say("판정: " + V["verdict"])
E7_INFO.update({f"val_{PRESET}": {"primary": {"fixed": f1, "broken": b1, "net": n1, "D": D1}, "verdict": V["verdict"], "yn_mass": round(_m, 4)}})
json.dump(E7_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

TEST = r'''
# RUN_TEST = True     # ← 판정과 상관없이 test를 돌리려면 이 줄 맨 앞의 #을 지우고 이 셀만 다시 실행
TAU_USE, ALPHA_USE = TAU, ALPHA   # 제출 후보에 쓸 값. 기본은 1차 가설 (바꾸려면 검증에서 확인한 값만)
DO_TEST = (RUN_TEST is True) or (RUN_TEST == "auto" and (AUTO_TEST_OK or bool(EXTRA_TEST)))   # 추가 채점 목록에 test 문항이 있으면 채점 (E7-b용)
if not DO_TEST:
    print(f"test 채점 생략 (RUN_TEST={RUN_TEST!r} | 판정: {V['verdict']})")
else:
    assert TAU_USE <= TAU_SCORE, f"TAU_USE는 채점해 둔 범위 {TAU_SCORE} 이하여야 합니다"
    set_resolution(PRESET)
    YN_TEST = predict_yn(TEST_GATED, "test")
    Q_TEST, _ = q_matrix(YN_TEST, len(TEST_IDS), {i: n for n, i in enumerate(TEST_IDS)})
    a_new, sel = mix(PB_TEST, Q_TEST, M_TEST, NEG_TEST, TAU_USE, ALPHA_USE)
    changed = a_new != A_TEST                     # E7 때문에 기준 앙상블 답이 바뀐 문항
    tag = f"{RUN_NAME}_t{TAU_USE}_a{ALPHA_USE}".replace(".", "")
    V["cands"] = []
    for name, fname in [(BASE["name"], BASE.get("file")), ("majority_vote", VOTE_FILE)]:
        base_ans = file_answers(fname) if fname else None
        if base_ans is None:
            if name != BASE["name"]: print(f"{fname}: 파일이 없어 건너뜁니다"); continue
            base_ans = A_TEST
        cand = np.where(changed, a_new, base_ans)
        cp = os.path.join(RUN_DIR, f"submission_{name}__{tag}.csv")
        sub_df[["id"]].assign(answer=[LETTERS[i] for i in cand]).to_csv(cp, index=False)
        V["cands"].append((os.path.basename(cp), int((cand != base_ans).sum())))
        print(f"저장: {cp} | 기존 {name} 제출과 다른 답 {int((cand != base_ans).sum())}개 / {len(cand)} (섞은 문항 {int(sel.sum())}개)")
    E7_INFO.update({f"test_{PRESET}": {"tau": TAU_USE, "alpha": ALPHA_USE, "candidates": V["cands"]}})
    json.dump(E7_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

SUMMARY = r'''
print(f"E7 | {MODEL_ID} 제로샷 {'4bit(NF4)' if use_4bit else 'bf16'} | PREFILL={PREFILL!r} | 기준 {BASE['name']} {acc_base:.2f}% | {PRESET} | {GPU_NAME}")
for _l in V["lines"]: print(_l)
for n, d in V.get("cands", []): print(f"제출 후보: {n} | 기존 제출과 다른 답 {d}개")
if torch.cuda.is_available(): print(f"최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB / {VRAM_GB:.0f}GB")
'''

INTRO = r'''
# E7 · 불확실 문항만 보기별 Yes/No 검증 (학습 없음, 제로샷)

**하는 일**
- 기준 앙상블(clean4_qg_hcx, 검증 97.7)에서 **1·2위 확률 차가 작은 문항만** 골라, 보기 4개를 하나씩 따로 "이 후보 답이 정답입니까? Yes/No"로 묻습니다. 보기 기호(a~d)를 고르는 방식이 아니어서, 기존 멤버들과 다른 종류의 신호를 얻습니다.
- 보기마다 s = log P(Yes) − log P(No)를 구해 q = softmax(s)로 바꾸고, 불확실 문항에서만 (1−α)·p앙상블 + α·q로 섞습니다. 부정형 질문("~가 아닌 것" 등)은 섞지 않습니다.
- 검증 1,000문항에서 기준 앙상블과 같은 문항끼리 고침·새로 틀림을 세어 판정합니다. 불확실하지 않은 문항은 건드리지 않으므로, 채점은 불확실 문항만 합니다.

**사전 기준 (결과를 보기 전에 고정)**
- 1차 가설 하나로 판정합니다: **τ = 0.5, α = 0.25 → 순이득 ≥ +3이면 채택**(test 채점 진행), +1~+2면 공개 1회 확인 후보, 0 이하면 불채택.
- τ ∈ {0.3, 0.5, 0.8} × α ∈ {0.25, 0.5, 1.0} 표는 참고용입니다. 표에서 가장 좋은 칸을 골라 채택하지 않습니다.

**검증 모델**: 기본은 Gemma 4 31B 제로샷(어댑터 없음, 지금까지 시험한 모델 중 제로샷 점수가 가장 높았음). `MODEL_ID`만 바꾸면 Qwen3.8-27B 등으로도 돌릴 수 있고, 결과 폴더는 모델 이름별로 따로 생깁니다.
**런타임**: G4면 bf16, A100 40GB면 자동으로 4bit(NF4). E1에서 NF4와 bf16 결과 차이가 없음을 확인했으므로 어느 쪽이든 괜찮습니다.
**시간(추정)**: 모델 받기·로드 약 10분 + 검증 불확실 문항 채점 5~10분 → 판정. 채택이면 test 불확실 문항 20~40분(진행 중 남은 시간 표시).
**결과**: `vqa_runs/<RUN_NAME>/`에만 씁니다. 기존 실행·제출 파일은 읽기만 합니다. 채택되면 clean4와 다수결(0.97676) 각각에 E7이 바꾼 문항만 반영한 제출 후보 2개를 만듭니다.

**실행**: 1번 셀을 확인하고 위에서부터 끝까지 실행하세요. 7번 셀에서 채점할 문항 수를 먼저 보여 줍니다(GPU 불필요).
'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 라이브러리 설치 (학습 때와 같은 버전)"), ("code", S(4)),
    ("markdown", "## 3. Drive 연결 → 데이터 준비"), ("code", DATA),
    ("markdown", "## 4. 공통 import · CSV 로드"), ("code", S(8)),
    ("markdown", "## 5. 고정 검증셋 분할 (기존 멤버 예측과 같아야 비교할 수 있음)"), ("code", SPLIT),
    ("markdown", "## 6. 프롬프트 (기존 + 보기별 Yes/No)"), ("code", PROMPT_YN),
    ("markdown", "## 7. 기준 앙상블 재구성 → 불확실 문항 고르기 (GPU 불필요)\n멤버 예측 파일로 clean4_qg_hcx의 검증(97.7 확인)·test 확률을 다시 만들고, 1·2위 확률 차가 작은 문항 수와 그 안의 기준 오답 수를 보여 줍니다."), ("code", BASECELL),
    ("markdown", "## 8. 검증 모델 로드 (제로샷, 어댑터 없음)"), ("code", MODEL),
    ("markdown", "## 9. 채점 함수 (보기 하나당 한 번, 마지막 위치의 Yes/No 확률)"), ("code", SCORING),
    ("markdown", "## 10. 채점 전 점검 (6문항 × 보기 4)\n다음 토큰이 Yes/No가 되도록 생성 프롬프트 뒤에 붙일 문자열을 고릅니다(Gemma는 빈 생각 블록을 붙인 경우도 비교). 정답은 쓰지 않고 Yes/No 확률 합만으로 고릅니다."), ("code", PRECHECK),
    ("markdown", "## 11. 검증 불확실 문항 채점"), ("code", VALRUN),
    ("markdown", "## 12. 판정: 기준 앙상블과 같은 문항끼리 비교\n- 검증 모델 단독 실력(불확실 문항에서 기준 앙상블과 비교), 탐색 표, 1차 가설 결과, 답이 바뀐 문항 목록(`e7_changed_up2.0.csv`)을 보여 줍니다.\n- **판정은 1차 가설 한 줄로만** 합니다."), ("code", EVAL),
    ("markdown", "## 13. test 불확실 문항 채점 → 제출 후보 (채택 시)\nclean4 제출과 다수결 제출(0.97676) 각각에서, E7 때문에 기준 앙상블 답이 바뀐 문항만 고친 후보를 만듭니다."), ("code", TEST),
    ("markdown", "## 14. 결과 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text))
nbformat.validate(nb); nbformat.write(nb, OUT_NB)
print("저장:", OUT_NB, "| 셀", len(nb.cells))
