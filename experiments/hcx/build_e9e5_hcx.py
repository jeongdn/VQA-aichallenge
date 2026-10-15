# -*- coding: utf-8 -*-
"""HCX 전용 E9·E5 노트북 빌더: HCX는 transformers 개발 버전이 필요해 E9·E5 노트북에서 빠졌음.
이 노트북은 HCX 환경에서 E9와 같은 애매한 문항을 (1) 보기 순서 4가지 × LoRA 켬/끔 (2) up3.0·up4.0으로 채점하고,
E9·E5 노트북이 채점한 다른 멤버 결과와 합쳐 HCX까지 넣은 앙상블을 같은 사전 기준으로 판정합니다.
설치·프롬프트·채점 함수는 팀의 HCX 학습 노트북 셀을, 게이트·판정은 E9·E5 빌더 코드를 그대로 가져옵니다.
"""
import sys, json, uuid, textwrap
import nbformat

OUT_NB = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/hcx/out/e9e5_hcx.ipynb"
SP = "/tmp/vqa_scratch"
HCX_NB = "/tmp/vqa_uploads/2fb9367d-stage2_hcx_32b_train_full.ipynb"

_n5 = {}
_argv = sys.argv; sys.argv = ["build_e5.py", f"{SP}/_e5_tmp.ipynb"]
exec(open("/home/user/vqa/e5/build_e5.py", encoding="utf-8").read(), _n5)   # build_e5 안에서 build_e9도 실행됨
sys.argv = _argv
_n9 = _n5["_ns"]
rep = _n9["rep"]
hcx_nb = json.load(open(HCX_NB, encoding="utf-8"))
def H(i):
    s = hcx_nb["cells"][i]["source"]; return "".join(s) if isinstance(s, list) else s
def indent(code, n=4): return textwrap.indent(code.strip("\n"), " " * n)

CFG = r'''
HCX_RUN   = "hcx32b_r16_native"   # 검증 1,000문항을 빼고 학습한 HCX 멤버 (clean4에서 가중치 2 = 전체의 1/3)
DO_TTA    = True    # E9와 같은 채점: up2.0에서 보기 순서 4가지 × LoRA 켬/끔 → vqa_runs/e9_tta/hcx32b_r16_native/
DO_HIRES  = True    # E5와 같은 채점: 검증 up2.0 재현 확인 + up3.0·up4.0 → vqa_runs/e5_hires/hcx32b_r16_native/
E9_OTHERS = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev"]   # E9 노트북이 채점한 멤버 (판정 때 함께 반영)
E5_OTHERS = ["q35_27b_r16_native", "q38_27b_r16_native"]                                               # E5 노트북이 채점한 멤버
E9_NAME, E5_NAME = "e9_tta", "e5_hires"   # E9·E5 노트북의 RUN_NAME (같은 결과 폴더를 씀. 다른 멤버 파일은 읽기만 함)
HR_PRESETS = ["up3.0", "up4.0"]   # E5와 같은 해상도. 메모리가 모자라거나 해상도가 적용되지 않으면 그 해상도만 자동으로 건너뜀
HR_BS      = {"up2.0": 8, "up3.0": 2, "up4.0": 1}
PRESET     = "up2.0"        # 기존 멤버 예측 파일의 해상도 (기준)
TAU_GATE   = 0.8            # E9·E5와 같은 애매한 문항 (기준 앙상블 3개 중 하나라도 1·2위 확률 차 < 0.8)
SCORE_BASE = True           # LoRA를 끈 원본 모델 점수도 저장 (E9의 H2용). 맨 마지막에 채점
RUN_TEST   = True           # test 애매한 문항도 채점 (제출 후보를 만들려면 필요)
RETRY_SKIPPED = False       # True: 예전 실행에서 건너뛴 고해상도를 다시 시도 (더 큰 GPU로 바꿨을 때)
RUN_NAME   = "e9e5_hcx"     # (기록용 이름)
MODEL_ID   = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"   # 2번 셀 설치 조건용 자리표시. 실제 베이스는 어댑터 기록에서 읽음

# ── 기준 앙상블 레시피 (E8·E9·E5와 같음) ──
Q35, Q38 = ("q35_27b_r16_native", PRESET), ("q38_27b_r16_native", PRESET)
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", PRESET), ("gemma4_31b_r16_up20", PRESET), ("hcx32b_r16_native", PRESET)
RECIPES = {
    "clean4_qg_hcx":    dict(file="submission_clean4_qg_hcx.csv",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], cal=False),
    "ens_q38_gemma4":   dict(file="submission_ens_q38_gemma4.csv",   members=[Q38, G],                 weights=[1, 1],          cal=False),
    "ens_q38_gdev_cal": dict(file="submission_ens_q38_gdev_cal.csv", members=[Q38, GDEV],              weights=[1, 1],          cal=True),
}
VOTE = dict(name="majority_vote", file="submission_majority_vote.csv", voters=["ens_q38_gemma4", "clean4_qg_hcx", "ens_q38_gdev_cal"])
PUBLIC = {"clean4_qg_hcx": 0.97646, "majority_vote": 0.97676}; N_PUBLIC = 3357
ENSEMBLE_SUBDIR = "ensembles"

# ── 사전 기준 (결과를 보기 전에 고정. E9·E5와 같은 값) ──
NET_MIN = 3        # 기준 대비: 검증 순이득 + 공개 순이득(clean4 후보) ≥ +3 이고 공개 순이득 ≥ 0 → 채택
BO = dict(gap=0.15, pmin=0.60, agree=0.75, delta=0.12, w=0.5)   # E9 H2(LoRA-off 결합): 미리 정한 기준값 그대로
BO_NET_MIN = 2
INC_NET_MIN = 2    # E9·E5(HCX 없음) 후보를 이미 채택했다면: HCX를 더한 후보로 바꾸려면 HCX 몫 검증 순이득 + 공개 순이득 ≥ +2 (공개 ≥ 0)

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
BATCH_SIZE   = "auto"
SEED, VAL_SIZE = 42, 1000       # 검증 분할용 — 팀 공통값
SKIP_MISSING_IMAGES = False
HF_TOKEN     = ""               # 비워 두면 Colab 보안 비밀 'HF_TOKEN'을 찾아봄

RES_PRESETS = {
    "baseline384": (384 * 384, 384 * 384),
    "mid":         (256 * 28 * 28, 512 * 28 * 28),
    "native":      (256 * 28 * 28, 1280 * 28 * 28),
    "up1.5":       (1_550_000, 2_100_000),
    "up2.0":       (2_700_000, 3_400_000),
    "up3.0":       (6_000_000, 7_600_000),      # E5와 같은 픽셀 범위 (720×960 사진 → 가로세로 약 2.9배)
    "up4.0":       (10_500_000, 12_500_000),    # (가로세로 약 3.9배)
}
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}
RES_PRESETS_INTERNVL = {"baseline384": 1, "mid": 4, "native": 6, "up1.5": 9, "up2.0": 12, "up2.5": 16}
RES_PRESET = PRESET
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
'''

DATA = r'''
import os, re, json, hashlib, subprocess, shutil
try:
    from google.colab import drive
    drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
# ── HCX 멤버 확인 (읽기만 함): 어댑터·베이스 모델·기존 예측 파일. 이름이 틀렸으면 압축을 풀기 전에 멈춤 ──
d = os.path.join(OUT_DIR, HCX_RUN); assert os.path.isdir(d), f"멤버 폴더가 없습니다: {d} → 1번 셀 HCX_RUN을 확인하세요"
fs = sorted(os.listdir(d))
lg = json.load(open(os.path.join(d, "train_log.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "train_log.json")) else {}
cfg = lg.get("config") or {}
assert not cfg.get("TRAIN_ON_FULL"), f"{HCX_RUN}은 전체 데이터로 학습한 실행입니다 → 검증셋을 뺀 멤버(hcx32b_r16_native)를 쓰세요"
rv = {int(m.group(1)): f for f in fs if (m := re.match(rf"^preds_final_ep(\d+)__{re.escape(PRESET)}__1000\.csv$", f))}
ea = {int(k): v for k, v in (lg.get("epoch_acc") or {}).items()}
ep = next(iter(rv)) if len(rv) == 1 else (max(ea, key=lambda e: (ea[e], e)) if ea else (max(rv) if rv else None))
assert ep in rv, f"{HCX_RUN}: {PRESET} 검증 예측 파일이 없습니다 (폴더 내용 일부: {fs[:8]})"
rt = next((f for f in fs if re.match(rf"^preds_final_ep{ep}_test__{re.escape(PRESET)}__6714\.csv$", f)), None)
ad = os.path.join(d, f"adapter_ep{ep}")
assert os.path.isfile(os.path.join(ad, "adapter_model.safetensors")), f"{HCX_RUN}: 어댑터가 없습니다: {ad}"
acfg = json.load(open(os.path.join(ad, "adapter_config.json"), encoding="utf-8"))
mid = str(acfg.get("base_model_name_or_path") or "")
if not mid or mid.startswith("/"): mid = str(cfg.get("MODEL_ID") or "")
assert mid and not mid.startswith("/"), f"{HCX_RUN}: 베이스 모델 이름을 찾지 못했습니다"
assert "hyperclova" in mid.lower() or "hcx" in HCX_RUN.lower(), f"{HCX_RUN}({mid}): 이 노트북은 HCX 전용입니다 → Qwen·Gemma는 E9·E5 노트북으로"
MEMBER_INFO = {HCX_RUN: dict(dir=d, ep=ep, adapter_dir=ad, model_id=mid, ref_val=os.path.join(d, rv[ep]), ref_test=os.path.join(d, rt) if rt else None,
                             md5=hashlib.md5(open(os.path.join(ad, "adapter_model.safetensors"), "rb").read(1 << 20)).hexdigest()[:8])}
print(f"  {HCX_RUN}: 베이스 {mid} | adapter_ep{ep} | 검증 {rv[ep]} | test {rt or '없음'}")
if RUN_TEST: assert rt, f"{HCX_RUN}: test 예측 파일이 없습니다 (RUN_TEST=True)"
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
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"{f} 가 {DATA_DIR} 에 없습니다"
E9_DIR, E5_DIR = os.path.join(OUT_DIR, E9_NAME), os.path.join(OUT_DIR, E5_NAME)
os.makedirs(os.path.join(E9_DIR, HCX_RUN), exist_ok=True); os.makedirs(os.path.join(E5_DIR, HCX_RUN), exist_ok=True)
def _read_json(p):
    try: return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    except Exception: return {}
# HCX 기록은 E9·E5 노트북의 기록 파일(e9_info.json·e5_info.json)과 따로 둠 → 두 노트북이 동시에 돌아도 서로 덮어쓰지 않음
INFO9_PATH, INFO5_PATH = os.path.join(E9_DIR, HCX_RUN, "hcx_info.json"), os.path.join(E5_DIR, HCX_RUN, "hcx_info.json")
HCX9_INFO, HCX5_INFO = _read_json(INFO9_PATH), _read_json(INFO5_PATH)
for _inf, _dir in ((HCX9_INFO, E9_DIR), (HCX5_INFO, E5_DIR)):
    old = (_inf.get("members") or {}).get(HCX_RUN)
    if old and old.get("md5") != MEMBER_INFO[HCX_RUN]["md5"]:
        raise RuntimeError(f"{_dir}/{HCX_RUN}에는 다른 HCX 어댑터로 만든 결과가 있습니다 → 그 폴더를 옮기고 다시 실행하세요")
# E9·E5 노트북 기록 (읽기만): 다른 멤버가 같은 해상도·같은 문턱으로 채점됐는지
for nm, rec, others, on in (("E9", _read_json(os.path.join(E9_DIR, "e9_info.json")), E9_OTHERS, DO_TTA), ("E5", _read_json(os.path.join(E5_DIR, "e5_info.json")), E5_OTHERS, DO_HIRES)):
    if not on: continue
    for r in others:
        m = (rec.get("members") or {}).get(r)
        if not m: print(f"  {nm} 노트북: {r} 채점 기록 없음 (아직 안 돌렸거나 진행 중) → 끝난 뒤 판정 셀만 다시 실행하면 됨")
        elif m.get("tau_gate") != TAU_GATE or m.get("preset", PRESET) != PRESET:
            print(f"  ⚠️ {nm} 노트북: {r}는 TAU_GATE {m.get('tau_gate')} / 해상도 {m.get('preset', PRESET)}로 채점됨 → 이 노트북 1번 셀과 같게 맞추세요")
print("데이터 OK:", DATA_DIR, "| 결과 폴더:", os.path.join(E9_DIR, HCX_RUN), "·", os.path.join(E5_DIR, HCX_RUN))
'''

SPLIT = _n9["SPLIT"]
assert SPLIT == rep(H(11),
    'if split_md5 == EXPECTED_SPLIT_MD5:\n    print("✅ 팀 공통 분할과 일치")\nelse:\n'
    '    print("⚠️⚠️ 검증 분할이 기준과 다릅니다 → 팀원 간 점수 비교 불가. train.csv 원본/SEED/VAL_SIZE를 확인하세요.")\n'
    'val_df[["id"]].to_csv(os.path.join(OUT_DIR, "valid_ids.csv"), index=False)',
    'assert split_md5 == EXPECTED_SPLIT_MD5, "⚠️ 검증 분할이 기준과 다릅니다 → 기존 멤버 예측과 비교할 수 없습니다"\n'
    'print("✅ 팀 공통 분할과 일치")')

GATE = _n5["GATE"]
GATE = rep(GATE, '''print(f"→ 멤버 {len(MEMBERS_HR)}개 × (검증 {len(VAL_G)} × 해상도 {1 + len(HR_PRESETS)} + test {len(TEST_G) if RUN_TEST else 0} × 해상도 {len(HR_PRESETS)})  (up2.0 검증은 기존 예측 재현 확인용)")''',
           '''print("→ HCX 채점: " + " | ".join(x for x in [
    (f"TTA 검증 {len(VAL_G)} + test {len(TEST_G) if RUN_TEST else 0} × 순서 4" + (" × LoRA 켬/끔" if SCORE_BASE else "")) if DO_TTA else "",
    (f"고해상도 검증 {len(VAL_G)} × {1 + len(HR_PRESETS)} + test {len(TEST_G) if RUN_TEST else 0} × {len(HR_PRESETS)}") if DO_HIRES else ""] if x))''')

_s17 = H(17)
_aug_start = _s17.index("from PIL import ImageEnhance"); _aug_end = _s17.index("def encode_batch")
SCORING = _s17
for old, new in [("set_resolution(TRAIN_PRESET)", "# (해상도는 모델을 올린 뒤 set_resolution(PRESET)으로 정함)"),
                 (_s17[_aug_start:_aug_end], ""),
                 ("    if _AUG_ON and IMG_AUG: images = [augment_image(im) for im in images]\n", ""),
                 ("\"run\": f\"{RUN_NAME}{'_dry' if DRY_RUN else ''}/{tag}\"", "\"run\": f\"{RUN_NAME}/{tag}\"")]:
    SCORING = rep(SCORING, old, new)

LOADER = r'''
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, load_peft_weights, get_peft_model_state_dict
import transformers, peft, gc, contextlib
print("transformers", transformers.__version__, "| peft", peft.__version__, "| torch", torch.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # 채점이 모두 끝났으면 판정만 할 때는 GPU가 없어도 됨
if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0); _cap = torch.cuda.get_device_capability(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
else:
    GPU_NAME, _cap, VRAM_GB = "cpu", (0, 0), 0.0
DTYPE = torch.bfloat16
BS = (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2) if BATCH_SIZE == "auto" else int(BATCH_SIZE)
CLOSE_THINK = False

def _free():
    for _v in ("model", "base", "enc", "logits", "out"):
        if _v in globals(): del globals()[_v]
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _norm_key(k):
    k = k.replace("base_model.model.", "", 1); i = k.rfind("language_model.")
    return k[i + len("language_model."):] if i >= 0 else k
def check_adapter_weights(model, adapter_dir):
    # 어댑터 파일의 LoRA 텐서가 전부 값 그대로 모델에 들어갔는지 (PEFT는 이름이 안 맞는 키를 조용히 건너뛸 수 있음)
    fw = load_peft_weights(adapter_dir, device="cpu"); mw = {k: v.detach() for k, v in get_peft_model_state_dict(model).items()}
    if set(fw) != set(mw): fw = {_norm_key(k): v for k, v in fw.items()}; mw = {_norm_key(k): v for k, v in mw.items()}
    assert set(fw) == set(mw), f"어댑터 파일과 모델의 LoRA 키가 다릅니다 (파일에만 {len(set(fw) - set(mw))}개, 모델에만 {len(set(mw) - set(fw))}개)"
    worst = max((v.to(mw[k].dtype).float() - mw[k].float().cpu()).abs().max().item() for k, v in fw.items())
    assert worst == 0.0, f"어댑터 파일과 모델의 LoRA 값이 다릅니다 (최대 차이 {worst:.3g})"
    assert any("lora_B" in k and v.abs().max().item() > 0 for k, v in fw.items()), "lora_B가 전부 0입니다 → 학습되지 않은 어댑터"
    return len(fw)

def _fix_prompt(t):
    # HCX 학습 노트북과 같음: 생각 모드를 연 채 끝나는 템플릿이면 빈 생각 블록을 닫아, 첫 토큰에서 바로 답하게 함
    if CLOSE_THINK and t.endswith("<think>\n"): return t + "\n</think>\n\n"
    if CLOSE_THINK and t.endswith("<think>"): return t + "\n\n</think>\n\n"
    return t

def load_member(run):
    # HCX를 bf16 베이스 + 기존 어댑터(merge 없음)로 올림. 프로세서·템플릿 처리는 HCX 학습 노트북 7번 셀과 같음
    global processor, tok, FAMILY, IMAGE_TOKEN_ID, TEMPLATE_KW, USE_SYSTEM_ROLE, SYSTEM_AS_STRING, CLOSE_THINK, base, model, LETTER_IDS, MODEL_ID
    assert torch.cuda.is_available(), "GPU 런타임이 아닙니다 (런타임 → 런타임 유형 변경)"
    assert _cap[0] >= 8, f"{GPU_NAME}은 bf16을 지원하지 않습니다 → G4 또는 A100 80GB를 쓰세요"
    inf = MEMBER_INFO[run]; MODEL_ID = inf["model_id"]
    _sz = re.search(r"(?<![\d.])(\d+)B", MODEL_ID, flags=re.I)
    need = int(_sz.group(1)) * 2.05 + 6 if _sz else 0.0
    if VRAM_GB < need: raise RuntimeError(f"bf16 {MODEL_ID}에는 약 {need:.0f}GB가 필요한데 {GPU_NAME}은 {VRAM_GB:.0f}GB입니다 → G4 또는 A100 80GB")
    processor = AutoProcessor.from_pretrained(MODEL_ID); processor.tokenizer.padding_side = "left"; tok = processor.tokenizer
    _pname = type(processor).__name__
    FAMILY = "gemma4" if _pname.startswith("Gemma4") else "internvl" if _pname.startswith("InternVL") else "qwen"
    IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
    if IMAGE_TOKEN_ID is None: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
    _tmpl = processor.chat_template if isinstance(processor.chat_template, str) else json.dumps(processor.chat_template)
    TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
    if not TEMPLATE_KW and _pname.startswith("HyperCLOVAX") and re.search(r"\bthinking\b", _tmpl or ""):
        TEMPLATE_KW = {"thinking": False}
    SYSTEM_AS_STRING = False
    _row = {"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}
    _raw_list = lambda t: "'type': 'text'" in t or '"type": "text"' in t
    try:
        _probe = processor.apply_chat_template(build_messages(_row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
        if _raw_list(_probe):
            SYSTEM_AS_STRING = True
            _probe = processor.apply_chat_template(build_messages(_row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
            if _raw_list(_probe): SYSTEM_AS_STRING = False; USE_SYSTEM_ROLE = False
    except Exception:
        USE_SYSTEM_ROLE = False
    _probe = processor.apply_chat_template(build_messages(_row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
    CLOSE_THINK = _probe.endswith("<think>\n") or _probe.endswith("<think>")
    print("  프롬프트 미리보기(끝):", repr(_fix_prompt(_probe)[-90:]), "| 생각 블록 닫기" if CLOSE_THINK else "")
    t0 = time.time()
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa", device_map={"": 0})
    base.config.use_cache = False
    assert not [n for n, m in base.named_modules() if any(k in type(m).__name__ for k in ("Linear4bit", "Linear8bit"))], "양자화된 층이 있습니다 (bf16이어야 함)"
    model = PeftModel.from_pretrained(base, inf["adapter_dir"], is_trainable=False).eval()
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    n_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and any(k in n for k in ("visual", "vision", "siglip")))
    assert n_lora > 0 and n_vis == 0, f"LoRA 층 {n_lora}개, 비전 타워 {n_vis}개 → 어댑터 부착 이상"
    nk = check_adapter_weights(model, inf["adapter_dir"])
    LETTER_IDS = []
    for L in LETTERS:
        ids = []
        for s in (L, L.upper(), " " + L, " " + L.upper()):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids: ids.append(t[0])
        assert ids and len(tok.encode(L, add_special_tokens=False)) == 1, f"'{L}' 가 단일 토큰이 아닙니다"
        LETTER_IDS.append(ids)
    set_resolution(PRESET)
    print(f"  모델 로드 {time.time() - t0:.0f}s | {type(base).__name__} | 프로세서 {_pname} | 계열 {FAMILY} | LoRA 키 {nk}개 값까지 일치 | 템플릿 인자 {TEMPLATE_KW} | system 역할 {USE_SYSTEM_ROLE}")
print(f"런타임: {GPU_NAME} ({VRAM_GB:.0f}GB) | bf16 | 평가 배치 {BS}")
'''

RUNCELL = r'''
# ── HCX 채점: 모델 로드 → 사전 점검 → ① TTA(LoRA 켬) → ② 고해상도(검증 up2.0 재현 → 검증·test up3.0·up4.0) → ③ TTA(LoRA 끔, H2용) ──
# 끊겨도 다시 실행하면 남은 것만. 파일 형식은 E9·E5 노트북과 같아서 판정 셀이 다른 멤버 결과와 그대로 합칩니다
PERM_COLS = ["id", "lora", "perm", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens"]
RES_COLS = ["id", "preset", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens"]
SPLIT_DF = {"val": VAL_G, "test": TEST_G}
SPLITS = ["val", "test"] if RUN_TEST else ["val"]
def _perm_path(run, split): return os.path.join(E9_DIR, run, f"perm_{split}__{PRESET}.csv")
def _res_path(run, split): return os.path.join(E5_DIR, run, f"res_{split}.csv")
def _done_perm(run, split):
    p = _perm_path(run, split)
    if not os.path.exists(p): return set()
    d = pd.read_csv(p); return set(zip(d.id, d.lora.astype(int), d.perm.astype(int)))
def _done_res(run, split):
    p = _res_path(run, split)
    if not os.path.exists(p): return set()
    d = pd.read_csv(p); return set(zip(d.id, d.preset))
def _skipped_hcx(): return set() if RETRY_SKIPPED else set((HCX5_INFO.get("infeasible") or {}).get(HCX_RUN, []))
def _plan():
    # 주 가설(H1)에 필요한 채점을 먼저, LoRA 끔(H2용)은 맨 뒤
    P = [("tta", s, 1) for s in SPLITS] if DO_TTA else []
    if DO_HIRES: P += [("res", "val", "up2.0")] + [("res", "val", pr) for pr in HR_PRESETS] + ([("res", "test", pr) for pr in HR_PRESETS] if RUN_TEST else [])
    if DO_TTA and SCORE_BASE: P += [("tta", s, 0) for s in SPLITS]
    return P
def _remaining():
    dp = {s: _done_perm(HCX_RUN, s) for s in SPLITS}; dr = {s: _done_res(HCX_RUN, s) for s in SPLITS}; sk = _skipped_hcx(); out = {}
    for kind, s, x in _plan():
        ids = SPLIT_DF[s].id
        if kind == "tta": out[(kind, s, x)] = sum((i, x, k) not in dp[s] for i in ids for k in range(len(TTA_PERMS)))
        else: out[(kind, s, x)] = 0 if x in sk else sum((i, x) not in dr[s] for i in ids)
    return out
def _label(k): return f"{k[1]}·" + (f"TTA LoRA{'켬' if k[2] else '끔'}" if k[0] == "tta" else k[2])
def _mark_skip(preset, why):
    HCX5_INFO.setdefault("infeasible", {}).setdefault(HCX_RUN, [])
    if preset not in HCX5_INFO["infeasible"][HCX_RUN]: HCX5_INFO["infeasible"][HCX_RUN].append(preset)
    HCX5_INFO.setdefault("infeasible_why", {})[preset] = why
    json.dump(HCX5_INFO, open(INFO5_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

def score_split(run, split, lora, save_every=10):
    path = _perm_path(run, split); os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _done_perm(run, split); rows = SPLIT_DF[split].to_dict("records"); recs, n, nb, t0 = [], 0, 0, time.time()
    todo = [(k, [r for r in rows if (r["id"], lora, k) not in done]) for k in range(len(TTA_PERMS))]
    total = sum(len(rs) for _, rs in todo)
    print(f"  [TTA {split} | LoRA {'켬' if lora else '끔'}] {total:,}건 남음", flush=True)
    for k, rs in todo:
        for j in range(0, len(rs), BS):
            batch = rs[j: j + BS]
            with (model.disable_adapter() if not lora else contextlib.nullcontext()):
                p, m, nt = score_rows(batch, [TTA_PERMS[k]] * len(batch))
            for r, pp, mm, tt in zip(batch, p, m, nt):
                recs.append({"id": r["id"], "lora": lora, "perm": k, "p_a": pp[0], "p_b": pp[1], "p_c": pp[2], "p_d": pp[3], "letter_mass": float(mm), "img_tokens": int(tt)})
            n += len(batch); nb += 1
            if nb % save_every == 0 or n >= total:
                pd.DataFrame(recs, columns=PERM_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False); recs = []
                el = time.time() - t0; print(f"    {n:>5}/{total} | {el / n:.2f}s/건 | 남은 시간 ~{(total - n) * el / n / 60:.0f}분", flush=True)
    if recs: pd.DataFrame(recs, columns=PERM_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def score_res(run, split, preset, save_every=10):
    global BS
    path = _res_path(run, split); os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _done_res(run, split); rows = [r for r in SPLIT_DF[split].to_dict("records") if (r["id"], preset) not in done]
    set_resolution(preset); bs0 = BS; BS = min(HR_BS.get(preset, 8), (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2))
    print(f"  [고해상도 {split} | {preset}] {len(rows):,}문항 남음 (배치 {BS})", flush=True)
    recs, n, nb, t0 = [], 0, 0, time.time()
    try:
        while n < len(rows):
            batch = rows[n: n + BS]
            p, m, nt = score_rows(batch)
            if n == 0 and preset != PRESET:
                # 해상도가 실제로 바뀌었는지: 모든 사진이 up2.0에선 최대 픽셀 이하, 고해상도에선 최소 픽셀 이상이어야 하므로 토큰 수로 바로 확인됨
                _lo = 0.85 * RES_PRESETS[preset][0] / RES_PRESETS[PRESET][1] * UP2_TOK_MAX
                if min(nt) < _lo:
                    _mark_skip(preset, f"해상도 미적용: 이미지 토큰 {list(nt)} < 기대 최소 {_lo:.0f} (up2.0 최대 {UP2_TOK_MAX})")
                    print(f"  ⚠️ {preset}: 해상도가 적용되지 않았습니다 (이미지 토큰 {list(nt)}) → HCX는 {preset}를 건너뜁니다")
                    set_resolution(PRESET); BS = bs0; return
            for r, pp, mm, tt in zip(batch, p, m, nt):
                recs.append({"id": r["id"], "preset": preset, "p_a": pp[0], "p_b": pp[1], "p_c": pp[2], "p_d": pp[3], "letter_mass": float(mm), "img_tokens": int(tt)})
            n += len(batch); nb += 1
            if nb % save_every == 0 or n >= len(rows):
                pd.DataFrame(recs, columns=RES_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False); recs = []
                el = time.time() - t0; print(f"    {n:>5}/{len(rows)} | {el / n:.2f}s/문항 | 이미지 토큰 약 {int(np.mean(nt)):,}개 | 남은 시간 ~{(len(rows) - n) * el / n / 60:.0f}분", flush=True)
    except RuntimeError as e:
        if "메모리 부족" not in str(e): raise
        if recs: pd.DataFrame(recs, columns=RES_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        _mark_skip(preset, "배치 1에서도 GPU 메모리 부족"); torch.cuda.empty_cache()
        print(f"  ⚠️ {preset}: 배치 1에서도 GPU 메모리 부족 → HCX는 {preset}를 건너뜁니다")
    set_resolution(PRESET); BS = bs0

rem = _remaining()
print(f"▶ {HCX_RUN} | 남은 채점: " + ", ".join(f"{_label(k)} {n:,}" for k, n in rem.items()))
if sum(rem.values()) == 0:
    print("  이미 끝났습니다 → 채점 건너뜀 (모델도 올리지 않음)")
else:
    _free(); load_member(HCX_RUN)
    # 사전 점검 (up2.0): 어댑터 켬/끔 차이, letter_mass, 이미지 토큰 수가 기존 예측과 같은지
    set_resolution(PRESET)
    _chk = VAL_G.head(4).to_dict("records") or val_df.head(4).to_dict("records")
    p_on, m_on, nt_on = score_rows(_chk)
    with model.disable_adapter(): p_off, _, _ = score_rows(_chk)
    _d = float(np.abs(p_on - p_off).max())
    _ref = pd.read_csv(MEMBER_INFO[HCX_RUN]["ref_val"]).drop_duplicates("id", keep="last").set_index("id").reindex([r["id"] for r in _chk])
    print(f"  사전 점검(up2.0): 어댑터 켬/끔 확률 최대 차이 {_d:.3f} | letter_mass {m_on.mean():.3f} | 이미지 토큰 {list(nt_on)}"
          + (f" (기존 {[int(t) for t in _ref.img_tokens]})" if "img_tokens" in _ref.columns else "")
          + f" | 기존 답과 같은 문항 {int((p_on.argmax(1) == _ref[['p_a', 'p_b', 'p_c', 'p_d']].to_numpy().argmax(1)).sum())}/{len(_chk)}")
    assert _d > 1e-3, "어댑터를 켜고 꺼도 출력이 같습니다 → 어댑터가 적용되지 않았습니다. 위 출력을 확인하세요"
    assert m_on.mean() > 0.9, "letter_mass가 낮습니다 → 프롬프트(생각 블록)·토큰 처리 문제. 위 출력을 확인하세요"
    if "img_tokens" in _ref.columns and not _ref.img_tokens.isna().any():
        assert [int(t) for t in _ref.img_tokens] == [int(t) for t in nt_on], "이미지 토큰 수가 기존 예측과 다릅니다 → 해상도·프로세서가 달라 비교할 수 없습니다"
    UP2_TOK_MAX = int(max(nt_on))
    _rec = {"md5": MEMBER_INFO[HCX_RUN]["md5"], "model_id": MODEL_ID, "adapter": MEMBER_INFO[HCX_RUN]["adapter_dir"], "precision": "bf16", "gpu": GPU_NAME,
            "transformers": transformers.__version__, "tau_gate": TAU_GATE}
    HCX9_INFO.setdefault("members", {})[HCX_RUN] = {**_rec, "preset": PRESET, "perms": [list(p) for p in TTA_PERMS]}
    HCX5_INFO.setdefault("members", {})[HCX_RUN] = {**_rec, "presets": {pr: list(RES_PRESETS[pr]) for pr in ["up2.0"] + HR_PRESETS}}
    for _inf, _p in ((HCX9_INFO, INFO9_PATH), (HCX5_INFO, INFO5_PATH)): json.dump(_inf, open(_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for kind, s, x in _plan():
        if kind == "tta": score_split(HCX_RUN, s, x)
        elif x in _skipped_hcx(): print(f"  [고해상도 {s} | {x}] 건너뜀 ({(HCX5_INFO.get('infeasible_why') or {}).get(x, '이전 실행 기록')})")
        else: score_res(HCX_RUN, s, x)
print("\n채점 끝 → 11번(TTA)·12번(고해상도) 셀에서 판정")
'''

# ── 판정 셀: E9·E5 판정 코드를 그대로 쓰고 (멤버 목록 = 다른 멤버 + HCX), 결과 이름·기록 파일만 HCX용으로 ──
_ev9 = _n9["EVAL"]
for o, n in [('fn, Dn = _write(f"{nm}__e9_tta4", nm, A1[nm]["test"], A0[nm]["test"])', 'fn, Dn = _write(f"{nm}__e9_tta4_hcx", nm, A1[nm]["test"], A0[nm]["test"])'),
             ('fn, Dn = _write(f"{nm}__e9_tta4_baseoff", nm, A2[nm]["test"], A0[nm]["test"])', 'fn, Dn = _write(f"{nm}__e9_tta4_baseoff_hcx", nm, A2[nm]["test"], A0[nm]["test"])'),
             ('    n1, D1 = RES["clean4_qg_hcx"]\n',
              '    n1, D1 = RES["clean4_qg_hcx"]\n'
              '    N_INC = None                                          # HCX 몫: 같은 TTA에서 HCX만 뺀 앙상블 대비\n'
              '    if HCX_RUN in OVER1 and len(OVER1) > 1:\n'
              '        A1x = build({k: v for k, v in OVER1.items() if k != HCX_RUN})\n'
              '        for nm in ("clean4_qg_hcx", VOTE["name"]):\n'
              '            f, b, n, D = paired(A1x[nm]["val"], A1[nm]["val"], Y_VAL)\n'
              '            if nm == "clean4_qg_hcx": N_INC = n\n'
              '            say(f"  (HCX 몫) {nm}: HCX를 뺀 TTA 대비 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}")\n'),
             ('        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 두 파일은 같은 TTA 결과로 만들었으므로 판정은 clean4 후보 한 줄로만 합니다")\n',
              '        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 두 파일은 같은 TTA 결과로 만들었으므로 판정은 clean4 후보 한 줄로만 합니다")\n'
              '        _inc_line(E9_DIR, "submission_clean4_qg_hcx__e9_tta4.csv", "submission_clean4_qg_hcx__e9_tta4_hcx.csv", N_INC, "E9")\n')]:
    _ev9 = rep(_ev9, o, n)

_ev5 = _n5["EVAL"]
for o, n in [('fn, Dn = _write(f"{nm}__e5_hires", nm, A1[nm]["test"], A0[nm]["test"])', 'fn, Dn = _write(f"{nm}__e5_hires_hcx", nm, A1[nm]["test"], A0[nm]["test"])'),
             ('_hp = os.path.join(E5_DIR, "e5_changed_val.html")', '_hp = os.path.join(E5_DIR, "e5_changed_val_hcx.html")'),
             ('parts = [f"<h2>E5 — 해상도 평균으로 clean4 답이 바뀐 검증 문항 {len(ch)}개</h2>"]', 'parts = [f"<h2>E5 + HCX — 해상도 평균으로 clean4 답이 바뀐 검증 문항 {len(ch)}개</h2>"]'),
             ('    n1, D1 = RES["clean4_qg_hcx"]\n',
              '    n1, D1 = RES["clean4_qg_hcx"]\n'
              '    N_INC = None                                          # HCX 몫: 같은 고해상도 결과에서 HCX만 뺀 앙상블 대비\n'
              '    if HCX_RUN in OVER1 and len(OVER1) > 1:\n'
              '        A1x = build({k: v for k, v in OVER1.items() if k != HCX_RUN})\n'
              '        for nm in ("clean4_qg_hcx", VOTE["name"]):\n'
              '            f, b, n, D = paired(A1x[nm]["val"], A1[nm]["val"], Y_VAL)\n'
              '            if nm == "clean4_qg_hcx": N_INC = n\n'
              '            say(f"  (HCX 몫) {nm}: HCX를 뺀 고해상도 대비 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}")\n'),
             ('        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 판정은 clean4 후보 한 줄로만 합니다")\n',
              '        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 판정은 clean4 후보 한 줄로만 합니다")\n'
              '        _inc_line(E5_DIR, "submission_clean4_qg_hcx__e5_hires.csv", "submission_clean4_qg_hcx__e5_hires_hcx.csv", N_INC, "E5")\n')]:
    _ev5 = rep(_ev5, o, n)
assert '"""' not in _ev9 + _ev5 and "'''" not in _ev9 + _ev5          # 들여쓰기로 감싸도 안전한지 (여러 줄 문자열 없음)

INC_HELP = r'''
def _inc_line(d, f_old, f_new, n_inc, tag):
    # E9·E5(HCX 없음) 후보를 이미 채택했을 때 쓰는 증분 기준을 출력
    po, pn = os.path.join(d, f_old), os.path.join(d, f_new)
    if n_inc is None or not (os.path.exists(po) and os.path.exists(pn)):
        say(f"  ({tag} 노트북의 HCX 없는 clean4 후보가 아직 없음 → 증분 기준은 그 후보가 생긴 뒤 이 셀을 다시 실행하면 출력)"); return
    rd = lambda p: pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS).answer.astype(str).str.strip().str.lower().to_numpy()
    k = max(0, INC_NET_MIN - n_inc)
    say(f"  ◀ {tag}(HCX 없음) clean4 후보를 이미 채택했다면: 이 clean4 후보가 그 후보보다 공개 {k}문항 이상 높을 때만 교체 "
        f"(HCX 몫 검증 {n_inc:+d} + 공개 ≥ +{INC_NET_MIN}, 공개 ≥ 0) | 두 후보의 다른 답 {int((rd(po) != rd(pn)).sum())}개")
'''

EVAL9 = ('# ── [TTA 판정] E9 판정 코드 그대로: 멤버 = E9 노트북의 4개 + HCX. 결과 이름에 _hcx (E9 노트북 결과는 건드리지 않음) ──\n'
         + INC_HELP.strip("\n") + "\n"
         + 'if not DO_TTA:\n    print("DO_TTA=False → TTA 판정 건너뜀"); V9, MEM9 = {"lines": [], "cands": []}, []\nelse:\n'
         + '    MEMBERS_TTA = E9_OTHERS + [HCX_RUN]\n    E9_INFO, INFO_PATH = HCX9_INFO, INFO9_PATH\n'
         + indent(_ev9) + "\n    V9, MEM9 = V, MEM_ROWS\n")

EVAL5 = ('# ── [고해상도 판정] E5 판정 코드 그대로: 멤버 = E5 노트북의 Q35·Q38 + HCX. 결과 이름에 _hcx (E5 노트북 결과는 건드리지 않음) ──\n'
         + INC_HELP.strip("\n") + "\n"
         + 'if not DO_HIRES:\n    print("DO_HIRES=False → 고해상도 판정 건너뜀"); V5, MEM5 = {"lines": [], "cands": []}, []\nelse:\n'
         + '    _E5REC = _read_json(os.path.join(E5_DIR, "e5_info.json"))      # E5 노트북 기록을 다시 읽음 (그사이 진행됐을 수 있음)\n'
         + '    def _skipped(run):\n'
         + '        if run == HCX_RUN: return _skipped_hcx()\n'
         + '        return set((_E5REC.get("infeasible") or {}).get(run, []))\n'
         + '    E5_INFO, INFO_PATH = HCX5_INFO, INFO5_PATH\n'
         + '    if not set(HR_PRESETS) - _skipped_hcx():\n'
         + '        print("HCX는 모든 고해상도를 건너뜀:", HCX5_INFO.get("infeasible_why")); V5 = {"lines": ["판정: HCX 고해상도 없음 — E5 노트북 판정을 그대로 쓰세요"], "cands": []}; MEM5 = []\n'
         + '        print(V5["lines"][0])\n'
         + '    else:\n'
         + '        MEMBERS_HR = E5_OTHERS + [HCX_RUN]\n'
         + indent(_ev5, 8) + "\n        V5, MEM5 = V, MEM_ROWS\n")

SUMMARY = r'''
print(f"E9·E5 + HCX | {HCX_RUN} | 채점 대상 검증 {len(VAL_G)} / test {len(TEST_G)} | {GPU_NAME} | TTA {DO_TTA} · 고해상도 {DO_HIRES}")
if HCX5_INFO.get("infeasible"): print("  HCX가 건너뛴 고해상도:", HCX5_INFO.get("infeasible_why"))
for title, on, mem, v in (("[TTA: E9 4멤버 + HCX]", DO_TTA, MEM9, V9), ("[고해상도: E5 Q35·Q38 + HCX]", DO_HIRES, MEM5, V5)):
    if not on: continue
    print(title)
    for r in mem: print("  ", r)
    for l in v["lines"]: print(l)
'''

INTRO = r'''# E9·E5 + HCX · 애매한 문항에서 HCX도 TTA·고해상도로 다시 채점 (학습 없음)

**왜**: HCX는 clean4에서 가중치 2(전체의 1/3)로 가장 큰 멤버인데, transformers 개발 버전이 있어야 올라가서 E9(TTA)·E5(고해상도) 노트북에서는 원래 확률을 그대로 썼습니다. HCX는 비전 인코더가 Qwen2.5-VL이라 해상도를 픽셀 수로 정하므로 고해상도도 됩니다.

**하는 일** (HCX 학습 노트북과 같은 환경·프롬프트·채점 함수)
- E9·E5와 **같은 애매한 문항**(기준 앙상블 3개 중 하나라도 1·2위 확률 차 < 0.8)만 HCX를 기존 어댑터 그대로 채점합니다.
  ① 보기 순서 4가지 (LoRA 켬) → ② 검증 up2.0 재현 확인 + up3.0·up4.0 → ③ 보기 순서 4가지 (LoRA 끔, E9의 H2용). 결과 파일 형식이 E9·E5와 같습니다.
- 판정은 **E9·E5 판정 코드를 그대로** 씁니다. 멤버 목록만 "E9·E5 노트북이 채점한 멤버 + HCX"로 바꿔 같은 레시피로 앙상블을 다시 만들고, 기준과 같은 문항끼리 비교합니다. HCX만 뺀 경우와의 차이(HCX 몫)도 따로 셉니다.
- E9·E5 노트북의 결과·기록 파일은 읽기만 합니다. 이 노트북의 제출 후보는 이름 끝에 `_hcx`가 붙습니다.

**사전 기준 (결과를 보기 전에 고정)**
- 기준(지금 제출 파일) 대비: E9·E5와 같음 — 검증 순이득 + clean4 후보 공개 순이득 ≥ +3이고 공개 순이득 ≥ 0 → 채택. 검증 순이득 ≤ −3이면 불채택. 바뀐 답 10개 미만이면 검증만으로 판단하지 않음.
- E9·E5(HCX 없음) 후보를 **이미 채택했다면**: HCX 몫 검증 순이득 + 그 후보 대비 공개 순이득 ≥ +2이고 공개 ≥ 0일 때만 `_hcx` 후보로 바꿈. 노트북이 필요한 공개 문항 수를 출력합니다.
- E9 판정이 불채택(검증 순이득 ≤ −3)이면 `DO_TTA=False`, E5가 불채택이면 `DO_HIRES=False`로 두고 돌리세요.

**실행**: **새 세션**에서 위에서부터 끝까지 (2번 셀이 transformers 개발 버전을 설치). 끊기면 다시 실행하면 남은 채점만 합니다. up4.0은 HCX 기준 문항당 이미지 토큰이 약 1만 3천~1만 6천 개라 배치 1로 돕니다. 메모리가 모자라거나 해상도가 적용되지 않으면 그 해상도만 건너뛰고 기록합니다.
E9·E5 노트북이 아직 돌고 있어도 됩니다. HCX 채점이 먼저 끝나면 판정은 "대기"로 나오고, E9·E5 채점이 끝난 뒤 다시 위에서부터 실행하면 채점은 건너뛰고(모델도 안 올림) 판정만 합니다.
**런타임**: G4 권장 (A100 80GB도 가능, up4.0은 메모리 부족이면 자동으로 건너뜀).'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 라이브러리 설치 (HCX 학습 때와 같은 버전 — transformers 개발 버전)"), ("code", H(4)),
    ("markdown", "## 3. Drive 연결 → HCX 멤버 확인 → 데이터 준비"), ("code", DATA),
    ("markdown", "## 4. 공통 import · CSV 로드"), ("code", H(9)),
    ("markdown", "## 5. 고정 검증셋 분할"), ("code", SPLIT),
    ("markdown", "## 6. 프롬프트 (HCX 학습 때와 동일)"), ("code", H(13)),
    ("markdown", "## 7. 기준 앙상블 → 애매한 문항 고르기 (GPU 불필요, E9·E5와 같은 문항)"), ("code", GATE),
    ("markdown", "## 8. 채점 함수 (HCX 학습 노트북과 동일, 증강만 제거)"), ("code", SCORING),
    ("markdown", "## 9. HCX 로드 함수 (bf16 베이스 + 기존 어댑터, merge 없음 · 템플릿 처리는 HCX 학습 노트북과 같음)"), ("code", LOADER),
    ("markdown", "## 10. HCX 채점: ① TTA(LoRA 켬) → ② 고해상도 → ③ TTA(LoRA 끔)\n사전 점검(어댑터 켬/끔 차이, letter_mass, 이미지 토큰 수, 기존 답과 같은지)을 통과해야 채점합니다."), ("code", RUNCELL),
    ("markdown", "## 11. [TTA 판정] E9 4멤버 + HCX → 제출 후보 (`*__e9_tta4_hcx.csv`)\n멤버별 `순서 그대로 재현(%)`이 85% 이상이어야 정상입니다."), ("code", EVAL9),
    ("markdown", "## 12. [고해상도 판정] E5 Q35·Q38 + HCX → 제출 후보 (`*__e5_hires_hcx.csv`)\n멤버별 `up2.0 재현(%)` 85% 이상, 해상도별 letter_mass 0.9 이상이어야 정상입니다. 답이 바뀐 검증 문항은 `e5_changed_val_hcx.html`에 이미지와 함께 저장됩니다."), ("code", EVAL5),
    ("markdown", "## 13. 결과 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    c = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]; nb.cells.append(c)
for c in nb.cells:
    if c.cell_type == "code":
        _src = "\n".join((l[: len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith("!") else l for l in c.source.split("\n"))
        try: compile(_src, "<cell>", "exec")
        except SyntaxError as e: raise SystemExit(f"문법 오류: {e}\n{c.source[:200]}")
nbformat.validate(nb); nbformat.write(nb, OUT_NB)
print("저장:", OUT_NB, "| 셀", len(nb.cells))
