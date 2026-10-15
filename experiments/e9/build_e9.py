# -*- coding: utf-8 -*-
"""E9 노트북 빌더: 기준 앙상블이 애매한 문항만 멤버별 TTA4(보기 순서 4가지)로 다시 채점 → 앙상블 비교 → 제출 후보.
같은 채점에서 LoRA를 끈 원본 모델 점수도 저장해 LoRA-off 결합 규칙(H2)을 따로 시험합니다.
설치·import·분할·프롬프트·채점 함수는 stage2 노트북 셀을 그대로 가져옵니다(학습·증강 부분만 제거).
"""
import json, sys, uuid
import nbformat

SRC_NB = "/home/user/vqa/src_stage2.ipynb"
OUT_NB = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e9/out/e9_tta_gated.ipynb"
src_nb = json.load(open(SRC_NB, encoding="utf-8"))
def S(i):
    s = src_nb["cells"][i]["source"]; return "".join(s) if isinstance(s, list) else s
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

CFG = r'''
MEMBERS_TTA = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev"]
#   TTA로 다시 채점할 멤버 (같은 베이스인 Gemma 둘은 연달아 → 베이스를 한 번만 받음). HCX는 설치가 달라 이 노트북으로는 안 됨 → 원래 확률 그대로
PRESET     = "up2.0"        # 채점 해상도 (멤버 예측 파일과 같음)
TAU_GATE   = 0.8            # 기준 앙상블 3개 중 하나라도 1·2위 확률 차 < 0.8인 문항만 채점
SCORE_BASE = True           # LoRA를 끈 원본 모델 점수도 저장 (H2 시험용). 멤버마다 LoRA 켠 채점(val → test)이 먼저 끝나고 그다음에 함. False면 채점량 절반
RUN_TEST   = True           # test 불확실 문항도 채점 (제출 후보를 만들려면 필요)
RUN_NAME   = "e9_tta"       # 결과 폴더: vqa_runs/e9_tta/<멤버>/  (기존 멤버 폴더는 읽기만 함)
MODEL_ID   = "Qwen/Qwen3.5-27B"   # 자리표시(2번 셀 설치 조건용). 멤버별 베이스 모델은 어댑터 기록에서 자동으로 읽음
DELETE_HF_CACHE = True      # 다른 베이스로 넘어갈 때 이전 베이스 파일을 지워 디스크 확보

# ── 기준 앙상블 레시피 (E8 노트북과 같음) ──
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

# ── 사전 기준 (결과를 보기 전에 고정) ──
NET_MIN = 3        # H1(TTA4): 검증 순이득 + 공개 순이득(clean4 후보) ≥ +3 이고 공개 순이득 ≥ 0 → 채택
BO = dict(gap=0.15, pmin=0.60, agree=0.75, delta=0.12, w=0.5)   # H2(LoRA-off 결합): 미리 정한 기준값 그대로
BO_NET_MIN = 2     # H2: H1 대비 검증 + 공개 순이득 ≥ +2 → 추가 채택 (test에서 H1과 다른 답이 5개 미만이면 시험하지 않음)

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
BATCH_SIZE   = "auto"
SEED, VAL_SIZE = 42, 1000       # 검증 분할용 — 팀 공통값
SKIP_MISSING_IMAGES = False
HF_TOKEN     = ""               # 비워 두면 Colab 보안 비밀 'HF_TOKEN'을 찾아봄 (Gemma 라이선스)

RES_PRESETS = {
    "baseline384": (384 * 384, 384 * 384),
    "mid":         (256 * 28 * 28, 512 * 28 * 28),
    "native":      (256 * 28 * 28, 1280 * 28 * 28),
    "up1.5":       (1_550_000, 2_100_000),
    "up2.0":       (2_700_000, 3_400_000),
}
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}
RES_PRESET = PRESET
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
'''

DATA = r'''
import os, re, json, hashlib, subprocess, shutil
try:
    from google.colab import drive
    drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
# ── 멤버 확인 (읽기만 함): 어댑터·베이스 모델·기존 예측 파일. 이름이 틀렸으면 압축을 풀기 전에 멈춤 ──
MEMBER_INFO = {}
for run in MEMBERS_TTA:
    d = os.path.join(OUT_DIR, run); assert os.path.isdir(d), f"멤버 폴더가 없습니다: {d} → 1번 셀 MEMBERS_TTA를 확인하세요"
    fs = sorted(os.listdir(d))
    lg = json.load(open(os.path.join(d, "train_log.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "train_log.json")) else {}
    cfg = lg.get("config") or {}
    assert not cfg.get("TRAIN_ON_FULL"), f"{run}은 전체 데이터로 학습한 실행입니다 → 이 노트북은 검증셋을 뺀 멤버만 씁니다"
    rv = {int(m.group(1)): f for f in fs if (m := re.match(rf"^preds_final_ep(\d+)__{re.escape(PRESET)}__1000\.csv$", f))}
    ea = {int(k): v for k, v in (lg.get("epoch_acc") or {}).items()}
    ep = next(iter(rv)) if len(rv) == 1 else (max(ea, key=lambda e: (ea[e], e)) if ea else (max(rv) if rv else None))
    assert ep in rv, f"{run}: {PRESET} 검증 예측 파일이 없습니다 (폴더 내용 일부: {fs[:8]})"
    rt = next((f for f in fs if re.match(rf"^preds_final_ep{ep}_test__{re.escape(PRESET)}__6714\.csv$", f)), None)
    ad = os.path.join(d, f"adapter_ep{ep}")
    assert os.path.isfile(os.path.join(ad, "adapter_model.safetensors")), f"{run}: 어댑터가 없습니다: {ad}"
    acfg = json.load(open(os.path.join(ad, "adapter_config.json"), encoding="utf-8"))
    mid = str(acfg.get("base_model_name_or_path") or "")
    if not mid or mid.startswith("/"): mid = str(cfg.get("MODEL_ID") or "")
    assert mid and not mid.startswith("/"), f"{run}: 베이스 모델 이름을 찾지 못했습니다"
    assert not ("hyperclova" in mid.lower() or "hcx" in run.lower()), f"{run}: HCX는 이 노트북으로 채점할 수 없습니다 → MEMBERS_TTA에서 빼세요"
    MEMBER_INFO[run] = dict(dir=d, ep=ep, adapter_dir=ad, model_id=mid, ref_val=os.path.join(d, rv[ep]), ref_test=os.path.join(d, rt) if rt else None,
                            md5=hashlib.md5(open(os.path.join(ad, "adapter_model.safetensors"), "rb").read(1 << 20)).hexdigest()[:8])
    print(f"  {run}: 베이스 {mid} | adapter_ep{ep} | 검증 {rv[ep]} | test {rt or '없음'}")
    if RUN_TEST: assert rt, f"{run}: test 예측 파일이 없습니다 (RUN_TEST=True)"
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
E9_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(E9_DIR, exist_ok=True)
INFO_PATH = os.path.join(E9_DIR, "e9_info.json")
E9_INFO = json.load(open(INFO_PATH, encoding="utf-8")) if os.path.exists(INFO_PATH) else {}
for run, inf in MEMBER_INFO.items():
    old = (E9_INFO.get("members") or {}).get(run)
    if old and old.get("md5") != inf["md5"]:
        raise RuntimeError(f"{E9_DIR}/{run}에는 다른 어댑터로 만든 결과가 있습니다 → 1번 셀 RUN_NAME을 바꾸세요")
print("데이터 OK:", DATA_DIR, "| 결과 폴더(새로 씀):", E9_DIR)
'''

SPLIT = rep(S(10),
    'if split_md5 == EXPECTED_SPLIT_MD5:\n    print("✅ 팀 공통 분할과 일치")\nelse:\n'
    '    print("⚠️⚠️ 검증 분할이 기준과 다릅니다 → 팀원 간 점수 비교 불가. train.csv 원본/SEED/VAL_SIZE를 확인하세요.")\n'
    'val_df[["id"]].to_csv(os.path.join(OUT_DIR, "valid_ids.csv"), index=False)',
    'assert split_md5 == EXPECTED_SPLIT_MD5, "⚠️ 검증 분할이 기준과 다릅니다 → 기존 멤버 예측과 비교할 수 없습니다"\n'
    'print("✅ 팀 공통 분할과 일치")')

GATE = r'''
# ── 기준 앙상블 3개를 멤버 예측 파일로 다시 만들고, 애매한 문항(1·2위 확률 차 < TAU_GATE)을 고릅니다 (GPU 불필요) ──
_CACHE = {}
def _probs(d): return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
def _load_preds(path, ids):
    d = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in ids if i not in d.index]
    assert not miss, f"{os.path.basename(path)}에 없는 문항 {len(miss)}개 (예: {miss[:3]})"
    return d.loc[ids]
def _read_log(run):
    p = os.path.join(OUT_DIR, run, "train_log.json")
    try: return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    except Exception: return {}
def member_file(run, preset, split):
    d = os.path.join(OUT_DIR, run)
    if not os.path.isdir(d): return None, "폴더 없음"
    cv, ct = {}, {}
    for f in os.listdir(d):
        m = re.match(rf"^preds_final_ep(\d+)__{re.escape(preset)}__1000\.csv$", f)
        if m: cv[int(m.group(1))] = f; continue
        m = re.match(rf"^preds_final_ep(\d+)_test__{re.escape(preset)}__(\d+)\.csv$", f)
        if m and int(m.group(2)) == len(test_df): ct[int(m.group(1))] = f
    base_ = cv or ct
    if not base_: return None, "예측 파일 없음"
    ea = {int(k): v for k, v in (_read_log(run).get("epoch_acc") or {}).items()}
    ep = max(ea, key=lambda e: (ea[e], e)) if ea else None
    ep = ep if ep in base_ else max(base_)
    pool = cv if split == "val" else ct
    return (os.path.join(d, pool[ep]), f"ep{ep}") if ep in pool else (None, f"ep{ep} {split} 파일 없음")
def load(run, preset, split):
    key = (run, preset, split)
    if key not in _CACHE:
        f, why = member_file(run, preset, split)
        if f is None: raise FileNotFoundError(f"{run}@{preset} {split}: {why}")
        _CACHE[key] = _probs(_load_preds(f, VAL_IDS if split == "val" else TEST_IDS))
    return _CACHE[key]
T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))          # Stage 3와 같은 온도 격자
def temp_scale(p, T):
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def fit_T(p, y):
    i = np.arange(len(y)); return float(T70[int(np.argmin([-np.mean(np.log(temp_scale(p, T)[i, y])) for T in T70]))])
def ens_probs(r, split, over=None):
    # over: {멤버: (검증 확률, test 확률)} → 그 멤버 확률을 바꿔서 같은 레시피로 앙상블 (온도는 바뀐 검증 확률로 다시 맞춤)
    w = np.asarray(r["weights"], float); w = w / w.sum(); out = 0.0
    for wi, (run, pre) in zip(w, r["members"]):
        Pv, Pt = over[run] if (over and run in over) else (load(run, pre, "val"), load(run, pre, "test"))
        P = Pv if split == "val" else Pt
        if r["cal"]: P = temp_scale(P, fit_T(Pv, Y_VAL))
        out = out + wi * P
    return out
def vote_of(arrs):
    A = np.stack(arrs, 1); cnt = np.stack([(A == k).sum(1) for k in range(4)], 1)
    return cnt.argmax(1)             # 동률이면 알파벳 앞 글자 = pandas mode(axis=1)[0]
def file_answers(fname):
    p = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR, fname)
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
def margin(P):
    s = np.sort(P, axis=1); return s[:, -1] - s[:, -2]
def paired(a_old, a_new, y):
    c0, c1 = a_old == y, a_new == y
    f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum()); return f, b, f - b, int((a_old != a_new).sum())
def score_of(count): return math.floor(count / N_PUBLIC * 1e5) / 1e5
def count_of(score): return int(round(score * N_PUBLIC))

VAL_IDS, TEST_IDS = val_df.id.tolist(), test_df.id.tolist()
Y_VAL = val_df.answer.map(LETTERS.index).to_numpy()
BASE = {name: {s: ens_probs(r, s) for s in ("val", "test")} for name, r in RECIPES.items()}
A0 = {name: {s: BASE[name][s].argmax(1) for s in ("val", "test")} for name in RECIPES}
A0[VOTE["name"]] = {s: vote_of([A0[v][s] for v in VOTE["voters"]]) for s in ("val", "test")}
GATE_V = np.any([margin(BASE[n]["val"]) < TAU_GATE for n in RECIPES], axis=0)
GATE_T = np.any([margin(BASE[n]["test"]) < TAU_GATE for n in RECIPES], axis=0)
VAL_G, TEST_G = val_df[GATE_V].reset_index(drop=True), test_df[GATE_T].reset_index(drop=True)
_fa = file_answers(RECIPES["clean4_qg_hcx"]["file"])
print(f"기준 clean4 검증 {(A0['clean4_qg_hcx']['val'] == Y_VAL).mean() * 100:.2f}% | 다수결(레시피) 검증 {(A0[VOTE['name']]['val'] == Y_VAL).mean() * 100:.2f}%"
      + ("" if _fa is None else f" | clean4 test 재현 {int((_fa == A0['clean4_qg_hcx']['test']).sum()):,}/{len(TEST_IDS):,}"))
print(f"채점 대상(세 앙상블 중 하나라도 1·2위 확률 차 < {TAU_GATE}): 검증 {len(VAL_G)}문항 (그중 clean4 오답 {int((GATE_V & (A0['clean4_qg_hcx']['val'] != Y_VAL)).sum())}개 / 전체 오답 {int((A0['clean4_qg_hcx']['val'] != Y_VAL).sum())}개) | test {len(TEST_G)}문항")
print(f"→ 멤버 {len(MEMBERS_TTA)}개 × (검증 {len(VAL_G)} + test {len(TEST_G) if RUN_TEST else 0}) × 순서 4가지" + (" × LoRA 켬/끔" if SCORE_BASE else ""))
# 예전에 해 본 TTA 기록 (stage2 결과 로그에서 같은 실행·해상도의 TTA 없음/있음 비교)
_lp = os.path.join(OUT_DIR, "results_log_s2.csv")
if os.path.exists(_lp):
    L = pd.read_csv(_lp)
    if {"run", "res", "tta", "n", "acc"} <= set(L.columns):
        L = L.drop_duplicates(["run", "res", "tta", "n"], keep="last"); t1 = L[L.tta == 1]
        m = t1.merge(L[L.tta == 0][["run", "res", "n", "acc"]], on=["run", "res", "n"], suffixes=("_tta", "_기본"))
        if len(m):
            m["차이(%p)"] = (m.acc_tta - m.acc_기본).round(2)
            print("\n예전 TTA 기록 (검증 전체 정확도):"); print(m[["run", "res", "n", "acc_기본", "acc_tta", "차이(%p)"]].to_string(index=False))
        else: print("\n예전 TTA 기록: 없음")
'''

_s16 = S(16)
_aug_start = _s16.index("from PIL import ImageEnhance"); _aug_end = _s16.index("def encode_batch")
SCORING = _s16
for old, new in [("set_resolution(TRAIN_PRESET)", "# (해상도는 멤버 모델을 올린 뒤 set_resolution(PRESET)으로 정함)"),
                 (_s16[_aug_start:_aug_end], ""),
                 ("    if _AUG_ON and IMG_AUG: images = [augment_image(im) for im in images]\n", ""),
                 ("\"run\": f\"{RUN_NAME}{'_dry' if DRY_RUN else ''}/{tag}\"", "\"run\": f\"{RUN_NAME}/{tag}\"")]:
    SCORING = rep(SCORING, old, new)

LOADER = r'''
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, load_peft_weights, get_peft_model_state_dict
import transformers, peft, gc, contextlib
print("transformers", transformers.__version__, "| peft", peft.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "GPU 런타임이 아닙니다 (런타임 → 런타임 유형 변경)"
DEVICE = "cuda"
GPU_NAME = torch.cuda.get_device_name(0); _cap = torch.cuda.get_device_capability(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
assert _cap[0] >= 8, f"{GPU_NAME}은 bf16을 지원하지 않습니다 → G4 또는 A100 80GB를 쓰세요"
DTYPE = torch.bfloat16
BS = (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2) if BATCH_SIZE == "auto" else int(BATCH_SIZE)

def _free():
    for _v in ("model", "base", "enc", "logits", "out"):
        if _v in globals(): del globals()[_v]
    gc.collect(); torch.cuda.empty_cache()

def _drop_hf_cache(model_id):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True); print(f"  디스크 확보: {model_id} 캐시 삭제")
    except Exception as e: print("  (캐시 삭제 생략:", e, ")")

def _norm_key(k):
    k = k.replace("base_model.model.", "", 1); i = k.rfind("language_model.")
    return k[i + len("language_model."):] if i >= 0 else k
def check_adapter_weights(model, adapter_dir):
    # E1과 같은 점검: 어댑터 파일의 LoRA 텐서가 전부 값 그대로 모델에 들어갔는지 (PEFT는 이름이 안 맞는 키를 조용히 건너뛸 수 있음)
    fw = load_peft_weights(adapter_dir, device="cpu"); mw = {k: v.detach() for k, v in get_peft_model_state_dict(model).items()}
    if set(fw) != set(mw): fw = {_norm_key(k): v for k, v in fw.items()}; mw = {_norm_key(k): v for k, v in mw.items()}
    assert set(fw) == set(mw), f"어댑터 파일과 모델의 LoRA 키가 다릅니다 (파일에만 {len(set(fw) - set(mw))}개, 모델에만 {len(set(mw) - set(fw))}개)"
    worst = max((v.to(mw[k].dtype).float() - mw[k].float().cpu()).abs().max().item() for k, v in fw.items())
    assert worst == 0.0, f"어댑터 파일과 모델의 LoRA 값이 다릅니다 (최대 차이 {worst:.3g})"
    assert any("lora_B" in k and v.abs().max().item() > 0 for k, v in fw.items()), "lora_B가 전부 0입니다 → 학습되지 않은 어댑터"
    return len(fw)

def load_member(run):
    # 멤버 하나를 bf16 베이스 + 기존 어댑터(merge 없음)로 올리고, 채점 함수가 쓰는 전역값을 그 멤버에 맞춤
    global processor, tok, FAMILY, IMAGE_TOKEN_ID, TEMPLATE_KW, USE_SYSTEM_ROLE, base, model, LETTER_IDS, MODEL_ID
    inf = MEMBER_INFO[run]; MODEL_ID = inf["model_id"]
    _sz = re.search(r"(?<![\d.])(\d+)B", MODEL_ID, flags=re.I)
    need = int(_sz.group(1)) * 2.05 + 6 if _sz else 0.0
    if VRAM_GB < need: raise RuntimeError(f"bf16 {MODEL_ID}에는 약 {need:.0f}GB가 필요한데 {GPU_NAME}은 {VRAM_GB:.0f}GB입니다 → G4 또는 A100 80GB")
    processor = AutoProcessor.from_pretrained(MODEL_ID); processor.tokenizer.padding_side = "left"; tok = processor.tokenizer
    FAMILY = "gemma4" if type(processor).__name__.startswith("Gemma4") else "qwen"
    IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
    if IMAGE_TOKEN_ID is None: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
    _tmpl = processor.chat_template if isinstance(processor.chat_template, str) else json.dumps(processor.chat_template)
    TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
    try:
        processor.apply_chat_template(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
    except Exception:
        USE_SYSTEM_ROLE = False
    t0 = time.time()
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa", device_map={"": 0})
    base.config.use_cache = False
    assert not [n for n, m in base.named_modules() if any(k in type(m).__name__ for k in ("Linear4bit", "Linear8bit"))], "양자화된 층이 있습니다 (bf16이어야 함)"
    model = PeftModel.from_pretrained(base, inf["adapter_dir"], is_trainable=False).eval()
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    n_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and any(k in n for k in ("visual", "vision_tower", "vision_model")))
    assert n_lora > 0 and n_vis == 0, f"LoRA 층 {n_lora}개, 비전 타워 {n_vis}개 → 어댑터 부착 이상"
    nk = check_adapter_weights(model, inf["adapter_dir"])
    if "qwen3_5" in str(getattr(base.config, "model_type", "")):
        import importlib.util
        if importlib.util.find_spec("fla") is None: print("  ⚠️⚠️ flash-linear-attention이 없습니다 → 2번 셀 실행 후 '세션 다시 시작'")
    LETTER_IDS = []
    for L in LETTERS:
        ids = []
        for s in (L, L.upper(), " " + L, " " + L.upper()):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids: ids.append(t[0])
        assert ids and len(tok.encode(L, add_special_tokens=False)) == 1, f"'{L}' 가 단일 토큰이 아닙니다"
        LETTER_IDS.append(ids)
    set_resolution(PRESET)
    print(f"  모델 로드 {time.time() - t0:.0f}s | {type(base).__name__} | 계열 {FAMILY} | LoRA 키 {nk}개 값까지 일치 | 템플릿 인자 {TEMPLATE_KW} | system 역할 {USE_SYSTEM_ROLE}")
print(f"GPU: {GPU_NAME} ({VRAM_GB:.0f}GB) | bf16 | 평가 배치 {BS}")
'''

RUNCELL = r'''
# ── 멤버마다: 모델 로드 → 사전 점검 → 애매한 문항을 보기 순서 4가지로 채점 (LoRA 켬 → 끔). 끊겨도 다시 실행하면 남은 것만 ──
PERM_COLS = ["id", "lora", "perm", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens"]
SPLIT_DF = {"val": VAL_G, "test": TEST_G}
def _perm_path(run, split): return os.path.join(E9_DIR, run, f"perm_{split}__{PRESET}.csv")
def _done(run, split):
    p = _perm_path(run, split)
    if not os.path.exists(p): return set()
    d = pd.read_csv(p); return set(zip(d.id, d.lora.astype(int), d.perm.astype(int)))
def _plan():
    return [(s, l) for l in ([1, 0] if SCORE_BASE else [1]) for s in (["val", "test"] if RUN_TEST else ["val"])]
def _remaining(run):
    dn = {s: _done(run, s) for s in {s for s, _ in _plan()}}
    return {(s, l): sum((i, l, k) not in dn[s] for i in SPLIT_DF[s].id for k in range(len(TTA_PERMS))) for s, l in _plan()}
def score_split(run, split, lora, save_every=10):
    path = _perm_path(run, split); os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _done(run, split); rows = SPLIT_DF[split].to_dict("records"); recs, n, nb, t0 = [], 0, 0, time.time()
    todo = [(k, [r for r in rows if (r["id"], lora, k) not in done]) for k in range(len(TTA_PERMS))]
    total = sum(len(rs) for _, rs in todo)
    print(f"  [{split} | LoRA {'켬' if lora else '끔'}] {total:,}건 남음", flush=True)
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

E9_INFO.setdefault("members", {})
for i, run in enumerate(MEMBERS_TTA):
    rem = _remaining(run)
    print(f"\n▶ {run} ({i + 1}/{len(MEMBERS_TTA)}) | 남은 채점: " + ", ".join(f"{s}·LoRA{'켬' if l else '끔'} {n:,}" for (s, l), n in rem.items()))
    if sum(rem.values()) == 0: print("  이미 끝났습니다 → 건너뜀"); continue
    _free(); load_member(run)
    # 사전 점검: 어댑터 켬/끔 차이, letter_mass, 이미지 토큰 수(기존 예측과 같은 해상도·프로세서인지)
    _chk = VAL_G.head(4).to_dict("records") or val_df.head(4).to_dict("records")
    p_on, m_on, nt_on = score_rows(_chk)
    with model.disable_adapter(): p_off, _, _ = score_rows(_chk)
    _d = float(np.abs(p_on - p_off).max())
    _ref = pd.read_csv(MEMBER_INFO[run]["ref_val"]).drop_duplicates("id", keep="last").set_index("id").reindex([r["id"] for r in _chk])
    print(f"  사전 점검: 어댑터 켬/끔 확률 최대 차이 {_d:.3f} | letter_mass {m_on.mean():.3f} | 이미지 토큰 {list(nt_on)}"
          + (f" (기존 {[int(t) for t in _ref.img_tokens]})" if "img_tokens" in _ref.columns else ""))
    assert _d > 1e-3, "어댑터를 켜고 꺼도 출력이 같습니다 → 어댑터가 적용되지 않았습니다. 위 출력을 확인하세요"
    assert m_on.mean() > 0.9, "letter_mass가 낮습니다 → 프롬프트·토큰 처리 문제. 위 출력을 확인하세요"
    if "img_tokens" in _ref.columns and not _ref.img_tokens.isna().any():
        assert [int(t) for t in _ref.img_tokens] == [int(t) for t in nt_on], "이미지 토큰 수가 기존 예측과 다릅니다 → 해상도·프로세서가 달라 비교할 수 없습니다"
    E9_INFO["members"][run] = {"md5": MEMBER_INFO[run]["md5"], "model_id": MODEL_ID, "adapter": MEMBER_INFO[run]["adapter_dir"], "precision": "bf16", "gpu": GPU_NAME,
                               "preset": PRESET, "tau_gate": TAU_GATE, "perms": [list(p) for p in TTA_PERMS]}
    json.dump(E9_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for s, l in _plan(): score_split(run, s, l)
    nxt = MEMBER_INFO[MEMBERS_TTA[i + 1]]["model_id"] if i + 1 < len(MEMBERS_TTA) else None
    if DELETE_HF_CACHE and nxt != MODEL_ID:
        _free(); _drop_hf_cache(MODEL_ID)
print("\n채점 끝 → 11번 셀에서 판정")
'''

EVAL = r'''
# ── 판정 (GPU 불필요): 멤버 확률을 애매한 문항에서만 TTA4 평균으로 바꿔 같은 레시피로 앙상블 → 같은 문항끼리 비교 ──
V = {"lines": [], "cands": []}
def say(s): print(s); V["lines"].append(s)
IDX = {"val": {i: n for n, i in enumerate(VAL_IDS)}, "test": {i: n for n, i in enumerate(TEST_IDS)}}
GIDS = {"val": VAL_G.id.tolist(), "test": TEST_G.id.tolist()}
def perm_stack(run, split, lora):
    # (문항 수, 순서 4, 보기 4) — 모든 애매한 문항·순서가 채점됐을 때만, 아니면 None
    p = _perm_path(run, split)
    if not os.path.exists(p): return None, None
    d = pd.read_csv(p); d = d[d.lora.astype(int) == lora].drop_duplicates(["id", "perm"], keep="last")
    if len(d) < len(GIDS[split]) * len(TTA_PERMS): return None, None
    d = d.set_index(["id", "perm"]); X = np.zeros((len(GIDS[split]), len(TTA_PERMS), 4))
    for k in range(len(TTA_PERMS)):
        X[:, k, :] = d.xs(k, level="perm").reindex(GIDS[split])[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
    if np.isnan(X).any(): return None, None                 # 애매한 문항 중 빠진 것이 있음 (TAU_GATE를 바꿨거나 채점 중단)
    return X, float(d.letter_mass.mean())
MEM_ROWS, OVER1, OVER2, health, BO_N = [], {}, {}, [], {"val": 0, "test": 0}
gv = np.array([IDX["val"][i] for i in GIDS["val"]], int); gt = np.array([IDX["test"][i] for i in GIDS["test"]], int)
for run in MEMBERS_TTA:
    pre = PRESET; Pv0, Pt0 = load(run, pre, "val"), load(run, pre, "test")
    Xv, mv = perm_stack(run, "val", 1); Xt, mt = perm_stack(run, "test", 1) if RUN_TEST else (None, None)
    if Xv is None or (RUN_TEST and Xt is None):
        MEM_ROWS.append({"멤버": run, "상태": "채점 미완료 → 원래 확률 사용"}); continue
    ref_v = Pv0[gv]; id_agree = float((Xv[:, 0, :].argmax(1) == ref_v.argmax(1)).mean() * 100)   # 순서 그대로(0번) 채점 = 기존 예측 재현 확인
    tta_v = Xv.mean(1); Pv1 = Pv0.copy(); Pv1[gv] = tta_v
    Pt1 = Pt0.copy()
    if RUN_TEST: Pt1[gt] = Xt.mean(1)
    yg = Y_VAL[gv]; f, b, n, D = paired(ref_v.argmax(1), tta_v.argmax(1), yg)
    row = {"멤버": run, "상태": "✅", "순서 그대로 재현(%)": round(id_agree, 1), "letter_mass": round(mv, 3),
           "멤버 단독(애매한 검증 문항)": f"고침 {f} / 새로 틀림 {b} / 순이득 {n:+d} (바뀐 답 {D})"}
    if id_agree < 85 or mv < 0.9:
        row["상태"] = "⚠️ 확인 필요"; health.append(f"{run}: 순서 그대로 재현 {id_agree:.1f}% / letter_mass {mv:.3f}")
    OVER1[run] = (Pv1, Pt1)
    # H2: LoRA를 끈 원본 모델 (LoRA-off 결합 규칙 — 원본이 다른 답을 고르고 네 조건을 모두 만족할 때만 0.5:0.5)
    Ov, _ = perm_stack(run, "val", 0) if SCORE_BASE else (None, None)
    Ot, _ = perm_stack(run, "test", 0) if (SCORE_BASE and RUN_TEST) else (None, None)
    if Ov is not None and (Ot is not None or not RUN_TEST):
        def _bo(P_ft, O):
            pb = O.mean(1); bt = pb.argmax(1); ft = P_ft.argmax(1); s = np.sort(P_ft, 1); gap = s[:, -1] - s[:, -2]
            agree = (O.argmax(2) == bt[:, None]).mean(1); ii = np.arange(len(P_ft))
            sel = (bt != ft) & (gap < BO["gap"]) & (pb[ii, bt] >= BO["pmin"]) & (agree >= BO["agree"]) & ((P_ft[ii, ft] - P_ft[ii, bt]) <= BO["delta"])
            out = P_ft.copy(); out[sel] = (1 - BO["w"]) * P_ft[sel] + BO["w"] * pb[sel]; return out, int(sel.sum())
        Pv2 = Pv1.copy(); Pv2[gv], nsv = _bo(tta_v, Ov)
        Pt2 = Pt1.copy(); nst = 0
        if RUN_TEST: Pt2[gt], nst = _bo(Xt.mean(1), Ot)
        OVER2[run] = (Pv2, Pt2); BO_N["val"] += nsv; BO_N["test"] += nst
        row["원본 결합 대상(검증/test)"] = f"{nsv}/{nst}"
    MEM_ROWS.append(row)
print(pd.DataFrame(MEM_ROWS).to_string(index=False))
COMPLETE = len(OVER1) == len(MEMBERS_TTA)
if not OVER1:
    say("TTA 채점이 끝난 멤버가 없습니다 → 10번 셀을 먼저 끝까지 실행하세요")
else:
    def build(over):
        P = {n: {s: ens_probs(r, s, over) for s in ("val", "test")} for n, r in RECIPES.items()}
        A = {n: {s: P[n][s].argmax(1) for s in ("val", "test")} for n in RECIPES}
        A[VOTE["name"]] = {s: vote_of([A[v][s] for v in VOTE["voters"]]) for s in ("val", "test")}
        return A
    A1 = build(OVER1)
    say(f"[H1: TTA4] 반영 멤버 {len(OVER1)}/{len(MEMBERS_TTA)}: {', '.join(OVER1)}" + ("" if COMPLETE else " (미완료 멤버는 원래 확률)"))
    RES = {}
    for nm in ("clean4_qg_hcx", VOTE["name"]):
        f, b, n, D = paired(A0[nm]["val"], A1[nm]["val"], Y_VAL); RES[nm] = (n, D)
        say(f"  {nm}: 검증 {(A0[nm]['val'] == Y_VAL).mean() * 100:.2f}% → {(A1[nm]['val'] == Y_VAL).mean() * 100:.2f}% | 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}"
            + ("  (바뀐 답 10개 미만 → 검증만으로는 판단 안 함)" if D < 10 else ""))
    n1, D1 = RES["clean4_qg_hcx"]
    H2_OK = SCORE_BASE and len(OVER2) == len(OVER1) == len(MEMBERS_TTA)
    if SCORE_BASE and not H2_OK and OVER1:
        say(f"[H2: + LoRA-off 결합] LoRA를 끈 채점이 끝난 멤버 {len(OVER2)}/{len(MEMBERS_TTA)} → 모두 끝나면 다시 실행 (H2 후보는 만들지 않음)")
    if H2_OK:
        A2 = build({**OVER1, **OVER2})
        f2, b2, n2, D2 = paired(A1["clean4_qg_hcx"]["val"], A2["clean4_qg_hcx"]["val"], Y_VAL)
        say(f"[H2: + LoRA-off 결합] 결합 대상 검증 {BO_N['val']} / test {BO_N['test']}문항(멤버 합) | clean4 검증 H1 대비 고침 {f2} / 새로 틀림 {b2} / 순이득 {n2:+d} (바뀐 답 {D2})")
    if health:
        say("판정: 확인 필요 — " + "; ".join(health) + " → 위 출력을 확인하세요")
    elif not COMPLETE:
        say("판정: 대기 — 모든 멤버의 TTA 채점이 끝나면 다시 실행")
    elif n1 <= -3:
        say(f"판정: 불채택 — H1 검증 순이득 {n1:+d} (≤ −3). 공개로 확인할 필요 없음")
    elif RUN_TEST:
        say(f"판정: 공개 확인 — 검증 순이득 {n1:+d}(바뀐 답 {D1}). 아래 clean4 후보의 공개 점수로 결정합니다")
        os.makedirs(E9_DIR, exist_ok=True)
        def _write(name, base_name, A_new, A_ref):
            # 기존 제출 파일에서, 이 실험 때문에 레시피 답이 바뀐 문항만 고침 (파일에 다른 차이가 있어도 이 실험의 효과만 반영)
            fa = file_answers(RECIPES[base_name]["file"] if base_name in RECIPES else VOTE["file"])
            base_ans = fa if fa is not None else A_ref
            ch = A_new != A_ref; cand = np.where(ch, A_new, base_ans)
            path = os.path.join(E9_DIR, f"submission_{name}.csv"); sub_df[["id"]].assign(answer=[LETTERS[i] for i in cand]).to_csv(path, index=False)
            return os.path.basename(path), int((cand != base_ans).sum())
        need = max(0, NET_MIN - n1)
        for nm in ("clean4_qg_hcx", VOTE["name"]):
            fn, Dn = _write(f"{nm}__e9_tta4", nm, A1[nm]["test"], A0[nm]["test"]); b0 = count_of(PUBLIC[nm]); V["cands"].append((fn, Dn))
            line = f"  제출 후보: {fn} | 기존 {nm}({PUBLIC[nm]:.5f} = {b0:,}문항)과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개)"
            if nm == "clean4_qg_hcx":
                line += f" | ◀ 판정용: 공개 {score_of(b0 + need):.5f}({b0 + need:,}문항) 이상이면 H1 채택 (검증 {n1:+d} + 공개 ≥ +{NET_MIN}, 공개 ≥ 0)"
            say(line)
        if H2_OK and BO_N["test"] > 0:
            ch2 = {nm: int((A2[nm]["test"] != A1[nm]["test"]).sum()) for nm in ("clean4_qg_hcx", VOTE["name"])}
            if ch2["clean4_qg_hcx"] < 5:
                say(f"  H2: test에서 H1과 다른 답 {ch2['clean4_qg_hcx']}개 (< 5) → 시험할 가치가 없어 후보를 만들지 않음")
            else:
                need2 = max(0, BO_NET_MIN - n2)
                for nm in ("clean4_qg_hcx", VOTE["name"]):
                    fn, Dn = _write(f"{nm}__e9_tta4_baseoff", nm, A2[nm]["test"], A0[nm]["test"]); V["cands"].append((fn, Dn))
                    say(f"  제출 후보(H2): {fn} | 기존 {nm}과 다른 답 {Dn}개 | H1 후보와 다른 답 {ch2[nm]}개"
                        + (f" | ◀ 판정용: H1 clean4 후보보다 공개 {need2}문항 이상 높으면 H2 추가 채택" if nm == "clean4_qg_hcx" else ""))
        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 두 파일은 같은 TTA 결과로 만들었으므로 판정은 clean4 후보 한 줄로만 합니다")
    else:
        say(f"판정: 검증만 확인 — 검증 순이득 {n1:+d}(바뀐 답 {D1}). 제출 후보를 만들려면 1번 셀 RUN_TEST=True로 10번 셀부터 다시 실행")
    E9_INFO.update({"verdict_lines": V["lines"], "cands": V["cands"]})
    json.dump(E9_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

SUMMARY = r'''
print(f"E9 | 불확실 문항 TTA4 | {PRESET} | 채점 대상 검증 {len(VAL_G)} / test {len(TEST_G)} | GPU {globals().get('GPU_NAME', '-')}")
for r in MEM_ROWS: print("  ", {k: v for k, v in r.items()})
for l in V["lines"]: print(l)
'''

INTRO = r'''# E9 · 애매한 문항만 보기 순서 TTA4로 다시 채점 (학습 없음)

**하는 일**
- 기준 앙상블 3개(clean4, ens_q38_gemma4, ens_q38_gdev_cal) 중 하나라도 1·2위 확률 차가 0.8보다 작은 문항만 골라, 멤버 4개(Q35·Q38·G·GDEV)를 **기존 어댑터 그대로** 보기 순서 4가지로 다시 채점합니다. 확신이 큰 문항은 TTA를 해도 답이 거의 안 바뀌므로 계산을 애매한 문항에 모읍니다.
- 그 문항들에서만 멤버 확률을 순서 4가지 평균으로 바꾸고, **같은 레시피로 앙상블을 다시 만들어** 기준과 같은 문항끼리 고침·새로 틀림을 셉니다. HCX는 설치가 달라 원래 확률을 그대로 씁니다.
- 같은 채점에서 **LoRA를 끈 원본 모델** 점수도 저장해, LoRA-off 결합 규칙(원본이 다른 답을 확신 있게 고르고 네 조건을 모두 만족할 때만 0.5:0.5)을 기준값 그대로 따로 시험합니다(H2).

**사전 기준 (결과를 보기 전에 고정)**
- 검증에서 바뀐 답이 10개 미만이면 검증만으로 판단하지 않습니다. 제출 후보를 공개로 한 번 확인합니다.
- **H1(TTA4)**: 검증 순이득 + 공개 순이득(clean4 후보) ≥ +3 이고 공개 순이득 ≥ 0 → 채택. 노트북이 채택에 필요한 공개 점수를 미리 출력합니다.
- **H2(+LoRA-off 결합)**: H1 대비 검증 + 공개 순이득 ≥ +2 → 추가 채택. test에서 H1과 다른 답이 5개 미만이면 시험하지 않습니다.

**실행**: 1번 셀을 확인하고 위에서부터 끝까지. 멤버 4개를 한 세션에서 차례로 올립니다(모델을 바꿀 때마다 GPU 메모리를 비우고, 베이스가 바뀌면 이전 베이스 파일을 지움). 끊기면 다시 실행하면 남은 채점만 합니다.
**런타임**: G4 또는 A100 80GB (bf16). 결과는 `vqa_runs/e9_tta/`에만 씁니다.'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 라이브러리 설치 (학습 때와 같은 버전)"), ("code", S(4)),
    ("markdown", "## 3. Drive 연결 → 멤버 확인 → 데이터 준비"), ("code", DATA),
    ("markdown", "## 4. 공통 import · CSV 로드"), ("code", S(8)),
    ("markdown", "## 5. 고정 검증셋 분할"), ("code", SPLIT),
    ("markdown", "## 6. 프롬프트 (학습 때와 동일)"), ("code", S(12)),
    ("markdown", "## 7. 기준 앙상블 → 애매한 문항 고르기 (GPU 불필요)\n예전에 해 본 TTA 기록(stage2 결과 로그)도 함께 보여 줍니다."), ("code", GATE),
    ("markdown", "## 8. 채점 함수 (stage2와 동일, 증강만 제거)"), ("code", SCORING),
    ("markdown", "## 9. 멤버 로드 함수 (bf16 베이스 + 기존 어댑터, merge 없음)"), ("code", LOADER),
    ("markdown", "## 10. 멤버별 채점: 보기 순서 4가지 × (LoRA 켬 → 끔)\n멤버마다 사전 점검(어댑터 켬/끔 차이, letter_mass, 이미지 토큰 수)을 통과해야 채점합니다."), ("code", RUNCELL),
    ("markdown", "## 11. 판정 → 제출 후보\n멤버별로 `순서 그대로 재현(%)`(0번 순서 = 기존 예측 재현)이 85% 이상이어야 정상입니다."), ("code", EVAL),
    ("markdown", "## 12. 결과 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    c = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]; nb.cells.append(c)
nbformat.validate(nb); nbformat.write(nb, OUT_NB)
print("저장:", OUT_NB, "| 셀", len(nb.cells))
