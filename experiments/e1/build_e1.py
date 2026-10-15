# -*- coding: utf-8 -*-
"""E1 노트북 빌더: 기존(NF4로 학습·평가된) LoRA 어댑터를 bf16 베이스에 merge 없이 얹어 다시 채점.

학습 코드는 없고, 프롬프트·채점 함수는 팀의 stage2 노트북 셀을 그대로 가져옵니다.
stage2에서 바꾼 부분은 rep()로 정확히 한 군데씩만 고치고, 바뀐 줄은 빌드할 때 diff로 출력합니다.
"""
import difflib, json, sys
import nbformat

SRC_NB = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/src_stage2.ipynb"
OUT_NB = sys.argv[2] if len(sys.argv) > 2 else "/home/user/vqa/e1/e1_bf16_reeval.ipynb"

src_nb = json.load(open(SRC_NB, encoding="utf-8"))
def S(i):
    s = src_nb["cells"][i]["source"]
    return "".join(s) if isinstance(s, list) else s

def rep(text, old, new, count=1):
    n = text.count(old)
    assert n == count, (n, old[:90])
    return text.replace(old, new)

SHOW_DIFF = []
def patched(i, pairs, label):
    orig = S(i); new = orig
    for old, nw in pairs:
        new = rep(new, old, nw)
    SHOW_DIFF.append((label, orig, new))
    return new

# ───────────────────────────── 1. 설정 ─────────────────────────────
CFG = r'''
SRC_RUN   = "q35_27b_r16_native"      # 다시 채점할 기존 실행 (vqa_runs 아래 폴더 이름). 이 폴더는 읽기만 합니다
MODEL_ID  = "Qwen/Qwen3.5-27B"        # 그 실행의 베이스 모델. 어댑터에 기록된 이름과 다르면 3번 셀에서 멈춥니다
PRESET    = "up2.0"                   # 채점 해상도. 기존 NF4 결과(preds_final_ep*__up2.0__1000.csv)와 같은 해상도로 비교합니다
SRC_EPOCH = "auto"                    # "auto": 기존 최종 평가에 쓰인 epoch의 어댑터 / 숫자를 넣으면 adapter_ep<숫자>
RUN_TEST  = "auto"                    # "auto": 판정을 통과할 때만 test 6,714문항 채점 / True: 항상 / False: 안 함
RUN_NAME  = f"{SRC_RUN}_bf16eval"     # 결과 폴더 이름 = Stage 3에서 쓸 멤버 이름
# 다른 멤버도 같은 방법으로 재채점: SRC_RUN·MODEL_ID 두 줄만 바꾸세요
#   예) "q38_27b_r16_native" + "Qwen/Qwen3.8-27B" / "gemma4_31b_r16_up20" + "google/gemma-4-31B-it"
#   HCX는 별도 노트북(개발 버전 transformers)으로 학습했으므로 이 노트북으로는 안 됩니다

# 판정 기준 (보고서 E1): 같은 1,000문항에서 순이득(고침 − 새로 틀림) ≥ +3 이고 NLL 감소
NET_MIN = 3

# 앙상블 관점: 현재 최고 제출을 검증에서 다시 만든 뒤, 그 안에서 SRC_RUN 자리만 bf16 결과로 바꿨을 때를 같은 1,000문항에서 비교합니다.
# 공개 최고 0.97676 = 아래 세 앙상블 제출의 다수결(동률이면 알파벳이 빠른 답 = pandas mode(axis=1)[0]).
# 각 앙상블 레시피가 맞는지는 멤버 test 예측으로 제출 파일을 다시 만들어 6,714문항이 모두 같은지로 확인합니다 (틀리면 ⚠️).
#   calibrate=True: 멤버마다 검증셋에서 온도를 맞춘 뒤 평균 (Stage 3 CALIBRATE와 같은 방식, 이름 끝 _cal)
# 가중 평균 하나만 기준으로 쓰려면 vote 없이 dict(name=..., members=..., weights=..., calibrate=False, file=...) 형태로 넣고, 건너뛰려면 None
Q35, Q38 = ("q35_27b_r16_native", "up2.0"), ("q38_27b_r16_native", "up2.0")
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", "up2.0"), ("gemma4_31b_r16_up20", "up2.0"), ("hcx32b_r16_native", "up2.0")
ENSEMBLE_REF = {"name": "majority_vote", "file": "submission_majority_vote.csv", "public": 0.97676,
                "vote": [dict(name="ens_q38_gemma4",   members=[Q38, G],                 weights=[1, 1],          calibrate=False, file="submission_ens_q38_gemma4.csv"),
                         dict(name="clean4_qg_hcx",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], calibrate=False, file="submission_clean4_qg_hcx.csv"),
                         dict(name="ens_q38_gdev_cal", members=[Q38, GDEV],              weights=[1, 1],          calibrate=True,  file="submission_ens_q38_gdev_cal.csv")]}
ENSEMBLE_SUBDIR = "ensembles"          # 제출 파일이 있는 폴더 (vqa_runs 아래)

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
BATCH_SIZE   = "auto"          # 평가용 배치 (OOM이면 자동으로 절반)
SEED, VAL_SIZE = 42, 1000      # 검증 분할용 — 팀 공통값이므로 절대 바꾸지 마세요
SKIP_MISSING_IMAGES = False
HF_TOKEN     = ""              # Gemma처럼 라이선스 승인이 필요한 모델일 때 HF 토큰 (비워 두면 Colab 보안 비밀 'HF_TOKEN'을 찾아봄)

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

# ───────────────── 3. Drive → 원본 실행 확인 → 데이터 ─────────────────
DATA = r'''
import os, re, json, hashlib, subprocess
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── 원본 실행 확인 (읽기만 합니다). 데이터 압축을 풀기 전에 확인해서, 이름이 틀렸으면 바로 멈춥니다 ──
SRC_DIR = os.path.join(OUT_DIR, SRC_RUN)
assert os.path.isdir(SRC_DIR), f"원본 실행 폴더가 없습니다: {SRC_DIR} → 1번 셀의 SRC_RUN을 확인하세요"
_files = sorted(os.listdir(SRC_DIR))
_rx_val = re.compile(rf"^preds_final_ep(\d+)__{re.escape(PRESET)}__1000\.csv$")
REF_VAL_BY_EP = {int(m.group(1)): os.path.join(SRC_DIR, f) for f in _files if (m := _rx_val.match(f))}
_adapters = sorted(int(m.group(1)) for f in _files if (m := re.match(r"^adapter_ep(\d+)$", f)))
_log_path = os.path.join(SRC_DIR, "train_log.json")
SRC_LOG = json.load(open(_log_path, encoding="utf-8")) if os.path.exists(_log_path) else {}
SRC_CFG = SRC_LOG.get("config") or {}
_ea = {int(k): v for k, v in (SRC_LOG.get("epoch_acc") or {}).items()}
if SRC_EPOCH != "auto":       SRC_EP = int(SRC_EPOCH)
elif len(REF_VAL_BY_EP) == 1: SRC_EP = next(iter(REF_VAL_BY_EP))          # 기존 최종 평가 파일 이름에 적힌 epoch
elif _ea:                     SRC_EP = max(_ea, key=lambda e: (_ea[e], e))  # stage2 11번 셀과 같은 규칙 (동률이면 뒤 epoch)
elif _adapters:               SRC_EP = _adapters[-1]
else: raise FileNotFoundError(f"{SRC_DIR}에 adapter_ep* 폴더가 없습니다. 폴더 내용: {_files}")
ADAPTER_DIR = os.path.join(SRC_DIR, f"adapter_ep{SRC_EP}")
_st = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
assert os.path.isfile(_st) and os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json")), \
    f"어댑터 파일이 없습니다: {ADAPTER_DIR} (원본 폴더 내용: {_files})"
REF_VAL = REF_VAL_BY_EP.get(SRC_EP)
_rx_test = re.compile(rf"^preds_final_ep{SRC_EP}_test__{re.escape(PRESET)}__\d+\.csv$")
REF_TEST = next((os.path.join(SRC_DIR, f) for f in _files if _rx_test.match(f)), None)

ADAPTER_CFG = json.load(open(os.path.join(ADAPTER_DIR, "adapter_config.json"), encoding="utf-8"))
_rec = str(ADAPTER_CFG.get("base_model_name_or_path") or "")
if _rec and not _rec.startswith("/") and _rec != MODEL_ID:
    raise ValueError(f"이 어댑터는 '{_rec}' 위에서 학습됐는데 MODEL_ID는 '{MODEL_ID}'입니다 → 1번 셀의 MODEL_ID를 '{_rec}'로 바꾸세요")
SRC_TRAIN_ON_FULL = bool(SRC_CFG.get("TRAIN_ON_FULL", False))
DO_VAL = not SRC_TRAIN_ON_FULL        # 검증셋까지 학습한 실행이면 검증 비교는 의미가 없음 → test만 채점

def _md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()
ADAPTER_MD5 = _md5(_st)
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME)
assert os.path.abspath(RUN_DIR) != os.path.abspath(SRC_DIR), "RUN_NAME이 원본 실행과 같습니다"
os.makedirs(RUN_DIR, exist_ok=True)
INFO_PATH = os.path.join(RUN_DIR, "e1_info.json")
E1_INFO = json.load(open(INFO_PATH, encoding="utf-8")) if os.path.exists(INFO_PATH) else {}
if E1_INFO and E1_INFO.get("adapter_md5") != ADAPTER_MD5:
    raise RuntimeError(f"{RUN_DIR}에는 다른 어댑터로 만든 결과가 있습니다 (기록 {str(E1_INFO.get('adapter_md5'))[:8]} ≠ 지금 {ADAPTER_MD5[:8]}) "
                       "→ 1번 셀의 RUN_NAME을 바꾸세요")
E1_INFO.update({"src_run": SRC_RUN, "adapter_dir": ADAPTER_DIR, "adapter_md5": ADAPTER_MD5, "model_id": MODEL_ID,
                "precision": "bf16 베이스 + 기존 어댑터 (merge 없음)"})
json.dump(E1_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

_nf4_rule = re.search(r"(?<![\d.])(2\d|3\d|7\d)B", _rec or MODEL_ID) is not None       # 기존 노트북의 USE_4BIT="auto" 규칙
print(f"원본 실행: {SRC_DIR}")
print(f"  어댑터 adapter_ep{SRC_EP} (md5 {ADAPTER_MD5[:8]}) | r={ADAPTER_CFG.get('r')}, alpha={ADAPTER_CFG.get('lora_alpha')} | "
      f"학습 해상도 {SRC_CFG.get('TRAIN_PRESET', '?')} | 기록된 베이스 {_rec or '?'}")
print(f"  비교 기준(기존) 검증 파일: {os.path.basename(REF_VAL) if REF_VAL else '없음 → 비교 불가'} | "
      f"기존 test 파일: {os.path.basename(REF_TEST) if REF_TEST else '없음'}")
print("  기존 노트북의 USE_4BIT='auto' 규칙이었다면 이 모델은 " +
      ("NF4로 학습·평가됐습니다 (이번 비교 대상)" if _nf4_rule else "bf16이었습니다 → 재채점해도 거의 달라지지 않을 것입니다"))
if SRC_TRAIN_ON_FULL: print("  ⚠️ 검증셋까지 학습한 실행입니다 → 검증 비교는 건너뛰고 test만 채점합니다")
print(f"결과 폴더(새로 씀): {RUN_DIR}")

# ── 데이터 (stage2와 동일) ──
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
print("데이터 OK:", DATA_DIR)
'''

# ───────────────── 5. 분할: stage2와 같되 불일치면 멈춤, 공용 파일은 쓰지 않음 ─────────────────
SPLIT = patched(10, [
    ('if split_md5 == EXPECTED_SPLIT_MD5:\n    print("✅ 팀 공통 분할과 일치")\nelse:\n'
     '    print("⚠️⚠️ 검증 분할이 기준과 다릅니다 → 팀원 간 점수 비교 불가. train.csv 원본/SEED/VAL_SIZE를 확인하세요.")\n'
     'val_df[["id"]].to_csv(os.path.join(OUT_DIR, "valid_ids.csv"), index=False)',
     'assert split_md5 == EXPECTED_SPLIT_MD5, "⚠️ 검증 분할이 기준과 다릅니다 → 기존 결과와 비교할 수 없습니다. train.csv 원본/SEED/VAL_SIZE를 확인하세요."\n'
     'print("✅ 팀 공통 분할과 일치")'),
], "5. 검증 분할")

# ───────────────── 7. 모델 로드 (학습 코드 없음) ─────────────────
MODEL = r'''
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, load_peft_weights, get_peft_model_state_dict
import transformers, peft, gc
print("transformers", transformers.__version__, "| peft", peft.__version__, "| torch", torch.__version__)

for _v in ("model", "base", "enc", "logits", "out"):       # 같은 세션에서 다시 실행할 때 이전 모델을 GPU에서 비움
    if _v in globals(): del globals()[_v]
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

assert torch.cuda.is_available(), "GPU 런타임이 아닙니다 (런타임 → 런타임 유형 변경)"
DEVICE = "cuda"
GPU_NAME = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
assert cap[0] >= 8, f"{GPU_NAME}은 bf16 연산을 지원하지 않습니다 → G4 또는 A100 80GB 런타임을 쓰세요"
DTYPE = torch.bfloat16
_sz = re.search(r"(?<![\d.])(\d+)B", MODEL_ID, flags=re.I)
NEED_GB = int(_sz.group(1)) * 2.05 + 6 if _sz else 0.0       # bf16 가중치(파라미터 x 2바이트) + 채점용 여유
if VRAM_GB < NEED_GB:
    raise RuntimeError(f"bf16 {MODEL_ID}에는 GPU 메모리가 약 {NEED_GB:.0f}GB 필요한데 {GPU_NAME}은 {VRAM_GB:.0f}GB입니다 "
                       "→ 런타임을 G4(96GB) 또는 A100 80GB로 바꾸세요")
BS = (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2) if BATCH_SIZE == "auto" else int(BATCH_SIZE)
print(f"GPU: {GPU_NAME} ({VRAM_GB:.0f}GB) | dtype={DTYPE} | 4bit=False (bf16 고정) | 평가 배치={BS}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"
tok = processor.tokenizer
FAMILY = "gemma4" if type(processor).__name__.startswith("Gemma4") else "qwen"     # 프로세서 계열 (해상도 인자·이미지 전달 방식이 다름)
IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
if IMAGE_TOKEN_ID is None: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
_tmpl = processor.chat_template if isinstance(processor.chat_template, str) else json.dumps(processor.chat_template)
TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
try:
    processor.apply_chat_template(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
except Exception as _e:
    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")
print(f"계열: {FAMILY} | 이미지 토큰 id: {IMAGE_TOKEN_ID} | chat template 추가 인자: {TEMPLATE_KW} | system 역할: {USE_SYSTEM_ROLE}")

t0 = time.time()
base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa", device_map={"": 0})
base.config.use_cache = False
_quant = [n for n, m in base.named_modules() if any(k in type(m).__name__ for k in ("Linear4bit", "Linear8bit"))]
assert not _quant, f"양자화된 층이 있습니다 (bf16이어야 함): {_quant[:3]}"
model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)      # merge 하지 않고 어댑터를 그대로 얹음
model.eval()
_dt = sorted({str(p.dtype) for n, p in base.named_parameters() if "lora_" not in n})
_mem = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
print(f"모델 로드 {time.time() - t0:.0f}s | {type(base).__name__} | 베이스 가중치 dtype {_dt} | 사용 VRAM {_mem:.1f}GB")
if "qwen3_5" in str(getattr(base.config, "model_type", "")):
    import importlib.util
    if importlib.util.find_spec("fla") is None:
        print("⚠️⚠️ flash-linear-attention이 없습니다 → 선형 어텐션이 순수 PyTorch로 돌아 매우 느립니다. 2번 셀을 실행하고 '세션 다시 시작' 후 처음부터 실행하세요.")
    else:
        print("✅ flash-linear-attention 커널 사용")

n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
n_lora_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and any(k in n for k in ("visual", "vision_tower", "vision_model")))
print(f"LoRA가 붙은 층: {n_lora}개 (비전 타워 {n_lora_vis}개 — 0이어야 정상)")
assert n_lora > 0 and n_lora_vis == 0

def _norm_key(k):
    k = k.replace("base_model.model.", "", 1)
    i = k.rfind("language_model.")
    return k[i + len("language_model."):] if i >= 0 else k

def check_adapter_weights(model, adapter_dir):
    # 어댑터 파일의 LoRA 텐서가 전부, 값 그대로 모델에 들어갔는지 확인 (PEFT는 이름이 안 맞는 키를 조용히 건너뛸 수 있음)
    fw = load_peft_weights(adapter_dir, device="cpu")
    mw = {k: v.detach() for k, v in get_peft_model_state_dict(model).items()}
    same_names = set(fw) == set(mw)
    if not same_names:                                     # 이름 형식(접두어)만 다른 경우를 대비해 언어 모델 기준 이름으로 맞춰 봄
        fw = {_norm_key(k): v for k, v in fw.items()}; mw = {_norm_key(k): v for k, v in mw.items()}
    only_f, only_m = sorted(set(fw) - set(mw)), sorted(set(mw) - set(fw))
    assert not only_f and not only_m, (f"어댑터 파일과 모델의 LoRA 키가 다릅니다 → 파일에만 {len(only_f)}개(예: {only_f[:2]}), "
                                       f"모델에만 {len(only_m)}개(예: {only_m[:2]})")
    worst, nz_b = 0.0, 0
    for k, v in fw.items():
        w = mw[k]
        assert tuple(v.shape) == tuple(w.shape), f"모양이 다릅니다: {k} {tuple(v.shape)} vs {tuple(w.shape)}"
        worst = max(worst, (v.to(w.dtype).float() - w.float().cpu()).abs().max().item())   # 불러올 때와 같은 dtype으로 맞추면 정확히 같아야 함
        if "lora_B" in k and v.abs().max().item() > 0: nz_b += 1
    n_b = sum("lora_B" in k for k in fw)
    assert worst == 0.0, f"어댑터 파일과 모델의 LoRA 값이 다릅니다 (최대 차이 {worst:.3g})"
    assert nz_b > 0, "어댑터 파일의 lora_B가 전부 0입니다 → 학습되지 않은 어댑터입니다"
    return {"keys": len(fw), "same_names": same_names, "nonzero_B": f"{nz_b}/{n_b}", "max_diff": worst}

_c = check_adapter_weights(model, ADAPTER_DIR)
print(f"✅ 어댑터 파일 ↔ 모델 가중치 일치: 키 {_c['keys']}개 전부 값까지 같음 | 0이 아닌 lora_B {_c['nonzero_B']}"
      + ("" if _c["same_names"] else " (이름 접두어만 달라 언어 모델 기준으로 맞춤)"))

LETTER_IDS = []                                     # 첫 번째(소문자·공백 없음)가 학습 목표, 나머지 변형은 확률 집계용
for L in LETTERS:
    ids = []
    for s in (L, L.upper(), " " + L, " " + L.upper()):
        t = tok.encode(s, add_special_tokens=False)
        if len(t) == 1 and t[0] not in ids: ids.append(t[0])
    assert ids and len(tok.encode(L, add_special_tokens=False)) == 1, f"'{L}' 가 단일 토큰이 아닙니다"
    LETTER_IDS.append(ids)
print("글자 토큰 id:", dict(zip(LETTERS, LETTER_IDS)))
'''

# ───────────────── 8. 채점 함수: stage2 16번 셀 그대로 (증강·학습 전용 부분만 제거) ─────────────────
_s16 = S(16)
_aug_start = _s16.index("from PIL import ImageEnhance")
_aug_end = _s16.index("def encode_batch")
SCORING = patched(16, [
    ("set_resolution(TRAIN_PRESET)", "set_resolution(PRESET)"),
    (_s16[_aug_start:_aug_end], ""),
    ("    if _AUG_ON and IMG_AUG: images = [augment_image(im) for im in images]\n", ""),
    ("\"run\": f\"{RUN_NAME}{'_dry' if DRY_RUN else ''}/{tag}\"", "\"run\": f\"{RUN_NAME}/{tag}\""),
], "8. 채점 함수")

# ───────────────── 9. 사전 점검 ─────────────────
PRECHECK = r'''
set_resolution(PRESET)
_chk = val_df.head(8)
_rows = _chk.to_dict("records")
t0 = time.time()
p_on, m_on, nt_on = score_rows(_rows)
with model.disable_adapter():
    p_off, m_off, _ = score_rows(_rows)
d_onoff = float(np.abs(p_on - p_off).max())
print(f"8문항 채점 {time.time() - t0:.0f}s | 어댑터 켬/끔 확률 최대 차이 {d_onoff:.3f} | letter_mass 평균 {m_on.mean():.3f} | 이미지 토큰 {nt_on}")
assert d_onoff > 1e-3, "어댑터를 켜고 꺼도 출력이 같습니다 → 어댑터가 적용되지 않았습니다. 위 출력을 확인하세요"
assert m_on.mean() > 0.9, "letter_mass가 낮습니다 → 프롬프트·토큰 처리 문제입니다. 위 출력을 확인하세요"
if REF_VAL:
    _ref = pd.read_csv(REF_VAL).drop_duplicates("id", keep="last").set_index("id").reindex(_chk.id)
    if _ref.pred.isna().any():
        print("⚠️ 기존 검증 파일에 이 8문항이 없습니다 → 사전 비교는 건너뜁니다")
    else:
        _pref = _ref[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
        _agree = int((p_on.argmax(1) == _pref.argmax(1)).sum())
        print(f"기존 결과와 비교(같은 8문항): 답 일치 {_agree}/8 | 확률 평균 차이 {np.abs(p_on - _pref).mean():.4f} | "
              f"letter_mass {_ref.letter_mass.mean():.3f} → {m_on.mean():.3f}")
        if "img_tokens" in _ref.columns:
            assert [int(t) for t in _ref.img_tokens] == [int(t) for t in nt_on], \
                f"이미지 토큰 수가 기존 결과와 다릅니다 ({[int(t) for t in _ref.img_tokens]} vs {list(nt_on)}) → 해상도·프로세서가 달라 비교할 수 없습니다"
            print("✅ 이미지 토큰 수가 기존 결과와 같음 (같은 해상도·프로세서)")
        if _agree < 6: print("⚠️ 답이 기존과 많이 다릅니다. 계속 진행하지만 결과를 확인하세요")
print("✅ 사전 점검 통과 → 본 채점을 시작합니다")
'''

# ───────────────── 10. 검증 채점 ─────────────────
_s22 = S(22)
ERR_FN = _s22[:_s22.index("if not DRY_RUN and state")].rstrip() + "\n"
SRC_ERR = ERR_FN  # stage2 22번 셀의 오답 리포트 함수를 그대로 사용
VALEVAL = ERR_FN + r'''
FINAL_TAG = f"final_ep{SRC_EP}"      # stage2와 같은 이름 규칙 → preds_final_ep{N}__{PRESET}__1000.csv / preds_final_ep{N}_test__{PRESET}__6714.csv
set_resolution(PRESET)
if DO_VAL:
    acc_new, m_new = evaluate(val_df, FINAL_TAG)          # 끊겨도 다시 실행하면 남은 문항만 채점
    export_error_report(m_new, os.path.join(RUN_DIR, f"error_report_{FINAL_TAG}_{PRESET}.html"))
else:
    print("검증셋까지 학습한 실행이라 검증 채점을 건너뜁니다 (test만 채점)")
'''

# ───────────────── 11. 비교·판정 ─────────────────
COMPARE = r'''
def _probs(d): return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
def _load_preds(path, ids):
    d = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in ids if i not in d.index]
    assert not miss, f"{os.path.basename(path)}에 없는 문항 {len(miss)}개 (예: {miss[:3]})"
    return d.loc[ids]
def _nll(P, y): return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())
def _paired(a_old, a_new, y):
    c0, c1 = a_old == y, a_new == y
    fixed, broken = int((~c0 & c1).sum()), int((c0 & ~c1).sum())
    return fixed, broken, fixed - broken, int((a_old != a_new).sum())
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
    # split="val": preds_final_ep{N}__{preset}__1000.csv / "test": 같은 epoch의 preds_final_ep{N}_test__{preset}__{개수}.csv
    f, why = find_val_file(run, preset)
    if f is None or split == "val": return f, why
    ep = re.match(r"^preds_final_ep(\d+)__", os.path.basename(f)).group(1)
    d = os.path.join(OUT_DIR, run)
    t = sorted(x for x in os.listdir(d) if re.match(rf"^preds_final_ep{ep}_test__{re.escape(preset)}__\d+\.csv$", x))
    return (os.path.join(d, t[0]), why) if t else (None, f"ep{ep} test 파일 없음")

VAL_IDS, Y_VAL, TEST_IDS = val_df.id.tolist(), val_df.answer.map(LETTERS.index).to_numpy(), test_df.id.tolist()
_MP = {}
def member_probs(run, preset, split):
    k = (run, preset, split)
    if k not in _MP:
        f, why = find_member_file(run, preset, split)
        if f is None: raise FileNotFoundError(f"{run}@{preset} {split}: {why}")
        _MP[k] = _probs(_load_preds(f, VAL_IDS if split == "val" else TEST_IDS))
    return _MP[k]
def temp_scale(p, T):                              # Stage 3 CALIBRATE와 같은 식 (답 순위는 그대로, 확신 정도만 조절)
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def fit_T(p, y):
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70)); idx = np.arange(len(y))
    nll = [-np.mean(np.log(temp_scale(p, T)[idx, y])) for T in grid]
    return float(grid[int(np.argmin(nll))])
def has_src(v): return (SRC_RUN, PRESET) in [tuple(m) for m in v["members"]]
def voter_probs(v, split, swap=None):
    # 앙상블 하나의 확률 (가중 평균). swap={"val": bf16 검증 확률, "test": bf16 test 확률}이면 SRC_RUN 멤버 자리를 그것으로 바꿈
    Ps = []
    for run, pre in [tuple(m) for m in v["members"]]:
        mine = swap is not None and (run, pre) == (SRC_RUN, PRESET)
        P = swap[split] if mine else member_probs(run, pre, split)
        if v.get("calibrate"):                     # 온도는 멤버마다 검증셋에서 맞춤 (bf16으로 바꾼 멤버는 bf16 검증 확률로 다시 맞춤)
            P = temp_scale(P, fit_T(swap["val"] if mine else member_probs(run, pre, "val"), Y_VAL))
        Ps.append(P)
    w = np.asarray(v["weights"], dtype=float)
    assert len(w) == len(Ps), f"{v['name']}: members와 weights 개수가 다릅니다"
    return sum(wi * Pi for wi, Pi in zip(w, Ps)) / w.sum()
def vote(answers):                                 # 다수결. 동률(1:1:1)이면 알파벳이 빠른 답 = pandas mode(axis=1)[0]과 같음
    A = np.stack(answers, 1)
    return np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1)
def file_answers(fname):
    p = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR, fname)
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)

V = {"verdict": "판정 불가", "net": None, "ens_net": None, "lines": []}
def say(s): print(s); V["lines"].append(s)
AUTO_TEST_OK = False
REF_VIEW = None
if not DO_VAL:
    V["verdict"] = "검증 생략 (검증셋까지 학습한 실행) → test만 채점"; AUTO_TEST_OK = True
else:
    ids = val_df.id.tolist(); y = val_df.answer.map(LETTERS.index).to_numpy()
    new = _load_preds(os.path.join(RUN_DIR, f"preds_{FINAL_TAG}__{PRESET}__1000.csv"), ids); P1 = _probs(new); a1 = P1.argmax(1)
    if REF_VAL is None:
        say(f"bf16 검증 정확도 {(a1 == y).mean() * 100:.2f}% — 기존 검증 파일이 없어 문항 단위 비교는 못 합니다")
        V["verdict"] = f"판정 불가 — {SRC_RUN}에 preds_final_ep{SRC_EP}__{PRESET}__1000.csv가 없어 NF4와 비교할 수 없습니다"
    else:
        ref = _load_preds(REF_VAL, ids); P0 = _probs(ref); a0 = P0.argmax(1)
        fixed, broken, net, D = _paired(a0, a1, y)
        nll0, nll1 = _nll(P0, y), _nll(P1, y)
        lm0, lm1 = float(ref.letter_mass.mean()), float(new.letter_mass.mean())
        dp = np.abs(P1 - P0).max(1)
        say(f"[단독] {SRC_RUN}@{PRESET}, 같은 1,000문항 |  기존(NF4) → bf16")
        say(f"  정확도 {(a0 == y).mean() * 100:.2f}% → {(a1 == y).mean() * 100:.2f}% | NLL {nll0:.4f} → {nll1:.4f} | letter_mass {lm0:.3f} → {lm1:.3f}")
        say(f"  고침 {fixed} | 새로 틀림 {broken} | 순이득 {net:+d} | 답이 바뀐 문항 D={D} (강한 근거 기준 2√D = {2 * math.sqrt(D):.1f})")
        say(f"  문항별 확률 변화(최대 |Δp|): 중앙값 {np.median(dp):.4f}, 상위 10% {np.quantile(dp, 0.9):.4f}")
        health = []                                                  # 비교 자체가 성립하는지 (하나라도 걸리면 판정하지 않음)
        if "img_tokens" in ref.columns:
            _same = int((ref.img_tokens.to_numpy() == new.img_tokens.to_numpy()).sum())
            say(f"  이미지 토큰 수 일치 {_same}/{len(ids)}")
            if _same != len(ids): health.append(f"이미지 토큰 수가 다른 문항 {len(ids) - _same}개(해상도 불일치)")
        if lm1 < lm0 - 0.01: health.append(f"letter_mass 하락 {lm0:.3f} → {lm1:.3f}(토큰화·템플릿 문제 가능성)")
        if D <= 2 and np.median(dp) < 0.002: say("  ℹ️ 기존 결과와 사실상 같습니다 → 기존 실행도 bf16이었을 가능성이 큽니다")
        _chg = val_df.assign(NF4=[LETTERS[i] for i in a0], bf16=[LETTERS[i] for i in a1],
                             p_gold_NF4=P0[np.arange(len(y)), y].round(3), p_gold_bf16=P1[np.arange(len(y)), y].round(3))
        _chg = _chg[_chg.NF4 != _chg.bf16].copy()
        _chg["결과"] = np.where(_chg.bf16 == _chg.answer, "고침", np.where(_chg.NF4 == _chg.answer, "새로 틀림", "둘 다 틀림"))
        _cols = ["id", "결과", "answer", "NF4", "bf16", "p_gold_NF4", "p_gold_bf16", "question", "a", "b", "c", "d"]
        _chg[_cols].to_csv(os.path.join(RUN_DIR, f"e1_changed_{PRESET}.csv"), index=False, encoding="utf-8-sig")
        if len(_chg):
            print("\n답이 바뀐 문항 (e1_changed_" + PRESET + ".csv에 전체 저장):")
            print(_chg[["id", "결과", "answer", "NF4", "bf16", "p_gold_NF4", "p_gold_bf16", "question"]].head(20).to_string(index=False))
        V.update({"acc_nf4": round(float((a0 == y).mean() * 100), 2), "acc_bf16": round(float((a1 == y).mean() * 100), 2),
                  "fixed": fixed, "broken": broken, "net": net, "D": D, "nll_nf4": round(nll0, 4), "nll_bf16": round(nll1, 4)})
        if health:
            V["verdict"] = f"확인 필요 — {'; '.join(health)} → 판정하지 않습니다. 위 출력을 확인하세요"
        elif net >= NET_MIN and nll1 < nll0:
            V["verdict"] = f"채택 — 순이득 {net:+d} ≥ +{NET_MIN}, NLL 감소 (보고서 E1 기준 통과)" + \
                           ("" if net >= 2 * math.sqrt(D) else " / 단, 2√D 미만이라 공개 제출로 한 번 더 확인 권장")
            AUTO_TEST_OK = True
        elif net >= 1:
            V["verdict"] = f"보류 — 순이득 {net:+d}" + (" (NLL은 늘어남)" if nll1 >= nll0 else "") + " → 기준 미달. 공개 제출로 확인하려면 test를 직접 켜세요"
            V["hold"] = True
        else:
            V["verdict"] = f"불채택 — 순이득 {net:+d}: bf16 재채점 이득 없음"

    # ── 앙상블 관점: 현재 최고 제출을 검증에서 다시 만든 뒤 SRC_RUN 자리만 bf16 결과로 교체 ──
    if ENSEMBLE_REF:
        voters = ENSEMBLE_REF.get("vote") or [ENSEMBLE_REF]
        rname = ENSEMBLE_REF["name"]
        try:
            b_ans = [voter_probs(v, "val").argmax(1) for v in voters]
        except FileNotFoundError as e:
            say(f"[앙상블] {rname}: 멤버 파일이 없어 건너뜁니다 → {e}")
        else:
            if not any(has_src(v) for v in voters):
                say(f"[앙상블] {rname}에 {SRC_RUN}@{PRESET}가 없어 교체 비교를 건너뜁니다")
            else:
                # (1) 레시피 확인: 멤버 test 예측으로 각 제출 파일을 다시 만들어 6,714문항이 모두 같은지
                rep_ok, rep_txt = True, []
                for v in voters:
                    fa = file_answers(v["file"]) if v.get("file") else None
                    try: ra, why = voter_probs(v, "test").argmax(1), ""
                    except FileNotFoundError as e: ra, why = None, str(e)
                    if fa is None or ra is None:
                        rep_ok = False; rep_txt.append(f"{v['name']} 확인 불가({'제출 파일 없음' if fa is None else why})")
                    else:
                        _m = int((fa == ra).sum()); rep_ok &= _m == len(fa)
                        rep_txt.append(f"{v['name']} {_m:,}/{len(fa):,}" + ("" if _m == len(fa) else " ⚠️"))
                if "vote" in ENSEMBLE_REF:
                    _fv = [file_answers(v["file"]) if v.get("file") else None for v in voters]
                    _fm = file_answers(ENSEMBLE_REF["file"]) if ENSEMBLE_REF.get("file") else None
                    if _fm is None or any(x is None for x in _fv):
                        rep_ok = False; rep_txt.append("다수결 파일 확인 불가")
                    else:
                        _m = int((vote(_fv) == _fm).sum()); rep_ok &= _m == len(_fm)
                        rep_txt.append(f"다수결 {_m:,}/{len(_fm):,}" + ("" if _m == len(_fm) else " ⚠️"))
                # (2) 검증 1,000문항: 기준 제출 vs SRC_RUN만 bf16으로 바꾼 제출
                s_ans = [voter_probs(v, "val", swap={"val": P1}).argmax(1) if has_src(v) else a for v, a in zip(voters, b_ans)]
                e0, e1 = vote(b_ans), vote(s_ans)
                acc_e0 = (e0 == y).mean() * 100
                _exp = ENSEMBLE_REF.get("expected_val_acc")
                if _exp is not None and abs(acc_e0 - _exp) >= 0.05:
                    rep_ok = False; rep_txt.append(f"검증 {acc_e0:.2f}% ≠ 기대 {_exp} ⚠️")
                f2, b2, n2, D2 = _paired(e0, e1, y)
                say(f"[앙상블] {rname}" + (f"(공개 {ENSEMBLE_REF['public']})" if ENSEMBLE_REF.get("public") else "") +
                    f" 검증 재구성 {acc_e0:.2f}% | 레시피 확인(test 재현): " + ", ".join(rep_txt))
                if not rep_ok:
                    say("  ⚠️ 재현이 안 된 부분이 있습니다 → 1번 셀 ENSEMBLE_REF의 멤버·가중치·calibrate·파일 이름을 확인하세요. 아래 줄은 참고만 하세요")
                if len(voters) > 1:
                    for v, _a0, _a1 in zip(voters, b_ans, s_ans):
                        if has_src(v):
                            _f, _b, _n, _D = _paired(_a0, _a1, y)
                            say(f"  · {v['name']} 하나만 보면: 고침 {_f} | 새로 틀림 {_b} | 순이득 {_n:+d} | 답이 바뀐 문항 {_D}")
                say(f"  {SRC_RUN} 자리만 bf16으로 바꾼 {rname} → {(e1 == y).mean() * 100:.2f}% | 고침 {f2} | 새로 틀림 {b2} | 순이득 {n2:+d} | 답이 바뀐 문항 {D2}")
                V["ens_net"] = n2
                REF_VIEW = {"voters": voters, "ok": rep_ok}
                if V.get("hold") and rep_ok and n2 >= NET_MIN:        # 단독이 +1~+2(음수 아님)이고 기준 제출에서 +3 이상일 때만
                    AUTO_TEST_OK = True; V["verdict"] += f" / {rname} 교체 순이득 {n2:+d} ≥ +{NET_MIN} → test 채점"
say("판정: " + V["verdict"])
E1_INFO.update({f"result_{PRESET}": {k: v for k, v in V.items() if k != "lines"}})
json.dump(E1_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

# ───────────────── 12. test ─────────────────
TEST = r'''
# RUN_TEST = True     # ← 판정과 상관없이 test를 돌리려면 이 줄 맨 앞의 #을 지우고 이 셀만 다시 실행 (1~1.5시간)
DO_TEST = (RUN_TEST is True) or (RUN_TEST == "auto" and AUTO_TEST_OK)
if not DO_TEST:
    print(f"test 추론 생략 (RUN_TEST={RUN_TEST!r} | 판정: {V['verdict']})")
else:
    set_resolution(PRESET)
    pred, TEST_PATH = predict_df(test_df, f"{FINAL_TAG}_test")        # 끊겨도 1번 셀부터 다시 실행하면 남은 문항만 채점
    sub = sub_df[["id"]].merge(pred[["id", "pred"]], on="id", how="left").rename(columns={"pred": "answer"})
    assert len(sub) == len(sub_df) and list(sub.id) == list(sub_df.id), "id 순서/개수 불일치"
    assert sub.answer.notna().all(), f"예측 누락 {int(sub.answer.isna().sum())}건"
    assert set(sub.answer.unique()) <= set(LETTERS), sub.answer.unique()
    sp = os.path.join(RUN_DIR, f"submission_{RUN_NAME}_{FINAL_TAG}_{PRESET}.csv"); sub.to_csv(sp, index=False)
    print("저장:", TEST_PATH, "\n저장:", sp, "| 예측 분포:", sub.answer.value_counts().sort_index().to_dict())
    V["test_file"] = TEST_PATH
    if REF_TEST:
        _t0 = pd.read_csv(REF_TEST).drop_duplicates("id", keep="last").set_index("id").reindex(sub.id)
        if _t0.pred.notna().all():
            V["test_diff"] = int((_t0.pred.to_numpy() != sub.answer.to_numpy()).sum())
            print(f"기존(NF4) test 예측과 답이 다른 문항: {V['test_diff']}개 / {len(sub)} (공개 절반에는 대략 그 절반)")
    # 기준 제출에서 SRC_RUN만 bf16으로 바꾼 제출 후보: bf16 교체로 답이 바뀐 문항만 고치고 나머지는 기존 제출 파일 그대로 둠
    if REF_VIEW and DO_VAL:
        PT1 = _probs(_load_preds(TEST_PATH, TEST_IDS))
        _ans, _base = [], []
        for v in REF_VIEW["voters"]:
            _rec = voter_probs(v, "test").argmax(1)
            _fa = file_answers(v["file"]) if v.get("file") else None
            _b = _rec if _fa is None else _fa
            _base.append(_b)
            if has_src(v):
                _sw = voter_probs(v, "test", swap={"val": P1, "test": PT1}).argmax(1)
                _ans.append(np.where(_sw != _rec, _sw, _b))
            else:
                _ans.append(_b)
        new_t = vote(_ans)
        base_t = file_answers(ENSEMBLE_REF["file"]) if ENSEMBLE_REF.get("file") else None
        if base_t is None: base_t = vote(_base)
        cp = os.path.join(RUN_DIR, f"submission_{ENSEMBLE_REF['name']}__{RUN_NAME}.csv")
        sub_df[["id"]].assign(answer=[LETTERS[i] for i in new_t]).to_csv(cp, index=False)
        V["cand_file"], V["cand_diff"] = cp, int((new_t != base_t).sum())
        print(f"{ENSEMBLE_REF['name']}에서 {SRC_RUN}만 bf16으로 바꾼 제출 후보 저장: {cp}\n  기존 제출과 다른 답 {V['cand_diff']}개 / {len(new_t)}"
              + (" → 기존과 같으니 제출할 필요 없습니다" if V["cand_diff"] == 0 else "")
              + ("" if REF_VIEW["ok"] else "  ⚠️ 기준 제출 재현이 안 돼 참고용입니다"))
    E1_INFO.update({f"test_{PRESET}": {"file": TEST_PATH, "diff_vs_nf4": V.get("test_diff"), "candidate": V.get("cand_file"), "candidate_diff": V.get("cand_diff")}})
    json.dump(E1_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

# ───────────────── 13. 요약 ─────────────────
SUMMARY = r'''
print(f"E1 | {SRC_RUN} → {RUN_NAME} | {MODEL_ID} bf16 + adapter_ep{SRC_EP} (md5 {ADAPTER_MD5[:8]}) | {PRESET} | {GPU_NAME}")
for _l in V["lines"]: print(_l)
if V.get("test_file"): print("test 예측:", os.path.basename(V["test_file"]), "| 기존 NF4와 다른 답:", V.get("test_diff", "비교 파일 없음"))
if V.get("cand_file"): print("제출 후보:", os.path.basename(V["cand_file"]), f"| 기존 {ENSEMBLE_REF['name']} 제출과 다른 답 {V['cand_diff']}개")
if torch.cuda.is_available(): print(f"최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB / {VRAM_GB:.0f}GB")
if V.get("test_file"): print(f'\nStage 3에서 쓰려면 멤버 ("{SRC_RUN}", "{PRESET}") 대신 ("{RUN_NAME}", "{PRESET}")')
if V["verdict"].startswith("채택"):
    print("다음: 같은 노트북으로 Q38·Gemma 어댑터도 재채점 (1번 셀 SRC_RUN·MODEL_ID만 변경) → Stage 3에서 bf16 멤버로 바꾼 조합 비교")
elif V["verdict"].startswith(("보류", "불채택")):
    print("다음: 위 출력을 확인하세요 (bf16 재학습을 할지 판단)")
'''

INTRO = r'''
# E1 · 기존 NF4 어댑터를 bf16 베이스에 얹어 다시 채점 (학습 없음)

**하는 일**
- `SRC_RUN`(기존 실행)의 어댑터를 **merge 없이** bf16 베이스 모델에 붙여 검증 1,000문항을 채점합니다. 학습은 하지 않습니다.
- 기존 NF4 결과(`preds_final_ep*__up2.0__1000.csv`)와 **문항 단위로 짝지어** 고침·새로 틀림·NLL을 비교하고, 보고서의 사전 기준(순이득 ≥ +3, NLL 감소)으로 판정합니다.
- **현재 최고 제출**(공개 0.97676 = 세 앙상블의 다수결)을 검증에서 다시 만든 뒤, 그 안에서 이 멤버만 bf16으로 바꿨을 때의 변화도 셉니다. 레시피가 맞는지는 test 제출 파일을 다시 만들어 6,714문항이 모두 같은지로 확인합니다.
- 판정을 통과하면 이어서 test 6,714문항을 채점하고, 최고 제출에서 이 멤버만 바꾼 **제출 후보 파일**도 만듭니다. 멤버 예측 파일 이름은 stage2와 같은 규칙이라 Stage 3에서 멤버 이름만 `…_bf16eval`로 바꿔 써도 됩니다.

**안전장치**
- 결과는 새 폴더 `vqa_runs/<SRC_RUN>_bf16eval/`에만 씁니다. 기존 실행 폴더는 읽기만 합니다 (공용 로그 `results_log_s2.csv`에는 한 줄씩 추가).
- 본 채점 전에 확인합니다: 어댑터 파일과 모델 가중치가 값까지 같은지, 어댑터를 켜고 끄면 출력이 달라지는지, 이미지 토큰 수가 기존 결과와 같은지(같은 해상도).

**런타임**: G4(96GB) 또는 A100 80GB. bf16 27B는 약 56GB라 A100 40GB에는 올라가지 않습니다(7번 셀에서 멈춤).
**시간(G4 기준 추정)**: 모델 내려받기·로드 약 10분 + 검증 15~20분 → 판정까지 30분 안팎. test까지 채점하면 1~1.5시간이 더 듭니다(진행 중 남은 시간 표시).

**실행**: 1번 셀의 `SRC_RUN`, `MODEL_ID`만 확인하고 위에서부터 끝까지 실행하세요.
'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 라이브러리 설치 (학습 때와 같은 버전)"), ("code", S(4)),
    ("markdown", "## 3. Drive 연결 → 원본 실행 확인 → 데이터 준비\n원본 실행 폴더·어댑터·기존 결과 파일을 먼저 확인하고, 이름이 틀렸으면 압축을 풀기 전에 멈춥니다."), ("code", DATA),
    ("markdown", "## 4. 공통 import · CSV 로드"), ("code", S(8)),
    ("markdown", "## 5. 고정 검증셋 분할 (기존 결과와 같아야 비교할 수 있음)"), ("code", SPLIT),
    ("markdown", "## 6. 프롬프트 (학습 때와 동일)"), ("code", S(12)),
    ("markdown", "## 7. bf16 베이스 로드 + 기존 어댑터 부착 + 적용 확인\n양자화 없이 bf16으로 올린 뒤 어댑터를 merge 없이 얹습니다. 어댑터 파일의 LoRA 텐서가 전부 값 그대로 들어갔는지 확인합니다."), ("code", MODEL),
    ("markdown", "## 8. 채점 함수 (학습 때와 동일한 로짓 비교)\nstage2의 채점 코드를 그대로 씁니다. 학습용 이미지 증강 부분만 뺐습니다."), ("code", SCORING),
    ("markdown", "## 9. 사전 점검 (8문항)\n어댑터를 켜고 끈 출력 차이, letter_mass, 기존 결과와의 이미지 토큰 수를 확인합니다. 하나라도 이상하면 여기서 멈춥니다."), ("code", PRECHECK),
    ("markdown", "## 10. 검증 1,000문항 채점"), ("code", VALEVAL),
    ("markdown", "## 11. 기존 NF4 결과와 문항 단위 비교 → 판정\n"
                 "- **채택**: 순이득 ≥ +3 이고 NLL 감소 (보고서 E1 기준) → test 채점으로 넘어감\n"
                 "- **보류**: 순이득은 +1 이상이지만 기준 미달(+1~+2, 또는 NLL 증가) → test는 자동으로 돌리지 않음. "
                 "단, 현재 최고 제출에서 이 멤버만 바꿨을 때 순이득이 +3 이상이면(제출 재현이 확인된 경우) test를 채점\n"
                 "- **불채택**: 순이득 0 이하\n"
                 "- **확인 필요**: letter_mass가 0.01 넘게 떨어졌거나 이미지 토큰 수가 기존과 다름 → 비교가 성립하지 않으므로 판정하지 않음\n\n"
                 "앙상블 관점 줄은 두 가지를 보여 줍니다. `· <앙상블> 하나만 보면`은 이 멤버가 들어 있는 앙상블 하나의 변화이고, "
                 "마지막 줄은 다수결까지 거친 최종 제출의 변화입니다. 다수결은 나머지 두 앙상블의 답이 서로 다를 때만 바뀝니다."), ("code", COMPARE),
    ("markdown", "## 12. test 추론 → 제출 파일 (판정 통과 시)\n"
                 "멤버 단독 제출 파일과 함께, 현재 최고 제출에서 이 멤버만 bf16으로 바꾼 제출 후보(`submission_<기준 이름>__<RUN_NAME>.csv`)를 만듭니다. "
                 "bf16 교체로 답이 바뀐 문항만 고치고 나머지는 기존 제출 파일 그대로 둡니다."), ("code", TEST),
    ("markdown", "## 13. 결과 요약"), ("code", SUMMARY),
]

nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    nb.cells.append(nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text))
nbformat.validate(nb)
nbformat.write(nb, OUT_NB)
print("저장:", OUT_NB, "| 셀", len(nb.cells))

for label, a, b in SHOW_DIFF:
    print(f"\n──── stage2 대비 바뀐 줄: {label} ────")
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++")): continue
        print(line[:170])
