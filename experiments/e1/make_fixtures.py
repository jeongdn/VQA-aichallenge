# -*- coding: utf-8 -*-
"""노트북 시험용 가짜 Drive 구조를 만든다 (실제 train/test CSV + 작은 이미지 + 실제 PEFT 어댑터 + 합성 예측 파일)."""
import os, sys, json, shutil, hashlib
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, "/home/user/vqa/e1")
from tiny_qwen import tiny_config, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

U = "/tmp/vqa_uploads"
ROOT = "/tmp/vqa_scratch/e1test"
DATA, RUNS, TINY = f"{ROOT}/content", f"{ROOT}/vqa_runs", f"{ROOT}/tiny_qwen35"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(DATA); os.makedirs(RUNS)

# ── 데이터: 실제 CSV + 작은 이미지 ──
for src, dst in [("6247e8ae-train.csv", "train.csv"), ("847ba753-test.csv", "test.csv"),
                 ("eb905257-sample_submission.csv", "sample_submission.csv"), ("359d3794-dev.csv", "dev.csv")]:
    shutil.copy(f"{U}/{src}", f"{DATA}/{dst}")
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
for p in list(train.path) + list(test.path):
    os.makedirs(os.path.dirname(f"{DATA}/{p}"), exist_ok=True)
    h = int(hashlib.md5(p.encode()).hexdigest()[:6], 16)
    Image.new("RGB", (24, 18), (h % 256, (h >> 8) % 256, (h >> 16) % 256)).save(f"{DATA}/{p}", quality=80)

# ── 검증 분할 (노트북과 같은 방식) ──
train["answer"] = train["answer"].astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True)
y = val.answer.map("abcd".index).to_numpy(); N = len(val)
assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"

# ── 작은 Qwen3.5 베이스 + stage2 방식 LoRA 어댑터 ──
torch.manual_seed(0)
Qwen3_5ForConditionalGeneration(tiny_config()).save_pretrained(TINY)
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"   # stage2와 동일
def make_adapter(path, zero_b=False, seed=1):
    base = Qwen3_5ForConditionalGeneration.from_pretrained(TINY, dtype=torch.float32)
    torch.manual_seed(seed)
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS))
    if not zero_b:
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n: p.normal_(0, 0.5)          # 학습된 것처럼 lora_B를 0이 아니게
    pm.save_pretrained(path)
    cfg = json.load(open(f"{path}/adapter_config.json")); cfg["base_model_name_or_path"] = "Qwen/Qwen3.5-27B"
    json.dump(cfg, open(f"{path}/adapter_config.json", "w"), indent=2)
    n_vis = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A") and "visual" in n)
    return sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A")), n_vis

# ── 합성 확률 ──
R = np.random.RandomState(0)
def prof(g, correct, conf=0.9):
    p = R.dirichlet([1, 1, 1, 1]) * (1 - conf)
    if correct: p[g] += conf
    else:
        w = R.choice([k for k in range(4) if k != g]); p = R.dirichlet([1, 1, 1, 1]) * 0.25; p[w] += 0.6; p[g] += 0.15
    return p / p.sum()
def probs_for(wrong_idx, n=N, yy=y):
    ws = set(wrong_idx); return np.stack([prof(yy[i], i not in ws) for i in range(n)])
def save_preds(path, ids, P, ntok=4, mass=0.999):
    pd.DataFrame({"id": ids, "pred": [ "abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3],
                  "letter_mass": mass, "img_tokens": ntok}).to_csv(path, index=False)

perm = R.permutation(N)
W0 = perm[:28]; S = W0[:4]; VV = W0[4:8]          # NF4 Q35 오답 28개. S: clean4 안에서만 뒤집힘(다수결은 그대로) / VV: 다수결까지 뒤집힘
others = {"q38_27b_r16_native": perm[20:50], "gemma4_31b_r16_up20_dev": perm[60:85],
          "gemma4_31b_r16_up20": perm[90:117], "hcx32b_r16_native": perm[120:146]}
P0 = probs_for(W0)
MEM = {k: probs_for(v) for k, v in others.items()}
for i in S:                                            # clean4는 근소하게 틀리지만 다른 두 앙상블은 맞히는 문항
    g_, w_ = y[i], (y[i] + 1) % 4
    P0[i] = 0.025; P0[i, w_] = 0.9; P0[i, g_] = 0.05
    for k in MEM: MEM[k][i] = 0.0; MEM[k][i, g_] = 0.55; MEM[k][i, w_] = 0.45
for i in VV:                                           # 세 앙상블이 w·w·g로 갈려 다수결이 틀리고, clean4가 뒤집히면 다수결도 뒤집히는 문항
    g_, w_ = y[i], (y[i] + 1) % 4
    P0[i] = 0.025; P0[i, w_] = 0.9; P0[i, g_] = 0.05
    for k, (pg, pw) in {"q38_27b_r16_native": (0.49, 0.51), "gemma4_31b_r16_up20": (0.2, 0.8),
                        "gemma4_31b_r16_up20_dev": (0.8, 0.2), "hcx32b_r16_native": (0.5, 0.5)}.items():
        MEM[k][i] = 0.0; MEM[k][i, g_] = pg; MEM[k][i, w_] = pw

NT = len(test); yt = R.randint(0, 4, NT); tperm = R.permutation(NT)
PT0 = probs_for(tperm[:200], n=NT, yy=yt)
MEMT = {k: probs_for(tperm[j * 150: j * 150 + 180], n=NT, yy=yt) for j, k in enumerate(others)}

src = f"{RUNS}/q35_27b_r16_native"; os.makedirs(src)
n_lora, n_vis = make_adapter(f"{src}/adapter_ep1")
json.dump({"epoch_acc": {"1": 0.972}, "history": [], "config": {"MODEL_ID": "Qwen/Qwen3.5-27B", "TRAIN_PRESET": "native",
           "EPOCHS": 1, "TRAIN_ON_FULL": False}}, open(f"{src}/train_log.json", "w"))
save_preds(f"{src}/preds_final_ep1__up2.0__1000.csv", val.id, P0)
save_preds(f"{src}/preds_final_ep1__up1.5__1000.csv", val.id, probs_for(perm[:40]))       # 다른 해상도 (골라지면 안 됨)
save_preds(f"{src}/preds_ep1__native__1000.csv", val.id, probs_for(perm[:40]))            # epoch 끝 평가 (골라지면 안 됨)
save_preds(f"{src}/preds_final_ep1__up2.0__tta4__1000.csv", val.id, probs_for(perm[:40])) # TTA 파일 (골라지면 안 됨)
save_preds(f"{src}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT0)
os.makedirs(f"{src}/ckpt_last/adapter")
for k, P in MEM.items():
    d = f"{RUNS}/{k}"; os.makedirs(d)
    save_preds(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, P)
    save_preds(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, MEMT[k])
# hcx에는 epoch 파일이 두 개 → train_log의 최고 epoch(1)를 골라야 함 (test도 ep1 것을 써야 함)
save_preds(f"{RUNS}/hcx32b_r16_native/preds_final_ep2__up2.0__1000.csv", val.id, probs_for(perm[:300]))
save_preds(f"{RUNS}/hcx32b_r16_native/preds_final_ep2_test__up2.0__{NT}.csv", test.id, probs_for(tperm[:900], n=NT, yy=yt))
json.dump({"epoch_acc": {"1": 0.975, "2": 0.970}}, open(f"{RUNS}/hcx32b_r16_native/train_log.json", "w"))
# q38도 원본 실행으로 쓸 수 있게 (세 앙상블 모두에 들어 있는 멤버)
q38 = f"{RUNS}/q38_27b_r16_native"
shutil.copytree(f"{src}/adapter_ep1", f"{q38}/adapter_ep1")
_c = json.load(open(f"{q38}/adapter_ep1/adapter_config.json")); _c["base_model_name_or_path"] = "Qwen/Qwen3.8-27B"
json.dump(_c, open(f"{q38}/adapter_ep1/adapter_config.json", "w"), indent=2)
json.dump({"epoch_acc": {"1": 0.968}, "config": {"MODEL_ID": "Qwen/Qwen3.8-27B", "TRAIN_PRESET": "native", "TRAIN_ON_FULL": False}}, open(f"{q38}/train_log.json", "w"))

# 전체 데이터 학습 실행 (검증 비교 생략·test만)
full = f"{RUNS}/q35_27b_r16_native_full"; os.makedirs(full)
shutil.copytree(f"{src}/adapter_ep1", f"{full}/adapter_ep1")
json.dump({"epoch_acc": {"1": 0.976}, "config": {"MODEL_ID": "Qwen/Qwen3.5-27B", "TRAIN_ON_FULL": True}}, open(f"{full}/train_log.json", "w"))
save_preds(f"{full}/preds_final_ep1__up2.0__1000.csv", val.id, probs_for(perm[:24]))
save_preds(f"{full}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT0)
# 기준 검증 파일이 없는 실행
noref = f"{RUNS}/q35_noref"; os.makedirs(noref)
shutil.copytree(f"{src}/adapter_ep1", f"{noref}/adapter_ep1")
save_preds(f"{noref}/preds_final_ep1__up1.5__1000.csv", val.id, probs_for(perm[:30]))
# lora_B가 전부 0인 어댑터 (단위 시험용)
make_adapter(f"{ROOT}/adapter_zeroB", zero_b=True)

# ── 앙상블 제출 파일: Stage 3 방식(가중 평균, _cal은 멤버별 온도 보정) + 팀 다수결 코드 그대로 ──
def temp_scale(p, T):
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def fit_T(p, yy):
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70)); idx = np.arange(len(yy))
    return float(grid[int(np.argmin([-np.mean(np.log(temp_scale(p, T)[idx, yy])) for T in grid]))])
VALP = {"q35_27b_r16_native": P0, **MEM}; TESTP = {"q35_27b_r16_native": PT0, **MEMT}
RECIPES = {"ens_q38_gemma4": (["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1], False),
           "clean4_qg_hcx": (["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"], [1, 1, 1, 1, 2], False),
           "ens_q38_gdev_cal": (["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], True)}
def ens(names, w, cal, which):
    Ps = []
    for n in names:
        P = (VALP if which == "val" else TESTP)[n]
        if cal: P = temp_scale(P, fit_T(VALP[n], y))
        Ps.append(P)
    return sum(wi * Pi for wi, Pi in zip(w, Ps)) / sum(w)
ED = f"{RUNS}/ensembles"; os.makedirs(ED)
for name, (names, w, cal) in RECIPES.items():
    pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in ens(names, w, cal, "test").argmax(1)]}).to_csv(f"{ED}/submission_{name}.csv", index=False)
def user_majority(E, files, out):                      # 팀이 실행한 다수결 코드와 같은 방식
    preds = [pd.read_csv(os.path.join(E, f)).set_index("id")["answer"] for f in files]
    mv = pd.concat(preds, axis=1).mode(axis=1)[0]
    pd.DataFrame({"id": mv.index, "answer": mv.values}).to_csv(os.path.join(E, out), index=False)
FILES = ["submission_ens_q38_gemma4.csv", "submission_clean4_qg_hcx.csv", "submission_ens_q38_gdev_cal.csv"]
user_majority(ED, FILES, "submission_majority_vote.csv")
# 레시피와 다른 제출 파일(누군가 덮어쓴 경우): ens_q38_gemma4만 20문항 바꾸고 다수결도 그 파일로 다시 만듦
EB = f"{RUNS}/ensembles_bad"; shutil.copytree(ED, EB)
bad = pd.read_csv(f"{EB}/submission_ens_q38_gemma4.csv")
bad.loc[:19, "answer"] = bad.loc[:19, "answer"].map({"a": "b", "b": "c", "c": "d", "d": "a"})
bad.to_csv(f"{EB}/submission_ens_q38_gemma4.csv", index=False)
user_majority(EB, FILES, "submission_majority_vote.csv")

W = [1, 1, 1, 1, 2]; order = ["q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
E = (P0 + sum(w * MEM[k] for w, k in zip(W[1:], order))) / sum(W)
ens_acc = float((E.argmax(1) == y).mean() * 100)
vote_val = pd.DataFrame({k: ["abcd"[i] for i in ens(*RECIPES[k], "val").argmax(1)] for k in RECIPES}).mode(axis=1)[0].map("abcd".index).to_numpy()
# 다른 멤버가 모두 맞히는 문항 (여기서 Q35 하나가 바뀌어도 어떤 앙상블도 그대로)
safe_R = [i for i in perm[146:] if all(MEM[k][i].argmax() == y[i] for k in MEM)]
safe_W = [i for i in W0[8:] if E[i].argmax() == y[i] and vote_val[i] == y[i]]
json.dump({"val_ids": list(val.id), "y": y.tolist(), "P0": P0.tolist(), "W0": W0.tolist(), "S": S.tolist(), "VV": VV.tolist(),
           "safe_R": [int(i) for i in safe_R[:50]], "safe_W": [int(i) for i in safe_W], "ens_acc": ens_acc,
           "vote_acc": float((vote_val == y).mean() * 100), "PT0": PT0.tolist(), "test_ids": list(test.id)}, open(f"{ROOT}/fixture.json", "w"))
print(f"fixtures OK | LoRA 층 {n_lora}개 (비전 {n_vis}) | clean4 검증 {ens_acc:.2f}% | 다수결 검증 {(vote_val == y).mean() * 100:.2f}% | "
      f"safe_W {len(safe_W)} safe_R {len(safe_R)}")
