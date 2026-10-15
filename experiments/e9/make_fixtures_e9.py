# -*- coding: utf-8 -*-
"""E9 시험용 Drive: 실제 CSV + 작은 이미지(E7 시험 데이터 재사용) + 멤버 5개(서로 다른 확률) + 멤버 4개의 작은 LoRA 어댑터 + 제출 파일."""
import os, sys, json, shutil, hashlib
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/user/vqa/e1")
from tiny_qwen import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
SP = "/tmp/vqa_scratch"
ROOT, SRC = f"{SP}/e9test", f"{SP}/e7test"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(ROOT)
os.symlink(f"{SRC}/content", f"{ROOT}/content")                 # CSV + 작은 이미지 (읽기만 함)
shutil.copytree(f"{SRC}/tiny_qwen35", f"{ROOT}/tiny_qwen35")
RUNS = f"{ROOT}/vqa_runs"; os.makedirs(f"{RUNS}/ensembles")
train = pd.read_csv(f"{ROOT}/content/train.csv"); test = pd.read_csv(f"{ROOT}/content/test.csv")
train["answer"] = train.answer.astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True)
assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"
y = val.answer.map("abcd".index).to_numpy(); N, NT = len(val), len(test)
R = np.random.RandomState(11); yt = R.randint(0, 4, NT)
dz_v, dz_t = R.normal(6.5, 2.2, N), R.normal(6.5, 2.2, NT)
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def member(seed, yy, dz):
    r = np.random.RandomState(seed); lg = r.normal(0, 1, (len(yy), 4)); lg[np.arange(len(yy)), yy] += dz + r.normal(0, 1.4, len(yy)); return lg
MEM = {"q35_27b_r16_native": "Qwen/Qwen3.5-27B", "q38_27b_r16_native": "Qwen/Qwen3.8-27B", "gemma4_31b_r16_up20_dev": "google/gemma-4-31B-it",
       "gemma4_31b_r16_up20": "google/gemma-4-31B-it", "hcx32b_r16_native": "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"}
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
def make_adapter(path, base_name, seed):
    b = Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", dtype=torch.float32); torch.manual_seed(seed)
    pm = get_peft_model(b, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.5)
    pm.save_pretrained(path)
    c = json.load(open(f"{path}/adapter_config.json")); c["base_model_name_or_path"] = base_name; json.dump(c, open(f"{path}/adapter_config.json", "w"), indent=2)
def save(path, ids, P):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3],
                  "letter_mass": 0.998, "img_tokens": 4}).to_csv(path, index=False)
PV, PT = {}, {}
for s, (run, mid) in enumerate(MEM.items()):
    d = f"{RUNS}/{run}"; os.makedirs(d)
    PV[run], PT[run] = sm(member(100 + s, y, dz_v)), sm(member(200 + s, yt, dz_t))
    save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, PV[run]); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT[run])
    json.dump({"epoch_acc": {"1": float((PV[run].argmax(1) == y).mean())}, "history": [],
               "config": {"MODEL_ID": mid, "TRAIN_PRESET": "native", "EPOCHS": 1, "TRAIN_ON_FULL": False}}, open(f"{d}/train_log.json", "w"))
    if "hcx" not in run: make_adapter(f"{d}/adapter_ep1", mid, 10 + s)
# 예전 TTA 기록 (stage2 결과 로그 형식)
pd.DataFrame([{"time": "09-22 10:00", "run": "q35_9b_r16_native/final_ep1", "model": "Qwen/Qwen3.5-9B", "res": "up1.5", "tta": 0, "n": 1000, "acc": 96.4},
              {"time": "09-22 10:40", "run": "q35_9b_r16_native/final_ep1", "model": "Qwen/Qwen3.5-9B", "res": "up1.5", "tta": 1, "n": 1000, "acc": 96.5},
              {"time": "09-23 11:00", "run": "q3vl_8b_r16_native/final_ep1", "model": "Qwen/Qwen3-VL-8B", "res": "native", "tta": 0, "n": 1000, "acc": 95.1}]
             ).to_csv(f"{RUNS}/results_log_s2.csv", index=False)
T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def temp(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def fitT(p): i = np.arange(len(y)); return T70[int(np.argmin([-np.mean(np.log(temp(p, T)[i, y])) for T in T70]))]
def ens(runs, w, cal):
    w = np.asarray(w, float) / sum(w); return sum(wi * (temp(PT[r], fitT(PV[r])) if cal else PT[r]) for wi, r in zip(w, runs)).argmax(1)
C4 = list(MEM)[:2] + ["gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
clean = ens(["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"], [1, 1, 1, 1, 2], False)
qg = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1], False); cal = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], True)
A = np.stack([qg, clean, cal], 1); vote = np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1)
for nm, a in [("clean4_qg_hcx", clean), ("ens_q38_gemma4", qg), ("ens_q38_gdev_cal", cal), ("majority_vote", vote)]:
    pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in a]}).to_csv(f"{RUNS}/ensembles/submission_{nm}.csv", index=False)
np.savez(f"{ROOT}/truth.npz", y=y, yt=yt, **{f"PV_{k}": v for k, v in PV.items()}, **{f"PT_{k}": v for k, v in PT.items()})
print("E9 fixtures OK |", {k.split('_')[0]: round(float((v.argmax(1) == y).mean() * 100), 1) for k, v in PV.items()})
