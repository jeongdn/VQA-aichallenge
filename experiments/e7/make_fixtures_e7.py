# -*- coding: utf-8 -*-
"""E7 시험용 Drive 구조: 멤버 5개(같은 확률 → clean4 = 설계한 확률), 설계한 불확실 문항, 제출 파일 2개."""
import os, json, shutil, hashlib
import numpy as np, pandas as pd
E1 = "/tmp/vqa_scratch/e1test"
ROOT = "/tmp/vqa_scratch/e7test"
RUNS = f"{ROOT}/vqa_runs"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(RUNS)
# ── 데이터: 실제 CSV + 작은 이미지, 작은 Qwen3.5 모델 (e7test 안에 따로 만듦) ──
import sys, torch
from PIL import Image
sys.path.insert(0, "/home/user/vqa/e1")
from tiny_qwen import tiny_config, Qwen3_5ForConditionalGeneration
U = "/tmp/vqa_uploads"
DATA = f"{ROOT}/content"; os.makedirs(DATA)
for src, dst in [("6247e8ae-train.csv", "train.csv"), ("847ba753-test.csv", "test.csv"),
                 ("eb905257-sample_submission.csv", "sample_submission.csv"), ("359d3794-dev.csv", "dev.csv")]:
    shutil.copy(f"{U}/{src}", f"{DATA}/{dst}")
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
for p_ in list(train.path) + list(test.path):
    os.makedirs(os.path.dirname(f"{DATA}/{p_}"), exist_ok=True)
    h = int(hashlib.md5(p_.encode()).hexdigest()[:6], 16)
    Image.new("RGB", (24, 18), (h % 256, (h >> 8) % 256, (h >> 16) % 256)).save(f"{DATA}/{p_}", quality=80)
torch.manual_seed(0); Qwen3_5ForConditionalGeneration(tiny_config()).save_pretrained(f"{ROOT}/tiny_qwen35")
train["answer"] = train["answer"].astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True)
assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"
y = val.answer.map("abcd".index).to_numpy(); N = len(val)
NEG = val.question.astype(str).str.contains(r"아닌|않는|없는|틀린|잘못").to_numpy()
R = np.random.RandomState(1)
nonneg = R.permutation(np.where(~NEG)[0]); negi = np.where(NEG)[0]

def row(g, top, p_top, p_sec, sec=None):
    p = np.full(4, 0.0); rest = [k for k in range(4) if k not in (top, sec if sec is not None else g)]
    p[top] = p_top
    s2 = g if sec is None else sec
    if s2 != top: p[s2] = p_sec
    left = 1 - p.sum(); p[[k for k in range(4) if p[k] == 0]] = left / sum(p == 0); return p
P = np.stack([row(y[i], y[i], 0.97, 0.01, (y[i] + 1) % 4) for i in range(N)])
sets = {}; c = 0
def take(n):
    global c; out = nonneg[c:c + n]; c += n; return out
sets["W_small"] = take(12); sets["R_small"] = take(25); sets["W_mid"] = take(3); sets["R_mid"] = take(17); sets["W_conf"] = take(7)
sets["NEG_W"], sets["NEG_R"] = negi[:1], negi[1:2]
for i in sets["W_small"]: w = (y[i] + 1) % 4; P[i] = row(y[i], w, 0.45, 0.40)                 # 차이 0.05, 기준 오답
for i in sets["R_small"]: w = (y[i] + 1) % 4; P[i] = row(y[i], y[i], 0.50, 0.40, w)           # 차이 0.10, 기준 정답
for i in sets["W_mid"]:   w = (y[i] + 1) % 4; P[i] = row(y[i], w, 0.75, 0.15)                 # 차이 0.60 (τ=0.5 밖, 0.8 안)
for i in sets["R_mid"]:   w = (y[i] + 1) % 4; P[i] = row(y[i], y[i], 0.75, 0.15, w)
for i in sets["W_conf"]:  w = (y[i] + 1) % 4; P[i] = row(y[i], w, 0.95, 0.02)                 # 확신 오답
for i in sets["NEG_W"]:   w = (y[i] + 1) % 4; P[i] = row(y[i], w, 0.45, 0.40)
for i in sets["NEG_R"]:   w = (y[i] + 1) % 4; P[i] = row(y[i], y[i], 0.50, 0.40, w)
acc = (P.argmax(1) == y).mean() * 100
NT = len(test); yt = R.randint(0, 4, NT)
PT = np.stack([row(yt[i], yt[i], 0.97, 0.01, (yt[i] + 1) % 4) for i in range(NT)])
TG = R.permutation(NT)[:300]
for i in TG: PT[i] = row(yt[i], (yt[i] + 1) % 4, 0.45, 0.40)                              # test 불확실 300문항 (차이 0.05)
def save(path, ids, Pm):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in Pm.argmax(1)], "p_a": Pm[:, 0], "p_b": Pm[:, 1], "p_c": Pm[:, 2], "p_d": Pm[:, 3],
                  "letter_mass": 0.999, "img_tokens": 4}).to_csv(path, index=False)
for run in ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]:
    os.makedirs(f"{RUNS}/{run}"); save(f"{RUNS}/{run}/preds_final_ep1__up2.0__1000.csv", val.id, P); save(f"{RUNS}/{run}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT)
ED = f"{RUNS}/ensembles"; os.makedirs(ED)
clean = PT.argmax(1)
pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in clean]}).to_csv(f"{ED}/submission_clean4_qg_hcx.csv", index=False)
vote = clean.copy(); vch = R.permutation(NT)[:15]; vote[vch] = (vote[vch] + 1) % 4
pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in vote]}).to_csv(f"{ED}/submission_majority_vote.csv", index=False)
json.dump({"sets": {k: [int(i) for i in v] for k, v in sets.items()}, "val_ids": list(val.id), "y": y.tolist(), "test_ids": list(test.id),
           "TG": [int(i) for i in TG], "yt": yt.tolist(), "clean": clean.tolist(), "vote": vote.tolist()}, open(f"{ROOT}/fixture.json", "w"))
print(f"E7 fixtures OK | 기준 검증 {acc:.2f}% | τ<0.5: {int(((np.sort(P,1)[:,-1]-np.sort(P,1)[:,-2]) < 0.5).sum())}문항 | 부정형 검증 {int(NEG.sum())}개")
