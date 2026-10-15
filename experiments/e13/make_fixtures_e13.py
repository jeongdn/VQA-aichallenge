# -*- coding: utf-8 -*-
"""E13 시험용 Drive: 4bit 멤버 5개 + bf16 멤버(Q35 좋아짐, Q38 비슷, Gemma 나빠짐, HCX 검증만 있음, GDEV 없음) + 지금 제출 파일."""
import os, json, shutil, hashlib
import numpy as np, pandas as pd
SP = "/tmp/vqa_scratch"
ROOT, SRC = f"{SP}/e13test", f"{SP}/e7test/content"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
DATA, RUNS = f"{ROOT}/content", f"{ROOT}/vqa_runs"; os.makedirs(DATA); os.makedirs(f"{RUNS}/ensembles")
for f in ["train.csv", "test.csv", "sample_submission.csv"]: shutil.copy(f"{SRC}/{f}", f"{DATA}/{f}")
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv"); train["answer"] = train.answer.astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True); assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"
y = val.answer.map("abcd".index).to_numpy(); N, NT = len(val), len(test)
R = np.random.RandomState(9); yt = R.randint(0, 4, NT); dz_v, dz_t = R.normal(6.5, 2.2, N), R.normal(6.5, 2.2, NT)
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def member(seed, yy, dz, noise=1.3, bonus=0.0):
    r = np.random.RandomState(seed); lg = r.normal(0, 1, (len(yy), 4)); lg[np.arange(len(yy)), yy] += dz + bonus + r.normal(0, noise, len(yy)); return sm(lg)
def save(path, ids, P, mass=0.998):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3], "letter_mass": mass, "img_tokens": 1200}).to_csv(path, index=False)
MEM = {"q35_27b_r16_native": "Qwen/Qwen3.5-27B", "q38_27b_r16_native": "Qwen/Qwen3.8-27B", "gemma4_31b_r16_up20_dev": "google/gemma-4-31B-it",
       "gemma4_31b_r16_up20": "google/gemma-4-31B-it", "hcx32b_r16_native": "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"}
def cfg(mid, tp="native", extra=None, full=False): return {"MODEL_ID": mid, "TRAIN_PRESET": tp, "EPOCHS": 1, "MAX_STEPS": None, "LR": 0.0001, "STEP_SAMPLES": 16, "LORA_R": 16, "LORA_ALPHA": 32,
                                                          "SHUFFLE_OPTIONS": True, "TRAIN_ON_FULL": full, "SEED": 42, "TRAIN_SEED": 42, "EXTRA_TRAIN_CSV": extra}
PV, PT = {}, {}
for s, (run, mid) in enumerate(MEM.items()):
    d = f"{RUNS}/{run}"; os.makedirs(d); tp = "up2.0" if "gemma" in run else "native"; ex = "/x/dev_pseudo.csv" if run.endswith("_dev") else None
    PV[run], PT[run] = member(100 + s, y, dz_v), member(200 + s, yt, dz_t)
    save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, PV[run]); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT[run])
    json.dump({"epoch_acc": {"1": 0.97}, "config": cfg(mid, tp, ex)}, open(f"{d}/train_log.json", "w"))
# bf16 실행: 원래 멤버와 상관된 확률(같은 레시피)에 작은 성능 차이 + 시드 추가 실행 1개 + (gbad 시나리오용) 크게 나빠진 Gemma
def bf16(run, seed, bonus, noise=0.5):
    r = np.random.RandomState(seed)
    Pv = sm(np.log(PV[run]) + r.normal(0, noise, PV[run].shape) + bonus * np.eye(4)[y]); Pt = sm(np.log(PT[run]) + r.normal(0, noise, PT[run].shape) + bonus * np.eye(4)[yt])
    return Pv, Pt
BF = [("q35_27b_r16_native", "_bf16", 701, 0.5, 42), ("q35_27b_r16_native", "_bf16_s43", 702, 0.3, 43), ("q38_27b_r16_native", "_bf16", 703, 0.1, 42),
      ("gemma4_31b_r16_up20", "_bf16", 704, 0.3, 42)]
for run, suf, sd, bonus, tseed in BF:
    d = f"{RUNS}/{run}{suf}"; os.makedirs(d); tp = "up2.0" if "gemma" in run else "native"; Pv, Pt = bf16(run, sd, bonus)
    save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, Pv); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pt)
    c = cfg(MEM[run], tp); c["TRAIN_SEED"] = tseed; json.dump({"epoch_acc": {"1": 0.97}, "config": c}, open(f"{d}/train_log.json", "w")); PV[run + suf], PT[run + suf] = Pv, Pt
Pv, Pt = bf16("gemma4_31b_r16_up20", 705, -1.6, 1.0); os.makedirs(f"{ROOT}/_alt/gemma_bad")
save(f"{ROOT}/_alt/gemma_bad/preds_final_ep1__up2.0__1000.csv", val.id, Pv); save(f"{ROOT}/_alt/gemma_bad/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pt)
os.makedirs(f"{RUNS}/q35_27b_r16_native_bf16_old"); os.makedirs(f"{RUNS}/q35_27b_r16_native_bf16x")          # 이름 규칙에 안 맞는 폴더 → 무시돼야 함
d = f"{RUNS}/hcx32b_r16_native_bf16"; os.makedirs(d); save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, member(900, y, dz_v))   # test 예측 아직 없음
json.dump({"epoch_acc": {"1": 0.97}, "config": cfg(MEM["hcx32b_r16_native"])}, open(f"{d}/train_log.json", "w"))
T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def temp(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def fitT(p): i = np.arange(N); return T70[int(np.argmin([-np.mean(np.log(temp(p, T)[i, y])) for T in T70]))]
def ens(runs, w, cal, src, vsrc): w = np.asarray(w, float) / sum(w); return sum(wi * (temp(src[r], fitT(vsrc[r])) if cal else src[r]) for wi, r in zip(w, runs)).argmax(1)
C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
c = ens(C4, [1, 1, 1, 1, 2], False, PT, PV); a = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1], False, PT, PV); b = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], True, PT, PV)
A = np.stack([a, c, b], 1); v = np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1)
for nm, x in [("clean4_qg_hcx", c), ("ens_q38_gemma4", a), ("ens_q38_gdev_cal", b), ("majority_vote", v)]:
    pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in x]}).to_csv(f"{RUNS}/ensembles/submission_{nm}.csv", index=False)
np.savez(f"{ROOT}/truth.npz", y=y, yt=yt, **{f"PV_{k}": v_ for k, v_ in PV.items()}, **{f"PT_{k}": v_ for k, v_ in PT.items()})
print("E13 fixtures OK |", {k: round(float((v_.argmax(1) == y).mean() * 100), 1) for k, v_ in PV.items()})
