# -*- coding: utf-8 -*-
"""E10 시험용 Drive: 실제 CSV + 멤버 8개(서로 다른 성능·상관) + 제외돼야 할 실행들 + 지금 제출 파일."""
import os, json, shutil, hashlib
import numpy as np, pandas as pd
SP = "/tmp/vqa_scratch"
ROOT, SRC = f"{SP}/e10test", f"{SP}/e7test/content"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
DATA, RUNS = f"{ROOT}/content", f"{ROOT}/vqa_runs"; os.makedirs(DATA); os.makedirs(f"{RUNS}/ensembles")
for f in ["train.csv", "test.csv", "sample_submission.csv"]: shutil.copy(f"{SRC}/{f}", f"{DATA}/{f}")
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
train["answer"] = train.answer.astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True)
assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"
y = val.answer.map("abcd".index).to_numpy(); N, NT = len(val), len(test)
R = np.random.RandomState(5); yt = R.randint(0, 4, NT)
dz_v, dz_t = R.normal(6.5, 2.3, N), R.normal(6.5, 2.3, NT)            # 문항 난이도 (멤버끼리 공유 → 오답이 겹침)
alt_v, alt_t = R.randint(1, 4, N), R.randint(1, 4, NT)                  # 어려운 문항에서 멤버들이 같이 끌리는 오답
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def member(seed, yy, dz, alt, noise, shared, temp=1.0):
    r = np.random.RandomState(seed); lg = r.normal(0, 1, (len(yy), 4)); ii = np.arange(len(yy))
    lg[ii, yy] += dz * shared + r.normal(0, noise, len(yy)) + (1 - shared) * 6.0
    lg[ii, (yy + alt) % 4] += np.clip(4.0 - dz, 0, None) * shared            # 공유된 함정 보기
    return sm(lg / temp)
def save(path, ids, P, tok=1200):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3],
                  "letter_mass": 0.998, "img_tokens": tok}).to_csv(path, index=False)
def logcfg(mid, full=False): return {"epoch_acc": {"1": 0.97}, "history": [], "config": {"MODEL_ID": mid, "TRAIN_PRESET": "native", "EPOCHS": 1, "TRAIN_ON_FULL": full}}
SPEC = {  # run: (모델, {해상도: (seed, noise, shared, temp)})
    "q35_27b_r16_native":      ("Qwen/Qwen3.5-27B", {"up2.0": (11, 1.3, 1.0, 1.0)}),
    "q38_27b_r16_native":      ("Qwen/Qwen3.8-27B", {"up2.0": (12, 1.4, 1.0, 1.3)}),
    "gemma4_31b_r16_up20_dev": ("google/gemma-4-31B-it", {"up2.0": (13, 1.3, 1.0, 0.8)}),
    "gemma4_31b_r16_up20":     ("google/gemma-4-31B-it", {"up2.0": (14, 1.3, 1.0, 1.0)}),
    "hcx32b_r16_native":       ("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", {"up2.0": (15, 1.4, 1.0, 1.0)}),
    "ivl35_38b_r16_native":    ("OpenGVLab/InternVL3_5-38B-HF", {"up2.0": (16, 2.2, 0.55, 1.0)}),          # 다른 계열: 오답이 덜 겹침
    "q3vl_8b_r16_native":      ("Qwen/Qwen3-VL-8B-Instruct", {"native": (17, 1.8, 1.0, 1.0), "up2.0": (18, 1.7, 1.0, 1.0)}),
    "q35_9b_r16_native":       ("Qwen/Qwen3.5-9B", {"up1.5": (19, 1.9, 1.0, 1.0)}),
}
PV, PT = {}, {}
for run, (mid, pres) in SPEC.items():
    d = f"{RUNS}/{run}"; os.makedirs(d); json.dump(logcfg(mid), open(f"{d}/train_log.json", "w"))
    for pre, (sd, noise, shared, temp) in pres.items():
        Pv, Pt = member(sd, y, dz_v, alt_v, noise, shared, temp), member(sd + 100, yt, dz_t, alt_t, noise, shared, temp)
        save(f"{d}/preds_final_ep1__{pre}__1000.csv", val.id, Pv); save(f"{d}/preds_final_ep1_test__{pre}__{NT}.csv", test.id, Pt)
        if pre == ("up2.0" if "up2.0" in pres else pre): PV[run], PT[run] = Pv, Pt
    if run == "q35_27b_r16_native":                                   # TTA 평가 파일(해상도 이름에 __tta4)도 있음 → up2.0이 우선
        save(f"{d}/preds_final_ep1__up2.0__tta4__1000.csv", val.id, PV[run]); save(f"{d}/preds_final_ep1_test__up2.0__tta4__{NT}.csv", test.id, PT[run])
# ── 어댑터 파일 (내용이 다르면 다른 모델) + test 예측이 없어 빠질 실행 ──
def adapter(run, seed):
    a = f"{RUNS}/{run}/adapter_ep1"; os.makedirs(a, exist_ok=True); open(f"{a}/adapter_model.safetensors", "wb").write(np.random.RandomState(seed).bytes(2 << 20))
for s_, run in enumerate(SPEC): adapter(run, 500 + s_)
VO = {}
for run, sd, ad, part in [("q35_27b_r16_e4ctrl_native", 40, 900, False), ("q35_27b_r16_native_bf16eval", 41, None, False),
                          ("q35_27b_r16_native_s050", 42, "dup", False), ("q35_27b_r16_review1_up20", 43, 901, True)]:
    d = f"{RUNS}/{run}"; os.makedirs(d); json.dump(logcfg("Qwen/Qwen3.5-27B"), open(f"{d}/train_log.json", "w"))
    VO[run] = member(sd, y, dz_v, alt_v, 1.3, 1.0); save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, VO[run])
    if part: save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id[:1600], member(sd + 100, yt, dz_t, alt_t, 1.3, 1.0)[:1600])
    if ad == "dup": os.makedirs(f"{d}/adapter_ep1"); shutil.copy(f"{RUNS}/q35_27b_r16_native/adapter_ep1/adapter_model.safetensors", f"{d}/adapter_ep1/")
    elif ad: adapter(run, ad)
# ── 제외돼야 할 실행 ──
d = f"{RUNS}/q3vl_32b_r16_native_full"; os.makedirs(d); json.dump(logcfg("Qwen/Qwen3-VL-32B-Instruct", True), open(f"{d}/train_log.json", "w"))
save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, member(30, y, dz_v + 4, alt_v, 0.5, 1.0)); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, member(31, yt, dz_t, alt_t, 1.3, 1.0))
d = f"{RUNS}/broken_run"; os.makedirs(d); open(f"{d}/train_log.json", "w").write("{broken json")
save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, member(32, y, dz_v, alt_v, 1.3, 1.0))                 # 검증만 있음
d = f"{RUNS}/othersplit_run"; os.makedirs(d); json.dump(logcfg("Qwen/Qwen3.5-9B"), open(f"{d}/train_log.json", "w"))
oth = train.drop(val.index if False else train.index[train.id.isin(val.id)]).head(1000)                   # 다른 문항 1,000개
save(f"{d}/preds_final_ep1__up2.0__1000.csv", oth.id, member(33, oth.answer.map("abcd".index).to_numpy(), np.full(1000, 6.0), np.ones(1000, int), 1.3, 1.0))
save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, member(34, yt, dz_t, alt_t, 1.3, 1.0))
d = f"{RUNS}/lowacc_run"; os.makedirs(d); json.dump(logcfg("Qwen/Qwen2-VL-2B"), open(f"{d}/train_log.json", "w"))
save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, member(35, y, dz_v - 5, alt_v, 3.0, 1.0)); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, member(36, yt, dz_t - 5, alt_t, 3.0, 1.0))
d = f"{RUNS}/q35_27b_r16_native_dry"; os.makedirs(d); save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, PV["q35_27b_r16_native"])
os.makedirs(f"{RUNS}/e9_tta/q35_27b_r16_native"); open(f"{RUNS}/e9_tta/q35_27b_r16_native/perm_val__up2.0.csv", "w").write("id\n")
# ── 지금 제출 파일 (실제 레시피로) ──
T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def temp(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def fitT(p): i = np.arange(N); return T70[int(np.argmin([-np.mean(np.log(temp(p, T)[i, y])) for T in T70]))]
def ens(runs, w, cal, split):
    w = np.asarray(w, float) / sum(w); src = PV if split == "val" else PT
    return sum(wi * (temp(src[r], fitT(PV[r])) if cal else src[r]) for wi, r in zip(w, runs)).argmax(1)
C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
out = {}
for sp in ("val", "test"):
    c = ens(C4, [1, 1, 1, 1, 2], False, sp); a = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1], False, sp)
    b = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], True, sp)
    A = np.stack([a, c, b], 1); v = np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1); out[sp] = dict(clean=c, qg=a, cal=b, vote=v)
for nm, k in [("clean4_qg_hcx", "clean"), ("ens_q38_gemma4", "qg"), ("ens_q38_gdev_cal", "cal"), ("majority_vote", "vote")]:
    pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in out["test"][k]]}).to_csv(f"{RUNS}/ensembles/submission_{nm}.csv", index=False)
np.savez(f"{ROOT}/truth.npz", y=y, **{f"VO_{k}": v for k, v in VO.items()}, **{f"PV_{k}": v for k, v in PV.items()}, **{f"PT_{k}": v for k, v in PT.items()}, vote_val=out["val"]["vote"], vote_test=out["test"]["vote"])
print("E10 fixtures OK |", {k.split('_')[0] + ('dev' if k.endswith('_dev') else ''): round(float((v.argmax(1) == y).mean() * 100), 1) for k, v in PV.items()},
      "| 다수결 검증", round(float((out['val']['vote'] == y).mean() * 100), 2), "| clean4 검증", round(float((out['val']['clean'] == y).mean() * 100), 2))
