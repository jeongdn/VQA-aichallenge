# -*- coding: utf-8 -*-
"""E8 앙상블 노트북 시험용 Drive: 서로 다른 멤버 5개 + 제출 파일 + (시나리오별) 전체 데이터 멤버.
사용: python3 make_fixtures_e8.py <scenario>
  base       : 모든 제출 파일이 선언 레시피로 재현됨, 전체 데이터 멤버 없음
  overwrite  : ens_q38_gemma4 파일이 실제로는 [Q38, GDEV]로 만들어짐 (다른 멤버 조합)
  corrupt    : ens_q38_gemma4 파일의 답 20개가 레시피와 다름 (재현 불가) → 다수결도 그 파일로 만듦
  calvar     : ens_q38_gdev_cal이 로그확률 평균 + T28 온도로 만들어짐
  full       : base + 전체 데이터 멤버 (Q38 정상, HCX 정상, G 고장(30% 무작위), GDEV 설정 다름, Q35 추론 전)
"""
import os, sys, json, shutil, hashlib
import numpy as np, pandas as pd
scen = sys.argv[1] if len(sys.argv) > 1 else "base"
ROOT = "/tmp/vqa_scratch/e8test"
SRC = "/tmp/vqa_scratch/e7test/content"
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
R = np.random.RandomState(7); yt = R.randint(0, 4, NT)
dz_v, dz_t = R.normal(5.0, 2.0, N), R.normal(5.0, 2.0, NT)          # 문항 난이도 (멤버끼리 공유 → 오답이 겹침)
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def member(seed, yy, dz, noise=1.3):
    r = np.random.RandomState(seed); lg = r.normal(0, 1, (len(yy), 4)); lg[np.arange(len(yy)), yy] += dz + r.normal(0, noise, len(yy)); return lg
MEM = {"q35_27b_r16_native": ("Qwen/Qwen3.5-27B", "native", None), "q38_27b_r16_native": ("Qwen/Qwen3.8-27B", "native", None),
       "gemma4_31b_r16_up20_dev": ("google/gemma-4-31B-it", "up2.0", "/content/drive/MyDrive/vqa_runs/stage5/dev_pseudo.csv"),
       "gemma4_31b_r16_up20": ("google/gemma-4-31B-it", "up2.0", None), "hcx32b_r16_native": ("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", "native", None)}
def cfg(mid, tp, extra, full=False, **kw):
    c = {"MODEL_ID": mid, "TRAIN_PRESET": tp, "EPOCHS": 1, "MAX_STEPS": None, "LR": 0.0001, "STEP_SAMPLES": 16, "LORA_R": 16, "LORA_ALPHA": 32,
         "SHUFFLE_OPTIONS": True, "TRAIN_ON_FULL": full, "SEED": 42, "TRAIN_SEED": 42, "EXTRA_TRAIN_CSV": extra, "IMG_AUG": False, "TRAIN_PRESETS": None}
    c.update(kw); return c
def save(path, ids, P):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3],
                  "letter_mass": 0.998, "img_tokens": 1200}).to_csv(path, index=False)
PV, PT, LT = {}, {}, {}
for s, (run, (mid, tp, extra)) in enumerate(MEM.items()):
    d = f"{RUNS}/{run}"; os.makedirs(d)
    PV[run] = sm(member(100 + s, y, dz_v)); LT[run] = member(200 + s, yt, dz_t); PT[run] = sm(LT[run])
    save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, PV[run]); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT[run])
    save(f"{d}/preds_final_ep1__native__1000.csv", val.id, sm(member(300 + s, y, dz_v)))            # 다른 해상도 파일 (무시돼야 함)
    _c0 = cfg(mid, tp, extra); _c0.pop("IMG_AUG"); _c0.pop("TRAIN_SEED")          # 예전 기록에는 이 두 항목이 없음 (실제와 같게)
    json.dump({"epoch_acc": {"1": float((PV[run].argmax(1) == y).mean())}, "history": [], "config": _c0}, open(f"{d}/train_log.json", "w"))
# q3vl_32b: clean4에 없는 추가 후보 멤버
d = f"{RUNS}/q3vl_32b_r16_native"; os.makedirs(d)
PV_q3, PT_q3 = sm(member(150, y, dz_v)), sm(member(250, yt, dz_t))
_w4 = np.array([1, 1, 1, 1, 2], float) / 6
_C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
if scen == "full_add":
    _a4 = sum(wi * PV[r] for wi, r in zip(_w4, _C4)).argmax(1)
    for i in np.where(_a4 != y)[0]: PV_q3[i] = np.full(4, 0.001); PV_q3[i, y[i]] = 0.997
save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, PV_q3); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, PT_q3)
json.dump({"epoch_acc": {"1": float((PV_q3.argmax(1) == y).mean())}, "config": {k: v for k, v in cfg("Qwen/Qwen3-VL-32B-Instruct", "native", None).items() if k not in ("IMG_AUG", "TRAIN_SEED")}},
          open(f"{d}/train_log.json", "w"))
os.makedirs(f"{RUNS}/unrelated_run"); open(f"{RUNS}/unrelated_run/train_log.json", "w").write("{broken json")
# ── 제출 파일 ──
def temp(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def fitT(p, grid): i = np.arange(len(y)); return grid[int(np.argmin([-np.mean(np.log(temp(p, T)[i, y])) for T in grid]))]
G70, G28 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70)), np.linspace(0.3, 3.0, 28)
def ens(runs, w, how="prob", grid=None):
    w = np.asarray(w, float) / sum(w); out = 0
    for wi, r in zip(w, runs):
        P = PT[r] if grid is None else temp(PT[r], fitT(PV[r], grid))
        out = out + wi * (P if how == "prob" else np.log(np.clip(P, 1e-12, 1)))
    return out.argmax(1)
C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
clean = ens(C4, [1, 1, 1, 1, 2])
qg = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"] if scen == "overwrite" else ["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1])
if scen == "corrupt":
    fl = R.permutation(NT)[:20]; qg[fl] = (qg[fl] + 1) % 4
cal = ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], "log", G28) if scen == "calvar" else ens(["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], "prob", G70)
A = np.stack([qg, clean, cal], 1); vote = np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1)
for nm, a in [("clean4_qg_hcx", clean), ("ens_q38_gemma4", qg), ("ens_q38_gdev_cal", cal), ("majority_vote", vote)]:
    pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in a]}).to_csv(f"{RUNS}/ensembles/submission_{nm}.csv", index=False)
# 참고: 팀의 pandas 다수결과 같은지
_df = pd.DataFrame({k: ["abcd"[i] for i in a] for k, a in [("x", qg), ("y", clean), ("z", cal)]})
assert (_df.mode(axis=1)[0].map("abcd".index).to_numpy() == vote).all()
FULLS = {}
if scen in ("full", "full_add"):
    MEM["q3vl_32b_r16_native"] = ("Qwen/Qwen3-VL-32B-Instruct", "native", None); LT["q3vl_32b_r16_native"] = np.log(PT_q3)
    for s, run in enumerate(["q38_27b_r16_native", "hcx32b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "q35_27b_r16_native", "q3vl_32b_r16_native"]):
        mid, tp, extra = MEM[run]; d = f"{RUNS}/{run}_full"; os.makedirs(d)
        r = np.random.RandomState(500 + s); Pf = sm(LT[run] + r.normal(0, 0.6, LT[run].shape))            # 원래 멤버와 대부분 같은 답
        if run == "gemma4_31b_r16_up20":                                                                  # 고장: 10% 무작위
            bad = r.permutation(NT)[: NT * 3 // 10]; Pf[bad] = sm(r.normal(0, 1, (len(bad), 4)))
        Vf = sm(member(600 + s, y, dz_v + 4.0, 0.5))                                                      # 학습에 쓴 검증 → 높음
        kw = {"LR": 0.0002} if run == "gemma4_31b_r16_up20_dev" else {}
        json.dump({"epoch_acc": {"1": float((Vf.argmax(1) == y).mean())}, "history": [], "config": cfg(mid, tp, extra, True, **kw)}, open(f"{d}/train_log.json", "w"))
        save(f"{d}/preds_final_ep1__up2.0__1000.csv", val.id, Vf)
        if run != "q35_27b_r16_native": save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pf); FULLS[run] = Pf
PT["q3vl_32b_r16_native"] = PT_q3
np.savez(f"{ROOT}/truth.npz", clean=clean, qg=qg, cal=cal, vote=vote, **{f"PT_{k}": v for k, v in PT.items()}, **{f"PF_{k}": v for k, v in FULLS.items()})
accs = {k.split("_")[0] + ("dev" if k.endswith("_dev") else ""): round(float((v.argmax(1) == y).mean() * 100), 1) for k, v in PV.items()}
w = np.array([1, 1, 1, 1, 2], float) / 6
print(f"[{scen}] 멤버 검증 {accs} | clean4 검증 {float((sum(wi * PV[r] for wi, r in zip(w, C4)).argmax(1) == y).mean() * 100):.2f}% | "
      f"다수결≠clean4 {int((vote != clean).sum())} | 전체 데이터 {list(FULLS)}")
