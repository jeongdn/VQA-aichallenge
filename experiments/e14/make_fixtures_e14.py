# -*- coding: utf-8 -*-
"""E14 시험용 Drive: E10 시험 Drive(멤버 8개 + 제외·대기 실행들) + bf16 실행(test 있음 3개, 검증만 1개, test 진행 중 1개) + train_log 없는 평가 폴더."""
import os, json, shutil, subprocess
import numpy as np, pandas as pd
SP = "/tmp/vqa_scratch"
subprocess.run(["python3", "/home/user/vqa/e10/make_fixtures_e10.py"], check=True, capture_output=True)
SRC, ROOT = f"{SP}/e10test", f"{SP}/e14test"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
shutil.copytree(SRC, ROOT); RUNS = f"{ROOT}/vqa_runs"
shutil.rmtree(f"{RUNS}/e10_search", ignore_errors=True)
T = dict(np.load(f"{ROOT}/truth.npz")); y = T["y"]
test = pd.read_csv(f"{ROOT}/content/test.csv"); train = pd.read_csv(f"{ROOT}/content/train.csv")
VID = pd.read_csv(f"{RUNS}/q35_27b_r16_native/preds_final_ep1__up2.0__1000.csv").id.to_numpy(); NT = len(test)
yt = np.argmax(T["PT_q35_27b_r16_native"] + T["PT_q38_27b_r16_native"] + T["PT_gemma4_31b_r16_up20"], 1)   # test '정답' 대용 (bf16 실행 생성용)
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def save(path, ids, P):
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3], "letter_mass": 0.998, "img_tokens": 1200}).to_csv(path, index=False)
def logcfg(mid, **kw): return {"epoch_acc": {"1": 0.97}, "history": [], "config": dict({"MODEL_ID": mid, "TRAIN_PRESET": "native", "EPOCHS": 1, "TRAIN_ON_FULL": False}, **kw)}
def bf16(base, seed, bonus, sharp=1.0):
    r = np.random.RandomState(seed)
    Pv = sm((np.log(T[f"PV_{base}"]) + r.normal(0, 0.8, (1000, 4)) + bonus * np.eye(4)[y]) * sharp)
    Pt = sm((np.log(T[f"PT_{base}"]) + r.normal(0, 0.8, (NT, 4)) + bonus * np.eye(4)[yt]) * sharp)
    return Pv, Pt
NEW = [("q35_27b_r16_native_bf16", "q35_27b_r16_native", "Qwen/Qwen3.5-27B", 901, 0.9, 1.0, "full"),
       ("q38_27b_r16_native_bf16", "q38_27b_r16_native", "Qwen/Qwen3.8-27B", 902, 0.3, 1.4, "full"),        # 과신(sharp) → 온도가 필요
       ("gemma4_31b_r16_up20_bf16", "gemma4_31b_r16_up20", "google/gemma-4-31B-it", 903, 0.5, 1.0, "full"),
       ("hcx32b_r16_native_bf16", "hcx32b_r16_native", "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", 904, 0.5, 1.0, "valonly"),
       ("gemma4_31b_r16_up20_dev_bf16", "gemma4_31b_r16_up20_dev", "google/gemma-4-31B-it", 905, 0.5, 1.0, "partial")]
for run, base, mid, sd, bonus, sharp, kind in NEW:
    d = f"{RUNS}/{run}"; os.makedirs(d); json.dump(logcfg(mid), open(f"{d}/train_log.json", "w"))
    Pv, Pt = bf16(base, sd, bonus, sharp); save(f"{d}/preds_final_ep1__up2.0__1000.csv", VID, Pv)
    if kind == "full": save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pt)
    if kind == "partial": save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id[:3000], Pt[:3000])
    T[f"PV_{run}"], T[f"PT_{run}"] = Pv, Pt
# train_log 없는 평가 폴더 (계열은 이름 끝을 떼어 q38_27b_r16_native의 기록에서 찾아야 함)
d = f"{RUNS}/q38_27b_r16_native_evalx"; os.makedirs(d); Pv, Pt = bf16("q38_27b_r16_native", 906, 0.2)
save(f"{d}/preds_final_ep1__up2.0__1000.csv", VID, Pv); save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pt); T["PV_q38_27b_r16_native_evalx"], T["PT_q38_27b_r16_native_evalx"] = Pv, Pt
# 전체 데이터(_full) 실행: 원래 실행과 비슷한 test 답 2개(정상) + 크게 다른 1개(고장) + test 없는 1개
def full_run(base, mid, seed, flip, with_test=True):
    d = f"{RUNS}/{base}_full"; os.makedirs(d); json.dump(logcfg(mid, TRAIN_ON_FULL=True), open(f"{d}/train_log.json", "w"))
    r = np.random.RandomState(seed); Pv = sm(np.log(T[f"PV_{base}"]) + 2.0 * np.eye(4)[y]); save(f"{d}/preds_final_ep1__up2.0__1000.csv", VID, Pv)   # 검증은 학습에 써서 거의 다 맞힘
    Pt = sm(np.log(T[f"PT_{base}"]) + r.normal(0, 0.3, (NT, 4)))
    if flip: m_ = r.rand(NT) < flip; Pt[m_] = np.roll(Pt[m_], 1, axis=1)
    if with_test: save(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", test.id, Pt); T[f"PT_{base}_full"] = Pt
full_run("q35_27b_r16_native", "Qwen/Qwen3.5-27B", 951, 0.0)
full_run("gemma4_31b_r16_up20", "google/gemma-4-31B-it", 952, 0.0)
full_run("hcx32b_r16_native", "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", 953, 0.2)      # 20% 답이 바뀜 → 고장으로 걸러져야 함
full_run("q38_27b_r16_native", "Qwen/Qwen3.8-27B", 954, 0.0, with_test=False)
os.symlink(f"{SP}/e7test/content/train", f"{ROOT}/content/train")          # 작은 가짜 이미지 6,714장 (오답 보고서 시험용)
np.savez(f"{ROOT}/truth.npz", **T)
acc = {k[3:]: round(float((v.argmax(1) == y).mean() * 100), 1) for k, v in T.items() if k.startswith("PV_")}
print("E14 fixtures OK |", acc)
