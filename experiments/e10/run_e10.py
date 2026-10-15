# -*- coding: utf-8 -*-
"""e10_ensemble_search.ipynb 코드 셀을 순서대로 실행하고, 멤버 탐색·중첩 CV·제출 후보를 독립 구현과 대조하는 시험 도구.
사용: python3 run_e10.py [key=value ...]"""
import sys, os, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import minimize
NB = "/home/user/vqa/e10/out/e10_ensemble_search.ipynb"
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/e10test"; RUNS = f"{ROOT}/vqa_runs"
over = dict(a.split("=", 1) for a in sys.argv[1:])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, DATA, HELP, POOL, MEMT, METH, CV, FINAL, EXTR, SUMM = cells
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
run(CFG, "1 설정")
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS})
for k, v in over.items(): g[k] = eval(v, g)
for n, s in [("2 데이터", DATA), ("3 함수", HELP), ("4 멤버 탐색", POOL), ("5 멤버 표", MEMT), ("6 방법", METH), ("7 중첩 CV", CV), ("8 후보", FINAL), ("9 추가 후보", EXTR), ("10 요약", SUMM)]: run(s, n)

print("\n━━━━ [시험 도구 확인] ━━━━")
ok = True
T = np.load(f"{ROOT}/truth.npz"); y = T["y"]
EXP = ["gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "hcx32b_r16_native", "ivl35_38b_r16_native", "q35_27b_r16_native", "q35_9b_r16_native", "q38_27b_r16_native", "q3vl_8b_r16_native"]
pool = g["MEMBERS_ALL"]; print("멤버:", pool)
ok &= sorted(r for r, _ in pool) == EXP
ok &= dict(pool)["q3vl_8b_r16_native"] == "up2.0" and dict(pool)["q35_9b_r16_native"] == "up1.5" and dict(pool)["q35_27b_r16_native"] == "up2.0"
sk = " ".join(g["_skip"]); print("제외:", g["_skip"])
for w in ["q3vl_32b_r16_native_full", "broken_run", "othersplit_run", "lowacc_run"]: ok &= w in sk
ok &= "q35_27b_r16_native_dry" not in sk and "e9_tta" not in sk and "ensembles" not in sk
# 독립 구현 (노트북 코드를 쓰지 않음)
names = [r for r, _ in pool]; Mn = len(names)
PV = np.stack([pd.read_csv(f"{RUNS}/{r}/preds_final_ep1__{p}__1000.csv").set_index("id").loc[g["VAL_IDS"], ["p_a", "p_b", "p_c", "p_d"]].to_numpy() for r, p in pool])
PT = np.stack([pd.read_csv(f"{RUNS}/{r}/preds_final_ep1_test__{p}__{len(g['TEST_IDS'])}.csv").set_index("id").loc[g["TEST_IDS"], ["p_a", "p_b", "p_c", "p_d"]].to_numpy() for r, p in pool])
LV, LT = np.log(np.clip(PV, 1e-6, 1)), np.log(np.clip(PT, 1e-6, 1))
ok &= np.allclose(PV, g["PV"]) and np.allclose(PT, g["PT"])
REF = T["vote_val"]; ok &= (g["REF"] == REF).all()
def nllf(P, yy): return float(-np.mean(np.log(np.clip(P[np.arange(len(yy)), yy], 1e-12, 1))))
def i_avg(tr): return ("p", np.ones(Mn))
def i_vote(tr): return ("v", None)
def i_greedy(tr):
    cnt = np.zeros(Mn)
    for t in range(g["GREEDY_STEPS"]):
        best = None
        for m in range(Mn):
            c = cnt.copy(); c[m] += 1; P = np.einsum("m,mnk->nk", c / c.sum(), PV[:, tr]); v = nllf(P, y[tr])
            if best is None or v < best[0]: best = (v, m)
        cnt[best[1]] += 1
    return ("p", cnt)
def i_subset(tr):
    nl = [nllf(PV[m, tr], y[tr]) for m in range(Mn)]; top = sorted(range(Mn), key=lambda m: nl[m])[: g["TOP_N"]]; best = None
    for k in range(2, min(g["MAX_SUBSET"], len(top)) + 1):
        for S in itertools.combinations(top, k):
            Q = PV[list(S)][:, tr].mean(0); key = (int((Q.argmax(1) == y[tr]).sum()), -nllf(Q, y[tr]))
            if best is None or key > best[0]: best = (key, S)
    w = np.zeros(Mn); w[list(best[1])] = 1; return ("p", w)
def i_stack(tr):                                                     # 수치 미분 없이 torch로 같은 목적함수를 최적화 (다른 구현)
    import torch
    L = torch.tensor(LV[:, tr]); yy = torch.tensor(y[tr])
    def obj(w):
        s = torch.einsum("m,mnk->nk", w, L); return torch.nn.functional.cross_entropy(s, yy) + g["STACK_L2"] * (w * w).sum()
    def fg(x):
        w = torch.tensor(x, requires_grad=True); v = obj(w); v.backward(); return float(v), w.grad.numpy().astype(float)
    r = minimize(fg, np.full(Mn, 1.0 / Mn), jac=True, method="L-BFGS-B", bounds=[(0, g["STACK_WMAX"])] * Mn, options={"maxiter": 500})
    return ("l", r.x)
def i_apply(m, P, L):
    kind, w = m
    if kind == "v": A = P.argmax(2); cnt = np.stack([(A == k).sum(0) for k in range(4)], 1); return (cnt + 1e-3 * P.mean(0)).argmax(1)
    if kind == "l": return np.einsum("m,mnk->nk", w, L).argmax(1)
    return np.einsum("m,mnk->nk", w / w.sum(), P).argmax(1)
IM = {"avg_all": i_avg, "vote_all": i_vote, "greedy_nll": i_greedy, "stack_log": i_stack, "best_subset": i_subset}
N = len(y); rs = np.random.RandomState(g["CV_SEED"]); OO = {k: np.zeros((g["CV_REPEATS"], N), int) for k in IM}
for r in range(g["CV_REPEATS"]):
    perm = rs.permutation(N); folds = np.array_split(perm, g["CV_FOLDS"])
    for te in folds:
        tr = np.array(sorted(set(range(N)) - set(te.tolist())))
        for k, fn in IM.items(): OO[k][r, te] = i_apply(fn(tr), PV[:, te], LV[:, te])
for k in IM:
    a_nb = next(rw for rw in g["CV_ROWS"] if rw["방법"] == k)["중첩 CV 정확도(%)"]; a_me = round(float(((OO[k] == y).mean(1) * 100).mean()), 2)
    same = (OO[k] == g["OOF"][k]).mean() * 100
    print(f"  {k}: 노트북 {a_nb} / 독립 {a_me} | OOF 답 일치 {same:.2f}%")
    ok &= (a_nb == a_me) if k != "stack_log" else abs(a_nb - a_me) <= 0.1
    ok &= same == 100.0 if k != "stack_log" else same >= 99.8
# 전체 검증으로 맞춘 최종 조합 → 제출 파일
for k, fn in IM.items():
    m = fn(np.arange(N)); a = i_apply(m, PT, LT)
    f = pd.read_csv(f"{RUNS}/e10_search/submission_e10_{k}.csv").answer.map("abcd".index).to_numpy(); same = int((a == f).sum())
    print(f"  submission_e10_{k}.csv = 독립 계산 {same}/{len(f)}"); ok &= same == len(f) if k != "stack_log" else same >= len(f) - 3
# 판정 줄
best = max(g["CV_ROWS"], key=lambda r: (r["중첩 CV 정확도(%)"], -[k for k in IM].index(r["방법"])))
vl = next(l for l in g["V"]["lines"] if l.startswith("판정")); print("판정 줄:", vl)
n_cv = int(round(best["순이득"])); need = max(0, 3 - n_cv); cnt = int(round(0.97676 * 3357)) + need
if n_cv <= -3: ok &= vl.startswith("판정: 불채택")
else: ok &= vl.startswith("판정: 공개 확인") and f"submission_e10_{best['방법']}.csv" in vl and f"({cnt:,}문항) 이상이면 채택" in vl
ok &= os.path.exists(f"{RUNS}/e10_search/e10_info.json")
# test 예측이 없는 실행: 어댑터가 있고 다른 멤버와 겹치지 않는 것만 합산
X = g["X_TABLE"].set_index("실행"); print(X[["자기 어댑터", "test 예측", "전체 평균에 추가(검증)"]].to_string())
ok &= X.loc["q35_27b_r16_native_bf16eval", "자기 어댑터"].startswith("없음") and X.loc["q35_27b_r16_native_s050", "자기 어댑터"] == "q35_27b_r16_native와 같음"
ok &= X.loc["q35_27b_r16_review1_up20", "test 예측"] == "일부만" and X.loc["q35_27b_r16_e4ctrl_native", "test 예측"] == "없음"
ok &= X.loc["lowacc_run", "test 예측"].startswith("있음") and "검증 읽기 실패" in X.loc["othersplit_run", "전체 평균에 추가(검증)"]
ok &= sorted(r for r, _, _ in g["EXTRA"]) == ["q35_27b_r16_e4ctrl_native", "q35_27b_r16_review1_up20"]
base = PV.mean(0).argmax(1); Pall = (PV.sum(0) + T["VO_q35_27b_r16_e4ctrl_native"] + T["VO_q35_27b_r16_review1_up20"]) / (Mn + 2)
c0, c1 = base == y, Pall.argmax(1) == y; want = f"고침 {int((~c0 & c1).sum())} / 새로 틀림 {int((c0 & ~c1).sum())}"
print("추가 줄:", g["X_LINE"], "| 독립 계산:", want); ok &= want in g["X_LINE"]
for run in ["q35_27b_r16_e4ctrl_native", "q35_9b_r16_native"]:
    if run in X.index:
        P1 = (PV.sum(0) + T[f"VO_{run}"]) / (Mn + 1); c1 = P1.argmax(1) == y
        ok &= f"고침 {int((~c0 & c1).sum())}/새로 틀림 {int((c0 & ~c1).sum())}" in X.loc[run, "전체 평균에 추가(검증)"]
print("시험 도구 판정:", "통과" if ok else "실패")
