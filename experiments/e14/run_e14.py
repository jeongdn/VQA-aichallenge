# -*- coding: utf-8 -*-
"""e14_val_ensemble.ipynb 시험 도구: 노트북 코드 셀을 순서대로 실행하고, 멤버 탐색·계열·온도·강한 풀·중첩 CV·1-SE 선택·제출 파일을
노트북 코드를 쓰지 않은 독립 구현과 대조. 사용: python3 run_e14.py <scenario> [key=value ...]
  base / force(FORCE_METHOD=stack_log) / delta0(POOL_DELTA=0 → 단순 방법이 한 실행만 씀 → 1-SE 선택이 뒤로 밀림) / mustmiss"""
import sys, os, json, math, shutil, hashlib
import numpy as np, pandas as pd
from scipy.optimize import minimize
NB = "/home/user/vqa/e14/out/e14_val_ensemble.ipynb"
SP = "/tmp/vqa_scratch"
SRC = f"{SP}/e14test"; WORK = f"{SP}/e14work"
scen = sys.argv[1]; over = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
if scen == "force": over.setdefault("FORCE_METHOD", "'stack_log'")
if scen == "delta0": over.setdefault("POOL_DELTA", "0.0")
if scen == "mustmiss": over.setdefault("MUST_INCLUDE", "[('ivl35_38b_r16_nativeX', 'up2.0')]")
shutil.rmtree(WORK, ignore_errors=True); shutil.copytree(f"{SRC}/vqa_runs", f"{WORK}/vqa_runs"); R = f"{WORK}/vqa_runs"
FULLALL = "fullall" in sys.argv[2:] or scen == "fullall"
if FULLALL:
    over.pop("fullall", None)
    MODEL = {"gemma4_31b_r16_up20_bf16": "google/gemma-4-31B-it", "gemma4_31b_r16_up20_dev": "google/gemma-4-31B-it", "ivl35_38b_r16_native": "OpenGVLab/InternVL3_5-38B-HF",
             "q35_27b_r16_native_bf16": "Qwen/Qwen3.5-27B", "q35_9b_r16_native": "Qwen/Qwen3.5-9B", "q38_27b_r16_native_bf16": "Qwen/Qwen3.8-27B",
             "q38_27b_r16_native_evalx": "Qwen/Qwen3.8-27B", "q3vl_8b_r16_native": "Qwen/Qwen3-VL-8B-Instruct"}
    rs0 = np.random.RandomState(77)
    for r, mid in MODEL.items():
        d = f"{R}/{r}_full"; os.makedirs(d, exist_ok=True); pre = "up1.5" if r == "q35_9b_r16_native" else "up2.0"
        src = pd.read_csv(f"{R}/{r}/preds_final_ep1_test__{pre}__6714.csv"); P_ = src[["p_a", "p_b", "p_c", "p_d"]].to_numpy() * np.exp(rs0.normal(0, 0.15, (len(src), 4)))
        src[["p_a", "p_b", "p_c", "p_d"]] = P_ / P_.sum(1, keepdims=True); src.to_csv(f"{d}/preds_final_ep1_test__{pre}__6714.csv", index=False)
        shutil.copy(f"{R}/{r}/preds_final_ep1__{pre}__1000.csv", f"{d}/preds_final_ep1__{pre}__1000.csv")
        json.dump({"epoch_acc": {"1": 0.99}, "config": {"MODEL_ID": mid, "TRAIN_ON_FULL": True}}, open(f"{d}/train_log.json", "w"))
    if scen == "fullall": scen = "base"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
def run(src, name): print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
run(cells[0], "1 설정"); g.update({"DATA_DIR": f"{SRC}/content", "OUT_DIR": R})
for k, v in over.items(): g[k] = eval(v, g)
names = ["2 데이터", "3 함수", "4 멤버", "5 멤버 표", "6 방법", "7 중첩 CV", "8 제출", "9 보고서", "10 요약"]
if scen == "noimg":                                               # 이미지가 없는 세션 (Colab 밖) → 글만 있는 보고서
    os.makedirs(f"{WORK}/content", exist_ok=True)
    for f in ("train.csv", "test.csv", "sample_submission.csv"): shutil.copy(f"{SRC}/content/{f}", f"{WORK}/content/{f}")
    g["DATA_DIR"] = f"{WORK}/content"
try:
    for i, nm in enumerate(names, 1): run(cells[i], nm)
except Exception as e:
    msg = f"{type(e).__name__}: {e}"; print("멈춤:", msg[:300])
    ok = scen == "mustmiss" and "반드시 찾을 멤버의 폴더가 없습니다" in msg
    print("시험 도구 판정:", "통과" if ok else "실패"); sys.exit()
if scen == "mustmiss": print("시험 도구 판정: 실패 (멈춰야 함)"); sys.exit()

print("\n━━━━ [시험 도구 확인] ━━━━"); ok = True
T = np.load(f"{SRC}/truth.npz"); y = T["y"]; N = len(y); VIDS, TIDS = g["VAL_IDS"], g["TEST_IDS"]
EXP = {"gemma4_31b_r16_up20": "google/gemma-4-31B-it", "gemma4_31b_r16_up20_dev": "google/gemma-4-31B-it", "gemma4_31b_r16_up20_bf16": "google/gemma-4-31B-it",
       "hcx32b_r16_native": "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", "ivl35_38b_r16_native": "OpenGVLab/InternVL3_5-38B-HF",
       "q35_27b_r16_native": "Qwen/Qwen3.5-27B", "q35_27b_r16_native_bf16": "Qwen/Qwen3.5-27B", "q35_9b_r16_native": "Qwen/Qwen3.5-9B",
       "q38_27b_r16_native": "Qwen/Qwen3.8-27B", "q38_27b_r16_native_bf16": "Qwen/Qwen3.8-27B", "q38_27b_r16_native_evalx": "Qwen/Qwen3.8-27B",
       "q3vl_8b_r16_native": "Qwen/Qwen3-VL-8B-Instruct"}
got = dict(zip(g["NAMES"], g["FAM"])); print("멤버·계열 일치:", got == EXP); ok &= got == EXP
ok &= dict(g["MEMBERS_ALL"])["q35_9b_r16_native"] == "up1.5" and dict(g["MEMBERS_ALL"])["q3vl_8b_r16_native"] == "up2.0"
W, S_ = " ".join(g["_wait"]), " ".join(g["_skip"]); print("대기:", g["_wait"]); print("제외:", g["_skip"])
for w, how in [("hcx32b_r16_native_bf16", "test 예측 없음"), ("gemma4_31b_r16_up20_dev_bf16", "진행 중(일부만 있음)"), ("q35_27b_r16_review1_up20", "진행 중"),
               ("q35_27b_r16_e4ctrl_native", "test 예측 없음"), ("q35_27b_r16_native_bf16eval", "test 예측 없음")]:
    seg = next((x for x in g["_wait"] if x.startswith(w + "@")), ""); ok &= how in seg
for s_, how in [("q3vl_32b_r16_native_full", "전체 데이터"), ("othersplit_run", "검증 읽기 실패"), ("lowacc_run", "고장")]:
    ok &= any(x.startswith(s_) and how in x for x in g["_skip"])
ok &= "q35_27b_r16_native_dry" not in W + S_
# ── 독립 구현 ──
names = list(EXP) ; names = g["NAMES"]; Mn = len(names); pres = dict(g["MEMBERS_ALL"])
def rd(run, split):
    f = f"{R}/{run}/preds_final_ep1{'_test' if split == 'test' else ''}__{pres[run]}__{1000 if split == 'val' else len(TIDS)}.csv"
    return pd.read_csv(f).set_index("id").loc[VIDS if split == "val" else TIDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
PV = np.stack([rd(r, "val") for r in names]); PT = np.stack([rd(r, "test") for r in names]); ok &= np.allclose(PV, g["PV"]) and np.allclose(PT, g["PT"])
LV, LT = np.log(np.clip(PV, 1e-6, 1)), np.log(np.clip(PT, 1e-6, 1)); fam = [EXP[r] for r in names]
GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def tsc(p, t):
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / t); return q / q.sum(1, keepdims=True)
def nllf(p, yy): return float(-np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1))))
def fitT(m, idx):
    best = None
    for t in GRID:
        v = nllf(tsc(PV[m, idx], t), y[idx])
        if best is None or v < best[0]: best = (v, t)
    return best[1]
DELTA = g["POOL_DELTA"]
def pool(idx):
    a = [(PV[m, idx].argmax(1) == y[idx]).mean() * 100 for m in range(Mn)]; ref = sum(sorted(a)[-3:]) / min(3, len(a))
    return [m for m in range(Mn) if a[m] >= ref - DELTA - 1e-9]
Tall = [fitT(m, np.arange(N)) for m in range(Mn)]; ok &= np.allclose(Tall, g["T_ALL"]); ok &= pool(np.arange(N)) == g["POOL_ALL"]
print("온도·강한 풀 일치:", np.allclose(Tall, g["T_ALL"]), pool(np.arange(N)) == g["POOL_ALL"], "| 풀:", [names[m] for m in pool(np.arange(N))])
def famavg(ms, Tm, P):
    groups = {}
    for m in ms: groups.setdefault(fam[m], []).append(m)
    fps = [np.mean([tsc(P[m], Tm[m]) for m in grp], 0) for grp in groups.values()]; return fps
def m_cal(tr):
    ms = pool(tr); Tm = {m: fitT(m, tr) for m in ms}; return lambda P, L: np.mean(famavg(ms, Tm, P), 0)
def m_avgfam(tr):
    ms = pool(tr); Tm = {m: 1.0 for m in ms}; return lambda P, L: np.mean(famavg(ms, Tm, P), 0)
def m_vote(tr):
    ms = pool(tr); Tm = {m: fitT(m, tr) for m in ms}
    def f(P, L):
        fps = famavg(ms, Tm, P); avg = np.mean(fps, 0); votes = np.zeros_like(avg)
        for q in fps: votes[np.arange(len(q)), q.argmax(1)] += 1
        return votes + 1e-3 * avg
    return f
def m_avgruns(tr):
    ms = pool(tr); return lambda P, L: np.mean([P[m] / P[m].sum(1, keepdims=True) for m in ms], 0)
def m_greedy(tr):
    Tm = [fitT(m, tr) for m in range(Mn)]; sel = []
    for t in range(g["GREEDY_STEPS"]):
        best = None
        for m in range(Mn):
            Q = np.mean([tsc(PV[j, tr], Tm[j]) for j in sel + [m]], 0); v = nllf(Q, y[tr])
            if best is None or v < best[0]: best = (v, m)
        sel.append(best[1])
    return lambda P, L: np.mean([tsc(P[j], Tm[j]) for j in sel], 0)
def m_stack(tr):
    import torch
    Lt = torch.tensor(LV[:, tr]); yy = torch.tensor(y[tr])
    def fg(x):
        w = torch.tensor(x, requires_grad=True); v = torch.nn.functional.cross_entropy(torch.einsum("m,mnk->nk", w, Lt), yy) + g["STACK_L2"] * (w * w).sum(); v.backward()
        return float(v.detach()), w.grad.numpy().astype(float)
    r = minimize(fg, np.full(Mn, 1.0 / Mn), jac=True, method="L-BFGS-B", bounds=[(0, g["STACK_WMAX"])] * Mn); w = r.x
    def f(P, L):
        s = np.einsum("m,mnk->nk", w, L); e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    return f
IM = [("cal_fam", m_cal, True), ("avg_fam", m_avgfam, True), ("vote_fam", m_vote, False), ("avg_runs", m_avgruns, True), ("greedy_cal", m_greedy, True), ("stack_log", m_stack, True)]
ok &= [k for k, _, _ in IM] == [k for k, _, _, _ in g["METHOD_LIST"]]
rs = np.random.RandomState(g["CV_SEED"]); RP = g["CV_REPEATS"]; OO = {k: np.zeros((RP, N), int) for k, _, _ in IM}; OP = {k: np.zeros((RP, N, 4)) for k, _, okn in IM if okn}
for r in range(RP):
    for te in np.array_split(rs.permutation(N), g["CV_FOLDS"]):
        tr = np.array(sorted(set(range(N)) - set(te.tolist())))
        for k, fn, okn in IM:
            Sx = fn(tr)(PV[:, te], LV[:, te]); OO[k][r, te] = Sx.argmax(1)
            if okn: OP[k][r, te] = Sx
cor = {k: (OO[k] == y).mean(0) for k in OO}; acc = {k: cor[k].mean() * 100 for k in OO}; order = [k for k, _, _ in IM]
for k in order:
    same = (OO[k] == g["OOF"][k]).mean() * 100; nb_acc = next(rw for rw in g["CV_ROWS"] if rw["방법"] == k)["중첩 CV 정확도(%)"]
    print(f"  {k}: 노트북 {nb_acc} / 독립 {acc[k]:.2f} | OOF 답 일치 {same:.2f}%")
    ok &= (same == 100.0 and nb_acc == round(acc[k], 2)) if k != "stack_log" else (same >= 99.8 and abs(nb_acc - acc[k]) <= 0.1)
best = max(order, key=lambda k: (round(acc[k], 6), -order.index(k)))
def se(k): d = cor[k] - cor[best]; return 0.0 if k == best else d.std(ddof=1) / math.sqrt(N) * 100
chosen = next(k for k in order if acc[k] >= acc[best] - se(k) - 1e-9)
print(f"  1등 {best} / 선택 {chosen} | 노트북 1등 {g['BEST']} / 선택 {g['CHOSEN']}"); ok &= (best, chosen) == (g["BEST"], g["CHOSEN"])
for rw in g["CV_ROWS"]:
    k = rw["방법"]; ok &= abs(rw["차이의 표준오차(%p)"] - round(se(k), 2)) <= 0.011
    if k in OP: ok &= abs(rw["중첩 CV NLL"] - round(np.mean([nllf(OP[k][r], y) for r in range(RP)]), 4)) <= (2e-3 if k == "stack_log" else 1e-9)
use = g["FORCE_METHOD"] or chosen; fnx = dict((k, fn) for k, fn, _ in IM)[use](np.arange(N)); ans = fnx(PT, LT).argmax(1)
tag = hashlib.md5(",".join(f"{r}@{p}" for r, p in g["MEMBERS_ALL"]).encode()).hexdigest()[:6]
fp = f"{R}/e14_val_ens/submission_e14_{use}_{Mn}runs_{tag}.csv"; ok &= os.path.exists(fp)
files = sorted(f for f in os.listdir(f"{R}/e14_val_ens") if f.startswith("submission_e14_")); print("  제출 파일:", files); ok &= files == [os.path.basename(fp)]
if os.path.exists(fp):
    f1 = pd.read_csv(fp); ok &= list(f1.id) == list(TIDS); same = int((f1.answer.map("abcd".index).to_numpy() == ans).sum())
    print(f"  {os.path.basename(fp)} = 독립 계산 {same}/{len(ans)}"); ok &= same >= (len(ans) - 3 if use == "stack_log" else len(ans))
ok &= any(l.startswith(f"[선택] {use}") for l in g["V"]["lines"])
if scen == "force": ok &= use == "stack_log" and "FORCE_METHOD로 지정" in g["V"]["lines"][0]
if scen == "delta0": print("  (delta0) 강한 풀 크기:", len(pool(np.arange(N))), "| 선택이 목록 첫 방법이 아님:", chosen != "cal_fam"); ok &= chosen != "cal_fam"
# ── 4bit → bf16 짝 (test 대기 실행 포함) ──
def rdv(run, pre="up2.0"): return pd.read_csv(f"{R}/{run}/preds_final_ep1__{pre}__1000.csv").set_index("id").loc[VIDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
for base, new, wait in [("q35_27b_r16_native", "q35_27b_r16_native_bf16", False), ("q38_27b_r16_native", "q38_27b_r16_native_bf16", False), ("gemma4_31b_r16_up20", "gemma4_31b_r16_up20_bf16", False),
                        ("hcx32b_r16_native", "hcx32b_r16_native_bf16", True), ("gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20_dev_bf16", True)]:
    a0, a1 = rdv(base).argmax(1), rdv(new).argmax(1); f, b = int(((a0 != y) & (a1 == y)).sum()), int(((a0 == y) & (a1 != y)).sum())
    want = f"  {base} → {new}{' (test 대기)' if wait else ''}: 검증 {(a0 == y).mean() * 100:.2f}% → {(a1 == y).mean() * 100:.2f}% | 고침 {f} · 새로 틀림 {b} · 순이득 {f - b:+d}"
    hit = any(l.startswith(want) for l in g["PAIRS"]); ok &= hit
    if not hit: print("  짝 줄 다름:", want)
ok &= len(g["PAIRS"]) == 5; print("  4bit→bf16 짝 줄:", len(g["PAIRS"]), "개 (test 대기 2개 포함)")
# ── 오답 보고서 ──
Sv = fnx(PV, LV); Pd = m_cal(np.arange(N))(PV, LV) if use == "vote_fam" else Sv / Sv.sum(1, keepdims=True)
av = Sv.argmax(1); srt = np.sort(Pd, 1); mg = srt[:, -1] - srt[:, -2]; wrong = np.where(av != y)[0]
near = [int(i) for i in np.argsort(mg, kind="stable") if av[i] == y[i]][: g["REPORT_NEAR"]]
items = [int(i) for i in wrong[np.argsort(-mg[wrong], kind="stable")]] + near
ok &= [i for i, _ in g["ITEMS"]] == items; E = g["ERR_TABLE"]; ok &= list(E.id) == [VIDS[i] for i in items]
ok &= all(E["정답"].iloc[j] == "abcd"[y[i]] and E["조합 답"].iloc[j].startswith("abcd"[av[i]]) for j, i in enumerate(items))
html_ = open(f"{R}/e14_val_ens/{g['REPORT_FILE']}", encoding="utf-8").read(); nimg = html_.count("data:image/jpeg;base64")
want_img = 0 if scen == "noimg" else len(items); print(f"  보고서: 오답 {len(wrong)} · 아슬 {len(near)} · 이미지 {nimg} (기대 {want_img})"); ok &= nimg == want_img == g["n_img"]
ok &= html_.count("class='it ") == len(items) and os.path.exists(f"{R}/e14_val_ens/{g['REPORT_FILE'].replace('.html', '.csv')}")
# ── 최종 후보 B (독립 구현): 고른 방법의 매개변수를 다시 구하고, 정상 _full만 같은 자리에 넣음 ──
FT = g["FULL_TABLE"].set_index("원래 실행"); print(FT[["상태"]].to_string())
if "hcx32b_r16_native" in FT.index: ok &= FT.loc["hcx32b_r16_native", "상태"].startswith("⚠️ 안 씀: test 일치")
if "q38_27b_r16_native" in FT.index: ok &= FT.loc["q38_27b_r16_native", "상태"].startswith("test 예측 없음")
def rdt(run): return pd.read_csv(f"{R}/{run}/preds_final_ep1_test__up2.0__{len(TIDS)}.csv").set_index("id").loc[TIDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
def rdt2(run, pre):
    f = f"{R}/{run}/preds_final_ep1_test__{pre}__{len(TIDS)}.csv"
    if not os.path.exists(f): return None
    d_ = pd.read_csv(f).drop_duplicates("id", keep="last").set_index("id")
    return None if not all(i in d_.index for i in TIDS) else d_.loc[TIDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
GOOD = {}
for m, r in enumerate(names):
    fr = r + "_full"
    if not os.path.isdir(f"{R}/{fr}"): continue
    Pf = rdt2(fr, pres[r]); cf = json.load(open(f"{R}/{fr}/train_log.json")).get("config", {})
    if Pf is None or not cf.get("TRAIN_ON_FULL") or cf.get("MODEL_ID") != EXP[r]: continue
    if (Pf.argmax(1) == PT[m].argmax(1)).mean() * 100 >= g["FULL_MIN_AGREE"]: GOOD[m] = Pf
print("  정상 _full (독립 판단):", sorted(names[m] for m in GOOD))
mode = g["FULL_MODE"]; allN = np.arange(N)
def with_full(m, Tm):
    if m in GOOD_USED: return [tsc(PT[m], Tm), tsc(GOOD_USED[m], Tm)] if mode == "mix" else [tsc(GOOD_USED[m], Tm)]
    return [tsc(PT[m], Tm)]
if use in ("cal_fam", "avg_fam", "vote_fam"):
    ms = pool(allN); Tm = {m: (fitT(m, allN) if use != "avg_fam" else 1.0) for m in ms}; GOOD_USED = {m: v for m, v in GOOD.items() if m in ms}
    groups = {}
    for m in ms: groups.setdefault(fam[m], []).append(m)
    fps = [np.mean([p for m in grp for p in with_full(m, Tm[m])], 0) for grp in groups.values()]; avg = np.mean(fps, 0)
    if use == "vote_fam":
        votes = np.zeros_like(avg)
        for q in fps: votes[np.arange(len(q)), q.argmax(1)] += 1
        avg = votes + 1e-3 * avg
    bans = avg.argmax(1)
elif use in ("avg_runs", "greedy_cal"):
    if use == "avg_runs": ms = pool(allN); wts = {m: 1.0 for m in ms}; Tm = {m: 1.0 for m in ms}
    else:
        Tl = [fitT(m, allN) for m in range(Mn)]; sel = []
        for t in range(g["GREEDY_STEPS"]):
            best_ = None
            for m in range(Mn):
                v = nllf(np.mean([tsc(PV[j], Tl[j]) for j in sel + [m]], 0), y)
                if best_ is None or v < best_[0]: best_ = (v, m)
            sel.append(best_[1])
        wts = {m: float(sel.count(m)) for m in set(sel)}; Tm = {m: Tl[m] for m in wts}
    GOOD_USED = {m: v for m, v in GOOD.items() if m in wts}; tot = sum(wts.values()); acc_ = 0
    for m, wm in wts.items(): acc_ = acc_ + wm / tot * np.mean(with_full(m, Tm[m]), 0)
    bans = acc_.argmax(1)
else:
    wst = np.asarray(g["FINAL_MODEL"]["w"], float); GOOD_USED = {m: v for m, v in GOOD.items() if wst[m] > 0.005}
    s_ = sum(wst[m] * LT[m] for m in range(Mn) if m not in GOOD_USED)
    for m, Pf in GOOD_USED.items():
        Lf = np.log(np.clip(Pf, 1e-6, 1)); s_ = s_ + (wst[m] / 2 * LT[m] + wst[m] / 2 * Lf if mode == "mix" else wst[m] * Lf)
    bans = s_.argmax(1)
if GOOD_USED:
    fp = f"{R}/e14_val_ens/{g['B_FILE']}"; ok &= os.path.exists(fp); bf = pd.read_csv(fp).answer.map("abcd".index).to_numpy()
    same = int((bf == bans).sum()); print(f"  B({use}, {mode}, _full {len(GOOD_USED)}개): 독립 계산 {same}/{len(bans)} | A와 다른 답 {int((bans != ans).sum())}")
    ok &= same >= (len(bans) - 3 if use == "stack_log" else len(bans)) and g["B_DIFF"] == int((bf != ans).sum())
else:
    ok &= g["B_FILE"] is None; print("  B: 쓸 _full 없음 (기대와 같음)")
print("시험 도구 판정:", "통과" if ok else "실패")
