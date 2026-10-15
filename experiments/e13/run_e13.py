# -*- coding: utf-8 -*-
"""e13_bf16_swap.ipynb 시험 도구: 시나리오마다 vqa_runs 복사본에서 노트북 코드 셀을 순서대로 실행하고, 결과를 독립 계산과 대조.
사용: python3 run_e13.py <scenario>
  base(Q35 bf16 +5·Q35 시드43 +4·Q38 −5·G +3·HCX 검증만·GDEV 없음) / gbad / q35only / cfgdiff / lowmass / same / full / none / allbad"""
import sys, os, json, math, shutil, re
import numpy as np, pandas as pd
NB = "/home/user/vqa/e13/out/e13_bf16_swap.ipynb"
SP = "/tmp/vqa_scratch"
SRC = f"{SP}/e13test"; WORK = f"{SP}/e13work"
scen = sys.argv[1]; over = dict(a.split("=", 1) for a in sys.argv[2:])
shutil.rmtree(WORK, ignore_errors=True); shutil.copytree(f"{SRC}/vqa_runs", f"{WORK}/vqa_runs"); R = f"{WORK}/vqa_runs"
Q35, Q38, G, GD, H = "q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "hcx32b_r16_native"
VF, TF = "preds_final_ep1__up2.0__1000.csv", "preds_final_ep1_test__up2.0__6714.csv"
def jl(run): return json.load(open(f"{R}/{run}/train_log.json"))
def jw(run, j): json.dump(j, open(f"{R}/{run}/train_log.json", "w"))
if scen == "gbad":
    for f in (VF, TF): shutil.copy(f"{SRC}/_alt/gemma_bad/{f}", f"{R}/{G}_bf16/{f}")
if scen == "q35only":
    for r in (Q38, G): shutil.rmtree(f"{R}/{r}_bf16")
if scen == "cfgdiff":
    j = jl(Q35 + "_bf16"); j["config"]["LR"] = 0.0002; jw(Q35 + "_bf16", j)
if scen == "lowmass":
    d = pd.read_csv(f"{R}/{Q35}_bf16/{VF}"); d["letter_mass"] = 0.62; d.to_csv(f"{R}/{Q35}_bf16/{VF}", index=False)
if scen == "same":
    for f in (VF, TF): shutil.copy(f"{R}/{Q38}/{f}", f"{R}/{Q38}_bf16/{f}")
if scen == "full":
    j = jl(Q35 + "_bf16"); j["config"]["TRAIN_ON_FULL"] = True; jw(Q35 + "_bf16", j)
if scen == "none":
    for r in (Q35 + "_bf16", Q35 + "_bf16_s43", Q38 + "_bf16", G + "_bf16", H + "_bf16"): shutil.rmtree(f"{R}/{r}")
if scen == "allbad":                                               # 모든 bf16 실행을 크게 나쁘게 (정답 확률을 다른 보기로 밀어냄)
    y_ = np.load(f"{SRC}/truth.npz")["y"]
    for r in (Q35 + "_bf16", Q35 + "_bf16_s43", Q38 + "_bf16", G + "_bf16"):
        d = pd.read_csv(f"{R}/{r}/{VF}"); P_ = d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(); m_ = np.arange(len(d)) % 12 == 0
        P_[m_] = np.roll(np.eye(4)[y_[m_]], 1, axis=1) * 0.9 + 0.025; d[["p_a", "p_b", "p_c", "p_d"]] = P_; d.to_csv(f"{R}/{r}/{VF}", index=False)
if scen == "extra":                                                # dev 의사 라벨을 더한 Q38 bf16 (EXTRA_RUNS) — 설정 차이는 경고 없이 표에만
    shutil.copytree(f"{R}/{Q35}_bf16", f"{R}/q38_27b_r16_dev_bf16"); T_ = np.load(f"{SRC}/truth.npz")
    for f, key in ((VF, "PV_q38_27b_r16_native"), (TF, "PT_q38_27b_r16_native")):
        d = pd.read_csv(f"{R}/q38_27b_r16_dev_bf16/{f}"); yy = T_["y"] if f == VF else T_["yt"]; rs_ = np.random.RandomState(8)
        lg = np.log(T_[key]) + rs_.normal(0, 0.5, T_[key].shape) + 0.6 * np.eye(4)[yy]; P_ = np.exp(lg - lg.max(1, keepdims=True)); P_ /= P_.sum(1, keepdims=True)
        d[["p_a", "p_b", "p_c", "p_d"]] = P_; d.to_csv(f"{R}/q38_27b_r16_dev_bf16/{f}", index=False)
    j = jl(Q38 + "_bf16"); j["config"]["EXTRA_TRAIN_CSV"] = "/content/drive/MyDrive/vqa_runs/dev_pseudo.csv"; jw("q38_27b_r16_dev_bf16", j)
if scen == "partial":                                              # Q38 bf16 test 추론이 이어하기 중 (앞 2,000문항만)
    d = pd.read_csv(f"{R}/{Q38}_bf16/{TF}"); d.head(2000).to_csv(f"{R}/{Q38}_bf16/{TF}", index=False)
if scen == "worst":                                                # bf16 실행 30%가 자신 있게 틀림 (게이트를 끄고 불채택 갈래 시험)
    T_ = np.load(f"{SRC}/truth.npz"); rs_ = np.random.RandomState(4)
    for r in (Q35 + "_bf16", Q35 + "_bf16_s43", Q38 + "_bf16", G + "_bf16"):
        for f, yy in ((VF, T_["y"]), (TF, T_["yt"])):
            d = pd.read_csv(f"{R}/{r}/{f}"); P_ = np.eye(4)[yy] * 0.97 + 0.01; bad = np.arange(len(d)) % 3 == 0   # 모든 실행이 같은 문항에서 틀림
            P_[bad] = np.roll(P_[bad], 1, axis=1); d[["p_a", "p_b", "p_c", "p_d"]] = P_; d.to_csv(f"{R}/{r}/{f}", index=False)
if scen == "bigwin":                                               # bf16 실행이 거의 다 맞힘 → 앙상블이 확실히 좋아짐
    T_ = np.load(f"{SRC}/truth.npz"); rs_ = np.random.RandomState(3)
    for r in (Q35 + "_bf16", Q35 + "_bf16_s43", Q38 + "_bf16", G + "_bf16"):
        for f, yy in ((VF, T_["y"]), (TF, T_["yt"])):
            d = pd.read_csv(f"{R}/{r}/{f}"); P_ = np.eye(4)[yy] * 0.9 + 0.025; bad = rs_.rand(len(d)) < 0.004
            P_[bad] = np.roll(P_[bad], 1, axis=1); d[["p_a", "p_b", "p_c", "p_d"]] = P_; d.to_csv(f"{R}/{r}/{f}", index=False)
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
def run(src, name): print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
run(cells[0], "1 설정"); g.update({"DATA_DIR": f"{SRC}/content", "OUT_DIR": R})
for k, v in over.items(): g[k] = eval(v, g)
SWG, MXG = g["SWAP_GATE"], g["MIX_GATE"]
for i, nm in enumerate(["2 데이터", "3 함수", "4 멤버별", "5 판정", "6 요약"], 1): run(cells[i], nm)

print("\n━━━━ [시험 도구 확인] ━━━━"); ok = True
T = np.load(f"{SRC}/truth.npz"); y = T["y"]; VIDS, TIDS = g["VAL_IDS"], g["TEST_IDS"]
def P(run, s): return pd.read_csv(f"{R}/{run}/{VF if s == 'val' else TF}").set_index("id").loc[VIDS if s == "val" else TIDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
def nllf(p): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
# 기대 AVAIL (독립: 폴더 규칙 + 상태)
base_av = {Q35: [Q35 + "_bf16", Q35 + "_bf16_s43"], Q38: [Q38 + "_bf16"], G: [G + "_bf16"]}
want = {"partial": {Q35: base_av[Q35], G: base_av[G]}, "q35only": {Q35: base_av[Q35]}, "full": {Q35: [Q35 + "_bf16_s43"], Q38: base_av[Q38], G: base_av[G]}, "none": {},
        "extra": {Q35: base_av[Q35], Q38: base_av[Q38] + ["q38_27b_r16_dev_bf16"], G: base_av[G]}}.get(scen, base_av)
got = g["AVAIL"]; print("AVAIL:", got); ok &= {k: sorted(v) for k, v in got.items()} == {k: sorted(v) for k, v in want.items()}
MT = g["MEM_TABLE"]; st = dict(zip(MT["bf16 실행"], MT["상태"]))
ok &= st.get(GD + "_bf16", "").startswith("폴더 없음") and not any(k.endswith(("_bf16_old", "_bf16x")) for k in st)
ok &= st.get("q35_27b_r16_dev_bf16", "").startswith("폴더 없음") and (scen == "extra") != st.get("q38_27b_r16_dev_bf16", "").startswith("폴더 없음")
if scen == "extra": ok &= MT[MT["bf16 실행"] == "q38_27b_r16_dev_bf16"].iloc[0]["레시피 차이"] == "EXTRA_TRAIN_CSV"
if scen != "none": ok &= st.get(H + "_bf16", "").startswith("test 예측 없음")
if scen == "full": ok &= "전체 데이터" in st[Q35 + "_bf16"]
if scen == "partial": ok &= st[Q38 + "_bf16"].startswith("test 예측 진행 중")
S = {}
for o, runs in want.items():                                     # 멤버별 숫자 + 기준 (독립 계산)
    for n_ in runs:
        a0, a1 = P(o, "val").argmax(1), P(n_, "val").argmax(1); c0, c1 = a0 == y, a1 == y
        f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum()); k = min(f, b); nn = f + b
        p = 1.0 if nn == 0 else min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / 2 ** nn)
        sw = (f - b) > SWG or ((f - b) == SWG and nllf(P(n_, "val")) < nllf(P(o, "val"))); mx = (f - b) >= MXG; S[n_] = (o, sw, mx)
        row = MT[MT["bf16 실행"] == n_].iloc[0]
        hit = (row["고침"], row["새로 틀림"], row["바뀐 답"]) == (f, b, int((a0 != a1).sum())) and row["부호 검정 p"] == ("<0.001" if p < 0.001 else f"{p:.3f}") \
              and row["교체 방식"] == ("넣음" if sw else "뺌") and row["평균 방식"] == ("넣음" if mx else "뺌") \
              and row["test 바뀐 답"] == int((P(o, "test").argmax(1) != P(n_, "test").argmax(1)).sum())
        print(f"  {n_}: 순이득 {f - b:+d} ({f}/{b}) p {p:.3f} 교체 {sw} 평균 {mx} → {'OK' if hit else '다름'}"); ok &= hit
SW = {}; MX = {}
for n_, (o, sw, mx) in S.items():
    if sw: SW.setdefault(o, []).append(n_)
    if mx: MX.setdefault(o, []).append(n_)
T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def ts(p, t): q = np.clip(p, 1e-12, 1) ** (1 / t); return q / q.sum(1, keepdims=True)
def vm(r, s, kind, m):
    if r in m: return np.mean([P(n_, s) for n_ in m[r]], 0) if kind == "swap" else np.mean([P(r, s)] + [P(n_, s) for n_ in m[r]], 0)
    return P(r, s)
def fitT(r, kind, m): p = vm(r, "val", kind, m); return T70[int(np.argmin([-np.mean(np.log(ts(p, t)[np.arange(len(y)), y])) for t in T70]))]
def ens(runs, w, cal, s, kind, m):
    w = np.asarray(w, float) / sum(w); return sum(wi * (ts(vm(r, s, kind, m), fitT(r, kind, m)) if cal else vm(r, s, kind, m)) for wi, r in zip(w, runs)).argmax(1)
def recipes(kind, m):
    out = {}
    for s in ("val", "test"):
        c = ens([Q35, Q38, GD, G, H], [1, 1, 1, 1, 2], False, s, kind, m); a = ens([Q38, G], [1, 1], False, s, kind, m); b = ens([Q38, GD], [1, 1], True, s, kind, m)
        A = np.stack([a, c, b], 1); v = np.stack([(A == k).sum(1) for k in range(4)], 1).argmax(1); out[s] = {"clean4_qg_hcx": c, "majority_vote": v}
    return out
B0 = recipes("swap", {}); L = g["V"]["lines"]; vl = next(l for l in L if l.startswith("판정")); print("판정 줄:", vl)
fa = {nm: pd.read_csv(f"{R}/ensembles/submission_{nm}.csv").answer.map("abcd".index).to_numpy() for nm in ("clean4_qg_hcx", "majority_vote")}
ok &= all((B0["test"][nm] == fa[nm]).all() for nm in fa)
e13 = f"{R}/e13_bf16"; files = sorted(f for f in os.listdir(e13) if f.endswith(".csv")) if os.path.isdir(e13) else []
if not want: ok &= vl.startswith("판정: 대기") and not files
elif not MX and not SW: ok &= vl.startswith("판정: 불채택") and "멤버 기준" in vl and not files
else:
    res = {}
    for kind, m in (("swap", SW), ("mix", MX)):
        if not m: continue
        B1 = recipes(kind, m); c0, c1 = B0["val"]["clean4_qg_hcx"] == y, B1["val"]["clean4_qg_hcx"] == y
        f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum()); res[kind] = (f - b, f, b, B1); print(f"  독립 계산 [{kind}] {m} clean4 순이득 {f - b:+d} ({f}/{b})")
        lbl = {"swap": "[교체]", "mix": "[평균]"}[kind]; i0 = next(i for i, l in enumerate(L) if l.startswith(lbl))
        ok &= f"고침 {f} | 새로 틀림 {b} | 순이득 {f - b:+d}" in L[i0 + 1] and L[i0 + 1].strip().startswith("clean4_qg_hcx:")
    best = max(res, key=lambda k: (res[k][0], k == "mix")); n1, f1, b1, B1 = res[best]; print("  고른 방식:", best, n1)
    warn = any("⚠️ 확인할 것" in l for l in L); want_warn = scen in ("cfgdiff", "lowmass", "same"); print("  경고 줄:", warn, "| 기대:", want_warn); ok &= warn == want_warn
    print("  갈래:", "불채택" if n1 <= -3 else ("공개 확인 (공개 하락 허용)" if 3 - n1 < 0 else "공개 확인"))
    if n1 <= -3: ok &= vl.startswith("판정: 불채택") and not files
    else:
        ok &= vl.startswith("판정: 공개 확인") and {"swap": "교체", "mix": "평균"}[best] + " 방식" in vl
        need = 3 - n1; cnt = 3278 + need; sc = math.floor(cnt / 3357 * 1e5) / 1e5
        jl_ = next(l for l in L if "◀ 판정용" in l); print("  판정용 줄:", jl_.strip()[-120:]); ok &= f"공개 {sc:.5f}({cnt:,}문항) 이상이면 채택" in jl_ and ((f"공개가 {-need}문항까지 떨어져도 채택" in jl_) == (need < 0))
        m = SW if best == "swap" else MX
        tag = f"e13_{best}_" + "_".join(o.split("_")[0] + ("dev" if o.endswith("_dev") else "") + (f"x{len(m[o])}" if len(m[o]) > 1 else "") for o in sorted(m))
        exp_files = sorted(f"submission_{nm}__{tag}.csv" for nm in ("clean4_qg_hcx", "majority_vote")); print("  후보 파일:", files); ok &= files == exp_files
        for nm in ("clean4_qg_hcx", "majority_vote"):
            fp = f"{e13}/submission_{nm}__{tag}.csv"
            if not os.path.exists(fp): continue
            got_ = pd.read_csv(fp).answer.map("abcd".index).to_numpy(); exp = np.where(B1["test"][nm] != B0["test"][nm], B1["test"][nm], fa[nm])
            same = int((got_ == exp).sum()); print(f"  {os.path.basename(fp)} = 독립 계산 {same}/{len(exp)} (기존과 다른 답 {int((exp != fa[nm]).sum())})"); ok &= same == len(exp)
ok &= os.path.exists(f"{e13}/e13_info.json")
print("시험 도구 판정:", "통과" if ok else "실패")
