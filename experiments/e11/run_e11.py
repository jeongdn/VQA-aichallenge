# -*- coding: utf-8 -*-
"""e11_combine_on_avg.ipynb 시험 도구 (E9 시험 데이터 재사용).
준비: python3 run_e11.py prep → E9·E5·HCX mock base 결과 + E10(해당 시험 데이터) → e10_info.json·avg_all
시나리오: python3 run_e11.py <scenario> [key=value ...]  (시나리오마다 vqa_runs 복사본에서 실행)
  base / partial(gemma4 test TTA 일부 빠짐) / nohcxhr(HCX 고해상도 파일 없음) / nohcxhr_ok(+ EXPECT_HR에서 HCX 뺌) /
  infeasible(HCX up4.0 건너뜀 기록) / health(q38 순서0 답 흔들림) / public(PUBLIC_BASE 지정)"""
import sys, os, json, shutil, subprocess
import numpy as np, pandas as pd
NB = "/home/user/vqa/e11/out/e11_combine_on_avg.ipynb"
SP = "/tmp/vqa_scratch"
SRC = f"{SP}/e9test"; WORK = f"{SP}/e11work"
MEM5 = ["gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "hcx32b_r16_native", "q35_27b_r16_native", "q38_27b_r16_native"]

if sys.argv[1] == "prep":
    def sh(c):
        print("$", c); r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-400:], r.stderr[-300:]); assert r.returncode == 0
    sh("cd /home/user/vqa/hcx && python3 run_hcx.py prep | tail -2")               # E9 시험 데이터 + HCX 어댑터 + E9·E5 mock base
    sh("cd /home/user/vqa/hcx && python3 run_hcx.py mock base | tail -1")          # HCX의 TTA·고해상도 파일
    cells = ["".join(c["source"]) for c in json.load(open("/home/user/vqa/e10/out/e10_ensemble_search.ipynb", encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "__main__"}
    exec(cells[0], g); g.update({"DATA_DIR": f"{SRC}/content", "OUT_DIR": f"{SRC}/vqa_runs", "MUST_INCLUDE": []})
    for c in cells[1:]: exec(c, g)
    print("E10 멤버:", g["MEMBERS_ALL"]); assert sorted(r for r, _ in g["MEMBERS_ALL"]) == MEM5
    print("준비 끝"); sys.exit()

scen = sys.argv[1]; over = dict(a.split("=", 1) for a in sys.argv[2:])
shutil.rmtree(WORK, ignore_errors=True); shutil.copytree(f"{SRC}/vqa_runs", f"{WORK}/vqa_runs"); R = f"{WORK}/vqa_runs"
if scen == "partial":
    p = f"{R}/e9_tta/gemma4_31b_r16_up20/perm_test__up2.0.csv"; d = pd.read_csv(p); d.drop(d[d.lora == 1].index[-5:]).to_csv(p, index=False)
if scen.startswith("nohcxhr"): shutil.rmtree(f"{R}/e5_hires/hcx32b_r16_native")
if scen == "infeasible":
    for s in ("val", "test"):
        p = f"{R}/e5_hires/hcx32b_r16_native/res_{s}.csv"; d = pd.read_csv(p); d[d.preset != "up4.0"].to_csv(p, index=False)
    ip = f"{R}/e5_hires/hcx32b_r16_native/hcx_info.json"; j = json.load(open(ip)); j["infeasible"] = {"hcx32b_r16_native": ["up4.0"]}; json.dump(j, open(ip, "w"))
if scen == "health":
    p = f"{R}/e9_tta/q38_27b_r16_native/perm_val__up2.0.csv"; d = pd.read_csv(p); m = (d.perm == 0) & (d.lora == 1) & (np.arange(len(d)) % 3 == 0)
    d.loc[m, ["p_a", "p_b", "p_c", "p_d"]] = d.loc[m, ["p_b", "p_c", "p_d", "p_a"]].to_numpy(); d.to_csv(p, index=False)
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
run(cells[0], "1 설정"); g.update({"DATA_DIR": f"{SRC}/content", "OUT_DIR": R})
if scen == "nohcxhr_ok": g["EXPECT_HR"] = ["q35_27b_r16_native", "q38_27b_r16_native"]
if scen == "public": g["PUBLIC_BASE"] = 0.97706
for k, v in over.items(): g[k] = eval(v, g)
for i, nm in enumerate(["2 데이터", "3 함수", "4 기준", "5 멤버별 결과", "6 판정", "7 요약"], 1): run(cells[i], nm)

print("\n━━━━ [시험 도구 확인] ━━━━")
ok = True
L = g["V"]["lines"]
def verdict(tag):
    head = {"tta": "[T:", "hires": "[R:", "tta_hires": "[TR:"}[tag]
    i = next(k for k, l in enumerate(L) if l.startswith(head))
    return L[i], (L[i + 1] if i + 1 < len(L) and L[i + 1].startswith("  판정") else L[i])
# ── 독립 계산 (노트북 코드를 쓰지 않고 CSV에서 직접) ──
T = np.load(f"{SRC}/truth.npz"); y = T["y"]
VIDS, TIDS = g["VAL_IDS"], g["TEST_IDS"]; VI = {i: n for n, i in enumerate(VIDS)}; TI = {i: n for n, i in enumerate(TIDS)}
names = [r for r, _ in g["MEMBERS"]]; ok &= sorted(names) == MEM5
P0 = {r: (T[f"PV_{r}"], T[f"PT_{r}"]) for r in names}
def avg_ans(Pm): return np.mean([Pm[r][0] for r in names], 0).argmax(1), np.mean([Pm[r][1] for r in names], 0).argmax(1)
base = avg_ans(P0); fa = pd.read_csv(f"{R}/e10_search/submission_e10_avg_all.csv").answer.map("abcd".index).to_numpy()
ok &= (fa == base[1]).all()
def perm_views(r, s):
    p = f"{R}/e9_tta/{r}/perm_{s}__up2.0.csv"
    if not os.path.exists(p): return None
    d = pd.read_csv(p); d = d[d.lora == 1].drop_duplicates(["id", "perm"], keep="last"); c = d.groupby("id").perm.nunique()
    if (c < 4).any(): return "partial"
    return {i: np.stack([d[(d.id == i) & (d.perm == k)][["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)[0] for k in range(4)]) for i in c.index}
def res_views(r, s, pres):
    p = f"{R}/e5_hires/{r}/res_{s}.csv"
    if not os.path.exists(p): return None
    d = pd.read_csv(p).drop_duplicates(["id", "preset"], keep="last")
    return {pr: {i: v for i, v in zip(d[d.preset == pr].id, d[d.preset == pr][["p_a", "p_b", "p_c", "p_d"]].to_numpy(float))} for pr in pres}
def member_T(r):
    out = {}
    for s_ in ("val", "test"):
        pv = perm_views(r, s_)
        if pv is None or pv == "partial": return None
        out[s_] = pv
    return out
def member_R(r):
    pres = [p for p in ["up3.0", "up4.0"] if not (scen == "infeasible" and r == "hcx32b_r16_native" and p == "up4.0")]
    out = {}
    for s_ in ("val", "test"):
        rv = res_views(r, s_, pres)
        if rv is None or any(len(rv[p]) == 0 for p in pres): return None
        out[s_] = rv
    return out, pres
def build(kind):
    Pm = {r: (P0[r][0].copy(), P0[r][1].copy()) for r in names}; used = []
    for r in names:
        t = member_T(r) if kind in ("T", "TR") else None
        rr = member_R(r) if kind in ("R", "TR") else None
        if not t and not rr: continue
        used.append(r)
        for si, s_ in enumerate(("val", "test")):
            idx = VI if s_ == "val" else TI
            for i in (set(t[s_]) if t else set(rr[0][s_][rr[1][0]])):
                if t and rr: views = list(t[s_][i]) + [rr[0][s_][p][i] for p in rr[1]]
                elif t: views = list(t[s_][i])
                else: views = [P0[r][si][idx[i]]] + [rr[0][s_][p][i] for p in rr[1]]
                Pm[r][si][idx[i]] = np.mean(views, 0)
    return avg_ans(Pm), used
exp_line = {}
for tag, kind in (("tta", "T"), ("hires", "R"), ("tta_hires", "TR")):
    (av, at), used = build(kind); c0, c1 = base[0] == y, av == y; f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum())
    head, v = verdict(tag); print(f"  {tag}: {v.strip()[:70]}… | 독립 계산 고침 {f}/새로 틀림 {b} (반영 {len(used)})")
    exp_line[tag] = (f, b, at, head, v)
want = {"base": {"tta": "공개 확인", "hires": "공개 확인", "tta_hires": "공개 확인"}, "public": {"tta": "공개 확인", "hires": "공개 확인", "tta_hires": "공개 확인"},
        "partial": {"tta": "대기", "hires": "공개 확인", "tta_hires": "대기"}, "nohcxhr": {"tta": "공개 확인", "hires": "대기", "tta_hires": "대기"},
        "nohcxhr_ok": {"tta": "공개 확인", "hires": "공개 확인", "tta_hires": "공개 확인"}, "infeasible": {"tta": "공개 확인", "hires": "공개 확인", "tta_hires": "공개 확인"},
        "health": {"tta": "확인 필요", "hires": "공개 확인", "tta_hires": "확인 필요"}}[scen]
for tag, (f, b, at, head, v) in exp_line.items():
    w = want[tag]
    if f"순이득 {f - b:+d}" in head and f - b <= -3 and w == "공개 확인": w = "불채택"
    hit = w in v; print(f"  {tag}: 기대 '{w}' → {'OK' if hit else '다름'}"); ok &= hit
    ok &= "반영 멤버" in head and f"고침 {f} | 새로 틀림 {b}" in head
    if w == "공개 확인":
        fp = f"{R}/e11_combine/submission_e11_{tag}.csv"; f1 = pd.read_csv(fp).answer.map("abcd".index).to_numpy()
        e1 = np.where(at != base[1], at, fa); same = int((f1 == e1).sum()); print(f"    {os.path.basename(fp)} = 독립 계산 {same}/{len(f1)}"); ok &= same == len(f1)
if scen == "public": ok &= any("0.977" in l and "이상이면 채택" in l for l in L)
print("시험 도구 판정:", "통과" if ok else "실패")
