# -*- coding: utf-8 -*-
"""manual_ensemble.ipynb 시험 (CPU): E14 시험용 가짜 Drive(멤버 14개 + 제외·대기·_full 실행 + 기존 제출 파일)로 셀을 순서대로 실행하고,
결과를 원본 예측 파일에서 따로 계산한 값과 대조. 슬라이더 패널은 ipywidgets를 화면 없이 만들어 값 바꾸기·버튼 누르기로 시험."""
import sys, os, json, glob, shutil, re, io, contextlib, hashlib
import numpy as np, pandas as pd

SP = "/tmp/vqa_scratch"
SRC = f"{SP}/e14test"; ROOT = f"{SP}/mantest"
NB = "/home/user/vqa/manual_ens/out/manual_ensemble.ipynb"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
shutil.copytree(f"{SRC}/vqa_runs", f"{ROOT}/vqa_runs"); os.symlink(f"{SRC}/content", f"{ROOT}/content")
RUNS_DIR, DATA = f"{ROOT}/vqa_runs", f"{ROOT}/content"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, DATA_C, HELP, POOLC, FUNCS, PANEL, CODE7, SAVE8, REPRO9, CMP10 = cells
g = {"__name__": "__main__"}
def run(src, name, capture=False):
    print(f"\n━━━━ [{name}] ━━━━", flush=True)
    if capture:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): exec(compile(src, f"<{name}>", "exec"), g)
        print(buf.getvalue()); return buf.getvalue()
    exec(compile(src, f"<{name}>", "exec"), g)

run(CFG, "1 설정"); g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS_DIR, "DRIVE_ZIP": "/nonexistent.zip"})
run(DATA_C, "2 데이터"); run(HELP, "3 함수"); run(POOLC, "4 멤버 탐색"); out5 = run(FUNCS, "5 멤버 표·조합 함수", capture=True)

# ── 독립 계산 도구 (노트북 함수를 쓰지 않음) ──
LET = "abcd"; val_ids, test_ids = g["VAL_IDS"], g["TEST_IDS"]; y = g["Y_VAL"]
def rd(run_, split):
    d = f"{RUNS_DIR}/{run_}"; pres = sorted({m.group(1) for f in os.listdir(d) if (m := re.match(r"^preds_final_ep\d+__(up[\d.]+|native)__1000\.csv$", f))})
    pre = "up2.0" if "up2.0" in pres else pres[0]
    fs = sorted(f for f in os.listdir(d) if re.match(r"^preds_final_ep(\d+)" + ("_test" if split == "test" else "") + "__" + re.escape(pre) + r"__(\d+)\.csv$", f))
    df = pd.read_csv(f"{d}/{fs[-1]}").drop_duplicates("id", keep="last").set_index("id").loc[val_ids if split == "val" else test_ids]
    return df[["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def ts(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def fitT(p): return float(GRID[int(np.argmin([-np.mean(np.log(ts(p, T)[np.arange(len(y)), y])) for T in GRID]))])
def indep(weights, method, temp, full, split):
    rs = [r for r, w in weights.items() if w > 0]; w = np.array([weights[r] for r in rs], float); w /= w.sum(); Ps = []
    for r in rs:
        P = rd(r, split)
        if split == "test" and full != "none" and r in FULLOK:
            Pf = rd(r + "_full", "test"); P = (P + Pf) / 2 if full == "mix" else Pf
        if temp == "val": P = ts(P, fitT(rd(r, "val")))
        Ps.append(P)
    if method == "prob": return sum(a * b for a, b in zip(w, Ps))
    if method == "logprob":
        s = sum(a * np.log(np.clip(b, 1e-6, 1)) for a, b in zip(w, Ps)); s -= s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    return sum(a * np.eye(4)[b.argmax(1)] for a, b in zip(w, Ps)) + 1e-3 * sum(a * b for a, b in zip(w, Ps))

# ── 5번 셀 검사 ──
NAMES, VALONLY, RUNS = g["NAMES"], g["VALONLY"], g["RUNS"]
print("\n━━━━ 검사 ━━━━")
assert not any(r.endswith("_dry") or r.endswith("_full") for r in RUNS), RUNS
for bad in ["lowacc_run", "othersplit_run"]: assert bad not in RUNS, bad
assert "broken_run" in RUNS and "broken_run" not in g["PTEST"] and g["FAMR"]["broken_run"] == "?broken_run"      # train_log가 깨진 검증만 있는 실행 → 검증 탐색만
assert "hcx32b_r16_native_bf16" in RUNS and "hcx32b_r16_native_bf16" not in g["PTEST"], "검증만 있는 실행이 표에 있고 test는 없어야 함"
assert "gemma4_31b_r16_up20_dev_bf16" in RUNS and "gemma4_31b_r16_up20_dev_bf16" not in g["PTEST"], "test 진행 중 실행은 검증 탐색만"
FULLOK = set(g["FULL_OF"])
assert FULLOK == {"q35_27b_r16_native", "gemma4_31b_r16_up20"}, FULLOK         # hcx _full(답 20% 뒤바뀜)·q38 _full(test 없음) 제외
for r in RUNS:
    assert abs(g["ACC"][r] - (rd(r, "val").argmax(1) == y).mean() * 100) < 1e-9 and abs(g["TT"][r] - fitT(rd(r, "val"))) < 1e-12, r
print(f"✅ 멤버 {len(RUNS)}개 (test 있음 {len(g['PTEST'])}) · _full 짝 {sorted(FULLOK)} · 정확도·온도 = 독립 계산")
# 기준 조합 → 기존 제출 파일(clean4)과 test 답이 같아야 함 (E8에서 재현 확인한 레시피)
ref = g["REF_CFG"]; at_ref = g["ens"](ref, "test").argmax(1)
clean4 = pd.read_csv(f"{RUNS_DIR}/ensembles/submission_clean4_qg_hcx.csv").set_index("id").loc[test_ids].answer.map(LET.index).to_numpy()
assert (at_ref == clean4).all(), int((at_ref != clean4).sum())
print(f"✅ 기준 조합 {g['REF_NAME']} = 기존 제출 파일 submission_clean4_qg_hcx.csv 와 test 답 6,714개 모두 같음 (검증 {g['REF_EVAL']['acc']:.2f}%)")
# 무작위 조합 × 방법 × 온도 × _full → 독립 계산과 대조
rng = np.random.RandomState(0); test_runs = [r for r in RUNS if r in g["PTEST"]]
n_chk = 0
for trial in range(6):
    ws = {r: float(rng.choice([0, 0, 0.5, 1, 1.5, 2, 3])) for r in test_runs}
    ws["q35_27b_r16_native"] = 1.0
    for method in ["prob", "logprob", "vote"]:
        for temp in ["none", "val"]:
            for full in ["none", "mix", "swap"]:
                c = {"weights": ws, "method": method, "temp": temp, "full": full}
                for split in ["val", "test"]:
                    if split == "val" and full != "none": continue
                    S, E = g["ens"](c, split), indep(ws, method, temp, full, split)
                    assert np.allclose(S, E, atol=1e-12), (trial, method, temp, full, split)
                    n_chk += 1
                e = g["evaluate"](c); assert abs(e["acc"] - (indep(ws, method, temp, "none", "val").argmax(1) == y).mean() * 100) < 1e-9
print(f"✅ 조합 점수 {n_chk}가지 (방법 3 × 온도 2 × _full 3 × 검증/test) = 독립 계산")
# _full 섞기는 검증에 영향 없음
c0 = {"weights": {"q35_27b_r16_native": 1, "gemma4_31b_r16_up20": 1}, "method": "prob", "temp": "none"}
assert (g["evaluate"]({**c0, "full": "swap"})["ans"] == g["evaluate"]({**c0, "full": "none"})["ans"]).all()
assert (g["ens"]({**c0, "full": "swap"}, "test").argmax(1) != g["ens"]({**c0, "full": "none"}, "test").argmax(1)).any()
# 빼기/넣기 표 한 줄 대조
ab = g["ablation"](ref); rowq = ab[(ab["실행"] == "hcx32b_r16_native") & ab["바꾸면"].str.startswith("빼기")].iloc[0]
w2 = dict(ref["weights"]); w2.pop("hcx32b_r16_native"); a1 = indep(w2, "prob", "none", "none", "val").argmax(1); a0 = at0 = indep(ref["weights"], "prob", "none", "none", "val").argmax(1)
fix, brk = int(((a1 == y) & (a0 != y)).sum()), int(((a1 != y) & (a0 == y)).sum())
assert (rowq["고침"], rowq["새로 틀림"], rowq["순이득"], rowq["바뀐 답"]) == (fix, brk, fix - brk, int((a1 != a0).sum())), rowq.to_dict()
assert set(ab["실행"]) == set(RUNS)
rowu = ab[(ab["실행"] == "hcx32b_r16_native") & ab["바꾸면"].str.startswith("올리기")].iloc[0]; w3 = dict(ref["weights"]); w3["hcx32b_r16_native"] = 2.5
a3 = indep(w3, "prob", "none", "none", "val").argmax(1); assert rowu["바꾸면"] == "올리기 (2→2.5)" and rowu["순이득"] == int(((a3 == y) & (a0 != y)).sum()) - int(((a3 != y) & (a0 == y)).sum())
assert len(ab) == sum(3 if (w_ := ref["weights"].get(r, 0)) > 0.5 else (2 if w_ > 0 else 1) for r in RUNS), len(ab)
print("✅ 빼기/넣기 표 = 독립 계산 (예: hcx 빼기 고침", fix, "· 새로 틀림", brk, ")")
# 부호 검정·부트스트랩 기본 성질
assert g["sign_p"](0, 0) == 1.0 and abs(g["sign_p"](5, 5) - 0.0625) < 1e-12 and g["sign_p"](3, 6) == 1.0
lo, hi = g["boot_ci"](g["REF_EVAL"]["ok"], g["REF_EVAL"]["ok"]); assert lo == hi == 0.0

# ── 6번 셀: 슬라이더 패널 (화면 없이) ──
out6 = run(PANEL, "6 슬라이더 패널", capture=True)
SL = g["SL"]; assert set(SL) == set(RUNS)
assert all(abs(SL[r].value - float(ref["weights"].get(r, 0))) < 1e-12 for r in RUNS), "시작값 = 기준 조합"
assert "▶ 슬라이더 조합" in out6 and "vs 기준" in out6
buf = io.StringIO()
with contextlib.redirect_stdout(buf): SL["q3vl_8b_r16_native"].value = 2.0
o = buf.getvalue(); assert "q3vl_8b_r16_native×2" in o, o[:400]
e_ui = g["evaluate"](g["ui_cfg"]()); w_exp = dict(ref["weights"]); w_exp["q3vl_8b_r16_native"] = 2.0
assert abs(e_ui["acc"] - (indep(w_exp, "prob", "none", "none", "val").argmax(1) == y).mean() * 100) < 1e-9
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g["B_POOL"].click()
assert {r for r, s in SL.items() if s.value > 0} == set(g["STRONG"]) and all(SL[r].value == 1.0 for r in g["STRONG"])
assert buf.getvalue().count("▶ 슬라이더 조합") == 1, "버튼 한 번에 한 번만 갱신"
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g["DD_M"].value = "logprob"; g["DD_T"].value = "val"
assert "로그확률" in buf.getvalue() and "온도 보정 (검증" in buf.getvalue()
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g["B_CODE"].click()
code_txt = buf.getvalue(); ns = {}; exec(code_txt.split("\n", 1)[1], ns)
assert ns["WEIGHTS"] == {r: 1.0 for r in g["STRONG"]} and (ns["METHOD"], ns["TEMP"], ns["FULL_MODE"]) == ("logprob", "val", "none")
g["TX"].value = "패널 저장/테스트"
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g["B_SAVE"].click()
assert "✅ 저장" in buf.getvalue(), buf.getvalue()
saved_ui = sorted(glob.glob(f"{RUNS_DIR}/manual_ens/submission_*.csv")); assert len(saved_ui) == 1 and "패널_저장_테스트" in saved_ui[0], saved_ui
buf = io.StringIO()
with contextlib.redirect_stdout(buf): SL["hcx32b_r16_native_bf16"].value = 1.0; g["B_SAVE"].click()
assert "test 예측이 없는 실행" in buf.getvalue() and len(glob.glob(f"{RUNS_DIR}/manual_ens/submission_*.csv")) == 1, "검증만 있는 실행이 들어가면 저장 막음"
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g["B_ZERO"].click()
assert "가중치가 모두 0" in buf.getvalue() and all(s.value == 0 for s in SL.values())
with contextlib.redirect_stdout(io.StringIO()): g["B_REF"].click()
assert all(abs(SL[r].value - float(ref["weights"].get(r, 0))) < 1e-12 for r in RUNS) and g["DD_M"].value == "prob" and g["DD_T"].value == "none"
print("✅ 패널: 시작값·슬라이더 갱신·버튼 5개(기준/강한 풀/모두 0/코드로 출력/저장)·검증만 있는 실행 저장 막기")

# ── 7·8번 셀: 오타·검증만 있는 실행 경고 + 저장 ──
code7 = CODE7.replace('WEIGHTS = dict(REF_CFG["weights"])', 'WEIGHTS = {"q35_27b_r16_native": 1, "gemma4_31b_r16_up20": 1.5, "q38_27b_r16_native_bf16": 1, "없는_실행_오타": 2}') \
             .replace('METHOD, TEMP, FULL_MODE = "prob", "none", "none"', 'METHOD, TEMP, FULL_MODE = "prob", "val", "mix"')
assert code7 != CODE7
out7 = run(code7, "7 코드 조합", capture=True)
assert "없는_실행_오타" in out7 and "무시" in out7 and "기준 조합과 답이 바뀐 검증 문항" in out7 and "멤버 하나씩 바꿔 보기 (순이득 큰 순)" in out7
out8 = run(SAVE8, "8 저장", capture=True)
p8 = g["SAVED_PATH"]; info = json.load(open(p8[:-4] + ".json", encoding="utf-8"))
assert info["cfg"]["weights"] == {"q35_27b_r16_native": 1.0, "gemma4_31b_r16_up20": 1.5, "q38_27b_r16_native_bf16": 1.0} and info["cfg"]["full"] == "mix"
exp = indep(info["cfg"]["weights"], "prob", "val", "mix", "test").argmax(1)
sub8 = pd.read_csv(p8); assert list(sub8.id) == list(g["sub_df"].id) and (sub8.answer.map(LET.index).to_numpy() == exp).all()
assert "full_test" in info["members"]["q35_27b_r16_native"] and "full_test" not in info["members"]["q38_27b_r16_native_bf16"]
print("✅ 7·8번: 오타 경고 · 자세한 보고 · 제출 파일 = 독립 계산(온도·_full 섞기 포함) · 설정 JSON(파일 경로·md5·온도)")

# ── 9번 셀: 재현 + 파일이 바뀐 경우 감지 ──
out9 = run(REPRO9, "9 재현", capture=True); assert out9.count("완전히 같음 ✅") == 2, out9
tf = f"{RUNS_DIR}/gemma4_31b_r16_up20/" + [f for f in os.listdir(f"{RUNS_DIR}/gemma4_31b_r16_up20") if "_test__" in f][0]
d = pd.read_csv(tf); d.loc[:, ["p_a", "p_b", "p_c", "p_d"]] = d[["p_d", "p_c", "p_b", "p_a"]].to_numpy(); d.to_csv(tf, index=False)   # 멤버 예측을 바꿔치기
for k in list(g["_CACHE"]): g["_CACHE"].pop(k)
g["PTEST"]["gemma4_31b_r16_up20"] = g["load"]("gemma4_31b_r16_up20", "up2.0", "test")
buf = io.StringIO()
with contextlib.redirect_stdout(buf): same = g["rebuild"](p8[:-4] + ".json")
assert "바뀌었거나" in buf.getvalue() and not same, buf.getvalue()
print("✅ 9번: 저장한 설정으로 다시 만들면 완전히 같음 · 멤버 예측 파일이 바뀌면 경고")

# ── 10번 셀: 비교 ──
out10 = run(CMP10, "10 비교", capture=True)
assert "[0] manual_ens/" in out10 and "submission_majority_vote.csv" in out10 and "test 답이 다른 문항 수" in out10

# ── 시작값 옵션 ──
for sf, chk in [("pool", lambda: {r for r, s in g["SL"].items() if s.value > 0} == set(g["STRONG"])),
                (os.path.relpath(p8[:-4] + ".json", RUNS_DIR), lambda: g["SL"]["gemma4_31b_r16_up20"].value == 1.5 and g["DD_F"].value == "mix")]:
    g["START_FROM"] = sf
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(PANEL, "<panel2>", "exec"), g)
    assert chk(), sf
print("✅ START_FROM: 강한 풀 · 저장한 설정 JSON")
import time
for tm in ["none", "val"]:
    c_t = {"weights": {r: 1.0 for r in g["STRONG"]}, "method": "prob", "temp": tm, "full": "mix"}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()): g["report"](c_t, "t", brief=True)
    print(f"패널 한 번 갱신 (강한 풀 {len(g['STRONG'])}개 · 온도 {tm}): {time.time() - t0:.2f}초")
print("\n✅✅ manual_ensemble 시험 통과")
