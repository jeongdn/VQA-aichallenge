# -*- coding: utf-8 -*-
"""ens_search.ipynb 시험 (CPU). E14 시험용 가짜 Drive에 두 실행을 더함:
  - q35_27b_r16_native_full_old: train_log에 TRAIN_ON_FULL이 없는 _full (이름으로 빠져야 함)
  - leak_run: 검증 정답을 거의 다 맞히는 실행 (검증 > 99% → 빠져야 함)
노트북 셀을 차례로 실행하고, 전수 검색·그리디·좌표 상승·중첩 CV를 노트북 함수를 쓰지 않는 독립 계산과 대조한다.
_full 짝(제출 파일에만 섞기): q35_27b_r16_native·gemma4_31b_r16_up20만 통과 (hcx 79% 일치 → 거부, q38 test 없음)
사용: python3 run_search.py [시나리오 ...]   (main / valonly / volist / noref / modes / pairs / timing)"""
import sys, os, json, shutil, re, io, contextlib, itertools, time, glob
import numpy as np, pandas as pd

SP = "/tmp/vqa_scratch"
NB = "/home/user/vqa/ens_search/out/ens_search.ipynb"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
assert len(cells) == 14, len(cells)
CFG, DATA_C, HELP, POOLC, MEMF, PREP, EXHC, GRDC, SUMC, PANEL, DETAIL, SAVEC, REPRO, CMP = cells
LET = "abcd"

def make_drive(root):
    shutil.rmtree(root, ignore_errors=True); os.makedirs(root)
    shutil.copytree(f"{SP}/e14test/vqa_runs", f"{root}/vqa_runs"); os.symlink(f"{SP}/e14test/content", f"{root}/content")
    return f"{root}/vqa_runs", f"{root}/content"

def add_runs(runs_dir, g):
    # (1) TRAIN_ON_FULL 표시가 없는 _full
    src, dst = f"{runs_dir}/q35_27b_r16_native_full", f"{runs_dir}/q35_27b_r16_native_full_old"
    shutil.copytree(src, dst); lg = json.load(open(f"{dst}/train_log.json")); lg["config"].pop("TRAIN_ON_FULL"); json.dump(lg, open(f"{dst}/train_log.json", "w"))
    # (2) 검증 정답 누출 실행
    d = f"{runs_dir}/leak_run"; os.makedirs(d)
    rng = np.random.RandomState(5); P = np.full((1000, 4), 0.01); P[np.arange(1000), g["Y_VAL"]] = 0.97
    flip = rng.choice(1000, 3, replace=False); P[flip] = np.roll(P[flip], 1, axis=1)          # 99.7%
    pd.DataFrame({"id": g["VAL_IDS"], "pred": [LET[i] for i in P.argmax(1)], **{f"p_{L}": P[:, i] for i, L in enumerate(LET)}, "letter_mass": 1.0, "img_tokens": 1}).to_csv(f"{d}/preds_final_ep1__up2.0__1000.csv", index=False)
    shutil.copy(f"{runs_dir}/q38_27b_r16_native/preds_final_ep1_test__up2.0__6714.csv", f"{d}/preds_final_ep1_test__up2.0__6714.csv")
    json.dump({"epoch_acc": {"1": 0.99}, "config": {"MODEL_ID": "Qwen/Qwen3.5-27B"}}, open(f"{d}/train_log.json", "w"))

def run_cells(g, srcs, quiet=True):
    for name, src in srcs:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf): exec(compile(src, f"<{name}>", "exec"), g)
        except Exception:
            print(buf.getvalue()[-3000:]); raise
        out = buf.getvalue()
        if not quiet: print(f"\n━━━━ [{name}] ━━━━\n{out}")
        g.setdefault("_OUT", {})[name] = out

def start(scn, **over):
    runs_dir, data_dir = make_drive(f"{SP}/search_{scn}")
    g = {"__name__": "__main__"}
    run_cells(g, [("1 설정", CFG)])
    g.update({"DATA_DIR": data_dir, "OUT_DIR": runs_dir, "DRIVE_ZIP": "/nonexistent.zip"}); g.update(over)
    run_cells(g, [("2 데이터", DATA_C), ("3 함수", HELP)])
    add_runs(runs_dir, g)
    return g, runs_dir

# ── 독립 계산 도구 ──
GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
def ts(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
def read_run(runs_dir, run, split, ids):
    d = f"{runs_dir}/{run}"; pres = sorted({m.group(1) for f in os.listdir(d) if (m := re.match(r"^preds_final_ep\d+__(up[\d.]+|native)__1000\.csv$", f))})
    pre = "up2.0" if "up2.0" in pres else pres[0]
    f = [x for x in os.listdir(d) if re.match(r"^preds_final_ep\d+" + ("_test" if split == "test" else "") + "__" + re.escape(pre) + r"__\d+\.csv$", x)][0]
    df = pd.read_csv(f"{d}/{f}").drop_duplicates("id", keep="last").set_index("id").loc[ids]
    return df[["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)

def indep_setup(g, runs_dir):
    y = g["Y_VAL"]; pool = g["SEARCH_POOL"]
    PV = {r: read_run(runs_dir, r, "val", g["VAL_IDS"]) for r in pool}
    T = {r: float(GRID[int(np.argmin([-np.mean(np.log(ts(PV[r], t)[np.arange(len(y)), y])) for t in GRID]))]) for r in pool}
    def stack(setting):
        A = np.stack([ts(PV[r], T[r]) if setting[1] == "val" else PV[r] for r in pool])
        return np.log(np.clip(A, 1e-6, 1)) if setting[0] == "logprob" else A
    return y, pool, stack

def okl(Sm, method, y):
    ar = np.arange(len(y))
    if method == "logprob": Z = Sm - Sm.max(-1, keepdims=True); lp = Z[..., ar, y] - np.log(np.exp(Z).sum(-1))
    else: lp = np.log(np.clip(Sm[..., ar, y], 1e-12, 1))
    return (Sm.argmax(-1) == y), -lp

FULLOK = {"q35_27b_r16_native": "q35_27b_r16_native_full", "gemma4_31b_r16_up20": "gemma4_31b_r16_up20_full"}
def comb_i(g, runs_dir, c, split, full=None, fullok=FULLOK):
    # 노트북 함수 없이: 멤버 확률(test면 _full 섞기) → 온도 → 가중 평균 (logprob은 로그 평균) → 행마다 argmax 가능한 점수
    y = g["Y_VAL"]; ar = np.arange(len(y)); full = c["full"] if full is None else full
    w = c["weights"]; runs = list(w); wv = np.array([w[r] for r in runs]); wv = wv / wv.sum(); Ps = []
    for r in runs:
        P = read_run(runs_dir, r, split, g["VAL_IDS"] if split == "val" else g["TEST_IDS"])
        if split == "test" and full != "none" and r in fullok:
            Pf = read_run(runs_dir, fullok[r], "test", g["TEST_IDS"]); P = (P + Pf) / 2 if full == "mix" else Pf
        if c["temp"] == "val":
            Pv = read_run(runs_dir, r, "val", g["VAL_IDS"]); Tr = float(GRID[int(np.argmin([-np.mean(np.log(ts(Pv, t)[ar, y])) for t in GRID]))]); P = ts(P, Tr)
        Ps.append(P)
    if c["method"] == "logprob": return sum(a * np.log(np.clip(b, 1e-6, 1)) for a, b in zip(wv, Ps))
    return sum(a * b for a, b in zip(wv, Ps))

def check_cands(g, runs_dir, fullok=FULLOK):
    # 9번 후보 표: 검증 정확도 · _full 칸 · _full로 바뀐 답 · test≠clean4 = 독립 계산 / 기준 줄 = 지금 제출 그대로
    y = g["Y_VAL"]; CT, CANDS, FM = g["CAND_TABLE"], g["CANDS"], g["FULL_MODE"]
    ref_t = pd.read_csv(f"{runs_dir}/ensembles/submission_clean4_qg_hcx.csv").set_index("id").loc[g["TEST_IDS"]].answer.map(LET.index).to_numpy()
    n_full_rows = 0
    for _, rw in CT.iterrows():
        c = CANDS[rw["후보"]]
        want_full = "none" if rw["후보"] == f"기준 {g['REF_NAME']}" else FM
        assert c["full"] == want_full, (rw["후보"], c["full"], want_full)
        assert abs(rw["검증(%)"] - round(float((comb_i(g, runs_dir, c, "val").argmax(1) == y).mean() * 100), 2)) < 1e-9, rw["후보"]
        nf = sum(r in fullok for r in c["weights"]) if c["full"] != "none" else 0
        assert rw["_full"] == (f"{c['full']} {nf}개" if nf else "-"), (rw["후보"], rw["_full"], nf)
        if rw["저장"] != "O": continue
        at = comb_i(g, runs_dir, c, "test").argmax(1)
        if nf:
            n_full_rows += 1
            assert rw["_full로 바뀐 답"] == int((at != comb_i(g, runs_dir, c, "test", "none").argmax(1)).sum()), rw["후보"]
        else: assert rw["_full로 바뀐 답"] == "-"
        assert rw["test≠clean4_qg_hcx"] == int((at != ref_t).sum()), (rw["후보"], rw["test≠clean4_qg_hcx"], int((at != ref_t).sum()))
    if g["REF_EVAL"] is not None:
        assert CT.iloc[0]["후보"] == f"기준 {g['REF_NAME']}" and CT.iloc[0]["test≠clean4_qg_hcx"] == 0 and CT.iloc[0]["_full"] == "-"
        rf = f"기준 {g['REF_NAME']} + _full {FM}"
        has_rf = rf in CANDS or any(rf in str(x) for x in CT["같은 조합"])
        assert has_rf == (FM != "none"), (FM, list(CANDS)[:3])
    return n_full_rows

def check_main():
    print("\n════════ 시나리오 main (기본 설정) ════════")
    g, runs_dir = start("main")
    run_cells(g, [("4 멤버 탐색", POOLC), ("5 멤버 표", MEMF), ("6 검색 준비", PREP)], quiet=False)
    MEM, RUNS = g["MEMBERS"], g["RUNS"]
    assert "q35_27b_r16_native_full_old" in g["NAMES"] and "q35_27b_r16_native_full_old" not in RUNS and "q35_27b_r16_native_full_old" in g["_full_out"]
    assert "leak_run" in g["NAMES"] and "leak_run" in g["_suspect"] and "leak_run" not in RUNS
    assert not any("full" in r.split("_") for r in RUNS) and all(r in g["PTEST"] for r in MEM) and not any(r in g["PTEST"] for r in RUNS if r not in MEM)
    assert g["SEARCH_POOL"] == MEM and len(MEM) == 12, (len(MEM), MEM)
    assert g["S_MAX"] == 12 and g["N_SUB"] == 4095
    HELD = g["HELD"]; K, R = g["CV_K"], g["CV_REPEATS"]
    for rp in range(R):
        cnt = np.sum([HELD[rp * K + k] for k in range(K)], 0); assert (cnt == 1).all() and all(HELD[rp * K + k].sum() == 200 for k in range(K))
    print("✅ _full 제외(train_log 표시 없는 것도 이름으로) · 누출 실행 제외 · 검색 풀 12개 · CV 접기(5겹 × 4번, 각 200문항)")
    o5 = g["_OUT"]["5 멤버 표"]
    assert set(g["FULL_OF"]) == set(FULLOK), sorted(g["FULL_OF"])
    for r, fr in FULLOK.items():
        assert np.allclose(g["FULL_OF"][r], read_run(runs_dir, fr, "test", g["TEST_IDS"])) and g["FULL_FILE"][r].endswith(f"{fr}/preds_final_ep1_test__up2.0__6714.csv")
    assert "_full hcx32b_r16_native → hcx32b_r16_native_full@up2.0: test 답 일치 79.1% ⚠️ 안 씀 (일치율 < 97.0%)" in o5, o5[:3000]
    assert "_full q38_27b_r16_native → q38_27b_r16_native_full: test 예측 없음 → 안 씀" in o5
    assert "짝이 있는 멤버 2개" in o5 and "(FULL_MODE='mix')" in o5
    mt = g["MEM_TABLE"].set_index("실행")["_full"]
    assert mt["q35_27b_r16_native"].startswith("q35_27b_r16_native_full@up2.0 99.0") and mt["hcx32b_r16_native"] == "" and mt["q38_27b_r16_native"] == ""
    print("✅ _full 짝: q35·gemma4 통과(99%대) · hcx 거부(79.1% 일치) · q38 test 없음 · 멤버 표 _full 칸")

    t0 = time.time(); run_cells(g, [("7 전수 검색", EXHC)], quiet=False); print(f"(7번 셀 {time.time() - t0:.1f}초)")
    y, pool, stack = indep_setup(g, runs_dir); M = len(pool); ar = np.arange(len(y))
    subs = [t for k in range(1, M + 1) for t in itertools.combinations(range(M), k)]
    for s in g["SEARCH_SETTINGS"]:
        A = stack(s); OK = np.zeros((len(subs), len(y)), bool); LL = np.zeros((len(subs), len(y)))
        for i, t in enumerate(subs): OK[i], LL[i] = okl(A[list(t)].mean(0), s[0], y)
        sc_acc = OK.sum(1) - 1e-5 * LL.sum(1); sc_nll = -LL.sum(1)
        for rule, sc in (("acc", sc_acc), ("nll", sc_nll)):
            order = sorted(range(len(subs)), key=lambda i: (-sc[i], len(subs[i]), subs[i]))
            mine = [sub for _, sub in g["EXH"][s]["top"][rule][:30]]; ref_ = [subs[i] for i in order[:30]]
            if mine != ref_:
                bad = [(k, a, b, sc[subs.index(a)], sc[subs.index(b)]) for k, (a, b) in enumerate(zip(mine, ref_)) if a != b][:3]
                assert all(abs(x[3] - x[4]) < 1e-6 for x in bad), (s, rule, bad)        # 순서가 다르면 점수가 사실상 같아야 함
            for f in range(len(HELD)):
                m = ~HELD[f]; tr = OK[:, m].sum(1) - 1e-5 * LL[:, m].sum(1) if rule == "acc" else -LL[:, m].sum(1)
                j = sorted(range(len(subs)), key=lambda i: (-tr[i], len(subs[i]), subs[i]))[0]; got = g["EXH"][s]["fb"][rule]["sub"][f]
                assert got == subs[j] or abs(tr[subs.index(got)] - tr[j]) < 1e-6, (s, rule, f, got, subs[j])
            # OOF 정확도 = 독립 계산
            oof = np.full((R, len(y)), -1)
            for f in range(len(HELD)):
                t = g["EXH"][s]["fb"][rule]["sub"][f]; Sm = A[list(t)].mean(0); oof[f // K, HELD[f]] = Sm.argmax(1)[HELD[f]]
            assert (oof == g["EXH"][s]["oof_" + rule]).all(), (s, rule)
        def cons_i(scores):
            order = sorted(range(len(subs)), key=lambda i: (-scores[i], len(subs[i]), subs[i]))[:100]
            cnt = np.zeros(M)
            for i in order: cnt[list(subs[i])] += 1
            k = max(1, int(round(float(np.median([len(subs[i]) for i in order])))))
            return tuple(sorted(sorted(range(M), key=lambda i: (-cnt[i], i))[:k]))
        assert g["EXH"][s]["cons"] == cons_i(sc_acc), (s, g["EXH"][s]["cons"], cons_i(sc_acc))
        oofc = np.full((R, len(y)), -1)
        for f in range(len(HELD)):
            m = ~HELD[f]; tr = OK[:, m].sum(1) - 1e-5 * LL[:, m].sum(1)
            t = cons_i(tr); oofc[f // K, HELD[f]] = A[list(t)].mean(0).argmax(1)[HELD[f]]
        assert (oofc == g["EXH"][s]["oof_cons"]).all(), s
    print("✅ 전수 검색: 설정 3개 × 규칙 2개 — 상위 30개 순서 · fold 20개마다 고른 조합 · OOF 답 · 합의 조합(전체·fold별) = 부분집합 4,095개 독립 계산")

    t0 = time.time(); run_cells(g, [("8 그리디·좌표", GRDC)], quiet=False); print(f"(8번 셀 {time.time() - t0:.1f}초)")
    def greedy_i(A, method, mask, steps):
        cnt = np.zeros(M); S = np.zeros(A.shape[1:]); best = (-np.inf, None)
        for _ in range(steps):
            scs = []
            for m in range(M):
                ok, ll = okl((S + A[m]) / (cnt.sum() + 1), method, y); ok, ll = (ok[mask], ll[mask]) if mask is not None else (ok, ll)
                scs.append(ok.sum() - 1e-5 * ll.sum())
            m = int(np.argmax(scs)); cnt[m] += 1; S = S + A[m]
            if scs[m] > best[0] + 1e-9: best = (scs[m], cnt.copy())
        return best[1]
    def coord_i(A, method, w0, mask, grid):
        w = np.array(w0, float)
        def sc(wv):
            if wv.sum() <= 0: return -np.inf
            ok, ll = okl(np.tensordot(wv, A, 1) / wv.sum(), method, y); ok, ll = (ok[mask], ll[mask]) if mask is not None else (ok, ll)
            return ok.sum() - 1e-5 * ll.sum()
        cur = sc(w)
        for _ in range(8):
            moved = False
            for m in range(M):
                best_v, best_s = None, cur
                for v in grid:
                    w2 = w.copy(); w2[m] = v; s2 = sc(w2)
                    if s2 > best_s + 1e-9 and (best_v is None or s2 > best_s): best_v, best_s = v, s2
                if best_v is not None: w[m] = best_v; cur = best_s; moved = True
            if not moved: break
        return w
    w_ref0 = np.array([float(g["REF_CFG"]["weights"].get(r, 0)) for r in pool])
    for s in g["SEARCH_SETTINGS"]:
        A = stack(s)
        assert np.allclose(greedy_i(A, s[0], None, g["GREEDY_STEPS"]), g["GRD"][s]["w"]), (s, g["GRD"][s]["w"])
        assert np.allclose(coord_i(A, s[0], w_ref0, None, g["WEIGHT_GRID"]), g["CRD"][s]["w"]), (s, g["CRD"][s]["w"])
        for f in (0, 7, 19):
            m = ~HELD[f]
            gw = greedy_i(A, s[0], m, g["GREEDY_STEPS"]); cw = coord_i(A, s[0], w_ref0, m, g["WEIGHT_GRID"])
            assert g["GRD"][s]["cv"][f]["weights"] == {pool[i]: float(v) for i, v in enumerate(gw) if v > 0}, (s, f)
            assert g["CRD"][s]["cv"][f]["weights"] == {pool[i]: float(v) for i, v in enumerate(cw) if v > 0}, (s, f)
    print("✅ 그리디·좌표 상승: 설정 3개 — 전체 검증 결과 + fold 3개(0·7·19) = 독립 구현")

    run_cells(g, [("9 결과 정리", SUMC)], quiet=False)
    CVT, CT, CANDS = g["CV_TABLE"], g["CAND_TABLE"], g["CANDS"]
    ref_ans = g["REF_EVAL"]["ans"]
    for _, row in CVT.iterrows():
        if row["방법"] == "모든 방법 중 최고": continue
        s = next(s for s in g["SEARCH_SETTINGS"] if g["set_name"](s) == row["설정"])
        oof = {"전수(정확도)": g["EXH"][s]["oof_acc"], "전수(NLL)": g["EXH"][s]["oof_nll"], "전수 합의": g["EXH"][s]["oof_cons"], "그리디": g["GRD"][s]["oof"], "좌표 상승": g["CRD"][s]["oof"]}[row["방법"]]
        assert abs(row["중첩 CV(%)"] - float((oof == y).mean() * 100)) <= 0.005 + 1e-9, (row["방법"], row["설정"], row["중첩 CV(%)"], float((oof == y).mean() * 100))
        fx = [((o == y) & (ref_ans != y)).sum() for o in oof]; bx = [((o != y) & (ref_ans == y)).sum() for o in oof]
        assert abs(row["CV 고침"] - round(float(np.mean(fx)), 1)) < 1e-9 and abs(row["CV 새로 틀림"] - round(float(np.mean(bx)), 1)) < 1e-9
    # '모든 방법 중 최고' OOF 독립 계산
    procs = []
    for s in g["SEARCH_SETTINGS"]:
        procs += [g["EXH"][s]["cv_acc"], g["EXH"][s]["cv_nll"], g["EXH"][s]["cv_cons"], g["GRD"][s]["cv"], g["CRD"][s]["cv"]]
    oof_all = np.full((R, len(y)), -1)
    for f in range(len(HELD)):
        m = ~HELD[f]; best, bs = None, -np.inf
        for pc in procs:
            S = g["ens"](pc[f], "val"); P = S / S.sum(1, keepdims=True)
            sc = (S.argmax(1) == y)[m].sum() - 1e-5 * (-np.log(np.clip(P[ar, y], 1e-12, 1))[m]).sum()
            if sc > bs: best, bs = pc[f], sc
        oof_all[f // K, HELD[f]] = g["ens"](best, "val").argmax(1)[HELD[f]]
    row = CVT[CVT["방법"] == "모든 방법 중 최고"].iloc[0]; assert abs(row["중첩 CV(%)"] - float((oof_all == y).mean() * 100)) <= 0.005 + 1e-9
    # 후보 표: 검증·_full·test 차이 = 독립 계산
    nfr = check_cands(g, runs_dir)
    assert nfr >= 2 and f"기준 {g['REF_NAME']} + _full mix" in CANDS and CANDS[f"기준 {g['REF_NAME']} + _full mix"]["weights"] == CANDS[f"기준 {g['REF_NAME']}"]["weights"]
    sj = json.load(open(glob.glob(f"{runs_dir}/ens_search/search_*.json")[0], encoding="utf-8")); assert sj["full"]["mode"] == "mix" and set(sj["full"]["pairs"]) == set(FULLOK)
    assert len(glob.glob(f"{runs_dir}/ens_search/search_*_cv.csv")) == 1 and len(glob.glob(f"{runs_dir}/ens_search/search_*_candidates.csv")) == 1
    print(f"✅ 9번: 중첩 CV 표 {len(CVT)}줄 · '모든 방법 중 최고' · 후보 {len(CT)}개(검증·_full 칸·_full로 바뀐 답·test 차이) = 독립 계산 · 기준 줄 = 지금 제출 · '기준 + _full mix' 줄 · 표 CSV·JSON 저장")

    # ── 10번 패널 (화면 없이) ──
    run_cells(g, [("10 패널", PANEL)])
    SL, DD_C, TX = g["SL"], g["DD_C"], g["TX"]
    assert set(SL) == set(RUNS) and "▶ 슬라이더 조합" in g["_OUT"]["10 패널"]
    DD_F = g["DD_F"]; assert DD_F.value == "mix" and "_full 섞은 멤버 2개" in g["_OUT"]["10 패널"]      # 처음: 기준 가중치 + FULL_MODE
    pick = [k for k in CANDS if k.startswith("그리디")][0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): DD_C.value = pick
    assert TX.value == pick and all(abs(SL[r].value - CANDS[pick]["weights"].get(r, 0.0)) < 1e-12 for r in SL) and g["DD_M"].value == CANDS[pick]["method"] and g["DD_T"].value == CANDS[pick]["temp"]
    assert DD_F.value == CANDS[pick]["full"] == "mix"
    assert buf.getvalue().count("▶ 슬라이더 조합") == 1
    with contextlib.redirect_stdout(io.StringIO()): g["B_SAVE"].click()
    f1 = glob.glob(f"{runs_dir}/ens_search/submission_*.csv"); assert len(f1) == 1 and re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", pick).strip("_") in f1[0], f1
    j1 = json.load(open(f1[0][:-4] + ".json", encoding="utf-8")); assert j1["search"]["방법"] == "그리디" and j1["cfg"] == CANDS[pick]
    for r, m in j1["members"].items(): assert ("full_test" in m) == (r in FULLOK), (r, m)
    assert (pd.read_csv(f1[0]).answer.map(LET.index).to_numpy() == comb_i(g, runs_dir, CANDS[pick], "test").argmax(1)).all()
    r0 = [r for r in MEM if SL[r].value == 0][0]
    with contextlib.redirect_stdout(io.StringIO()): SL[r0].value = 1.0; g["B_SAVE"].click()
    f2 = sorted(glob.glob(f"{runs_dir}/ens_search/submission_*.csv")); assert len(f2) == 2 and any("수정" in x for x in f2), f2
    vo = [r for r in RUNS if r not in g["PTEST"]][0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): SL[vo].value = 1.0; g["B_SAVE"].click()
    assert "test 예측이 없는 실행" in buf.getvalue() and len(glob.glob(f"{runs_dir}/ens_search/submission_*.csv")) == 2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): g["B_CODE"].click()
    ns = {}; exec(buf.getvalue().split("\n", 1)[1], ns); assert ns["WEIGHTS"][vo] == 1 and ns["WEIGHTS"][r0] == 1
    with contextlib.redirect_stdout(io.StringIO()): DD_F.value = "swap"; g["B_REF"].click()
    assert all(abs(SL[r].value - g["REF_CFG"]["weights"].get(r, 0)) < 1e-12 for r in SL) and DD_F.value == "swap"   # 버튼은 test 메뉴를 그대로 둠
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): g["B_CODE"].click()
    ns = {}; exec(buf.getvalue().split("\n", 1)[1], ns); assert ns["FULL"] == "swap" and ns["METHOD"] == g["REF_CFG"]["method"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): DD_F.value = "none"
    assert "_full 섞은 멤버" not in buf.getvalue() and "test = 원래 실행" in buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): DD_C.value = f"기준 {g['REF_NAME']} + _full mix"
    assert DD_F.value == "mix" and "_full 섞은 멤버 2개" in buf.getvalue() and buf.getvalue().count("▶ 슬라이더 조합") == 1
    with contextlib.redirect_stdout(io.StringIO()): DD_C.value = f"기준 {g['REF_NAME']}"
    assert DD_F.value == "none"
    print("✅ 10번 패널: 후보 불러오기(슬라이더·결합·온도·test·저장 이름) · 저장 JSON에 검색 기록·_full 파일 · 저장 답 = 독립 계산 · 고치면 '수정' 이름 · 검증만 있는 실행 저장 막기 · 코드로 출력(FULL 포함) · 버튼은 test 메뉴 유지")

    # ── 11~14번 ──
    top1 = [k for k in CANDS if k.startswith("전수 1위")][0]
    det = DETAIL.replace("PICK = None ", f'PICK = "{top1}" ', 1); assert det != DETAIL
    run_cells(g, [("11 자세히", det), ("12 저장", SAVEC)], quiet=False)
    assert "검색 기록: 전수(정확도)" in g["_OUT"]["11 자세히"] and "test = 원래 + _full 평균" in g["_OUT"]["11 자세히"]
    sub12 = pd.read_csv(g["SAVED_PATH"]); c = CANDS[top1]; assert g["CFG_NOW"] == c and g["NAME"] == top1
    assert (sub12.answer.map(LET.index).to_numpy() == g["ens"](c, "test").argmax(1)).all()
    assert (sub12.answer.map(LET.index).to_numpy() == comb_i(g, runs_dir, c, "test").argmax(1)).all()
    j12 = json.load(open(g["SAVED_PATH"][:-4] + ".json", encoding="utf-8")); assert j12["search"]["방법"] == "전수(정확도)" and j12["cfg"]["full"] == "mix"
    # FULL = "none"으로 덮어쓰기 → 이름에 표시 · 검색 기록 유지 · _full 파일 없음 · 답 = 섞지 않은 독립 계산
    det2 = det.replace("FULL = FULL_MODE ", 'FULL = "none" ', 1); assert det2 != det
    run_cells(g, [("11 자세히 (none)", det2), ("12 저장 (none)", SAVEC)])
    assert g["NAME"] == f"{top1} · test none" and g["CFG_NOW"]["full"] == "none"
    j12n = json.load(open(g["SAVED_PATH"][:-4] + ".json", encoding="utf-8"))
    assert j12n["search"]["방법"] == "전수(정확도)" and not any("full_test" in m for m in j12n["members"].values())
    assert (pd.read_csv(g["SAVED_PATH"]).answer.map(LET.index).to_numpy() == comb_i(g, runs_dir, c, "test", "none").argmax(1)).all()
    # PICK 없이 WEIGHTS만 → 검색 기록 없음
    det3 = DETAIL.replace('NAME = "my_search"', 'NAME = "manual_x"', 1)
    run_cells(g, [("11 직접", det3), ("12 저장 (직접)", SAVEC)])
    j3 = json.load(open(g["SAVED_PATH"][:-4] + ".json", encoding="utf-8")); assert "search" not in j3 and j3["cfg"]["full"] == "mix" and j3["cfg"]["weights"] == CANDS[f"기준 {g['REF_NAME']}"]["weights"]
    # 기준 + _full mix 저장 (q35·gemma4 둘 다 섞임) → JSON에 _full 파일·md5. 11번 '직접'(기준 가중치 + mix)과 설정이 같아 같은 지문
    rf = f"기준 {g['REF_NAME']} + _full mix"
    with contextlib.redirect_stdout(io.StringIO()): prf = g["save"](CANDS[rf], rf)
    assert prf[-10:] == g["SAVED_PATH"][-10:], (prf, g["SAVED_PATH"])
    jrf = json.load(open(prf[:-4] + ".json", encoding="utf-8"))
    for r, fr in FULLOK.items():
        m = jrf["members"][r]; assert m["full_test"] == f"{fr}/preds_final_ep1_test__up2.0__6714.csv" and m["full_test_md5"] == g["md5file"](f"{runs_dir}/{m['full_test']}")
    assert "search" not in jrf and (pd.read_csv(prf).answer.map(LET.index).to_numpy() == comb_i(g, runs_dir, CANDS[rf], "test").argmax(1)).all()
    run_cells(g, [("13 재현", REPRO), ("14 비교", CMP)], quiet=False)
    nj = len(glob.glob(f"{runs_dir}/ens_search/submission_*.json"))
    assert nj == 6 and g["_OUT"]["13 재현"].count("완전히 같음 ✅") == nj and g["_OUT"]["13 재현"].count("test = 원래 + _full 평균") == 5, g["_OUT"]["13 재현"]
    assert "test 답이 다른 문항 수" in g["_OUT"]["14 비교"] and "submission_majority_vote.csv" in g["_OUT"]["14 비교"] and g["_OUT"]["14 비교"].count("· test에 _full mix") == 5
    # _full 파일이 저장 뒤 바뀌면 재현 점검이 알려야 함
    fp = f"{runs_dir}/gemma4_31b_r16_up20_full/preds_final_ep1_test__up2.0__6714.csv"; df = pd.read_csv(fp)
    df.loc[:30, ["p_a", "p_b", "p_c", "p_d"]] = df.loc[:30, ["p_d", "p_c", "p_b", "p_a"]].to_numpy(); df.to_csv(fp, index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): g["rebuild"](prf[:-4] + ".json")
    assert "저장한 뒤 예측 파일이 바뀌었거나 다시 써진 실행: ['gemma4_31b_r16_up20']" in buf.getvalue(), buf.getvalue()
    print("✅ 11~14번: 후보 자세히(검색 기록·_full) · 저장 = 조합 test 답 = 독립 계산 · FULL 덮어쓰기(이름 표시·기록 유지) · 직접 가중치 · JSON에 _full 파일·md5 · 재현 6개 모두 같음 · 비교 표 · _full 파일 변조 감지")
    return g

def check_valonly():
    print("\n════════ 시나리오 valonly (검증만 있는 실행도 검색, 설정 1개, 크기 ≤ 3) ════════")
    g, runs_dir = start("valonly", SEARCH_INCLUDE_VALONLY=True, SEARCH_SETTINGS=[("prob", "none")], MAX_SIZE=3)
    run_cells(g, [("4", POOLC), ("5", MEMF), ("6", PREP), ("7", EXHC), ("8", GRDC), ("9", SUMC)])
    vo = [r for r in g["SEARCH_POOL"] if r not in g["PTEST"]]
    assert len(vo) == 7 and g["S_MAX"] == 3 and g["N_SUB"] == 19 + 171 + 969, (len(vo), g["S_MAX"], g["N_SUB"])
    ct = g["CAND_TABLE"]; flagged = ct[ct["저장"] == "test 없음"]
    for _, rw in ct.iterrows():
        has_vo = any(r in vo for r in g["CANDS"][rw["후보"]]["weights"])
        assert (rw["저장"] == "test 없음") == has_vo and (rw["test≠기준"] == "-") == has_vo
    check_cands(g, runs_dir)
    print(g["_OUT"]["9"][-1500:])
    print(f"✅ 검증만 있는 실행 7개 포함 검색 · 조합 크기 상한 3 → {g['N_SUB']:,}개 · 그런 실행이 든 후보 {len(flagged)}개는 '저장: test 없음'")

def check_volist():
    print("\n════════ 시나리오 volist (검증만 있는 실행을 이름 목록으로 골라 검색에 넣기 · 못 넣은 이유 출력) ════════")
    assert 'SEARCH_INCLUDE_VALONLY = ["q38flash_r16_native_bf16"]' in CFG, "기본값: Flash-Next 이름 목록"
    g, runs_dir = start("volist", SEARCH_SETTINGS=[("prob", "none")], MAX_SIZE=3)
    os.makedirs(f"{runs_dir}/badname_run"); shutil.copy(f"{runs_dir}/q38_27b_r16_native/preds_final_ep1__up2.0__1000.csv", f"{runs_dir}/badname_run/preds_ep1_up2.0.csv")
    run_cells(g, [("4", POOLC), ("5", MEMF)])
    vo_all = [r for r in g["RUNS"] if r not in g["PTEST"]]; mem0 = g["MEMBERS"][0]
    assert len(vo_all) == 7, vo_all
    # ① 기본값(목록에 Flash-Next만, 이 가짜 Drive에는 없음) → 경고만, 검색 풀 = 멤버
    run_cells(g, [("6 기본", PREP)])
    o = g["_OUT"]["6 기본"]
    assert g["SEARCH_POOL"] == g["MEMBERS"] and "q38flash_r16_native_bf16: 검색에 못 넣음 → Drive에 폴더가 없습니다" in o, o[:600]
    # ② 목록: 검증만 있는 실행 1개 + 없는 이름 + 보통 멤버 + 파일 이름이 틀린 폴더
    g["SEARCH_INCLUDE_VALONLY"] = [vo_all[0], "없는_실행", mem0, "badname_run", vo_all[0]]
    run_cells(g, [("6", PREP), ("7", EXHC), ("8", GRDC), ("9", SUMC)])
    o = g["_OUT"]["6"]
    assert g["SEARCH_POOL"] == g["MEMBERS"] + [vo_all[0]], g["SEARCH_POOL"]
    assert f"검증만 있는 실행 1개를 검색에 넣음: ['{vo_all[0]}']" in o and "없는_실행: 검색에 못 넣음 → Drive에 폴더가 없습니다" in o
    assert f"({mem0}: test까지 있는 보통 멤버" in o and "badname_run: 검색에 못 넣음 → 검증 예측 파일 이름이 규칙과 다릅니다" in o, o[:1500]
    assert g["N_SUB"] == sum(__import__("math").comb(len(g["SEARCH_POOL"]), k) for k in (1, 2, 3))
    ct = g["CAND_TABLE"]
    for _, rw in ct.iterrows():
        has_vo = vo_all[0] in g["CANDS"][rw["후보"]]["weights"]
        assert (rw["저장"] == "test 없음") == has_vo, (rw["후보"], rw["저장"])
    check_cands(g, runs_dir)
    # ③ True = 전부 · False = 안 넣음
    g["SEARCH_INCLUDE_VALONLY"] = True; run_cells(g, [("6 전부", PREP)]); assert g["SEARCH_POOL"] == g["MEMBERS"] + vo_all
    g["SEARCH_INCLUDE_VALONLY"] = False; run_cells(g, [("6 없음", PREP)]); assert g["SEARCH_POOL"] == g["MEMBERS"] and "검색에 못 넣음" not in g["_OUT"]["6 없음"]
    print(o[:900])
    print(f"✅ 이름 목록: 검증만 있는 실행 1개만 추가({len(g['SEARCH_POOL'])}개) · 없는 이름/파일 이름 틀림은 이유 출력 · 보통 멤버는 그대로 · 중복 이름 1번 · 그 실행이 든 후보만 '저장: test 없음' · True/False 동작 그대로")

def check_noref():
    print("\n════════ 시나리오 noref (기준 조합 실행이 하나도 없음) ════════")
    g, runs_dir = start("noref", SEARCH_SETTINGS=[("logprob", "none")])
    g["REF_CFG"] = {"weights": {"없는_실행": 1}, "method": "prob", "temp": "none", "full": "none"}
    run_cells(g, [("4", POOLC), ("5", MEMF), ("6", PREP), ("7", EXHC), ("8", GRDC), ("9", SUMC), ("10", PANEL)])
    assert g["REF_EVAL"] is None and "모두 1" in g["_OUT"]["8"] and not any(k.startswith("기준") for k in g["CANDS"])
    assert (g["W_REF0"] == 1).all() and g["START_CFG"] == next(iter(g["CANDS"].values()))
    check_cands(g, runs_dir); assert all(c["full"] == "mix" for c in g["CANDS"].values())
    print("✅ 기준 없음: 좌표 상승은 '모두 1'에서 시작 · 후보·패널 정상")

def check_modes():
    for mode in ("swap", "none"):
        print(f"\n════════ 시나리오 modes: FULL_MODE='{mode}' (확률+온도, 크기 ≤ 3) ════════")
        g, runs_dir = start("mode_" + mode, FULL_MODE=mode, SEARCH_SETTINGS=[("prob", "val")], MAX_SIZE=3)
        run_cells(g, [("4", POOLC), ("5", MEMF), ("6", PREP), ("7", EXHC), ("8", GRDC), ("9", SUMC), ("10", PANEL)])
        nfr = check_cands(g, runs_dir)
        assert (nfr > 0) == (mode != "none") and g["DD_F"].value == mode
        rf = f"기준 {g['REF_NAME']} + _full {mode}"
        if mode == "none": assert rf not in g["CANDS"] and all(c["full"] == "none" for c in g["CANDS"].values())
        else:
            det = DETAIL.replace("PICK = None ", f'PICK = "{rf}" ', 1)
            run_cells(g, [("11", det), ("12", SAVEC), ("13", REPRO)])
            assert (pd.read_csv(g["SAVED_PATH"]).answer.map(LET.index).to_numpy() == comb_i(g, runs_dir, g["CANDS"][rf], "test").argmax(1)).all()
            assert "완전히 같음 ✅" in g["_OUT"]["13"] and "test = _full로 교체" in g["_OUT"]["13"]
        print(g["_OUT"]["9"][-900:])
        print(f"✅ FULL_MODE='{mode}': 후보 표 _full 칸·test 차이 = 독립 계산" + ("" if mode == "none" else " · 교체 저장·재현"))

def check_pairs():
    print("\n════════ 시나리오 pairs (FULL_PAIRS: 이름 규칙 밖 짝 · 해상도 다른 짝 · 거부되는 짝) ════════")
    g, runs_dir = start("pairs", FULL_PAIRS={"q3vl_8b_r16_native": "q3vl8_moved_full",          # 이름이 다른 짝 (up1.5만 있음, 99% 일치) → 통과
                                            "q38_27b_r16_native": "q3vl_32b_r16_native_full",  # 다른 모델 (94.7% 일치) → 거부
                                            "q35_27b_r16_native": "gemma4_31b_r16_up20_full",  # 기본 짝을 덮어씀 (96.0%) → 거부 = q35는 짝 없음
                                            "없는_실행": "q35_27b_r16_native_full"})           # 멤버가 아님 → 조용히 건너뜀
    d = f"{runs_dir}/q3vl8_moved_full"; os.makedirs(d)
    src = pd.read_csv(f"{runs_dir}/q3vl_8b_r16_native/preds_final_ep1_test__up2.0__6714.csv")
    rng = np.random.RandomState(3); flip = rng.choice(len(src), 60, replace=False)
    src.loc[flip, ["p_a", "p_b", "p_c", "p_d"]] = np.roll(src.loc[flip, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(), 1, axis=1)
    src["pred"] = [LET[i] for i in src[["p_a", "p_b", "p_c", "p_d"]].to_numpy().argmax(1)]
    src.to_csv(f"{d}/preds_final_ep1_test__up1.5__6714.csv", index=False)
    json.dump({"config": {"MODEL_ID": "Qwen/Qwen3-VL-8B-Instruct", "TRAIN_ON_FULL": True}}, open(f"{d}/train_log.json", "w"))
    run_cells(g, [("4", POOLC), ("5", MEMF)], quiet=False)
    o5 = g["_OUT"]["5"]
    assert set(g["FULL_OF"]) == {"q3vl_8b_r16_native", "gemma4_31b_r16_up20"}, sorted(g["FULL_OF"])
    assert "_full q3vl_8b_r16_native → q3vl8_moved_full@up1.5 (해상도 다름: 원래 up2.0)" in o5 and "✅ 섞을 수 있음" in o5
    assert "_full q38_27b_r16_native → q3vl_32b_r16_native_full@up2.0: test 답 일치 94.7% ⚠️ 안 씀" in o5
    assert "_full q35_27b_r16_native → gemma4_31b_r16_up20_full@up2.0: test 답 일치 96.0% ⚠️ 안 씀" in o5 and "없는_실행" not in o5
    Pm = src.set_index("id").loc[g["TEST_IDS"]][["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
    assert np.allclose(g["FULL_OF"]["q3vl_8b_r16_native"], Pm)
    # TRAIN_ON_FULL 표시가 없으면 일치율이 높아도 거부
    lg = json.load(open(f"{d}/train_log.json")); lg["config"].pop("TRAIN_ON_FULL"); json.dump(lg, open(f"{d}/train_log.json", "w"))
    run_cells(g, [("5 flag 없음", MEMF)])
    assert "q3vl_8b_r16_native" not in g["FULL_OF"] and "train_log에 TRAIN_ON_FULL 없음" in g["_OUT"]["5 flag 없음"]
    print("✅ FULL_PAIRS: 이름이 다른 짝(해상도 달라도 찾음) 통과 · 다른 모델 짝 거부 · 기본 짝 덮어쓰기 · 멤버 아닌 항목 무시 · TRAIN_ON_FULL 없으면 거부")

def check_timing(n_fake=20):
    print(f"\n════════ 시나리오 timing (가짜 실행 {n_fake}개 → 조합 약 100만 개) ════════")
    g, runs_dir = start("timing", SEARCH_SETTINGS=[("prob", "none"), ("logprob", "none")])
    run_cells(g, [("4", POOLC), ("5", MEMF)])
    rng = np.random.RandomState(1); base = g["MEMBERS"]; fake = []
    for i in range(n_fake):
        r0 = base[i % len(base)]; P = g["PVAL"][r0] * np.exp(rng.normal(0, 0.6, g["PVAL"][r0].shape)); P /= P.sum(1, keepdims=True)
        nm = f"fake{i:02d}"; g["PVAL"][nm] = P; g["TT"][nm] = 1.0; g["ACC"][nm] = float((P.argmax(1) == g["Y_VAL"]).mean() * 100); g["FAMR"][nm] = g["FAMR"][r0]; fake.append(nm)
    g["SEARCH_ONLY"] = fake
    run_cells(g, [("6", PREP), ("7 함수만", EXHC.split("EXH, _t_all")[0])])
    print(f"  조합 {g['N_SUB']:,}개 (크기 1~{g['S_MAX']})")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()): top, fb, _ = g["scan"](("prob", "none"))
    t1 = time.time() - t0; t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()): g["scan"](("logprob", "none"))
    t2 = time.time() - t0
    # 표본 몇 개를 독립 계산과 대조
    y = g["Y_VAL"]; A = np.stack([g["PVAL"][r] for r in fake])
    for sc, sub in top["acc"][:5]:
        ok, ll = okl(A[list(sub)].mean(0), "prob", y); assert abs(sc - (ok.sum() - 1e-5 * ll.sum())) < 1e-3, (sc, sub)
    print(f"✅ 전수 검색 {g['N_SUB']:,}개: 확률 {t1:.0f}초 · 로그확률 {t2:.0f}초 (이 기계 CPU) · 상위 5개 점수 = 독립 계산")

if __name__ == "__main__":
    scns = sys.argv[1:] or ["main", "valonly", "volist", "noref", "modes", "pairs", "timing"]
    for s in scns: {"main": check_main, "valonly": check_valonly, "volist": check_volist, "noref": check_noref, "modes": check_modes, "pairs": check_pairs, "timing": check_timing}[s]()
    print("\n✅✅ ens_search 시험 통과:", scns)
