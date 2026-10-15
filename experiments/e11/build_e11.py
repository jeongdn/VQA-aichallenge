# -*- coding: utf-8 -*-
"""E11 노트북 빌더 (CPU): E10에서 고른 '멤버 전체 평균'을 기준으로, E9(TTA)·E5(고해상도)·HCX 노트북이 멤버별로 다시 채점한
애매한 문항 결과를 넣었을 때 좋아지는지 판정한다. 파일만 읽음 (GPU 불필요).
- 기준: e10_search/e10_info.json의 멤버 목록 → 동일 가중 확률 평균 (submission_e10_avg_all.csv 재현 확인)
- T: TTA 완료 멤버의 애매한 문항 확률을 보기 순서 4가지 평균으로 / R: 고해상도 완료 멤버를 {기존 up2.0, up3.0, up4.0} 평균으로 / TR: 둘 다
데이터·예측 파일 읽기 함수는 E8 노트북 코드를 그대로 씀.
"""
import sys, uuid
import nbformat

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e11/out/e11_combine_on_avg.ipynb"
SP = "/tmp/vqa_scratch"
_n8 = {}
_argv = sys.argv; sys.argv = ["build_e8_ens.py", f"{SP}/_e8_tmp.ipynb"]
exec(open("/home/user/vqa/e8/build_e8_ens.py", encoding="utf-8").read(), _n8)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

CFG = r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"
E10_SUBDIR, E9_NAME, E5_NAME, E11_SUBDIR = "e10_search", "e9_tta", "e5_hires", "e11_combine"
SEED, VAL_SIZE = 42, 1000

# ── 기준: E10의 멤버 전체 평균 ──
BASE_FILE   = "submission_e10_avg_all.csv"   # e10_search 폴더의 기준 제출 파일
PUBLIC_BASE = None     # 기준 파일의 공개 점수 (예: 0.97706). 넣으면 판정선을 출력. 모르면 None
N_PUBLIC    = 3357

# ── 기다릴 멤버: 이 멤버들의 채점이 모두 끝나야 판정 (진행 중이면 '대기') ──
EXPECT_TTA = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "hcx32b_r16_native"]   # E9 4개 + HCX 노트북
EXPECT_HR  = ["q35_27b_r16_native", "q38_27b_r16_native", "hcx32b_r16_native"]                                                  # E5 2개 + HCX 노트북
#   HCX 노트북을 안 돌리기로 하면 두 목록에서 "hcx32b_r16_native"를 빼세요. 목록 밖 멤버도 채점 파일이 모두 있으면 함께 반영합니다

# ── 사전 기준 (결과를 보기 전에 고정. E9·E5와 같은 값) ──
PRESET, HR_PRESETS = "up2.0", ["up3.0", "up4.0"]
NET_MIN = 3          # 기준 대비 검증 순이득 + 공개 순이득 ≥ +3 이고 공개 ≥ 0 → 채택. 검증 순이득 ≤ −3이면 불채택
REPRO_MIN, MASS_MIN = 85.0, 0.9   # 멤버 건강 점검: 다시 채점한 up2.0(순서 그대로) 답이 기존 예측과 85% 이상 같고, letter_mass ≥ 0.9
'''

DATA = rep(_n8["DATA"], 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, E11_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E11_SUBDIR); os.makedirs(E11_DIR, exist_ok=True)')
HELPERS = _n8["HELPERS"]

BASECELL = r'''
# ── 기준 = E10 멤버 전체 평균 (e10_info.json의 멤버 목록 그대로) ──
def _read_json(p):
    try: return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    except Exception: return {}
def _answers_at(p):
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"].astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
_info = _read_json(os.path.join(OUT_DIR, E10_SUBDIR, "e10_info.json"))
assert _info.get("members"), f"{E10_SUBDIR}/e10_info.json이 없습니다 → E10 노트북을 먼저 실행하세요"
MEMBERS = [tuple(m) for m in _info["members"]]; M = len(MEMBERS); NAMES = [r for r, _ in MEMBERS]
PV0 = np.stack([load(r, p, "val") for r, p in MEMBERS]); PT0 = np.stack([load(r, p, "test") for r, p in MEMBERS])
A_BASE = {"val": PV0.mean(0).argmax(1), "test": PT0.mean(0).argmax(1)}
FA_BASE = _answers_at(os.path.join(OUT_DIR, E10_SUBDIR, BASE_FILE))
assert FA_BASE is not None, f"{E10_SUBDIR}/{BASE_FILE}이 없거나 형식이 다릅니다"
_same = int((FA_BASE == A_BASE["test"]).sum())
print(f"기준: 멤버 {M}개 전체 평균 | 검증 {(A_BASE['val'] == Y_VAL).mean() * 100:.2f}% | {BASE_FILE} 재현 {_same:,}/{len(TEST_IDS):,}")
assert _same == len(TEST_IDS), "기준 파일이 e10_info.json 멤버의 평균과 다릅니다 → E10을 다시 실행해 두 파일을 맞추세요"
IDX = {"val": {i: n for n, i in enumerate(VAL_IDS)}, "test": {i: n for n, i in enumerate(TEST_IDS)}}
'''

VIEWS = r'''
# ── 멤버별로 다시 채점한 결과 읽기: TTA(e9_tta/<멤버>/perm_*.csv) · 고해상도(e5_hires/<멤버>/res_*.csv). 파일 형식은 E9·E5·HCX 노트북과 같음 ──
COLS = ["p_a", "p_b", "p_c", "p_d"]
def _read(p):
    try: return pd.read_csv(p) if os.path.exists(p) else None
    except Exception: return None
def _order(ids, s): return sorted(ids, key=lambda i: IDX[s][i])
TTA, HRV = {}, {}
for run, pre in MEMBERS:
    for s in ("val", "test"):
        d = _read(os.path.join(OUT_DIR, E9_NAME, run, f"perm_{s}__{PRESET}.csv"))
        if d is not None and pre == PRESET and len(d):
            d = d[d.lora.astype(int) == 1].drop_duplicates(["id", "perm"], keep="last")
            cnt = d.groupby("id").perm.nunique(); ids = _order([i for i in cnt[cnt == 4].index if i in IDX[s]], s)
            X = np.stack([d[d.perm == k].set_index("id").loc[ids, COLS].to_numpy(float) for k in range(4)], 1) if ids else np.zeros((0, 4, 4))
            TTA.setdefault(run, {})[s] = dict(ids=ids, X=X, mass=float(d.letter_mass.mean()), partial=len(set(d.id) - set(ids)))
        r = _read(os.path.join(OUT_DIR, E5_NAME, run, f"res_{s}.csv"))
        if r is not None and pre == PRESET and len(r):
            r = r.drop_duplicates(["id", "preset"], keep="last")
            for pr in ["up2.0"] + HR_PRESETS:
                q = r[r.preset == pr]
                if len(q):
                    ids = _order([i for i in q.id if i in IDX[s]], s)
                    HRV.setdefault(run, {}).setdefault(s, {})[pr] = dict(ids=ids, P=q.set_index("id").loc[ids, COLS].to_numpy(float), mass=float(q.letter_mass.mean()), tok=float(q.img_tokens.mean()))
_e5rec = _read_json(os.path.join(OUT_DIR, E5_NAME, "e5_info.json"))
def infeasible(run):
    a = set((_e5rec.get("infeasible") or {}).get(run, []))
    b = set((_read_json(os.path.join(OUT_DIR, E5_NAME, run, "hcx_info.json")).get("infeasible") or {}).get(run, []))
    return a | b
# 애매한 문항 목록 = 채점 파일들에 있는 문항의 합집합 (E9·E5·HCX는 같은 문턱·같은 옛 레시피로 골랐으므로 모두 같아야 함)
REF_T = {s: _order(set().union(*[set(v[s]["ids"]) for v in TTA.values() if s in v]), s) if TTA else [] for s in ("val", "test")}
REF_R = {s: _order(set().union(*[set(v[s][pr]["ids"]) for v in HRV.values() if s in v for pr in HR_PRESETS if pr in v[s]]), s) if HRV else [] for s in ("val", "test")}
gT = {s: np.array([IDX[s][i] for i in REF_T[s]], int) for s in ("val", "test")}; gR = {s: np.array([IDX[s][i] for i in REF_R[s]], int) for s in ("val", "test")}
print(f"애매한 문항 (TTA 파일): 검증 {len(REF_T['val'])} / test {len(REF_T['test'])} | (고해상도 파일): 검증 {len(REF_R['val'])} / test {len(REF_R['test'])}")

ROWS, OK_T, OK_R, WAIT_T, WAIT_R, HEALTH = [], {}, {}, [], [], []
yT, yR = Y_VAL[gT["val"]], Y_VAL[gR["val"]]
for i, (run, pre) in enumerate(MEMBERS):
    row = {"멤버": run}
    t = TTA.get(run, {})
    if t:
        done = all(s in t and t[s]["ids"] == REF_T[s] and t[s]["partial"] == 0 for s in ("val", "test"))
        if done:
            X = t["val"]["X"]; rp = float((X[:, 0].argmax(1) == PV0[i][gT["val"]].argmax(1)).mean() * 100); f, b, n, D = paired(PV0[i][gT["val"]].argmax(1), X.mean(1).argmax(1), yT)
            row.update({"TTA": "✅", "TTA 재현(%)": round(rp, 1), "TTA 단독(애매한 검증)": f"고침 {f}/새로 틀림 {b}/순이득 {n:+d}"})
            if rp < REPRO_MIN or t["val"]["mass"] < MASS_MIN: HEALTH.append((run, "T", f"{run} TTA: 재현 {rp:.1f}% / letter_mass {t['val']['mass']:.3f}")); row["TTA"] = "⚠️"
            Pv, Pt = PV0[i].copy(), PT0[i].copy(); Pv[gT["val"]] = X.mean(1); Pt[gT["test"]] = t["test"]["X"].mean(1); OK_T[i] = (Pv, Pt)
        else: row["TTA"] = "채점 중(미완료)"; WAIT_T.append(run)
    elif run in EXPECT_TTA: row["TTA"] = "파일 없음"; WAIT_T.append(run)
    h = HRV.get(run, {})
    F = [pr for pr in HR_PRESETS if pr not in infeasible(run)]
    if h:
        done = bool(F) and all(s in h and pr in h[s] and h[s][pr]["ids"] == REF_R[s] for s in ("val", "test") for pr in F) and "up2.0" in h.get("val", {}) and h["val"]["up2.0"]["ids"] == REF_R["val"]
        if done:
            rp = float((h["val"]["up2.0"]["P"].argmax(1) == PV0[i][gR["val"]].argmax(1)).mean() * 100)
            Rv = np.mean([PV0[i][gR["val"]]] + [h["val"][pr]["P"] for pr in F], 0); Rt = np.mean([PT0[i][gR["test"]]] + [h["test"][pr]["P"] for pr in F], 0)
            f, b, n, D = paired(PV0[i][gR["val"]].argmax(1), Rv.argmax(1), yR)
            row.update({"고해상도": "✅ " + "+".join(F), "up2.0 재현(%)": round(rp, 1), "고해상도 단독(애매한 검증)": f"고침 {f}/새로 틀림 {b}/순이득 {n:+d}"})
            bad = [f"{pr} letter_mass {h['val'][pr]['mass']:.3f}" for pr in ["up2.0"] + F if h["val"][pr]["mass"] < MASS_MIN]
            if rp < REPRO_MIN or bad: HEALTH.append((run, "R", f"{run} 고해상도: up2.0 재현 {rp:.1f}% " + " ".join(bad))); row["고해상도"] = "⚠️"
            Pv, Pt = PV0[i].copy(), PT0[i].copy(); Pv[gR["val"]] = Rv; Pt[gR["test"]] = Rt; OK_R[i] = (Pv, Pt)
        elif not F: row["고해상도"] = "모든 고해상도 건너뜀(기록)"
        else: row["고해상도"] = "채점 중(미완료)"; WAIT_R.append(run)
    elif run in EXPECT_HR: row["고해상도"] = "파일 없음"; WAIT_R.append(run)
    ROWS.append(row)
MEM_TABLE = pd.DataFrame(ROWS).fillna("")
print(MEM_TABLE.to_string(index=False))
'''

JUDGE = r'''
# ── 판정: 멤버 확률을 바꾼 뒤 같은 '전체 평균'으로 → 기준과 같은 문항끼리 비교 → 제출 후보 ──
V = {"lines": [], "cands": []}
def say(s): print(s); V["lines"].append(s)
def ens(over):
    Pv, Pt = PV0.copy(), PT0.copy()
    for i, (a, b) in over.items(): Pv[i], Pt[i] = a, b
    return Pv.mean(0).argmax(1), Pt.mean(0).argmax(1)
def _write(tag, At):
    cand = np.where(At != A_BASE["test"], At, FA_BASE)                  # 이 실험 때문에 바뀐 답만 기준 파일에서 고침
    p = os.path.join(E11_DIR, f"submission_e11_{tag}.csv"); sub_df[["id"]].assign(answer=[LETTERS[k] for k in cand]).to_csv(p, index=False)
    return os.path.basename(p), int((cand != FA_BASE).sum())
# TR: TTA와 고해상도가 둘 다 끝난 멤버는 모든 관점(보기 순서 4가지 + 고해상도)의 평균, 한쪽만 끝난 멤버는 그쪽 결과
OK_TR = {}
for i in set(OK_T) | set(OK_R):
    run = NAMES[i]
    if i in OK_T and i in OK_R and REF_T["val"] == REF_R["val"] and REF_T["test"] == REF_R["test"]:
        F = [pr for pr in HR_PRESETS if pr not in infeasible(run)]
        Pv, Pt = PV0[i].copy(), PT0[i].copy()
        Pv[gT["val"]] = np.mean([TTA[run]["val"]["X"][:, k] for k in range(4)] + [HRV[run]["val"][pr]["P"] for pr in F], 0)
        Pt[gT["test"]] = np.mean([TTA[run]["test"]["X"][:, k] for k in range(4)] + [HRV[run]["test"][pr]["P"] for pr in F], 0)
        OK_TR[i] = (Pv, Pt)
    else: OK_TR[i] = OK_T.get(i) or OK_R.get(i)
b0 = count_of(PUBLIC_BASE) if PUBLIC_BASE else None
RES = {}
for tag, lab, over, wait, kinds in [("tta", "T: TTA(보기 순서 4가지 평균)", OK_T, WAIT_T, "T"), ("hires", "R: 고해상도({기존 up2.0, up3.0, up4.0} 평균)", OK_R, WAIT_R, "R"),
                                    ("tta_hires", "TR: TTA + 고해상도", OK_TR, WAIT_T + WAIT_R, "TR")]:
    if not over:
        say(f"[{lab}] 판정: 대기 — 채점이 끝난 멤버가 없습니다" + (f" (기다리는 멤버: {', '.join(sorted(set(wait)))})" if wait else "")); continue
    Av, At = ens(over); f, b, n, D = paired(A_BASE["val"], Av, Y_VAL); RES[tag] = n
    say(f"[{lab}] 반영 멤버 {len(over)}개({', '.join(NAMES[i] for i in sorted(over))}) | 검증 {(A_BASE['val'] == Y_VAL).mean() * 100:.2f}% → {(Av == Y_VAL).mean() * 100:.2f}% "
        f"| 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}" + ("  (바뀐 답 10개 미만 → 검증만으로는 판단 안 함)" if D < 10 else ""))
    _used = {NAMES[i] for i in over}; bad = [m for r, k, m in HEALTH if r in _used and k in kinds]
    if bad: say(f"  판정: 확인 필요 — {'; '.join(bad)} → 위 출력을 확인하세요"); continue
    if wait: say(f"  판정: 대기 — 채점이 안 끝난 멤버 {', '.join(sorted(set(wait)))} (1번 셀 EXPECT 목록). 끝나면 다시 실행"); continue
    if n <= -3: say(f"  판정: 불채택 — 검증 순이득 {n:+d} (≤ −3). 공개로 확인할 필요 없음"); continue
    fn, Dn = _write(tag, At); V["cands"].append((fn, Dn))
    line = f"  판정: 공개 확인 — 제출 후보 {fn} | 기준({BASE_FILE})과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개)"
    if b0 is not None: line += f" | ◀ 공개 {score_of(b0 + max(0, NET_MIN - n)):.5f}({b0 + max(0, NET_MIN - n):,}문항) 이상이면 채택 (검증 {n:+d} + 공개 ≥ +{NET_MIN}, 공개 ≥ 0)"
    else: line += " | 1번 셀 PUBLIC_BASE에 기준 파일의 공개 점수를 넣고 다시 실행하면 판정선이 나옵니다"
    if tag == "tta_hires": line += " | T와 R이 둘 다 채택된 뒤에만 제출"
    say(line)
json.dump({"base": BASE_FILE, "public_base": PUBLIC_BASE, "members": MEMBERS, "table": ROWS, "verdict": V["lines"], "cands": V["cands"]},
          open(os.path.join(E11_DIR, "e11_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

SUMMARY = r'''
print(f"E11 | 기준 {BASE_FILE} (멤버 {M}개 평균, 공개 {PUBLIC_BASE}) | 애매한 문항 검증 {len(REF_T['val']) or len(REF_R['val'])}")
print(MEM_TABLE.to_string(index=False))
for l in V["lines"]: print(l)
'''

INTRO = r'''# E11 · 새 기준(멤버 전체 평균) 위에서 TTA·고해상도 판정 (CPU, 파일만 읽음)

**왜**: E10에서 "고정 분할 멤버 전체의 동일 가중 평균"이 가장 좋은 조합 방법으로 나왔습니다. E9(TTA)·E5(고해상도)·HCX 노트북은 옛 레시피(clean4·다수결) 기준으로 판정하므로, 기준이 바뀌면 같은 채점 결과를 새 기준 위에서 다시 판정해야 합니다. 새로 채점하지 않고 그 노트북들이 Drive에 저장한 파일만 읽습니다.

**하는 일**
- 기준: `e10_search/e10_info.json`의 멤버 목록 → 동일 가중 확률 평균. `submission_e10_avg_all.csv`와 같은지 먼저 확인합니다.
- 멤버마다 채점 파일 상태(완료·진행 중·없음)와 건강 점검(다시 채점한 up2.0 답이 기존과 85% 이상 같은지, letter_mass ≥ 0.9)을 보여 줍니다.
- **T**: TTA가 끝난 멤버의 애매한 문항 확률 → 보기 순서 4가지 평균 (E9 H1과 같은 방식)
- **R**: 고해상도가 끝난 멤버 → {기존 up2.0, up3.0, up4.0} 평균 (E5 H1과 같은 방식)
- **TR**: 둘 다 끝난 멤버는 모든 관점(보기 순서 4가지 + 고해상도)의 평균
- 멤버 확률만 바꾸고 조합은 그대로(전체 평균) → 기준과 같은 문항끼리 고침·새로 틀림을 셉니다.

**사전 기준 (E9·E5와 같음)**: 검증 순이득 + 공개 순이득 ≥ +3이고 공개 순이득 ≥ 0 → 채택. 검증 순이득 ≤ −3이면 불채택. 바뀐 답 10개 미만이면 검증만으로 판단하지 않음. **TR은 T와 R이 둘 다 채택된 뒤에만** 제출합니다. 1번 셀 `EXPECT_*`의 멤버가 채점 중이면 '대기'로 나옵니다.

**실행**: GPU 없는 새 세션에서 위에서부터 끝까지. `submission_e10_avg_all.csv`의 공개 점수가 나오면 1번 셀 `PUBLIC_BASE`에 넣고 다시 실행하세요. 결과는 `vqa_runs/e11_combine/`에만 씁니다.'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 데이터 (CSV만) · 고정 검증 분할"), ("code", DATA),
    ("markdown", "## 3. 예측 파일 읽기 함수 (E8과 같음)"), ("code", HELPERS),
    ("markdown", "## 4. 기준: E10 멤버 전체 평균"), ("code", BASECELL),
    ("markdown", "## 5. 멤버별 다시 채점한 결과 (TTA · 고해상도) — 상태와 건강 점검"), ("code", VIEWS),
    ("markdown", "## 6. 판정 → 제출 후보"), ("code", JUDGE),
    ("markdown", "## 7. 결과 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    c = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]; nb.cells.append(c)
for c in nb.cells:
    if c.cell_type == "code": compile(c.source, "<cell>", "exec")
nbformat.validate(nb); nbformat.write(nb, OUT)
print("저장:", OUT, "| 셀", len(nb.cells))
