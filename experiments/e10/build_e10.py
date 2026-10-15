# -*- coding: utf-8 -*-
"""E10 노트북 빌더 (CPU): 지금 레시피에 얽매이지 않고, 고정 분할로 학습한 모든 멤버에서 조합을 새로 찾는다.
- 멤버 자동 탐색 (검증·test 예측이 모두 있는 실행, 전체 데이터 학습 제외)
- 미리 정한 조합 방법 5개 (전체 평균 / 전체 다수결 / NLL 가중 로그 결합 / Caruana 탐욕 선택 / 최고 부분집합)
- 중첩 교차검증(10겹 × 5회): '검증으로 조합을 고르는 절차' 자체를 고르는 데 쓰지 않은 문항으로 평가 → 1,000문항 과적합 방지
- 전체 검증으로 다시 맞춘 조합 → test 제출 후보 + 공개 점수 판정선
데이터·예측 파일 읽기 함수는 E8 노트북 코드를 그대로 씀.
"""
import sys, uuid
import nbformat

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e10/out/e10_ensemble_search.ipynb"
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
ENSEMBLE_SUBDIR = "ensembles"      # 기존 제출 파일 폴더 (읽기만 함)
E10_SUBDIR   = "e10_search"        # 새 제출 후보를 쓰는 폴더
SEED, VAL_SIZE = 42, 1000          # 검증 분할 — 팀 공통값

# ── 반드시 넣을 멤버: 파일이 없거나 이름이 다르면 멈추고 그 폴더의 예측 파일 목록을 보여 줌 (검증 정확도 하한도 적용 안 함) ──
MUST_INCLUDE = [("ivl35_38b_r16_native", "up2.0")]   # InternVL3.5-38B — 5714/1000 고정 분할로 학습, up2.0 예측

# ── 나머지 멤버: vqa_runs 전체에서 자동 탐색 (검증 1,000·test 6,714 예측이 모두 있고, 전체 데이터 학습이 아닌 실행) ──
PRESET_PREF = ["up2.0"]    # 실행마다 이 해상도 예측을 우선 사용 (없으면 검증 정확도가 가장 높은 해상도)
INCLUDE = []               # 비우면 자동 탐색. 넣으면 이 목록만: [("ivl35_38b_r16_native", "up2.0"), ...]
EXCLUDE = []               # 자동 탐색에서 뺄 실행 이름
MIN_VAL_ACC = 90.0         # 검증 정확도가 이보다 낮은 실행은 제외 (고장·점검용 실행 방지)

# ── 조합 방법 설정 (결과를 보기 전에 고정) ──
CV_FOLDS, CV_REPEATS, CV_SEED = 10, 5, 2026   # 중첩 교차검증: 10겹 × 5회 반복
TOP_N, MAX_SUBSET = 10, 7   # 최고 부분집합: 학습 폴드 NLL 상위 10개 멤버에서 2~7개 조합을 모두 시험
GREEDY_STEPS = 20           # Caruana 탐욕 선택 횟수 (같은 멤버를 여러 번 고를 수 있음 = 가중치)
STACK_L2, STACK_WMAX = 1e-3, 3.0   # 가중 로그 결합: 가중치 L2 규제, 가중치 상한

# ── 넘어야 할 기준 = 지금 제출 레시피 (조합 후보 목록이 아님. 새 조합은 위의 모든 멤버에서 찾음) ──
#    이 레시피도 검증을 보고 고른 것이라 검증 점수는 실제보다 높게 나올 수 있음 → 공개 점수가 공정한 비교
PRESET = "up2.0"
Q35, Q38 = ("q35_27b_r16_native", PRESET), ("q38_27b_r16_native", PRESET)
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", PRESET), ("gemma4_31b_r16_up20", PRESET), ("hcx32b_r16_native", PRESET)
RECIPES = {
    "clean4_qg_hcx":    dict(file="submission_clean4_qg_hcx.csv",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], how="prob", cal=None),
    "ens_q38_gemma4":   dict(file="submission_ens_q38_gemma4.csv",   members=[Q38, G],                 weights=[1, 1],          how="prob", cal=None),
    "ens_q38_gdev_cal": dict(file="submission_ens_q38_gdev_cal.csv", members=[Q38, GDEV],              weights=[1, 1],          how="prob", cal="T70"),
}
VOTE = dict(name="majority_vote", file="submission_majority_vote.csv", voters=["ens_q38_gemma4", "clean4_qg_hcx", "ens_q38_gdev_cal"])
PUBLIC = {"clean4_qg_hcx": 0.97646, "majority_vote": 0.97676}; N_PUBLIC = 3357
BEST_NOW = "majority_vote"  # 지금 공개 최고 — 새 후보는 이것을 넘어야 함
NET_MIN = 3                 # 채택: 중첩 CV 순이득(지금 최고 대비) + 공개 순이득 ≥ +3 이고 공개 순이득 ≥ 0. 중첩 CV 순이득 ≤ −3이면 불채택
'''

DATA = _n8["DATA"]
DATA = rep(DATA, 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, E10_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E10_SUBDIR); os.makedirs(E10_DIR, exist_ok=True)')
HELPERS = _n8["HELPERS"]

POOL = r'''
# ── 멤버 후보 자동 탐색 ──
import itertools
def _presets(run):
    d = os.path.join(OUT_DIR, run)
    return sorted({m.group(2) for f in os.listdir(d) if (m := re.match(r"^preds_final_ep(\d+)__(.+)__1000\.csv$", f))})
MEMBERS_ALL, _skip = [], []
for run, pre in MUST_INCLUDE:
    d = os.path.join(OUT_DIR, run); assert os.path.isdir(d), f"반드시 넣을 멤버의 폴더가 없습니다: {d} → 1번 셀 MUST_INCLUDE의 이름을 확인하세요"
    assert not (read_log(run).get("config") or {}).get("TRAIN_ON_FULL"), f"{run}의 train_log에 TRAIN_ON_FULL=True로 기록돼 있습니다 → 검증을 학습에 쓴 실행은 넣을 수 없습니다"
    fv, wv = member_file(run, pre, "val"); ft, wt = member_file(run, pre, "test")
    if not (fv and ft):
        raise FileNotFoundError(f"{run}@{pre}: 검증 {'OK' if fv else wv} / test {'OK' if ft else wt}\n  폴더의 예측 파일: {sorted(f for f in os.listdir(d) if f.startswith('preds'))[:30]}\n"
                                f"  필요한 이름: preds_final_ep<N>__{pre}__1000.csv 와 preds_final_ep<N>_test__{pre}__{len(TEST_IDS)}.csv → 다르면 위 출력을 확인하세요")
    load(run, pre, "val"); load(run, pre, "test")          # 문항 id가 검증 1,000·test 6,714와 같은지까지 확인 (다르면 여기서 멈춤)
    MEMBERS_ALL.append((run, pre))
_must = {r for r, _ in MUST_INCLUDE}
_runs = sorted({r for r, _ in INCLUDE}) if INCLUDE else sorted(x for x in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, x)))
for run in _runs:
    if run in _must or run in EXCLUDE or run.endswith("_dry"): continue
    cfg = read_log(run).get("config") or {}
    pres = [p for r, p in INCLUDE if r == run] if INCLUDE else _presets(run)
    if not pres: continue                                             # 예측 파일이 없는 폴더 (실험 결과 폴더 등)
    if cfg.get("TRAIN_ON_FULL"): _skip.append(f"{run} (전체 데이터 학습 — 검증을 학습에 씀)"); continue
    ok, failed = [], False
    for pre in pres:
        fv, _ = member_file(run, pre, "val"); ft, _ = member_file(run, pre, "test")
        if not (fv and ft): continue
        try: P = load(run, pre, "val"); load(run, pre, "test")
        except Exception as e: _skip.append(f"{run}@{pre} (읽기 실패: {str(e)[:70]})"); failed = True; continue
        ok.append((pre, float((P.argmax(1) == Y_VAL).mean() * 100)))
    if not ok:
        if not failed: _skip.append(f"{run} (검증·test 예측이 둘 다 있는 해상도 없음)")
        continue
    pre = next((p for p in PRESET_PREF if p in dict(ok)), max(ok, key=lambda t: t[1])[0])
    if dict(ok)[pre] < MIN_VAL_ACC: _skip.append(f"{run}@{pre} (검증 {dict(ok)[pre]:.1f}% < {MIN_VAL_ACC})"); continue
    MEMBERS_ALL.append((run, pre))
_need = sorted({m for r in RECIPES.values() for m in r["members"]})
_miss = [m for m in _need if m not in MEMBERS_ALL]
assert not _miss, f"지금 레시피 멤버가 후보에 없습니다: {_miss} → EXCLUDE·INCLUDE·PRESET_PREF를 확인하세요"
M = len(MEMBERS_ALL); NAMES = [r for r, _ in MEMBERS_ALL]
PV = np.stack([load(r, p, "val") for r, p in MEMBERS_ALL]); PT = np.stack([load(r, p, "test") for r, p in MEMBERS_ALL])
LV, LT = np.log(np.clip(PV, 1e-6, 1.0)), np.log(np.clip(PT, 1e-6, 1.0))
print(f"멤버 후보 {M}개:", ", ".join(f"{r}@{p}" for r, p in MEMBERS_ALL))
for r, p in MUST_INCLUDE:
    print(f"✅ 반드시 포함: {r}@{p} | 검증 {(load(r, p, 'val').argmax(1) == Y_VAL).mean() * 100:.2f}% | test 예측 {len(load(r, p, 'test')):,}문항")
if _skip: print("제외:", *_skip, sep="\n  · ")

# ── 지금 제출 레시피 (검증·test 답) ──
A0 = {n: {s: ens_scores(r["members"], r["weights"], r["how"], r["cal"], split=s).argmax(1) for s in ("val", "test")} for n, r in RECIPES.items()}
A0[VOTE["name"]] = {s: vote_of([A0[v][s] for v in VOTE["voters"]]) for s in ("val", "test")}
for n in ("clean4_qg_hcx", VOTE["name"]):
    fa = file_answers(RECIPES[n]["file"] if n in RECIPES else VOTE["file"])
    print(f"  {n}: 검증 {(A0[n]['val'] == Y_VAL).mean() * 100:.2f}%" + ("" if fa is None else f" | 제출 파일 재현 {int((fa == A0[n]['test']).sum()):,}/{len(TEST_IDS):,}"))
REF = A0[BEST_NOW]["val"]
'''

MEMTABLE = r'''
# ── 멤버 표: 멤버마다 지금 최고(다수결)가 틀린 문항을 몇 개 맞히는지 = 새 조합에 보탤 수 있는 정보 ──
_c4 = [m for m in RECIPES["clean4_qg_hcx"]["members"]]
rows = []
for i, (run, pre) in enumerate(MEMBERS_ALL):
    a = PV[i].argmax(1); cfg = read_log(run).get("config") or {}
    r = {"멤버": run, "해상도": pre, "모델": str(cfg.get("MODEL_ID", "?")).split("/")[-1], "검증(%)": round(float((a == Y_VAL).mean() * 100), 2),
         "NLL": round(nll(PV[i], Y_VAL), 4), f"{BEST_NOW} 오답 중 맞힘": int(((REF != Y_VAL) & (a == Y_VAL)).sum()),
         f"{BEST_NOW} 정답 중 틀림": int(((REF == Y_VAL) & (a != Y_VAL)).sum()), "clean4에 있음": "O" if (run, pre) in _c4 else ""}
    if (run, pre) not in _c4:                                   # 참고: clean4에 가중치 1로 넣으면 (전체 검증, 같은 문항끼리)
        _c = RECIPES["clean4_qg_hcx"]; A5 = ens_scores(_c["members"] + [(run, pre)], _c["weights"] + [1], "prob", None, split="val").argmax(1)
        f, b, n, D = paired(A0["clean4_qg_hcx"]["val"], A5, Y_VAL); r["clean4+이 멤버(가중치 1)"] = f"고침 {f}/새로 틀림 {b}/순이득 {n:+d}"
    rows.append(r)
MEM_TABLE = pd.DataFrame(rows).sort_values("NLL").reset_index(drop=True).fillna("")
print(MEM_TABLE.to_string(index=False))
C = PV.argmax(2) == Y_VAL[None, :]
print(f"\n멤버 중 하나라도 맞힌 검증 문항 {int(C.any(0).sum())} (상한) | 전원 오답 {int((~C.any(0)).sum())} | 지금 최고({BEST_NOW}) 정답 {int((REF == Y_VAL).sum())}")
if M <= 14:
    D_ = np.array([[int((PV[i].argmax(1) != PV[j].argmax(1)).sum()) for j in range(M)] for i in range(M)])
    print("\n검증에서 두 멤버의 답이 다른 문항 수:"); print(pd.DataFrame(D_, index=NAMES, columns=[n[:12] for n in NAMES]).to_string())
'''

METHODS = r'''
# ── 조합 방법 (결과를 보기 전에 고정). fit(학습 문항 번호) → 모델, apply(모델, 확률, 로그확률) → 답 ──
from scipy.optimize import minimize
def _nll(P, y): return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))
def fit_avg(tr): return {"w": np.ones(M)}                                   # 전체 평균 (학습 없음)
def fit_vote(tr): return {"vote": True}                                     # 전체 다수결 (동률이면 평균 확률)
def fit_stack(tr):                                                          # 가중 로그 결합: log p ∝ Σ w_m log p_m, 학습 폴드 NLL 최소 (가중치 = 멤버별 온도 역할도 함)
    L, y = LV[:, tr], Y_VAL[tr]; ii = np.arange(len(tr)); Ly = L[:, ii, y]
    def f(w):
        s = np.tensordot(w, L, 1); s = s - s.max(1, keepdims=True); lse = np.log(np.exp(s).sum(1)); p = np.exp(s - lse[:, None])
        return (-np.mean(s[ii, y] - lse) + STACK_L2 * float(w @ w), -np.mean(Ly - np.einsum("mnk,nk->mn", L, p), axis=1) + 2 * STACK_L2 * w)
    r = minimize(f, np.full(M, 1.0 / M), jac=True, method="L-BFGS-B", bounds=[(0.0, STACK_WMAX)] * M)
    return {"w": r.x, "log": True}
def fit_greedy(tr):                                                         # Caruana: 학습 폴드 NLL이 가장 낮아지는 멤버를 하나씩 추가 (중복 허용)
    P, y = PV[:, tr], Y_VAL[tr]; cnt = np.zeros(M); cur = None
    for t in range(GREEDY_STEPS):
        cands = [P[m] if cur is None else (cur * t + P[m]) / (t + 1) for m in range(M)]
        j = int(np.argmin([_nll(c, y) for c in cands])); cnt[j] += 1; cur = cands[j]
    return {"w": cnt}
def fit_subset(tr):                                                         # 최고 부분집합: 학습 폴드 정확도 최대 (동률이면 NLL), 동일 가중 확률 평균
    P, y = PV[:, tr], Y_VAL[tr]; order = [int(i) for i in np.argsort([_nll(P[m], y) for m in range(M)])[:TOP_N]]; best = None
    for k in range(2, min(MAX_SUBSET, len(order)) + 1):
        for S in itertools.combinations(order, k):
            Q = P[list(S)].mean(0); key = (int((Q.argmax(1) == y).sum()), -_nll(Q, y))
            if best is None or key > best[0]: best = (key, S)
    w = np.zeros(M); w[list(best[1])] = 1.0; return {"w": w}
def apply(model, P, L):
    if model.get("vote"):
        A = P.argmax(2); cnt = np.stack([(A == k).sum(0) for k in range(4)], 1); return (cnt + 1e-3 * P.mean(0)).argmax(1)
    w = np.asarray(model["w"], float)
    if model.get("log"): return np.tensordot(w, L, 1).argmax(1)
    return np.tensordot(w / w.sum(), P, 1).argmax(1)
def describe(model):
    if model.get("vote"): return "전체 멤버 다수결"
    w = np.asarray(model["w"], float); w = w / w.sum()
    return " + ".join(f"{NAMES[i]}×{w[i]:.2f}" for i in np.argsort(-w) if w[i] > 0.005) + (" (로그확률)" if model.get("log") else " (확률)")
METHOD_LIST = [("avg_all", "전체 평균", fit_avg), ("vote_all", "전체 다수결", fit_vote), ("greedy_nll", f"탐욕 선택(Caruana, NLL, {GREEDY_STEPS}회)", fit_greedy),
               ("stack_log", "가중 로그 결합(NLL로 가중치 학습)", fit_stack), ("best_subset", f"최고 부분집합(상위 {TOP_N}개 중 2~{MAX_SUBSET}개, 정확도)", fit_subset)]
print("방법:", " | ".join(f"{k} = {lab}" for k, lab, _ in METHOD_LIST))
'''

CV = r'''
# ── 중첩 교차검증: 방법마다 '학습 폴드로 조합을 고르고 → 남긴 폴드를 예측'을 10겹 × 5회 반복 → 고를 때 쓰지 않은 문항으로 잰 정확도 ──
import time
t0 = time.time(); N = len(Y_VAL); rs = np.random.RandomState(CV_SEED)
OOF = {k: np.zeros((CV_REPEATS, N), int) for k, _, _ in METHOD_LIST}
for r in range(CV_REPEATS):
    for te in np.array_split(rs.permutation(N), CV_FOLDS):
        tr = np.setdiff1d(np.arange(N), te)
        for k, _, fn in METHOD_LIST: OOF[k][r, te] = apply(fn(tr), PV[:, te], LV[:, te])
    print(f"  반복 {r + 1}/{CV_REPEATS} 끝 ({time.time() - t0:.0f}s)", flush=True)
CV_ROWS = []
for k, lab, _ in METHOD_LIST:
    acc = (OOF[k] == Y_VAL[None, :]).mean(1) * 100
    fb = [paired(REF, OOF[k][r], Y_VAL) for r in range(CV_REPEATS)]
    CV_ROWS.append({"방법": k, "설명": lab, "중첩 CV 정확도(%)": round(float(acc.mean()), 2), "반복 간 표준편차": round(float(acc.std()), 2),
                    f"{BEST_NOW} 대비 고침": round(float(np.mean([x[0] for x in fb])), 1), "새로 틀림": round(float(np.mean([x[1] for x in fb])), 1),
                    "순이득": round(float(np.mean([x[2] for x in fb])), 1), "바뀐 답": round(float(np.mean([x[3] for x in fb])), 1)})
CV_TABLE = pd.DataFrame(CV_ROWS)
print(CV_TABLE.to_string(index=False))
print(f"(비교 기준) {BEST_NOW} 검증 {(REF == Y_VAL).mean() * 100:.2f}% · clean4 {(A0['clean4_qg_hcx']['val'] == Y_VAL).mean() * 100:.2f}% — 둘 다 검증을 보고 고른 레시피라 실제보다 높게 나올 수 있음 (공개 점수가 공정한 비교)")
'''

FINAL = r'''
# ── 전체 검증 1,000문항으로 다시 맞춘 조합 → test 답 → 제출 후보. 판정은 중첩 CV 1등 방법 하나로만 (사전 기준) ──
V = {"lines": [], "cands": []}
def say(s): print(s); V["lines"].append(s)
FINAL_M, FA = {}, {}
fa_best = file_answers(RECIPES[BEST_NOW]["file"] if BEST_NOW in RECIPES else VOTE["file"])
base_best = fa_best if fa_best is not None else A0[BEST_NOW]["test"]
for k, lab, fn in METHOD_LIST:
    mdl = fn(np.arange(len(Y_VAL))); FINAL_M[k] = mdl; FA[k] = apply(mdl, PT, LT)
    p = os.path.join(E10_DIR, f"submission_e10_{k}.csv"); sub_df[["id"]].assign(answer=[LETTERS[i] for i in FA[k]]).to_csv(p, index=False)
    say(f"  {k}: {describe(mdl)} | 전체 검증 {(apply(mdl, PV, LV) == Y_VAL).mean() * 100:.2f}% (고른 문항으로 잰 값 — 참고만) | test에서 {BEST_NOW} 제출과 다른 답 {int((FA[k] != base_best).sum())}개")
_order = {k: i for i, (k, _, _) in enumerate(METHOD_LIST)}
best = max(CV_ROWS, key=lambda r: (r["중첩 CV 정확도(%)"], -_order[r["방법"]]))["방법"]     # 동률이면 더 단순한(앞쪽) 방법
row = next(r for r in CV_ROWS if r["방법"] == best); n_cv = int(round(row["순이득"])); Dn = int((FA[best] != base_best).sum())
b0 = count_of(PUBLIC[BEST_NOW]); need = max(0, NET_MIN - n_cv)
say(f"[주 후보] 중첩 CV 1등: {best} ({row['설명']}) — 중첩 CV {row['중첩 CV 정확도(%)']:.2f}% | {BEST_NOW} 대비 순이득 {row['순이득']:+.1f} (바뀐 답 {row['바뀐 답']:.1f})")
if Dn == 0:
    say(f"판정: 제출 불필요 — test에서 {BEST_NOW}와 답이 같음")
elif n_cv <= -3:
    say(f"판정: 불채택 — 중첩 CV 순이득 {n_cv:+d} (≤ −3). 공개로 확인할 필요 없음")
else:
    fn_ = f"submission_e10_{best}.csv"; V["cands"].append((fn_, Dn))
    say(f"판정: 공개 확인 — 제출 후보 {fn_} | {BEST_NOW}({PUBLIC[BEST_NOW]:.5f} = {b0:,}문항)과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개) | "
        f"◀ 공개 {score_of(b0 + need):.5f}({b0 + need:,}문항) 이상이면 채택 (중첩 CV {n_cv:+d} + 공개 ≥ +{NET_MIN}, 공개 ≥ 0)")
    say("  다른 방법의 파일은 참고용입니다 (여러 개를 공개로 시험해 가장 좋은 것을 고르면 공개 점수에 과적합됨)")
json.dump({"members": MEMBERS_ALL, "skipped": _skip, "cv": CV_ROWS, "final": {k: describe(m) for k, m in FINAL_M.items()},
           "weights": {k: (np.asarray(m["w"], float).round(4).tolist() if "w" in m else "vote") for k, m in FINAL_M.items()},
           "verdict": V["lines"], "cands": V["cands"], "settings": {"CV": [CV_FOLDS, CV_REPEATS, CV_SEED], "TOP_N": TOP_N, "MAX_SUBSET": MAX_SUBSET,
           "GREEDY_STEPS": GREEDY_STEPS, "STACK_L2": STACK_L2, "STACK_WMAX": STACK_WMAX, "PRESET_PREF": PRESET_PREF}},
          open(os.path.join(E10_DIR, "e10_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

EXTRAS = r'''
# ── 참고: test 예측이 없어 빠진 실행을 '전체 평균'에 넣으면 검증이 어떻게 되나 → test 추론을 돌릴 가치가 있는지 (GPU 쓰기 전에 확인) ──
import glob
def _adapter_md5(run):
    fs = sorted(glob.glob(os.path.join(OUT_DIR, run, "adapter_ep*", "adapter_model.safetensors")))
    return hashlib.md5(open(fs[-1], "rb").read(1 << 20)).hexdigest()[:8] if fs else None
_pool_md5 = {r: _adapter_md5(r) for r in NAMES}
_base = PV.mean(0).argmax(1); XROWS, EXTRA = [], []
for run in sorted(x for x in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, x))):
    if run in NAMES or run in EXCLUDE or run.endswith("_dry") or (read_log(run).get("config") or {}).get("TRAIN_ON_FULL"): continue
    pres = _presets(run); pre = next((p for p in PRESET_PREF if p in pres), pres[0] if pres else None)
    if pre is None: continue
    try: P = load(run, pre, "val")
    except Exception as e: XROWS.append({"실행": run, "해상도": pre, "검증(%)": "", "자기 어댑터": "", "test 예측": "", "전체 평균에 추가(검증)": f"검증 읽기 실패: {str(e)[:50]}"}); continue
    md5 = _adapter_md5(run); dup = [r for r, m in _pool_md5.items() if md5 and m == md5]
    ft, _ = member_file(run, pre, "test"); tst = "없음"
    if ft:
        try: load(run, pre, "test"); tst = "있음(다른 이유로 제외)"
        except Exception: tst = "일부만"
    a = P.argmax(1); f, b, n, D = paired(_base, ((PV.sum(0) + P) / (M + 1)).argmax(1), Y_VAL)
    XROWS.append({"실행": run, "해상도": pre, "검증(%)": round(float((a == Y_VAL).mean() * 100), 2),
                  "자기 어댑터": "없음(평가만 한 폴더)" if md5 is None else (f"{dup[0]}와 같음" if dup else "있음"),
                  "지금 멤버와 다른 답(최소)": int(min((a != PV[i].argmax(1)).sum() for i in range(M))), "test 예측": tst,
                  "전체 평균에 추가(검증)": f"고침 {f}/새로 틀림 {b}/순이득 {n:+d}"})
    if md5 and not dup and (a == Y_VAL).mean() * 100 >= MIN_VAL_ACC: EXTRA.append((run, pre, P))
X_TABLE = pd.DataFrame(XROWS).fillna("")
print(X_TABLE.to_string(index=False) if len(X_TABLE) else "test 예측이 없어 빠진 실행 없음")
if EXTRA:
    f, b, n, D = paired(_base, ((PV.sum(0) + sum(P for _, _, P in EXTRA)) / (M + len(EXTRA))).argmax(1), Y_VAL)
    X_LINE = f"자기 어댑터가 있는 {len(EXTRA)}개({', '.join(r for r, _, _ in EXTRA)})를 모두 넣으면: 고침 {f} / 새로 틀림 {b} / 순이득 {n:+d} (바뀐 답 {D})"
else:
    X_LINE = "전체 평균에 더 넣을 수 있는 실행 없음"
print(X_LINE)
'''

SUMMARY = r'''
print(f"E10 | 멤버 {M}개 | 중첩 CV {CV_FOLDS}겹 × {CV_REPEATS}회 | 지금 최고 {BEST_NOW} 검증 {(REF == Y_VAL).mean() * 100:.2f}%")
print("멤버:", ", ".join(f"{r}@{p}" + (" (반드시 포함)" if (r, p) in MUST_INCLUDE else "") for r, p in MEMBERS_ALL))
print(MEM_TABLE.drop(columns=["모델"]).to_string(index=False))
print(CV_TABLE.drop(columns=["설명"]).to_string(index=False))
for l in V["lines"]: print(l)
print("[test 예측이 없어 빠진 실행]"); print(X_TABLE.to_string(index=False) if len(X_TABLE) else "없음"); print(X_LINE)
'''

INTRO = r'''# E10 · 모든 멤버에서 앙상블 조합 새로 찾기 (CPU, 학습 없음)

**왜**: 지금 제출 레시피(clean4 가중치 [1,1,1,1,2], 다수결)는 최종 답이 아닙니다. 거기에 없는 멤버(InternVL 등)도, 레시피에 있는 멤버를 빼거나 가중치를 바꾸는 것도 모두 후보입니다.
다만 검증이 1,000문항이고 지금 오답이 20개 남짓이라, **조합을 많이 시험해 검증 1등을 고르면 그 1등은 대부분 운**입니다. 그래서 조합을 고르는 **절차 자체**를 중첩 교차검증으로 평가합니다: 학습 폴드(900문항)로 조합을 고르고, 고를 때 안 본 100문항으로 채점 → 10겹 × 5회 반복.

**하는 일**
1. **InternVL(`ivl35_38b_r16_native@up2.0`)은 반드시 넣습니다** (1번 셀 `MUST_INCLUDE`). 파일을 못 찾으면 그 폴더의 예측 파일 목록을 보여 주고 멈춥니다. 나머지는 `vqa_runs`에서 고정 분할로 학습한 실행을 **모두** 자동으로 찾습니다 (검증·test 예측이 둘 다 있고 전체 데이터 학습이 아닌 것). 제외한 실행은 이유와 함께 출력합니다.
2. 멤버 표: 검증 정확도·NLL, **지금 최고(다수결)가 틀린 문항 중 몇 개를 맞히는지**, clean4에 가중치 1로 넣었을 때의 변화(참고).
3. 미리 정한 조합 방법 5개 — 전체 평균 / 전체 다수결 / Caruana 탐욕 선택(NLL) / 가중 로그 결합(NLL로 가중치 학습) / 최고 부분집합(정확도) — 을 중첩 교차검증으로 비교합니다.
4. 방법마다 전체 검증으로 다시 맞춰 test 제출 파일을 만들고, **중첩 CV 1등 방법 하나만** 공개로 확인합니다.

**사전 기준 (결과를 보기 전에 고정)**: 중첩 CV 순이득(지금 최고 대비, 반복 평균) + 공개 순이득 ≥ +3이고 공개 순이득 ≥ 0 → 채택. 중첩 CV 순이득 ≤ −3이면 공개 확인 없이 불채택. 노트북이 필요한 공개 점수를 미리 출력합니다.
지금 레시피의 검증 점수도 검증을 보고 고른 것이라 실제보다 높게 나올 수 있어, 이 비교는 새 방법에 불리한 쪽(보수적)입니다.

**실행**: GPU 없는 새 세션에서 위에서부터 끝까지. 기존 폴더는 읽기만 하고 `vqa_runs/e10_search/`에만 씁니다.'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 데이터 (CSV만) · 고정 검증 분할"), ("code", DATA),
    ("markdown", "## 3. 예측 파일 읽기 함수 (E8과 같음)"), ("code", HELPERS),
    ("markdown", "## 4. 멤버 후보 자동 탐색 + 지금 레시피 재현"), ("code", POOL),
    ("markdown", "## 5. 멤버 표 — 지금 최고가 틀린 문항을 누가 맞히나"), ("code", MEMTABLE),
    ("markdown", "## 6. 조합 방법 (미리 고정)"), ("code", METHODS),
    ("markdown", "## 7. 중첩 교차검증 (10겹 × 5회)"), ("code", CV),
    ("markdown", "## 8. 전체 검증으로 다시 맞춘 조합 → 제출 후보 · 판정"), ("code", FINAL),
    ("markdown", "## 9. 참고: test 예측이 없어 빠진 실행을 전체 평균에 넣으면?\n자기 어댑터가 있는(평가만 한 폴더가 아닌) 실행만 합산합니다. 좋아지면 그 실행들의 test 추론을 돌릴 가치가 있습니다."), ("code", EXTRAS),
    ("markdown", "## 10. 결과 요약"), ("code", SUMMARY),
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
