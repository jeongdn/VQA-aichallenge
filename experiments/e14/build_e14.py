# -*- coding: utf-8 -*-
"""E14 노트북 빌더 (CPU): 지금 제출 조합·공개 점수를 전혀 쓰지 않고, 검증 1,000문항만으로 앙상블을 처음부터 고른다.
- 멤버 자동 탐색 (4bit·bf16·시드·dev 추가 실행 모두 — 검증·test 예측이 다 있는 실행)
- 같은 기반 모델(MODEL_ID)의 실행끼리 '계열'로 묶음 (4bit·bf16 두 번 학습한 모델이 두 표를 갖지 않도록)
- 미리 정한 조합 방법 6개를 중첩 교차검증(10겹 × 5회)으로 비교 → '1-표준오차 규칙'으로 하나를 고름
  (1등과의 차이가 우연 범위(1 SE) 안이면 더 단순한 방법 — 문항 몇 개 차이를 실력으로 착각하지 않도록)
- 고른 방법을 검증 전체로 다시 맞춰 test 제출 파일 1개
데이터·예측 파일 읽기 함수는 E8 노트북 코드를 그대로 씀.
"""
import sys, uuid
import nbformat

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e14/out/e14_val_ensemble.ipynb"
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
ENSEMBLE_SUBDIR = "ensembles"      # (이 노트북은 읽지 않음 — E8 공용 코드의 경로 변수라 남겨 둠)
E14_SUBDIR   = "e14_val_ens"       # 제출 파일을 쓰는 폴더
SEED, VAL_SIZE = 42, 1000          # 검증 분할 — 팀 공통값

# ── 멤버: vqa_runs 전체에서 자동 탐색 (고정 분할로 학습, 검증 1,000·test 6,714 예측이 모두 있는 실행. 4bit·bf16·시드·dev 추가 모두) ──
MUST_INCLUDE = [("ivl35_38b_r16_native", "up2.0")]   # 파일을 못 찾으면 멈춤 (조합에 쓸지는 아래 규칙이 검증으로 정함)
PRESET_PREF = ["up2.0"]    # 실행마다 이 해상도 예측을 우선 사용 (없으면 검증 정확도가 가장 높은 해상도)
INCLUDE = []               # 비우면 자동 탐색. 넣으면 이 목록만: [("q35_27b_r16_native", "up2.0"), ...]
EXCLUDE = []               # 자동 탐색에서 뺄 실행 이름
MIN_VAL_ACC = 90.0         # 이보다 낮은 실행은 고장으로 보고 모든 방법에서 제외

# ── 조합 규칙 (결과를 보기 전에 고정) ──
POOL_DELTA = 1.5           # '강한 풀': 검증 정확도가 상위 3개 실행 평균보다 1.5%p 넘게 낮지 않은 실행 (평균·다수결 방법에 적용, 폴드마다 다시 계산)
T_GRID = "T70"             # 멤버별 온도 격자 (0.25~8, 로그 간격 70개) — 폴드마다 학습 문항 NLL 최소
CV_FOLDS, CV_REPEATS, CV_SEED = 10, 5, 2026   # 중첩 교차검증: 10겹 × 5회 반복
GREEDY_STEPS = 20          # Caruana 탐욕 선택 횟수 (같은 실행을 여러 번 고를 수 있음 = 가중치)
STACK_L2, STACK_WMAX = 1e-3, 3.0              # 가중 로그 결합: 가중치 L2 규제, 가중치 상한
FORCE_METHOD = None        # 1-SE 규칙 대신 특정 방법으로 제출 파일을 만들 때만 (예: "stack_log"). 보통은 None

# ── 최종 후보 B: A와 같은 방법 + 전체 데이터(6,714, 검증 포함)로 다시 학습한 실행(<실행>_full)을 같은 자리에 넣음 ──
FULL_SUFFIX = "_full"      # e8_full_retrain.ipynb가 만든 폴더 이름 규칙 (bf16, 원래 멤버와 같은 레시피)
FULL_MODE = "mix"          # "mix": 원래 실행과 _full을 같은 비중으로 함께 씀 / "swap": _full이 있으면 원래 실행 대신 _full만
FULL_MIN_AGREE = 97.0      # _full의 test 답이 원래 실행과 이 비율(%) 이상 같아야 씀 (학습 고장 감지 — 검증으로는 못 잼)
'''

DATA = _n8["DATA"]
DATA = rep(DATA, 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'E14_DIR = os.path.join(OUT_DIR, E14_SUBDIR); os.makedirs(E14_DIR, exist_ok=True)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 결과 폴더: {E14_DIR}")')
HELPERS = _n8["HELPERS"]

POOL = r'''
# ── 멤버 자동 탐색 ──
def _presets(run):
    d = os.path.join(OUT_DIR, run)
    return sorted({m.group(2) for f in os.listdir(d) if (m := re.match(r"^preds_final_ep(\d+)__(.+)__1000\.csv$", f))})
def model_of(run):
    # 기반 모델 ID: 자기 train_log → 없으면 이름 끝 조각을 하나씩 떼며 원래 실행의 train_log에서 (평가만 한 폴더 등)
    r = run
    for _ in range(4):
        mid = (read_log(r).get("config") or {}).get("MODEL_ID")
        if mid: return str(mid)
        if "_" not in r: break
        r = r.rsplit("_", 1)[0]
    return "?" + run
MEMBERS_ALL, _skip, _wait, VALONLY = [], [], [], {}
for run, pre in MUST_INCLUDE:
    d = os.path.join(OUT_DIR, run); assert os.path.isdir(d), f"반드시 찾을 멤버의 폴더가 없습니다: {d} → 1번 셀 MUST_INCLUDE의 이름을 확인하세요"
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
    ok, why_wait, vo = [], None, None
    for pre in pres:
        fv, _ = member_file(run, pre, "val"); ft, _ = member_file(run, pre, "test")
        if not fv: continue
        try: P = load(run, pre, "val")
        except Exception as e: _skip.append(f"{run}@{pre} (검증 읽기 실패: {str(e)[:70]})"); continue
        if vo is None or pre in PRESET_PREF: vo = (pre, P)
        if not ft: why_wait = f"{run}@{pre} (검증 {(P.argmax(1) == Y_VAL).mean() * 100:.2f}% · test 예측 없음 → test 추론이 끝나면 자동 포함)"; continue
        try: load(run, pre, "test")
        except Exception as e:
            why_wait = (f"{run}@{pre} (검증 {(P.argmax(1) == Y_VAL).mean() * 100:.2f}% · test 예측 진행 중(일부만 있음) → 끝나면 자동 포함)" if "없는 문항" in str(e)
                        else f"{run}@{pre} (test 읽기 실패: {str(e)[:60]})"); continue
        ok.append((pre, float((P.argmax(1) == Y_VAL).mean() * 100)))
    if not ok:
        if why_wait: _wait.append(why_wait); VALONLY[run] = vo
        continue
    pre = next((p for p in PRESET_PREF if p in dict(ok)), max(ok, key=lambda t: t[1])[0])
    if dict(ok)[pre] < MIN_VAL_ACC: _skip.append(f"{run}@{pre} (검증 {dict(ok)[pre]:.1f}% < {MIN_VAL_ACC} → 고장으로 봄)"); continue
    MEMBERS_ALL.append((run, pre))
M = len(MEMBERS_ALL); NAMES = [r for r, _ in MEMBERS_ALL]
FAM = [model_of(r) for r in NAMES]                      # 계열 = 기반 모델 ID
def short_fam(f): return f.split("/")[-1]
def prec_of(i):
    r = NAMES[i]
    if "bf16eval" in r: return "4bit 학습·bf16 평가"
    if "_bf16" in r: return "bf16"
    return "4bit" if re.search(r"(?<![\d.])(2\d|3\d|7\d)B", FAM[i]) else "bf16"   # 학습 노트북의 USE_4BIT="auto" 규칙
PV = np.stack([load(r, p, "val") for r, p in MEMBERS_ALL]); PT = np.stack([load(r, p, "test") for r, p in MEMBERS_ALL])
LV, LT = np.log(np.clip(PV, 1e-6, 1.0)), np.log(np.clip(PT, 1e-6, 1.0))
print(f"멤버 {M}개 · 계열 {len(set(FAM))}개")
for r, p in MUST_INCLUDE: print(f"✅ 찾음: {r}@{p} | 검증 {(load(r, p, 'val').argmax(1) == Y_VAL).mean() * 100:.2f}% (조합에 쓸지는 규칙이 정함)")
if _wait: print("test 예측이 없어 빠진 실행 (test 추론이 끝난 뒤 이 노트북을 다시 실행하면 자동 포함):", *_wait, sep="\n  · ")
if _skip: print("제외:", *_skip, sep="\n  · ")
'''

MEMTABLE = r'''
# ── 멤버 표 · 계열 표 · bf16 짝 비교 (검증 전체) ──
TG = T_GRIDS[T_GRID]
def fit_T_idx(m, idx):
    p, y = PV[m, idx], Y_VAL[idx]; ii = np.arange(len(idx))
    return float(TG[int(np.argmin([-np.mean(np.log(temp_scale(p, T)[ii, y])) for T in TG]))])
def strong_pool(idx):
    # 기준 = 상위 3개 실행의 평균 정확도 (한 실행만 튀게 높아도 풀이 갑자기 줄지 않도록)
    acc = [(PV[m, idx].argmax(1) == Y_VAL[idx]).mean() * 100 for m in range(M)]; ref = float(np.mean(sorted(acc)[-3:]))
    return [m for m in range(M) if acc[m] >= ref - POOL_DELTA - 1e-9]
ALL = np.arange(len(Y_VAL)); T_ALL = [fit_T_idx(m, ALL) for m in range(M)]; POOL_ALL = strong_pool(ALL)
rows = []
for i in range(M):
    a = PV[i].argmax(1)
    rows.append({"실행": NAMES[i], "계열(기반 모델)": short_fam(FAM[i]), "정밀도": prec_of(i), "해상도": MEMBERS_ALL[i][1],
                 "검증(%)": round(float((a == Y_VAL).mean() * 100), 2), "NLL": round(nll(PV[i], Y_VAL), 4), "온도 T": round(T_ALL[i], 2),
                 "보정 NLL": round(nll(temp_scale(PV[i], T_ALL[i]), Y_VAL), 4), "강한 풀": "O" if i in POOL_ALL else f"X (상위3 평균−{POOL_DELTA}%p 미만)"})
MEM_TABLE = pd.DataFrame(rows).sort_values(["계열(기반 모델)", "실행"]).reset_index(drop=True)
print(MEM_TABLE.to_string(index=False))
# bf16 짝: 이름이 <원래 실행>_bf16... 이면 같은 문항끼리 비교 (4bit 학습 vs bf16 학습)
PAIRS = []
_PVAL = {**{r: PV[i] for i, r in enumerate(NAMES)}, **{r: v[1] for r, v in VALONLY.items() if v is not None}}
_FAMV = {**dict(zip(NAMES, FAM)), **{r: model_of(r) for r in VALONLY}}
for r in list(NAMES) + sorted(VALONLY):
    if r not in _PVAL or "_bf16" not in r or "bf16eval" in r: continue
    base = r.split("_bf16")[0]
    base = base if base in _PVAL else next((b for b in _PVAL if _FAMV[b] == _FAMV[r] and "_bf16" not in b and b.startswith(base.rsplit("_", 1)[0])), None)
    if base is None: continue
    a0, a1 = _PVAL[base].argmax(1), _PVAL[r].argmax(1); f, b, n, D = paired(a0, a1, Y_VAL)
    PAIRS.append(f"  {base} → {r}{' (test 대기)' if r in VALONLY else ''}: 검증 {(a0 == Y_VAL).mean() * 100:.2f}% → {(a1 == Y_VAL).mean() * 100:.2f}% | 고침 {f} · 새로 틀림 {b} · 순이득 {n:+d} (바뀐 답 {D})")
if PAIRS: print("\n4bit → bf16 (같은 문항끼리):", *PAIRS, sep="\n")
# 계열 표: 강한 풀 안에서 계열마다 (온도 보정) 실행 평균
_g = {}
for m in POOL_ALL: _g.setdefault(FAM[m], []).append(m)
FAM_ROWS = []
for f, ms in sorted(_g.items(), key=lambda t: short_fam(t[0])):
    Pf = np.mean([temp_scale(PV[m], T_ALL[m]) for m in ms], 0)
    FAM_ROWS.append({"계열": short_fam(f), "실행 수": len(ms), "실행": ", ".join(NAMES[m] for m in ms), "계열 평균 검증(%)": round(float((Pf.argmax(1) == Y_VAL).mean() * 100), 2),
                     "계열 평균 NLL": round(nll(Pf, Y_VAL), 4)})
FAM_TABLE = pd.DataFrame(FAM_ROWS)
print("\n계열 (강한 풀, 온도 보정 후 실행 평균):"); print(FAM_TABLE.to_string(index=False))
C = PV.argmax(2) == Y_VAL[None, :]
print(f"\n멤버 중 하나라도 맞힌 검증 문항 {int(C.any(0).sum())} (조합으로 가능한 상한) | 전원 오답 {int((~C.any(0)).sum())}")
'''

METHODS = r'''
# ── 조합 방법 6개 (결과를 보기 전에 고정). 목록 순서 = 단순한(덜 맞추는) 순서 = 1-SE 규칙에서 먼저 고르는 순서 ──
#    fit(학습 문항 번호) → 모델 / scores(모델, 확률, 로그확률) → (문항 × 4) 점수, argmax가 답
from scipy.optimize import minimize
def _nll(P, y): return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))
def _groups(runs):
    g = {}
    for m in runs: g.setdefault(FAM[m], []).append(m)
    return g
def fit_cal_fam(tr):          # 강한 풀 · 실행별 온도 → 계열 안에서 평균 → 계열 동일 가중 확률 평균
    pool = strong_pool(tr); return {"kind": "fam", "groups": _groups(pool), "T": {m: fit_T_idx(m, tr) for m in pool}}
def fit_avg_fam(tr):          # 강한 풀 · 온도 없이 계열 동일 가중
    pool = strong_pool(tr); return {"kind": "fam", "groups": _groups(pool), "T": {m: 1.0 for m in pool}}
def fit_vote_fam(tr):         # 강한 풀 · 계열마다 (온도 보정 평균) 답 → 다수결, 동률이면 cal_fam 확률
    mdl = fit_cal_fam(tr); mdl["vote"] = True; return mdl
def fit_avg_runs(tr):         # 강한 풀 · 온도 없이 실행 동일 가중 (E10 전체 평균과 같은 방식)
    pool = strong_pool(tr); return {"kind": "runs", "w": {m: 1.0 for m in pool}, "T": {m: 1.0 for m in pool}}
def fit_greedy_cal(tr):       # 전체 멤버 · 실행별 온도 → Caruana 탐욕 선택 (학습 문항 NLL, 중복 허용 = 가중치)
    T = {m: fit_T_idx(m, tr) for m in range(M)}; Pc = [temp_scale(PV[m, tr], T[m]) for m in range(M)]; y = Y_VAL[tr]
    cnt = np.zeros(M); cur = None
    for t in range(GREEDY_STEPS):
        cands = [Pc[m] if cur is None else (cur * t + Pc[m]) / (t + 1) for m in range(M)]
        j = int(np.argmin([_nll(c, y) for c in cands])); cnt[j] += 1; cur = cands[j]
    return {"kind": "runs", "w": {m: float(cnt[m]) for m in range(M) if cnt[m] > 0}, "T": T}
def fit_stack(tr):            # 전체 멤버 · 가중 로그 결합: log p ∝ Σ w_m log p_m, 학습 문항 NLL + L2 최소 (가중치가 온도 역할도 함)
    L, y = LV[:, tr], Y_VAL[tr]; ii = np.arange(len(tr)); Ly = L[:, ii, y]
    def f(w):
        s = np.tensordot(w, L, 1); s = s - s.max(1, keepdims=True); lse = np.log(np.exp(s).sum(1)); p = np.exp(s - lse[:, None])
        return (-np.mean(s[ii, y] - lse) + STACK_L2 * float(w @ w), -np.mean(Ly - np.einsum("mnk,nk->mn", L, p), axis=1) + 2 * STACK_L2 * w)
    r = minimize(f, np.full(M, 1.0 / M), jac=True, method="L-BFGS-B", bounds=[(0.0, STACK_WMAX)] * M)
    return {"kind": "log", "w": r.x}
def scores(mdl, P, L):
    if mdl["kind"] == "log":
        s = np.tensordot(mdl["w"], L, 1); s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    if mdl["kind"] == "runs":
        w = mdl["w"]; tot = sum(w.values()); return sum(w[m] / tot * temp_scale(P[m], mdl["T"][m]) for m in w)
    fp = [np.mean([temp_scale(P[m], mdl["T"][m]) for m in ms], 0) for ms in mdl["groups"].values()]
    avg = np.mean(fp, 0)
    if mdl.get("vote"):
        A = np.stack([q.argmax(1) for q in fp]); cnt = np.stack([(A == c).sum(0) for c in range(4)], 1).astype(float)
        return cnt + 1e-3 * avg
    return avg
def describe(mdl):
    if mdl["kind"] == "log":
        w = mdl["w"]; return "가중 로그 결합: " + " + ".join(f"{NAMES[i]}×{w[i]:.2f}" for i in np.argsort(-w) if w[i] > 0.005)
    if mdl["kind"] == "runs":
        w = mdl["w"]; tot = sum(w.values())
        return "실행 가중 평균: " + " + ".join(f"{NAMES[m]}×{w[m] / tot:.2f}(T{mdl['T'][m]:.2f})" for m in sorted(w, key=lambda m: -w[m]))
    return ("계열 다수결 " if mdl.get("vote") else "계열 동일 가중 ") + f"({len(mdl['groups'])}계열): " + " | ".join(
        f"{short_fam(f)}[" + ", ".join(f"{NAMES[m]} T{mdl['T'][m]:.2f}" for m in ms) + "]" for f, ms in mdl["groups"].items())
METHOD_LIST = [   # (이름, 설명, fit, NLL 계산 여부)
    ("cal_fam",    "강한 풀 · 실행별 온도 → 계열 동일 가중 확률 평균", fit_cal_fam, True),
    ("avg_fam",    "강한 풀 · 온도 없이 계열 동일 가중", fit_avg_fam, True),
    ("vote_fam",   "강한 풀 · 계열 다수결 (동률이면 cal_fam 확률)", fit_vote_fam, False),
    ("avg_runs",   "강한 풀 · 실행 동일 가중 (E10 전체 평균 방식)", fit_avg_runs, True),
    ("greedy_cal", f"전체 멤버 · 온도 보정 후 Caruana 탐욕 선택 {GREEDY_STEPS}회 (NLL)", fit_greedy_cal, True),
    ("stack_log",  "전체 멤버 · 가중 로그 결합 (NLL로 가중치 학습)", fit_stack, True),
]
print("방법 (1-SE 규칙에서 먼저 고르는 순서):", *[f"{i + 1}. {k} — {lab}" for i, (k, lab, _, _) in enumerate(METHOD_LIST)], sep="\n  ")
'''

CV = r'''
# ── 중첩 교차검증: 방법마다 '학습 폴드(900문항)로 풀·온도·가중치를 정하고 → 남긴 100문항을 예측'을 10겹 × 5회 반복 ──
import time
t0 = time.time(); N = len(Y_VAL); rs = np.random.RandomState(CV_SEED)
OOF = {k: np.zeros((CV_REPEATS, N), int) for k, _, _, _ in METHOD_LIST}
OOFP = {k: np.zeros((CV_REPEATS, N, 4)) for k, _, _, ok in METHOD_LIST if ok}
POOL_SIZES = []
for r in range(CV_REPEATS):
    for te in np.array_split(rs.permutation(N), CV_FOLDS):
        tr = np.setdiff1d(np.arange(N), te); POOL_SIZES.append(len(strong_pool(tr)))
        for k, _, fn, ok in METHOD_LIST:
            S = scores(fn(tr), PV[:, te], LV[:, te]); OOF[k][r, te] = S.argmax(1)
            if ok: OOFP[k][r, te] = S
    print(f"  반복 {r + 1}/{CV_REPEATS} 끝 ({time.time() - t0:.0f}s)", flush=True)
CORR = {k: (OOF[k] == Y_VAL[None, :]).mean(0) for k in OOF}          # 문항별 정답률 (반복 평균)
ACC = {k: float(CORR[k].mean() * 100) for k in OOF}
_ord = [k for k, _, _, _ in METHOD_LIST]
BEST = max(_ord, key=lambda k: (round(ACC[k], 6), -_ord.index(k)))
def se_vs_best(k):
    d = CORR[k] - CORR[BEST]; return float(d.std(ddof=1) / np.sqrt(len(d)) * 100) if k != BEST else 0.0
CV_ROWS = []
for k, lab, _, ok in METHOD_LIST:
    se = se_vs_best(k)
    CV_ROWS.append({"방법": k, "설명": lab, "중첩 CV 정확도(%)": round(ACC[k], 2), "중첩 CV NLL": round(float(np.mean([_nll(OOFP[k][r], Y_VAL) for r in range(CV_REPEATS)])), 4) if ok else "",
                    "1등과 차이(%p)": round(ACC[k] - ACC[BEST], 2), "차이의 표준오차(%p)": round(se, 2),
                    "1등과 다른 답(평균)": round(float((OOF[k] != OOF[BEST]).sum(1).mean()), 1), "1 SE 안": "O" if ACC[k] >= ACC[BEST] - se - 1e-9 else ""})
CV_TABLE = pd.DataFrame(CV_ROWS)
CHOSEN = next(r["방법"] for r in CV_ROWS if r["1 SE 안"] == "O")      # 1 SE 안에 드는 방법 중 목록에서 가장 앞(단순한) 것
print(CV_TABLE.to_string(index=False))
print(f"강한 풀 크기 (폴드별): {min(POOL_SIZES)}~{max(POOL_SIZES)}개 | 중첩 CV 1등: {BEST} | 1-SE 규칙 선택: {CHOSEN}")
'''

FINAL = r'''
# ── 고른 방법을 검증 1,000문항 전체로 다시 맞춤 → test 답 → 제출 파일 1개 ──
V = {"lines": []}
def say(s): print(s); V["lines"].append(s)
USE = FORCE_METHOD or CHOSEN
assert USE in _ord, f"FORCE_METHOD={FORCE_METHOD!r}는 방법 목록에 없습니다: {_ord}"
fit_fn = dict((k, fn) for k, _, fn, _ in METHOD_LIST)[USE]
FINAL_MODEL = fit_fn(ALL)
S_val, S_test = scores(FINAL_MODEL, PV, LV), scores(FINAL_MODEL, PT, LT)
ANS = S_test.argmax(1)
tag = hashlib.md5(",".join(f"{r}@{p}" for r, p in MEMBERS_ALL).encode()).hexdigest()[:6]
FILE = f"submission_e14_{USE}_{M}runs_{tag}.csv"
sub_df[["id"]].assign(answer=[LETTERS[i] for i in ANS]).to_csv(os.path.join(E14_DIR, FILE), index=False)
row = next(r for r in CV_ROWS if r["방법"] == USE)
say(f"[선택] {USE} — {row['설명']}" + (" (FORCE_METHOD로 지정)" if FORCE_METHOD else f" (1-SE 규칙: 중첩 CV 1등 {BEST}와 {row['1등과 차이(%p)']:+.2f}%p, 표준오차 {row['차이의 표준오차(%p)']:.2f}%p)"))
say(f"  중첩 CV 정확도 {row['중첩 CV 정확도(%)']:.2f}% ← 고를 때 안 본 문항으로 잰 값 (이 조합의 실력 추정)")
say(f"  검증 전체로 맞춘 조합: {describe(FINAL_MODEL)}")
say(f"  (참고) 같은 1,000문항으로 맞추고 잰 검증 {(S_val.argmax(1) == Y_VAL).mean() * 100:.2f}% — 고를 때 본 문항이라 실제보다 높게 나옴")
say(f"  제출 파일: {E14_SUBDIR}/{FILE} | test 답 분포 {dict(zip(LETTERS, np.bincount(ANS, minlength=4).tolist()))}")
json.dump({"members": MEMBERS_ALL, "families": dict(zip(NAMES, FAM)), "wait": _wait, "skipped": _skip, "cv": CV_ROWS, "best": BEST, "chosen": USE,
           "forced": bool(FORCE_METHOD), "final": describe(FINAL_MODEL), "file": FILE, "T_all": dict(zip(NAMES, T_ALL)), "pool_all": [NAMES[m] for m in POOL_ALL],
           "settings": {"POOL_DELTA": POOL_DELTA, "T_GRID": T_GRID, "CV": [CV_FOLDS, CV_REPEATS, CV_SEED], "GREEDY_STEPS": GREEDY_STEPS,
                        "STACK_L2": STACK_L2, "STACK_WMAX": STACK_WMAX, "MIN_VAL_ACC": MIN_VAL_ACC, "PRESET_PREF": PRESET_PREF}},
          open(os.path.join(E14_DIR, f"e14_info_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

REPORT = r'''
# ── 10. 검증 오답 보고서: 고른 조합이 틀린 문항 + 아슬아슬하게 맞힌 문항 → 이미지 포함 HTML ──
import base64, io, glob, html as _html, subprocess
from PIL import Image
REPORT_NEAR = 15                          # 맞혔지만 1·2위 확률 차이가 가장 작은 문항 수 (조합이 흔들리는 영역)
IMG_MAX = {"오답": 1600, "아슬": 1024}    # 이미지 긴 변(픽셀) — 작은 글씨가 보이도록 오답은 크게
Pd = scores(fit_cal_fam(ALL), PV, LV) if USE == "vote_fam" else S_val / S_val.sum(1, keepdims=True)
ans_v = S_val.argmax(1); _srt = np.sort(Pd, 1); margin = _srt[:, -1] - _srt[:, -2]
WRONG = np.where(ans_v != Y_VAL)[0]
NEAR = [int(i) for i in np.argsort(margin, kind="stable") if ans_v[i] == Y_VAL[i]][:REPORT_NEAR]
ITEMS = [(int(i), "오답") for i in WRONG[np.argsort(-margin[WRONG], kind="stable")]] + [(i, "아슬") for i in NEAR]
FAMS_ALL = sorted(set(FAM), key=short_fam)                  # 계열별 답: 계열의 모든 실행(풀 밖 포함)을 온도 보정 후 평균
FAMP = {f: np.mean([temp_scale(PV[m], T_ALL[m]) for m in range(M) if FAM[m] == f], 0) for f in FAMS_ALL}
RUNC = PV.argmax(2) == Y_VAL[None, :]
def _find(p):
    q = os.path.join(DATA_DIR, p)
    if os.path.exists(q): return q
    h = glob.glob(os.path.join(DATA_DIR, "**", os.path.basename(p)), recursive=True); return h[0] if h else None
_need = sorted({os.path.basename(val_df.path.iloc[i]) for i, _ in ITEMS if _find(val_df.path.iloc[i]) is None})
if _need and IN_COLAB and os.path.exists(DRIVE_ZIP):          # 필요한 이미지만 압축에서 꺼냄 → 안 되면 전체 해제
    print(f"이미지 {len(_need)}개를 압축에서 꺼냅니다...")
    subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP] + _need + ["-r"], capture_output=True, text=True)
    if any(_find(val_df.path.iloc[i]) is None for i, _ in ITEMS):
        print("일부를 못 찾아 전체 압축을 풉니다 (몇 분)"); subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP], capture_output=True, text=True)
def _img64(p, kind):
    f = _find(p)
    if f is None: return None
    im = Image.open(f).convert("RGB"); im.thumbnail((IMG_MAX[kind], IMG_MAX[kind])); b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode(), im.size
rows_txt, parts, n_img = [], [], 0
for i, kind in ITEMS:
    r = val_df.iloc[i]; g, a = int(Y_VAL[i]), int(ans_v[i]); k = int(RUNC[:, i].sum())
    fams = " · ".join(f"{short_fam(f)} {LETTERS[int(FAMP[f][i].argmax())]}({FAMP[f][i].max():.2f})" + ("" if any(FAM[m] == f and m in POOL_ALL for m in range(M)) else "*") for f in FAMS_ALL)
    rows_txt.append({"구분": kind, "id": r.id, "정답": LETTERS[g], "조합 답": f"{LETTERS[a]}({Pd[i, a]:.2f})", "정답 확률": round(float(Pd[i, g]), 2), "맞힌 실행": f"{k}/{M}", "질문": str(r.question)[:60]})
    im = _img64(r.path, kind); n_img += im is not None
    opts = "".join(f"<li class='{'g' if j == g else ('w' if j == a else '')}'><b>{L}.</b> {_html.escape(str(r[L]))} <span class='p'>{Pd[i, j]:.3f}</span></li>" for j, L in enumerate(LETTERS))
    runs_ok = ", ".join(NAMES[m] for m in range(M) if RUNC[m, i]) or "없음"
    parts.append(f"<div class='it {kind}'><h3>{kind} · {_html.escape(str(r.id))} · 정답 {LETTERS[g]} / 조합 {LETTERS[a]} · 맞힌 실행 {k}/{M}</h3>"
                 + (f"<img src='data:image/jpeg;base64,{im[0]}' width='{im[1][0]}'>" if im else "<p class='noimg'>(이미지 없음)</p>")
                 + f"<p class='q'>{_html.escape(str(r.question))}</p><ol type='a'>{opts}</ol><p class='f'>계열 답: {_html.escape(fams)}</p><p class='f'>맞힌 실행: {_html.escape(runs_ok)}</p></div>")
ERR_TABLE = pd.DataFrame(rows_txt)
REPORT_FILE = f"e14_val_errors_{tag}.html"
_css = "body{font-family:sans-serif;max-width:1700px;margin:auto}.it{border:1px solid #ccc;margin:16px 0;padding:12px}.오답 h3{color:#b00}.아슬 h3{color:#a60}img{max-width:100%;height:auto}li.g{background:#dfd}li.w{background:#fdd}.p{color:#666;font-size:90%}.q{font-size:120%}.f{font-size:90%;color:#333}"
open(os.path.join(E14_DIR, REPORT_FILE), "w", encoding="utf-8").write(
    f"<html><head><meta charset='utf-8'><style>{_css}</style></head><body><h1>E14 검증 오답 {len(WRONG)}개 + 아슬 {len(NEAR)}개 ({USE}, 실행 {M}개)</h1>"
    f"<p>초록 = 정답, 빨강 = 조합이 고른 오답, 숫자 = 조합 확률. 계열 답의 * = 강한 풀 밖 계열(조합에 안 들어감)</p>{''.join(parts)}</body></html>")
ERR_TABLE.to_csv(os.path.join(E14_DIR, f"e14_val_errors_{tag}.csv"), index=False)
print(ERR_TABLE.to_string(index=False))
print(f"\\n보고서: {E14_SUBDIR}/{REPORT_FILE} (이미지 {n_img}/{len(ITEMS)}개) → 이 HTML 파일을 확인하세요")
'''

BCELL = r'''
# ── 11. 최종 후보 B: A와 같은 방법·같은 계열 구조에 _full 실행(검증 포함 6,714문항, bf16)을 넣음 ──
#    _full은 검증을 학습에 써서 검증으로 잴 수 없음 → 온도는 원래 실행 것을 그대로 쓰고, test 답 일치율·letter_mass로 고장만 걸러냄
FULLS, FROWS = {}, []
_runs_in = sorted({m for ms in FINAL_MODEL.get("groups", {}).values() for m in ms} | set(FINAL_MODEL.get("w", {}) if isinstance(FINAL_MODEL.get("w"), dict) else [])
                  | (set(np.where(np.asarray(FINAL_MODEL["w"]) > 0.005)[0].tolist()) if FINAL_MODEL["kind"] == "log" else set()))
for m in _runs_in:
    run, pre = MEMBERS_ALL[m]; fr = run + FULL_SUFFIX; row = {"원래 실행": run, "전체 데이터 실행": fr}
    if not os.path.isdir(os.path.join(OUT_DIR, fr)): row["상태"] = "없음"; FROWS.append(row); continue
    cfg_f = read_log(fr).get("config") or {}
    ft, why = member_file(fr, pre, "test")
    if ft is None: row["상태"] = f"test 예측 없음 ({why})"; FROWS.append(row); continue
    try: Pf = load(fr, pre, "test")
    except Exception as e: row["상태"] = f"test 진행 중/읽기 실패 ({str(e)[:40]})"; FROWS.append(row); continue
    agree = float((Pf.argmax(1) == PT[m].argmax(1)).mean() * 100)
    mass = float(pd.read_csv(ft).letter_mass.mean()) if "letter_mass" in pd.read_csv(ft, nrows=1).columns else float("nan")
    bad = [x for x, c in [(f"test 일치 {agree:.1f}% < {FULL_MIN_AGREE}", agree < FULL_MIN_AGREE), (f"letter_mass {mass:.3f}", mass == mass and mass < 0.9),
                          ("TRAIN_ON_FULL이 아님", not cfg_f.get("TRAIN_ON_FULL")), (f"모델 다름 ({cfg_f.get('MODEL_ID')})", model_of(fr) != FAM[m])] if c]
    row.update({"test 일치(%)": round(agree, 2), "letter_mass": round(mass, 3), "상태": "✅ 사용" if not bad else "⚠️ 안 씀: " + "; ".join(bad)})
    if not bad: FULLS[m] = Pf
    FROWS.append(row)
FULL_TABLE = pd.DataFrame(FROWS).fillna("")
print(FULL_TABLE.to_string(index=False) if len(FULL_TABLE) else "조합에 쓰인 실행이 없습니다")
if not FULLS:
    B_FILE, B_DIFF = None, 0; print("\\n최종 후보 B: 쓸 수 있는 _full 실행이 없어 만들지 않음 (A와 같음)")
else:
    idx = {m: M + k for k, m in enumerate(sorted(FULLS))}                              # 확장 배열에서 _full의 번호
    PT_X = np.concatenate([PT, np.stack([FULLS[m] for m in sorted(FULLS)])]); LT_X = np.log(np.clip(PT_X, 1e-6, 1.0))
    import copy
    BM = copy.deepcopy(FINAL_MODEL)
    if BM["kind"] == "fam":
        for f, ms in BM["groups"].items():
            new = []
            for m in ms:
                if m in FULLS:
                    BM["T"][idx[m]] = BM["T"][m]; new += ([m, idx[m]] if FULL_MODE == "mix" else [idx[m]])
                else: new.append(m)
            BM["groups"][f] = new
    elif BM["kind"] == "runs":
        w = {}
        for m, wm in BM["w"].items():
            if m in FULLS:
                BM["T"][idx[m]] = BM["T"][m]
                if FULL_MODE == "mix": w[m], w[idx[m]] = wm / 2, wm / 2
                else: w[idx[m]] = wm
            else: w[m] = wm
        BM["w"] = w
    else:
        w = np.concatenate([np.asarray(BM["w"], float), np.zeros(len(FULLS))])
        for m in FULLS:
            if FULL_MODE == "mix": w[idx[m]] = w[m] / 2; w[m] = w[m] / 2
            else: w[idx[m]] = w[m]; w[m] = 0.0
        BM["w"] = w
    B_ANS = scores(BM, PT_X, LT_X).argmax(1); B_DIFF = int((B_ANS != ANS).sum())
    B_FILE = f"submission_e14B_{USE}_{FULL_MODE}_{len(FULLS)}full_{tag}.csv"
    sub_df[["id"]].assign(answer=[LETTERS[i] for i in B_ANS]).to_csv(os.path.join(E14_DIR, B_FILE), index=False)
    print(f"\\n최종 후보 B: {E14_SUBDIR}/{B_FILE} | A와 같은 방법({USE}) · _full {len(FULLS)}개 {FULL_MODE} "
          f"({', '.join(NAMES[m] + FULL_SUFFIX for m in sorted(FULLS))}) | A와 test 답이 다른 문항 {B_DIFF}개")
'''

SUMMARY = r'''
print(f"E14 | 멤버 {M}개 · 계열 {len(set(FAM))}개 | 강한 풀(검증 전체 기준) {len(POOL_ALL)}개 | 중첩 CV {CV_FOLDS}겹 × {CV_REPEATS}회")
print(MEM_TABLE.to_string(index=False))
if PAIRS: print("4bit → bf16:", *PAIRS, sep="\n")
print(FAM_TABLE.drop(columns=["실행"]).to_string(index=False))
print(CV_TABLE.drop(columns=["설명"]).to_string(index=False))
for l in V["lines"]: print(l)
if _wait: print("대기:", *_wait, sep="\n  · ")
if _skip: print("제외:", *_skip, sep="\n  · ")
print(f"오답 보고서: {E14_SUBDIR}/{REPORT_FILE} (오답 {len(WRONG)} · 아슬 {len(NEAR)} · 이미지 {n_img}개)")
print("[최종 후보 B: _full 실행]"); print(FULL_TABLE.to_string(index=False) if len(FULL_TABLE) else "없음")
print(f"B 파일: {B_FILE} (A와 다른 답 {B_DIFF}개)" if B_FILE else "B 파일: 없음 (쓸 수 있는 _full 없음)")
'''

INTRO = r'''# E14 · 검증만으로 앙상블 새로 고르기 (CPU, 학습 없음)

**원칙**: 지금 제출 조합과 공개 점수는 **전혀 쓰지 않습니다.** 고정 분할(5,714/1,000)로 학습한 모든 실행 — 4bit, bf16, 시드·dev 추가 실행까지 — 을 멤버 후보로 놓고, 검증 1,000문항만으로 조합을 처음부터 고릅니다.

**문제**: 검증 오답이 20여 개뿐이라 조합을 여러 개 시험해 검증 1등을 고르면 그 1등은 대부분 운입니다. 그래서
1. **계열로 묶기**: 같은 기반 모델(MODEL_ID)의 실행(예: Q35 4bit와 Q35 bf16)은 한 계열입니다. 두 번 학습한 모델이 표를 두 개 갖지 않도록 계열 안에서 먼저 평균합니다.
2. **강한 풀**: 검증 정확도가 상위 3개 실행 평균보다 1.5%p 넘게 낮은 실행은 평균·다수결 방법에서 뺍니다 (작은 모델이 큰 모델과 같은 한 표를 갖지 않도록). 가중치를 학습하는 두 방법은 모든 멤버를 보고 스스로 정합니다.
3. **방법 6개를 미리 고정**: 계열 온도 보정 평균 / 계열 평균 / 계열 다수결 / 실행 평균 / 탐욕 선택 / 가중 로그 결합.
4. **중첩 교차검증 (10겹 × 5회)**: 900문항으로 풀·온도·가중치를 정하고, 안 본 100문항으로 채점 → 고르는 절차 자체의 실력을 잽니다.
5. **1-표준오차 규칙**: 1등 방법과의 차이가 표준오차(우연 범위) 안인 방법 중 **가장 단순한 것**을 고릅니다. 문항 한두 개 차이를 실력으로 착각하지 않기 위한 규칙입니다.
6. 고른 방법을 검증 전체로 다시 맞춰 **test 제출 파일 1개**를 만들고, 그 조합이 틀린 검증 문항을 이미지와 함께 모은 **오답 보고서(HTML)**도 만듭니다. 다른 방법의 파일은 만들지 않습니다 (여러 개를 공개로 시험해 고르면 다시 공개 점수에 맞추는 셈).

**실행**: GPU 없는 세션에서 위에서부터 끝까지. bf16 등 새 실행의 test 예측이 끝날 때마다 다시 실행하면 자동으로 포함합니다 (파일 이름에 멤버 수·목록 해시가 붙어 이전 결과를 덮어쓰지 않음). 기존 폴더는 읽기만 하고 `vqa_runs/e14_val_ens/`에만 씁니다.'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정 (규칙은 결과를 보기 전에 고정)"), ("code", CFG),
    ("markdown", "## 2. 데이터 (CSV만) · 고정 검증 분할"), ("code", DATA),
    ("markdown", "## 3. 예측 파일 읽기 함수 (E8과 같음)"), ("code", HELPERS),
    ("markdown", "## 4. 멤버 자동 탐색"), ("code", POOL),
    ("markdown", "## 5. 멤버 표 · 4bit→bf16 비교 · 계열 표"), ("code", MEMTABLE),
    ("markdown", "## 6. 조합 방법 6개 (미리 고정)"), ("code", METHODS),
    ("markdown", "## 7. 중첩 교차검증 → 1-SE 규칙으로 방법 선택"), ("code", CV),
    ("markdown", "## 8. 고른 방법 → 검증 전체로 다시 맞춤 → 제출 파일"), ("code", FINAL),
    ("markdown", "## 9. 검증 오답 보고서 (이미지 포함 HTML)\n고른 조합이 틀린 문항과 아슬아슬하게 맞힌 문항을 이미지·질문·보기·계열별 답과 함께 모읍니다. 필요한 이미지만 압축에서 꺼냅니다. **만들어진 HTML 파일을 확인하세요.**"), ("code", REPORT),
    ("markdown", "## 10. 최종 후보 B — A와 같은 방법 + 전체 데이터(_full) 실행\nA를 고른 방법과 계열 구조는 그대로 두고, 조합에 쓰인 실행마다 `<실행>_full`(검증까지 합친 6,714문항, bf16)이 있으면 같은 자리에 넣습니다 (`FULL_MODE=\"mix\"`: 원래 실행과 같은 비중). _full은 검증으로 잴 수 없어서 온도는 원래 실행 것을 쓰고, test 답 일치율·letter_mass로 학습 고장만 거릅니다."), ("code", BCELL),
    ("markdown", "## 11. 결과 요약"), ("code", SUMMARY),
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
