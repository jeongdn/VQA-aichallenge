# -*- coding: utf-8 -*-
"""E8 앙상블 노트북 빌더 (CPU, 새 세션에서 단독 실행).
A) 지금: 멤버 표 · 제출 파일 재현 확인(레시피 후보 탐색) · 다수결 재현
B) 전체 데이터 재학습이 끝난 뒤: 이상 징후 점검 → 전체 데이터 버전 앙상블·다수결 제출 후보 + 공개 점수 판정선
"""
import sys, uuid
import nbformat

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e8/out/e8_ensemble.ipynb"

INTRO = r'''# E8 · 전체 데이터 버전 앙상블 + 제출 파일 재현 확인 (CPU)

**GPU가 필요 없습니다.** 새 Colab 세션(CPU)에서 이 노트북만 위에서부터 실행하면 됩니다. 기존 폴더는 읽기만 하고, 새 파일은 `vqa_runs/e8_full/`에만 씁니다.

**A. 지금 바로 (전체 데이터 학습 전)**
- 4번 셀: 멤버별 학습 설정·검증 정확도·멤버 간 불일치 표. `e8_full_retrain.ipynb`에 넣을 `RECIPE_FROM`과 설정 확인용입니다.
- 5번 셀: 제출 파일(clean4, 다수결 재료 3개, 다수결)이 **멤버 예측 파일만으로 똑같이 다시 만들어지는지** 확인합니다. 파일끼리 비교하는 게 아니라, 파일마다 **자기 레시피**(`_cal`은 온도 보정 포함)로 다시 만들어 비교합니다. Stage 3의 정확한 방식을 몰라도 되도록, 확률 평균/로그확률 평균 × 온도 보정 없음/두 가지 온도 격자를 모두 시험해 **어느 레시피로 똑같이 나오는지** 알려 줍니다.

**B. 전체 데이터 재학습(`<멤버>_full`)이 하나라도 끝난 뒤**
- 6번 셀: 전체 데이터 멤버마다 이상 징후 점검(설정이 원래 멤버와 같은지, test 답 일치율, 학습 확인용 검증 점수) → 통과한 멤버만 바꿔 넣어 **clean4·다수결의 전체 데이터 버전**과 반반 섞은 버전을 만듭니다.
- 제출 전에 공개 점수 판정선(이 점수보다 낮으면 고장 의심)을 먼저 출력합니다. 전체 데이터 멤버는 검증셋을 학습에 썼으므로 검증으로는 판단하지 않습니다.

**실행**: 위에서부터 끝까지. 전체 데이터 학습이 더 끝나면 다시 실행하면 됩니다(멤버를 자동으로 찾음).'''

CFG = r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"      # 기존 제출 파일 폴더 (읽기만 함)
E8_SUBDIR    = "e8_full"           # 새 제출 후보를 쓰는 폴더
SEED, VAL_SIZE = 42, 1000          # 검증 분할 — 팀 공통값
PRESET = "up2.0"

# ── 멤버와 제출 파일 레시피 ──
Q35, Q38 = ("q35_27b_r16_native", PRESET), ("q38_27b_r16_native", PRESET)
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", PRESET), ("gemma4_31b_r16_up20", PRESET), ("hcx32b_r16_native", PRESET)
MEMBERS = [Q35, Q38, GDEV, G, HCX]
RECIPES = {   # cal=True: 멤버마다 검증으로 온도를 맞춘 뒤 평균. alt_members: 파일이 다른 멤버 조합으로 만들어졌는지도 시험
    "clean4_qg_hcx":    dict(file="submission_clean4_qg_hcx.csv",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], cal=False),
    "ens_q38_gemma4":   dict(file="submission_ens_q38_gemma4.csv",   members=[Q38, G],    weights=[1, 1], cal=False, alt_members=[[Q38, GDEV]]),
    "ens_q38_gdev_cal": dict(file="submission_ens_q38_gdev_cal.csv", members=[Q38, GDEV], weights=[1, 1], cal=True),
}
VOTE = dict(name="majority_vote", file="submission_majority_vote.csv", voters=["ens_q38_gemma4", "clean4_qg_hcx", "ens_q38_gdev_cal"])
EXPECTED_VAL = {"clean4_qg_hcx": 97.7}                 # 기록된 검증 정확도 (재현 확인용)
PUBLIC = {"clean4_qg_hcx": 0.97646, "majority_vote": 0.97676}   # 기존 제출의 공개 점수
N_PUBLIC = 3357                                        # 공개 채점 문항 수

# ── 전체 데이터 재학습 멤버 ──
FULL_SUFFIX = "_full"      # e8_full_retrain.ipynb 기본값: 원래 멤버 이름 + _full
FULL_MAP = {}              # 다른 이름으로 만들었으면 직접: {"q38_27b_r16_native": "q38_full_v2"}
NOISE_EXTRA = ["q3vl_32b_r16_native_full"]   # 7번 셀(라벨 노이즈 후보)에만 쓰는 추가 전체 데이터 실행 (없으면 건너뜀)
# ── clean4에 넣어 볼 멤버 (5-1번 셀에서 검증으로 판단 → 통과하면 6번 셀이 그 멤버를 넣은 후보도 만듦) ──
ADD_MEMBERS = [("q3vl_32b_r16_native", PRESET)]
ADD_WEIGHT, ADD_NET_MIN = 1, 3                   # 가중치 1로 넣어 검증 순이득 ≥ +3이고 바뀐 답 ≥ 10이면 추가 후보 (결과를 보기 전에 고정)
AGREE_FLOOR = 90.0         # 원래 멤버와의 test 답 일치율 하한. 실제 기준은 "원래 멤버가 가장 다른 멤버와 일치하는 비율"(6번 셀에서 계산)
'''

DATA = r'''
import os, re, json, math, hashlib, subprocess
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
_need = ["train.csv", "test.csv", "sample_submission.csv"]
def _have(): return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _need)
if IN_COLAB and not _have():
    print("압축에서 CSV 3개만 꺼냅니다...")
    subprocess.run(["7z", "e", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP] + _need + ["-r"], capture_output=True, text=True)
    if not _have():
        print("전체 압축 해제로 다시 시도합니다 (몇 분 걸립니다)")
        r = subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
assert _have(), f"{_need}가 {DATA_DIR}에 없습니다 → DATA_DIR / 압축 경로를 확인하세요"
assert os.path.isdir(OUT_DIR), f"{OUT_DIR}가 없습니다 → OUT_DIR을 확인하세요"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
train_df["answer"] = train_df["answer"].astype(str).str.strip().str.lower()
assert list(test_df.id) == list(sub_df.id), "test.csv와 sample_submission.csv의 id 순서가 다릅니다"
LETTERS = "abcd"
def make_split(df, val_size=VAL_SIZE, seed=SEED):      # stage2 노트북과 같은 함수
    rng = np.random.RandomState(seed); val_idx = []
    for letter in "abcd":
        idx = np.where(df.answer.values == letter)[0]; rng.shuffle(idx); val_idx.extend(idx[: val_size // 4].tolist())
    val_idx = sorted(val_idx); mask = np.zeros(len(df), dtype=bool); mask[val_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
tr_df, val_df = make_split(train_df)
assert hashlib.md5(",".join(val_df.id.tolist()).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2", "검증 분할이 팀 공통 분할과 다릅니다"
VAL_IDS, TEST_IDS = val_df.id.tolist(), test_df.id.tolist()
Y_VAL = val_df.answer.map(LETTERS.index).to_numpy()
ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)
print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")
'''

HELPERS = r'''
_CACHE, _TCACHE = {}, {}
def _probs(d): return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
def _load_preds(path, ids):
    d = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in ids if i not in d.index]
    assert not miss, f"{os.path.basename(path)}에 없는 문항 {len(miss)}개 (예: {miss[:3]})"
    return d.loc[ids]
def read_log(run):
    p = os.path.join(OUT_DIR, run, "train_log.json")
    try: return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    except Exception: return {}
def member_file(run, preset, split):
    # 검증: preds_final_ep{N}__{preset}__1000.csv / test: preds_final_ep{N}_test__{preset}__6714.csv (epoch는 E7과 같은 규칙으로 고름)
    d = os.path.join(OUT_DIR, run)
    if not os.path.isdir(d): return None, "폴더 없음"
    rv = re.compile(rf"^preds_final_ep(\d+)__{re.escape(preset)}__1000\.csv$")
    rt = re.compile(rf"^preds_final_ep(\d+)_test__{re.escape(preset)}__(\d+)\.csv$")
    cv, ct = {}, {}
    for f in os.listdir(d):
        m = rv.match(f)
        if m: cv[int(m.group(1))] = f; continue
        m = rt.match(f)
        if m and int(m.group(2)) == len(TEST_IDS): ct[int(m.group(1))] = f
    base = cv or ct
    if not base: return None, "예측 파일 없음"
    ea = {int(k): v for k, v in (read_log(run).get("epoch_acc") or {}).items()}
    ep = max(ea, key=lambda e: (ea[e], e)) if ea else None
    ep = ep if ep in base else max(base)
    pool = cv if split == "val" else ct
    if ep not in pool: return None, f"ep{ep} {split} 파일 없음"
    return os.path.join(d, pool[ep]), f"ep{ep}"
def load(run, preset, split):
    key = (run, preset, split)
    if key not in _CACHE:
        f, why = member_file(run, preset, split)
        if f is None: raise FileNotFoundError(f"{run}@{preset} {split}: {why}")
        d = _load_preds(f, VAL_IDS if split == "val" else TEST_IDS)
        _CACHE[key] = (_probs(d), f, float(d["letter_mass"].mean()) if "letter_mass" in d.columns else float("nan"))
    return _CACHE[key][0]
def file_answers(fname):
    p = os.path.join(ENS_DIR, fname)
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
T_GRIDS = {"T70": np.exp(np.linspace(np.log(0.25), np.log(8.0), 70)),   # 온도 격자 두 가지 (Stage 3가 어느 쪽이었든 찾도록)
           "T28": np.linspace(0.3, 3.0, 28)}
def temp_scale(p, T):
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def fit_T(run, preset, grid):
    # 멤버 하나의 온도: 원래(검증 제외) 멤버의 검증 예측으로 NLL 최소
    key = (run, preset, grid)
    if key not in _TCACHE:
        p, i = load(run, preset, "val"), np.arange(len(Y_VAL))
        nll = [-np.mean(np.log(temp_scale(p, T)[i, Y_VAL])) for T in T_GRIDS[grid]]
        _TCACHE[key] = float(T_GRIDS[grid][int(np.argmin(nll))])
    return _TCACHE[key]
def ens_scores(members, weights, how="prob", cal=None, split="test", swap=None, mix=False):
    # how: "prob" 확률 가중 평균 / "log" 로그확률 가중 평균. swap: {원래 멤버: 전체 데이터 멤버} → test 확률만 교체(mix면 둘의 평균)
    w = np.asarray(weights, dtype=float); assert len(w) == len(members), "members와 weights 개수가 다릅니다"
    w = w / w.sum(); out = 0.0
    for wi, (run, pre) in zip(w, members):
        P = load(run, pre, split)
        if swap and run in swap and split == "test":
            Pf = load(swap[run], pre, "test"); P = (P + Pf) / 2 if mix else Pf
        if cal: P = temp_scale(P, fit_T(run, pre, cal))
        out = out + wi * (P if how == "prob" else np.log(np.clip(P, 1e-12, 1.0)))
    return out
def vote_of(arrs):
    A = np.stack(arrs, 1); cnt = np.stack([(A == k).sum(1) for k in range(4)], 1)
    return cnt.argmax(1)             # 동률(셋 다 다름)이면 알파벳 앞 글자 — pandas mode(axis=1)[0]와 같음
def nll(P, y): return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))
def paired(a_old, a_new, y):
    c0, c1 = a_old == y, a_new == y
    f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum()); return f, b, f - b, int((a_old != a_new).sum())
CFG_DEFAULTS = {"TRAIN_SEED": 42, "MAX_STEPS": None, "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None,
                "AUX_GEN_CSV": None, "AUX_GEN_EVERY": 5, "AUX_GEN_WEIGHT": 1.0}     # 예전 train_log에 없는 항목 = 이 값 (stage2 이어하기 규칙과 같음)
def cfg_val(c, k):
    v = c[k] if k in c else CFG_DEFAULTS.get(k)
    if v is None: v = CFG_DEFAULTS.get(k)
    return None if (isinstance(v, (list, tuple)) and not v) else v
def rname(members, how, cal):
    return "+".join(r.split("_")[0] + ("dev" if r.endswith("_dev") else "") for r, _ in members) + f" | {'확률' if how == 'prob' else '로그확률'} 평균" + (f" | 온도 {cal}" if cal else "")
def score_of(count): return math.floor(count / N_PUBLIC * 1e5) / 1e5          # 리더보드 표시(소수 5자리 버림)
def count_of(score): return int(round(score * N_PUBLIC))
print("✅ 함수 준비 완료")
'''

MEMTABLE = r'''
# ── A-1. 멤버 표: 학습 설정 · 검증 정확도 · 멤버 간 불일치 ──
_rows, VAL_ANS = [], {}
for run, pre in MEMBERS + [m for m in ADD_MEMBERS if m not in MEMBERS]:
    lg = read_log(run); cfg = lg.get("config") or {}
    fv, why = member_file(run, pre, "val"); ft, _ = member_file(run, pre, "test")
    r = {"멤버": run + ("  (추가 후보)" if (run, pre) not in MEMBERS else ""), "모델": str(cfg.get("MODEL_ID", "?")).split("/")[-1], "학습 해상도": cfg.get("TRAIN_PRESETS") or cfg.get("TRAIN_PRESET", "?"),
         "EPOCHS": cfg.get("EPOCHS", "?"), "사용 epoch": why if fv else "-", "LR": cfg.get("LR", "?"), "r/α": f"{cfg.get('LORA_R', '?')}/{cfg.get('LORA_ALPHA', '?')}",
         "dev 추가": "O" if cfg.get("EXTRA_TRAIN_CSV") else "", "MAX_STEPS": cfg.get("MAX_STEPS") or "", "전체데이터": "O" if cfg.get("TRAIN_ON_FULL") else ""}
    if fv:
        P = load(run, pre, "val"); VAL_ANS[run] = P.argmax(1)
        r["검증(%)"], r["NLL"] = round(float((P.argmax(1) == Y_VAL).mean() * 100), 2), round(nll(P, Y_VAL), 4)
    else:
        r["검증(%)"], r["NLL"] = "없음", ""
    r["test"] = "O" if ft else "없음"
    _rows.append(r)
MEMBER_TABLE = pd.DataFrame(_rows)
print(MEMBER_TABLE.to_string(index=False))
if len(VAL_ANS) >= 2:
    names = list(VAL_ANS); short = [n.split("_")[0] + ("dev" if n.endswith("_dev") else "") for n in names]
    D = pd.DataFrame([[int((VAL_ANS[a] != VAL_ANS[b]).sum()) for b in names] for a in names], index=short, columns=short)
    print("\n검증 1,000문항에서 두 멤버의 답이 다른 문항 수:"); print(D.to_string())
    C = np.stack([VAL_ANS[n] == Y_VAL for n in names], 1)
    print(f"멤버 중 하나라도 맞힌 문항 {int(C.any(1).sum())} | 전원 오답 {int((~C.any(1)).sum())} | 전원 정답 {int(C.all(1).sum())}")
_fulls = []
for d in sorted(os.listdir(OUT_DIR)):
    if os.path.isdir(os.path.join(OUT_DIR, d)) and (read_log(d).get("config") or {}).get("TRAIN_ON_FULL"):
        _fulls.append(d)
print("\n전체 데이터(TRAIN_ON_FULL=True)로 학습한 실행:", _fulls if _fulls else "없음")
'''

REPRO = r'''
# ── A-2. 제출 파일 재현 확인: 파일마다 자기 레시피로 다시 만들어 같은지 (파일끼리 비교하는 것이 아님) ──
def _declared(r): return (r["members"], "prob", "T70" if r["cal"] else None)
REPRO, _lines = {}, []
def _say(s): print(s); _lines.append(s)
for name, r in RECIPES.items():
    fa = file_answers(r["file"])
    if fa is None:
        _say(f"· {r['file']}: 파일 없음 → 선언 레시피로만 씀"); REPRO[name] = dict(variant=_declared(r), exact=None, ans=None); continue
    res = []
    for mem in [r["members"]] + r.get("alt_members", []):
        for how in ("prob", "log"):
            for cal in (None, "T70", "T28"):
                try: a = ens_scores(mem, r["weights"], how, cal).argmax(1)
                except FileNotFoundError as e: _say(f"  ⚠️ {e}"); continue
                res.append(((mem, how, cal), int((a == fa).sum())))
    if not res:
        _say(f"· {r['file']}: 멤버 예측 파일이 없어 확인 불가"); REPRO[name] = dict(variant=_declared(r), exact=None, ans=fa); continue
    dec = next((n for v, n in res if v == _declared(r)), None)
    exact = [v for v, n in res if n == len(TEST_IDS)]
    if dec == len(TEST_IDS):
        var, msg = _declared(r), f"✅ 재현됨 ({rname(*_declared(r))}) {dec:,}/{len(TEST_IDS):,}"
    elif exact:
        var = exact[0]
        msg = (f"✅ 재현됨 — 단, 선언과 다른 레시피: {' / '.join(rname(*v) for v in exact[:3])}{' 외' if len(exact) > 3 else ''} {len(TEST_IDS):,}/{len(TEST_IDS):,}"
               + (f" (선언 레시피는 {dec:,})" if dec is not None else ""))
    else:
        bv, bn = max(res, key=lambda x: x[1]); var = _declared(r)
        msg = f"❌ 재현 안 됨 — 가장 가까운 레시피 {rname(*bv)}도 {len(TEST_IDS) - bn}문항 다름 (선언 레시피 {len(TEST_IDS) - (dec or 0)}문항 다름)"
    REPRO[name] = dict(variant=var, exact=bool(dec == len(TEST_IDS) or exact), ans=fa)
    _say(f"· {r['file']}: {msg}")
    if name in EXPECTED_VAL:
        va = (ens_scores(var[0], r["weights"], var[1], var[2], split="val").argmax(1) == Y_VAL).mean() * 100
        _say(f"    검증 {va:.2f}% (기록 {EXPECTED_VAL[name]})" + ("" if abs(va - EXPECTED_VAL[name]) < 0.05 else " ⚠️ 기록과 다름"))
# 다수결: ① 지금 있는 파일 3개로 다시 투표 ② 레시피로 다시 만든 3개로 투표
fv = file_answers(VOTE["file"])
arr_f = [REPRO[v]["ans"] for v in VOTE["voters"]]
if fv is None:
    _say(f"· {VOTE['file']}: 파일 없음")
else:
    if all(a is not None for a in arr_f):
        n1 = int((vote_of(arr_f) == fv).sum())
        _say(f"· {VOTE['file']}: 재료 파일 3개로 다시 투표 {n1:,}/{len(TEST_IDS):,} " + ("✅" if n1 == len(TEST_IDS) else "❌ (투표 방식이 다르거나 재료 파일이 그 뒤에 바뀜)"))
    arr_r = [ens_scores(REPRO[v]["variant"][0], RECIPES[v]["weights"], REPRO[v]["variant"][1], REPRO[v]["variant"][2]).argmax(1) for v in VOTE["voters"]]
    n2 = int((vote_of(arr_r) == fv).sum())
    _say(f"  멤버 예측 → 레시피 3개 → 투표 {n2:,}/{len(TEST_IDS):,} " + ("✅ 다수결 제출은 멤버 예측 파일만으로 재현됩니다" if n2 == len(TEST_IDS) else "❌ 멤버 예측만으로는 재현되지 않습니다"))
REPRO_LINES = _lines
'''

ADDCELL = r'''
# ── A-3. 멤버 추가 검증: clean4에 ADD_MEMBERS를 가중치 ADD_WEIGHT로 넣었을 때 (검증 1,000문항, 같은 문항끼리) ──
ADDED, ADD_LINES = [], []
_c4 = RECIPES["clean4_qg_hcx"]; _m4, _h4, _k4 = REPRO["clean4_qg_hcx"]["variant"]
_A4 = ens_scores(_m4, _c4["weights"], _h4, _k4, split="val").argmax(1)
for m in ADD_MEMBERS:
    run, pre = m
    if not (member_file(run, pre, "val")[0] and member_file(run, pre, "test")[0]):
        print(f"· {run}: 검증 또는 test 예측 파일이 없어 건너뜁니다"); continue
    Pm = load(run, pre, "val"); am = Pm.argmax(1)
    _A5 = ens_scores(_m4 + [m], _c4["weights"] + [ADD_WEIGHT], _h4, _k4, split="val").argmax(1)
    f, b, n, D = paired(_A4, _A5, Y_VAL)
    ok = n >= ADD_NET_MIN and D >= 10
    dis = ", ".join(f"{r_.split('_')[0] + ('dev' if r_.endswith('_dev') else '')} {int((VAL_ANS[r_] != am).sum())}" for r_ in VAL_ANS if r_ != run)
    line = (f"· clean4 + {run} (가중치 {ADD_WEIGHT}): 단독 검증 {(am == Y_VAL).mean() * 100:.2f}% | clean4 {(_A4 == Y_VAL).mean() * 100:.2f}% → {(_A5 == Y_VAL).mean() * 100:.2f}% "
            f"| 고침 {f} / 새로 틀림 {b} / 순이득 {n:+d} (바뀐 답 {D}) → " + ("✅ 추가 후보 (6번 셀이 이 멤버를 넣은 후보도 만듦)" if ok else f"추가하지 않음 (사전 기준: 순이득 ≥ +{ADD_NET_MIN}, 바뀐 답 ≥ 10)"))
    print(line); print(f"    다른 멤버와 검증 답이 다른 문항 수: {dis}"); ADD_LINES.append(line)
    if ok: ADDED.append(m)
if not ADD_MEMBERS: print("ADD_MEMBERS가 비어 있습니다")
'''

FULLCELL = r'''
# ── B. 전체 데이터 멤버 점검 → 전체 데이터 버전 제출 후보 ──
ALL_MEM = MEMBERS + [m for m in globals().get("ADDED", []) if m not in MEMBERS]
FULL = dict(FULL_MAP)
for run, pre in ALL_MEM:
    if run not in FULL and os.path.isdir(os.path.join(OUT_DIR, run + FULL_SUFFIX)): FULL[run] = run + FULL_SUFFIX
PRE_OF = dict(ALL_MEM)
OK_FULL, FULL_ROWS, E8 = {}, [], {"lines": [], "cands": []}
TA = {run: load(run, pre, "test").argmax(1) for run, pre in ALL_MEM if member_file(run, pre, "test")[0]}
def _agree(a, b): return float((a == b).mean() * 100)
def say(s): print(s); E8["lines"].append(s)
for run, fr in FULL.items():
    pre = PRE_OF[run]
    ft, why = member_file(fr, pre, "test")
    row = {"원래 멤버": run, "전체 데이터": fr}
    if ft is None:
        row["판정"] = f"대기 — test 예측 없음({why}). 학습·추론이 끝나면 다시 실행"; FULL_ROWS.append(row); continue
    c0, c1 = read_log(run).get("config") or {}, read_log(fr).get("config") or {}
    diff = {k: (cfg_val(c0, k), cfg_val(c1, k)) for k in sorted(set(c0) | set(c1)) if k not in ("TRAIN_ON_FULL", "MAX_STEPS") and cfg_val(c0, k) != cfg_val(c1, k)}
    P0, P1 = load(run, pre, "test"), load(fr, pre, "test")
    agree = _agree(P0.argmax(1), P1.argmax(1))
    thr = max(AGREE_FLOOR, min([_agree(TA[run], TA[o]) for o in TA if o != run] or [AGREE_FLOOR]))   # 같은 레시피라면 다른 멤버보다는 더 비슷해야 함
    hv = float((load(run, pre, "val").argmax(1) == Y_VAL).mean() * 100)
    fv_, _ = member_file(fr, pre, "val")
    fva = float((load(fr, pre, "val").argmax(1) == Y_VAL).mean() * 100) if fv_ else float("nan")
    lm = _CACHE[(fr, pre, "test")][2]
    flags = []
    if not c1.get("TRAIN_ON_FULL"): flags.append("train_log에 TRAIN_ON_FULL=True가 아님")
    if diff: flags.append("설정이 원래와 다름 " + ", ".join(f"{k}: {a}→{b}" for k, (a, b) in diff.items()))
    if agree < thr: flags.append(f"원래 멤버와 test 일치율 {agree:.1f}%가 기준 {thr:.1f}%(원래 멤버와 가장 다른 멤버의 일치율)보다 낮음")
    if fv_ and fva < hv: flags.append(f"학습에 쓴 검증에서도 {fva:.1f}%로 원래 멤버({hv:.1f}%)보다 낮음")
    if lm == lm and lm < 0.9: flags.append(f"letter_mass {lm:.3f} < 0.9")
    row.update({"test 일치율(%)": round(agree, 2), "기준(%)": round(thr, 2), "test 다른 답": int((P0.argmax(1) != P1.argmax(1)).sum()),
                "검증(학습 포함, %)": round(fva, 2) if fva == fva else "없음", "원래 검증(%)": round(hv, 2),
                "판정": "✅ 사용" if not flags else "⚠️ 제외 — " + "; ".join(flags)})
    FULL_ROWS.append(row)
    if not flags: OK_FULL[run] = fr
if not FULL:
    print("전체 데이터 재학습 폴더(<멤버>_full)가 아직 없습니다 → e8_full_retrain.ipynb 학습이 끝난 뒤 이 노트북을 다시 실행하세요")
else:
    print(pd.DataFrame(FULL_ROWS).to_string(index=False))
if OK_FULL:
    os.makedirs(E8_DIR, exist_ok=True)
    say(f"\n전체 데이터로 바꾼 멤버 {len(OK_FULL)}/{len(ALL_MEM)}: " + ", ".join(OK_FULL))
    CANDS, _R = {}, {k: (v, REPRO[k]["variant"]) for k, v in RECIPES.items()}
    if globals().get("ADDED"):                       # 5-1번 셀을 통과한 멤버를 clean4에 넣은 레시피 (다수결에서는 clean4 자리를 대신함)
        _c = RECIPES["clean4_qg_hcx"]; _v = REPRO["clean4_qg_hcx"]["variant"]
        _R["clean4_plus"] = (dict(_c, members=_c["members"] + ADDED, weights=_c["weights"] + [ADD_WEIGHT] * len(ADDED)), (_v[0] + ADDED, _v[1], _v[2]))
    for name, (r, (mem, how, cal)) in _R.items():
        for tag, mix in (("full", False), ("mix", True)):
            CANDS[f"{name}_{tag}"] = ens_scores(mem, r["weights"], how, cal, "test", swap=OK_FULL, mix=mix).argmax(1)
    for tag in ("full", "mix"):
        CANDS[f"{VOTE['name']}_{tag}"] = vote_of([CANDS[f"{v}_{tag}"] for v in VOTE["voters"]])
        if "clean4_plus" in _R:
            CANDS[f"{VOTE['name']}_plus_{tag}"] = vote_of([CANDS[f"{'clean4_plus' if v == 'clean4_qg_hcx' else v}_{tag}"] for v in VOTE["voters"]])
    base_ans = {"clean4_qg_hcx": REPRO["clean4_qg_hcx"]["ans"], VOTE["name"]: file_answers(VOTE["file"])}
    say("\n제출 후보 (공개 판정선은 제출 전에 고정: 정답 수가 판정선 이하이면 고장 의심 → 채택하지 않고 원인 확인)")
    for base, nm0 in [("clean4_qg_hcx", "clean4_qg_hcx"), (VOTE["name"], VOTE["name"])] + ([("clean4_qg_hcx", "clean4_plus"), (VOTE["name"], f"{VOTE['name']}_plus")] if "clean4_plus" in _R else []):
        ref = base_ans[base]
        if ref is None: say(f"  {base}: 기존 제출 파일이 없어 비교 생략"); continue
        for tag in ("full", "mix"):
            nm = f"{nm0}_{tag}"; a = CANDS[nm]; Dn = int((a != ref).sum())
            path = os.path.join(E8_DIR, f"submission_{nm}.csv")
            sub_df[["id"]].assign(answer=[LETTERS[i] for i in a]).to_csv(path, index=False)
            b = count_of(PUBLIC[base]); k = max(3, math.ceil(math.sqrt(Dn / 2))) if Dn else 0
            line = (f"  {os.path.basename(path)} | 기존 {base}({PUBLIC[base]:.5f} = {b:,}문항)과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개)"
                    + (f" | 판정선: {score_of(b - k):.5f}({b - k:,}문항) 이하면 고장 의심" if Dn else " | 기존과 같음 → 제출 불필요"))
            say(line); E8["cands"].append({"file": os.path.basename(path), "base": base, "D": Dn, "alarm_count": b - k if Dn else None})
    say(f"  권장 제출: submission_{VOTE['name']}_full.csv, submission_clean4_qg_hcx_full.csv 두 개 (mix는 참고용 — full이 판정선 근처일 때만)"
        + ("" if "clean4_plus" not in _R else f" | 5-1번 셀 추가 멤버 후보: submission_{VOTE['name']}_plus_full.csv, submission_clean4_plus_full.csv"))
    if not all(REPRO[v]["exact"] is not False for v in VOTE["voters"]):
        say("  ⚠️ 다수결 재료 중 재현 안 되는 파일이 있어, 다수결 후보는 선언 레시피로 만들었습니다 (기존 다수결과 레시피가 다를 수 있음)")
    json.dump({"ok_full": OK_FULL, "rows": FULL_ROWS, "repro": {k: {"variant": rname(*v["variant"]), "exact": v["exact"]} for k, v in REPRO.items()},
               "cands": E8["cands"]}, open(os.path.join(E8_DIR, "e8_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

NOISE = r"""
# ── (참고) 라벨 노이즈 후보: 전체 데이터 멤버가 '학습한' 검증 문항인데도 틀린 문항 ──
# 전체 데이터 멤버의 검증 예측 = 이미 학습한 문항을 다시 푼 결과. 여기서도 틀리면 라벨이 틀렸거나, 한 번 보고는 못 배울 만큼 어려운 문항입니다.
NOISE_ROWS = []
_fr = {run: fr for run, fr in (OK_FULL or FULL).items() if member_file(fr, PRE_OF[run], "val")[0]}
for _x in NOISE_EXTRA:
    if member_file(_x, PRESET, "val")[0] and (read_log(_x).get("config") or {}).get("TRAIN_ON_FULL"): _fr[_x] = _x; PRE_OF.setdefault(_x, PRESET)
if not _fr:
    print("전체 데이터 멤버의 검증 예측 파일이 없어 건너뜁니다")
else:
    _Pin = {fr: load(fr, PRE_OF[run], "val") for run, fr in _fr.items()}
    _Win = np.stack([P.argmax(1) != Y_VAL for P in _Pin.values()], 1)
    _mem, _how, _cal = REPRO["clean4_qg_hcx"]["variant"]
    _A_out = ens_scores(_mem, RECIPES["clean4_qg_hcx"]["weights"], _how, _cal, split="val").argmax(1)     # 원래 멤버 앙상블 (이 문항들을 학습하지 않음)
    for fr, w in zip(_Pin, _Win.T): print(f"  {fr}: 학습한 검증 문항 중 틀린 것 {int(w.sum())}개")
    _cand = np.where(_Win.any(1))[0]
    for i in _cand[np.argsort(-_Win[_cand].sum(1), kind="stable")]:
        r = val_df.iloc[i]; gold = LETTERS[Y_VAL[i]]
        NOISE_ROWS.append({"id": r.id, "전체멤버 오답 수": f"{int(_Win[i].sum())}/{len(_Pin)}",
                           "전체멤버 답(확률)": " ".join(f"{LETTERS[int(P[i].argmax())]}({P[i].max():.2f})" for P in _Pin.values()),
                           "정답": f"{gold}: {str(r[gold])[:24]}", "원래 앙상블": LETTERS[int(_A_out[i])] + (" ✗" if _A_out[i] != Y_VAL[i] else " ✓"),
                           "질문": str(r.question)[:48]})
    if NOISE_ROWS:
        print(pd.DataFrame(NOISE_ROWS).to_string(index=False))
    _all = int(_Win.all(1).sum()); _out = int((_A_out != Y_VAL).sum())
    print(f"\n전체 데이터 멤버가 모두 틀린 문항 {_all}개 | 원래 앙상블 오답 {_out}개 중 전체 데이터 멤버가 하나라도 여전히 틀린 문항 {int(((_A_out != Y_VAL) & _Win.any(1)).sum())}개")
    print("이미지 확인: 각 _full 폴더의 error_report_final_ep*_up2.0.html (학습에 쓴 검증 문항 중 틀린 것을 이미지와 함께 보여 줌)")
    print("→ 위 문항을 이미지로 보고, 라벨이 명백히 틀린 문항이 몇 개인지 알려 주세요")
"""

SUMMARY = r'''
print("E8 요약 | 멤버", len(MEMBERS), "| 재현 확인:")
for l in REPRO_LINES: print(l)
for l in globals().get("ADD_LINES", []): print(l)
if "FULL_ROWS" in globals() and FULL_ROWS:
    print("전체 데이터 멤버:")
    for r in FULL_ROWS: print(f"  {r['전체 데이터']}: {r['판정']}" + (f" | test 일치 {r['test 일치율(%)']}%, 다른 답 {r['test 다른 답']}" if "test 일치율(%)" in r else ""))
for l in E8["lines"]: print(l)
'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. Drive 연결 → CSV 준비 → 고정 검증 분할"), ("code", DATA),
    ("markdown", "## 3. 함수 (멤버 예측 파일 찾기 · 앙상블 · 온도 보정 · 다수결)"), ("code", HELPERS),
    ("markdown", "## 4. 멤버 표 — 학습 설정 · 검증 정확도 · 멤버 간 불일치\n`e8_full_retrain.ipynb`의 3-1번 셀이 가져올 설정과 같은지 확인하는 용도입니다."), ("code", MEMTABLE),
    ("markdown", "## 5. 제출 파일 재현 확인\n파일마다 **자기 레시피**로 다시 만들어 같은지 봅니다. 선언 레시피(확률 평균, `_cal`은 온도 보정)로 안 맞으면 다른 레시피 후보를 모두 시험해 어느 것으로 똑같이 나오는지 알려 줍니다."), ("code", REPRO),
    ("markdown", "## 5-1. 멤버 추가 검증 (q3vl_32b 등)\n원래(검증 제외) 멤버로 clean4에 넣었을 때 검증에서 이득이 있는지 봅니다. 기준을 통과한 멤버만 6번 셀에서 전체 데이터 버전으로 넣습니다."), ("code", ADDCELL),
    ("markdown", "## 6. 전체 데이터 멤버 점검 → 제출 후보\n`<멤버>_full` 폴더를 찾아 이상 징후를 점검하고, 통과한 멤버만 바꿔 넣습니다. 아직 학습 중인 멤버는 원래 멤버를 그대로 씁니다."), ("code", FULLCELL),
    ("markdown", "## 7. (참고) 라벨 노이즈 후보\n전체 데이터 멤버는 검증셋도 학습했으므로, 검증 문항을 다시 풀게 하면 '학습한 문항을 또 틀리는지' 볼 수 있습니다. 여기서 틀린 문항이 라벨 오류인지, 단지 어려운 문항인지 이미지로 확인하는 용도입니다."), ("code", NOISE),
    ("markdown", "## 8. 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    c = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]; nb.cells.append(c)
nbformat.validate(nb); nbformat.write(nb, OUT)
print("저장:", OUT, "| 셀", len(nb.cells))
