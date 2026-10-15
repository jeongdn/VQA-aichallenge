# -*- coding: utf-8 -*-
"""조합 검색 노트북 빌더 (CPU) — _full 실행 없이, 나머지 실행들로 좋은 조합을 찾음. 제출 파일(test 답)에만 _full을 섞음 (FULL_MODE none/mix/swap).
- 멤버 탐색·데이터·예측 파일 읽기: E8/E14 코드 그대로 (manual_ensemble과 같은 멤버)
- _full 제외: 이름 조각 'full' / train_log TRAIN_ON_FULL / 검증 > SUSPECT_ACC
- 검색: 전수(같은 가중치 부분집합) · 그리디(Caruana, 가중치 1씩) · 좌표 상승(기준에서) × 결합 설정(확률/확률+온도/로그확률)
- 중첩 CV(5겹×4번)로 '검색 방법 자체'의 기대 정확도를 잼 — 전수 검색은 한 번 훑을 때 fold별 최고 조합까지 같이 구함
- 후보 표 → 슬라이더 패널(후보 불러오기) → 저장(CSV+JSON) → 재현 점검 → 제출 파일 비교
사용: python3 build_ens_search.py [출력 경로]"""
import sys, json, uuid, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/ens_search/out/ens_search.ipynb"
SP = "/tmp/vqa_scratch"
_n14 = {}
_argv = sys.argv; sys.argv = ["build_e14.py", f"{SP}/_e14_tmp_s.ipynb"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read(), _n14)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

DATA = _n14["_n8"]["DATA"]
DATA = rep(DATA, 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, SAVE_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, SAVE_SUBDIR); os.makedirs(SAVE_DIR, exist_ok=True)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 저장 폴더: {SAVE_DIR}")')
HELPERS = _n14["_n8"]["HELPERS"]
POOL = _n14["POOL"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# 조합 검색 (CPU) — 검색은 `_full` 없이, 제출 파일에는 `_full`을 섞어서

**하는 일**
- manual_ensemble과 같은 규칙으로 멤버를 찾은 뒤 `_full` 실행을 모두 뺍니다.
  - 이름에 `_full`이 들어간 실행과 train_log에 `TRAIN_ON_FULL`이 있는 실행을 뺍니다.
  - 검증이 99%를 넘는 실행도 검증을 학습에 쓴 것으로 보고 뺍니다.
- 남은 실행들로 좋은 조합을 찾습니다. 검색 방법 3가지를 결합 설정 3가지(확률 평균 / 확률+멤버별 온도 / 로그확률 평균)마다 돌립니다.
  1. **전수 검색**: 실행 부분집합을 같은 가중치로 모두 계산합니다(조합 수 상한 안에서). 1위, NLL 1위, 상위 100개의 합의 조합을 후보로 올립니다.
  2. **그리디**(Caruana 방식): 가중치를 1씩 더하며, 가장 좋아지는 실행을 계속 추가합니다.
  3. **좌표 상승**: 기준 조합에서 시작해 실행 하나씩 가중치(0~3)를 바꿔 가며 올라갑니다.
- **중첩 CV로 검색 자체를 채점합니다.**
  - 800문항으로 검색해 고른 조합을 남은 200문항으로 채점합니다. 이것을 5겹 × 4번 반복합니다.
  - 검색 1위의 검증 점수는 많은 조합 중 1,000문항에서 가장 운 좋은 것을 고른 값이라 부풀려져 있습니다.
  - 중첩 CV 값이 그 방법으로 고른 조합의 실제 기대 정확도입니다.
- 찾은 후보는 슬라이더 패널에서 불러와 직접 다듬을 수 있습니다.
  - 저장하면 `vqa_runs/ens_search/`에 제출 CSV와 설정 JSON이 남습니다. JSON만 있으면 똑같이 다시 만들 수 있습니다.
- **제출 파일(test 답)에는 `_full`을 섞습니다** (`FULL_MODE`, 기본 `"mix"`).
  - 조합에 든 멤버에 `<이름>_full` 짝이 있으면, test 확률을 원래 실행과 `_full`의 평균으로 바꿉니다(`"swap"`이면 `_full`로 교체).
  - 이름이 규칙과 다른 짝은 1번 셀 `FULL_PAIRS`에 적습니다. 짝이 없는 멤버는 원래 실행 그대로 들어갑니다.
  - 검색·검증 수치는 항상 원래 실행으로 계산합니다. `_full`은 검증 문항으로 학습해서 검증으로는 잴 수 없기 때문입니다.
  - 짝은 test 답이 원래 실행과 97% 이상 같고, train_log에 `TRAIN_ON_FULL`이 있을 때만 씁니다(학습 고장·엉뚱한 짝 방지).

**실행**: 위에서부터 차례로 실행합니다. 전수 검색은 실행 수에 따라 몇 초에서 몇 분 걸립니다(6번 셀이 조합 수를 먼저 알려 줌). GPU가 필요 없습니다.
""")

md("## 1. 설정")
code(r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"          # 기존 제출 파일 폴더 (읽기만 함)
SAVE_SUBDIR     = "ens_search"         # 이 노트북이 제출 파일·설정·검색 결과를 쓰는 폴더
COMPARE_SUBDIRS = ["manual_ens"]       # 14번 비교에 같이 넣을 폴더 (manual_ensemble 저장본)
SEED, VAL_SIZE = 42, 1000              # 검증 분할 — 팀 공통값
N_PUBLIC = 3357                        # 공개 채점 문항 수 (test 6,714의 절반)

# ── 멤버 자동 탐색 (E14와 같은 규칙) + _full 제외 ──
MUST_INCLUDE = []                      # 반드시 있어야 하는 멤버 [(실행, 해상도)] — 없으면 멈춤. 보통 비워 둠
PRESET_PREF = ["up2.0"]                # 실행마다 이 해상도 예측을 우선 사용
INCLUDE = []                           # 비우면 자동 탐색. 넣으면 이 목록만: [("q35_27b_r16_native", "up2.0"), ...]
EXCLUDE = []                           # 자동 탐색에서 뺄 실행 이름
MIN_VAL_ACC = 90.0                     # 이보다 낮은 실행은 고장으로 보고 뺌
T_GRID = "T70"                         # 멤버별 온도 격자 (0.25~8, 검증 NLL 최소)
FULL_TOKEN = "full"                    # 이름을 '_'로 나눈 조각에 이게 있으면 뺌 (q35_27b_r16_native_full, x_full_bf16 …)
SUSPECT_ACC = 99.0                     # 검증이 이보다 높으면 검증을 학습에 쓴 실행으로 보고 뺌
POOL_DELTA = 1.5                       # 패널 '강한 풀' 버튼: 상위 3개 평균보다 1.5%p 넘게 낮지 않은 실행

# ── 제출 파일에 _full 섞기 (검색·검증은 항상 _full 없이) ──
FULL_MODE = "mix"                      # test 답을 만들 때: "none" 원래 실행만 / "mix" 원래와 _full 확률 평균 / "swap" _full로 교체
FULL_SUFFIX = "_full"                  # 짝 찾기 규칙: <멤버 이름>_full
FULL_PAIRS = {                         # 이름 규칙 밖의 짝 {멤버: _full 실행}. 목록에 없는 멤버는 그냥 건너뜀
    "q3vl_32b_old_up15": "q3vl_32b_r16_native_full",               # 옮겨 온 Qwen3-VL-32B ↔ E8 _full
    # "gemma4_31b_r16_up20_bf16": "gemma4_31b_r16_up20_full",      # bf16 학습 멤버에 4bit _full을 섞으려면 주석을 푸세요 (학습 설정이 달라 기본은 안 섞음)
    # "q35_27b_r16_native_bf16": "q35_27b_r16_native_full",
}
FULL_MIN_AGREE = 97.0                  # _full의 test 답이 원래 실행과 이 비율(%) 이상 같아야 씀

# ── 검색 ──
SEARCH_SETTINGS = [("prob", "none"), ("prob", "val"), ("logprob", "none")]   # (결합 방법, 온도): 확률 / 확률+멤버별 온도 / 로그확률
SEARCH_EXCLUDE = []                    # 검색에서만 뺄 실행 (패널에는 남음)
SEARCH_ONLY = []                       # 비우면 전체. 넣으면 이 실행들로만 검색
SEARCH_INCLUDE_VALONLY = ["q38flash_r16_native_bf16"]   # test가 아직 없는(검증만 있는) 실행도 검색에 넣기: 실행 이름 목록 / True = 전부 / False = 안 넣음
                                       #   그런 실행이 든 조합은 저장 불가 → test 추론이 끝나 Drive에 오면 자동으로 보통 멤버가 됨 (목록은 그대로 둬도 됨)
MAX_SUBSETS = 1_000_000                # 전수 검색 조합 수 상한 → 넘으면 조합 크기를 줄임 (실행 20개 = 약 100만, 설정마다 1분 안팎)
MAX_SIZE = None                        # 조합 최대 크기 (None = 상한 안에서 최대)
GREEDY_STEPS = 25                      # 그리디: 가중치 1씩 최대 이만큼 더함
WEIGHT_GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3]   # 좌표 상승 가중치 후보
CV_K, CV_REPEATS, CV_SEED = 5, 4, 0    # 중첩 CV: 5겹 × 4번
TOP_SHOW = 15                          # 전수 검색 상위 몇 개를 보여 줄지
CAND_TOP = 5                           # 설정마다 전수 상위 몇 개를 후보로 올릴지
CONS_TOP = 100                         # '전수 합의': 상위 100개 조합에 가장 자주 든 실행을 그 조합들의 크기(중앙값)만큼 모아 같은 가중치로

# ── 비교 기준 ──
REF_NAME = "clean4_qg_hcx"             # 지금 제출 레시피 (검증 97.7%, 공개 0.97646). 폴더가 없는 실행은 자동으로 빠짐
REF_CFG = {"weights": {"q35_27b_r16_native": 1, "q38_27b_r16_native": 1, "gemma4_31b_r16_up20_dev": 1,
                       "gemma4_31b_r16_up20": 1, "hcx32b_r16_native": 2},
           "method": "prob", "temp": "none", "full": "none"}
REF_SUBMISSIONS = {"majority_vote (공개 0.97676)": "ensembles/submission_majority_vote.csv",     # test 답을 비교할 기존 제출 (OUT_DIR 기준)
                   "clean4_qg_hcx (공개 0.97646)": "ensembles/submission_clean4_qg_hcx.csv"}
''')

md("## 2. 데이터 (CSV만 꺼냄) · 고정 검증 분할")
code(DATA)

md("## 3. 예측 파일 읽기 함수 (E8·E14와 같음)")
code(HELPERS)

md("## 4. 멤버 자동 탐색 (E14와 같은 규칙)")
code(POOL)

md("""
## 5. 멤버 표 (`_full` 제외) · 조합 함수
- **온도 T**: 검증 NLL을 최소로 하는 값입니다. 1보다 크면 과신, 작으면 소심한 멤버입니다.
- **혼자 맞힘**: 다른 어떤 실행도 못 맞혔는데 이 실행만 맞힌 검증 문항 수입니다.
- test가 아직 없는 실행은 패널에서 검증 탐색만 할 수 있습니다. 검색과 저장에서는 빠집니다.
- **`_full` 짝**: 표의 `_full` 칸에 짝 이름과 test 답 일치율이 나옵니다. 짝이 있는 멤버만 제출 파일에서 `_full`이 섞입니다.
""")
code(r'''
import datetime, itertools, time
TG = T_GRIDS[T_GRID]
def fit_T_P(P):
    ii = np.arange(len(Y_VAL)); return float(TG[int(np.argmin([-np.mean(np.log(temp_scale(P, T)[ii, Y_VAL])) for T in TG]))])
def is_full(r): return FULL_TOKEN in str(r).split("_")
def _acc_of(P): return float((P.argmax(1) == Y_VAL).mean() * 100)
_vo = {r: v for r, v in VALONLY.items() if v is not None and r not in NAMES}
_pv0 = {**{r: PV[i] for i, r in enumerate(NAMES)}, **{r: v[1] for r, v in _vo.items()}}
_full_out = sorted(r for r in _pv0 if is_full(r))
_suspect = sorted(r for r in _pv0 if not is_full(r) and _acc_of(_pv0[r]) > SUSPECT_ACC)
_vo_low = sorted(r for r in _vo if _acc_of(_vo[r][1]) < MIN_VAL_ACC)
MEMBERS = [r for r in NAMES if r not in _full_out and r not in _suspect]                       # test까지 있는 멤버 = 검색·저장 대상
RUNS = MEMBERS + sorted(r for r in _vo if r not in _full_out and r not in _suspect and r not in _vo_low)   # + 검증만 있는 실행 (패널 탐색용)
if _full_out: print(f"_full 실행 제외 (이름에 '_{FULL_TOKEN}'): {_full_out}")
if _suspect: print(f"⚠️ 검증 > {SUSPECT_ACC}% → 검증을 학습에 쓴 실행으로 보고 제외: {_suspect}")
if len(RUNS) > len(MEMBERS):
    _vo_names = RUNS[len(MEMBERS):]
    _vo_in = _vo_names if SEARCH_INCLUDE_VALONLY is True else [r for r in _vo_names if r in (SEARCH_INCLUDE_VALONLY or [])]
    print(f"검증만 있는 실행 {len(_vo_names)}개 (test 없음 → 패널에 넣고 저장은 불가): {_vo_names}"
          + (f"\n  → 이 중 {_vo_in}은 SEARCH_INCLUDE_VALONLY에 있어 검색에도 넣음 (6번 셀)" if _vo_in else "\n  → 검색에서는 뺌 (넣으려면 1번 셀 SEARCH_INCLUDE_VALONLY에 이름을 적으세요)"))
if _vo_low: print(f"검증만 있는 실행 중 검증 < {MIN_VAL_ACC}%라서 뺀 것: {_vo_low}")
assert len(MEMBERS) >= 2, "test까지 있는 멤버가 2개 미만입니다 → OUT_DIR와 4번 셀 출력을 확인하세요"
PVAL = {r: _pv0[r] for r in RUNS}
PTEST = {r: PT[NAMES.index(r)] for r in MEMBERS}
PRE = dict(MEMBERS_ALL); PRE.update({r: _vo[r][0] for r in RUNS if r not in PRE})
FAMR = dict(zip(NAMES, FAM)); FAMR.update({r: model_of(r) for r in RUNS if r not in FAMR})
TT = {r: fit_T_P(PVAL[r]) for r in RUNS}
ACC = {r: _acc_of(PVAL[r]) for r in RUNS}
CORRECT = {r: PVAL[r].argmax(1) == Y_VAL for r in RUNS}

# ── _full 짝 (제출 파일에만 씀) ──
FULL_LABEL = {"none": "test = 원래 실행", "mix": "test = 원래 + _full 평균", "swap": "test = _full로 교체"}
assert FULL_MODE in FULL_LABEL, f"FULL_MODE는 none/mix/swap 중 하나: {FULL_MODE}"
FULL_OF, FULL_FILE, FULL_INFO = {}, {}, {}
_pairs = {r: r + FULL_SUFFIX for r in MEMBERS}
for _k, _v in FULL_PAIRS.items():
    if _k in MEMBERS: _pairs[_k] = _v
for r, fr in _pairs.items():
    dfr = os.path.join(OUT_DIR, fr)
    if not os.path.isdir(dfr): continue
    _tp = sorted({m.group(2) for f in os.listdir(dfr) if (m := re.match(r"^preds_final_ep(\d+)_test__(.+)__(\d+)\.csv$", f)) and int(m.group(3)) == len(TEST_IDS)})
    _order = [PRE[r]] + [p for p in PRESET_PREF if p != PRE[r]] + [p for p in _tp if p != PRE[r] and p not in PRESET_PREF]
    pre_f = next((p for p in _order if p in _tp), None)
    if pre_f is None: print(f"  _full {r} → {fr}: test 예측 없음 → 안 씀"); continue
    try: Pf = load(fr, pre_f, "test")
    except Exception as ex: print(f"  _full {r} → {fr}@{pre_f}: 읽기 실패 ({str(ex)[:60]}) → 안 씀"); continue
    agree = float((Pf.argmax(1) == PTEST[r].argmax(1)).mean() * 100)
    flag = bool((read_log(fr).get("config") or {}).get("TRAIN_ON_FULL"))
    ok = agree >= FULL_MIN_AGREE and flag
    why = "" if ok else (f"일치율 < {FULL_MIN_AGREE}%" if agree < FULL_MIN_AGREE else "train_log에 TRAIN_ON_FULL 없음")
    print(f"  _full {r} → {fr}@{pre_f}" + (f" (해상도 다름: 원래 {PRE[r]})" if pre_f != PRE[r] else "") + f": test 답 일치 {agree:.1f}% " + ("✅ 섞을 수 있음" if ok else f"⚠️ 안 씀 ({why})"))
    if ok: FULL_OF[r], FULL_FILE[r], FULL_INFO[r] = Pf, _CACHE[(fr, pre_f, "test")][1], f"{fr}@{pre_f} {agree:.1f}%"
print(f"제출 파일의 _full 섞기: {FULL_LABEL[FULL_MODE]} (FULL_MODE='{FULL_MODE}') · 짝이 있는 멤버 {len(FULL_OF)}개" + (f": {sorted(FULL_OF)}" if FULL_OF else ""))

_rows = []
for r in sorted(RUNS, key=lambda r: (short_fam(FAMR[r]), -ACC[r])):
    others = np.sum([CORRECT[q] for q in RUNS if q != r], 0)
    _rows.append({"실행": r, "계열": short_fam(FAMR[r]), "검증(%)": round(ACC[r], 2), "오답": int((~CORRECT[r]).sum()), "NLL": round(nll(PVAL[r], Y_VAL), 4),
                  "온도 T": round(TT[r], 2), "혼자 맞힘": int((CORRECT[r] & (others == 0)).sum()), "test": "O" if r in PTEST else "없음(검증만)",
                  "_full": FULL_INFO.get(r, "")})
MEM_TABLE = pd.DataFrame(_rows)
print(MEM_TABLE.to_string(index=False))
_acc_sorted = sorted(ACC[r] for r in MEMBERS); POOL_REF = float(np.mean(_acc_sorted[-3:]))
STRONG = [r for r in MEMBERS if ACC[r] >= POOL_REF - POOL_DELTA - 1e-9]
_anyc = np.any([CORRECT[r] for r in MEMBERS], 0)
print(f"\n멤버 {len(MEMBERS)}개 · 계열 {len(set(short_fam(FAMR[r]) for r in MEMBERS))}개 | 어떤 멤버라도 맞힌 검증 문항 {int(_anyc.sum())} = 조합으로 가능한 상한 | 전원 오답 {int((~_anyc).sum())}")
_miss_ref = [r for r in REF_CFG["weights"] if r not in PVAL]
if _miss_ref: print(f"⚠️ 기준 조합 {REF_NAME}의 실행 중 찾지 못한 것: {_miss_ref} → 빼고 비교합니다")
REF_CFG = {**REF_CFG, "weights": {r: w for r, w in REF_CFG["weights"].items() if r in PVAL}, "full": "none"}

# ── 조합 함수 ──
METHOD_LABEL = {"prob": "확률 가중 평균", "logprob": "로그확률 가중 평균 (기하 평균)", "vote": "가중 다수결 (동률은 확률 평균)"}
TEMP_LABEL = {"none": "온도 보정 없음", "val": "멤버별 온도 보정 (검증으로 맞춘 T)"}
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}          # E14 조합의 검증 오답 24개 (분류는 사람이 본 결과)
E14_ERR = {k: [f"train_{x}.jpg" for x in v.split()] for k, v in E14_ERR.items()}
_VPOS = {i: k for k, i in enumerate(VAL_IDS)}

def norm_cfg(cfg, warn=True):
    w = {}
    for r, v in (cfg.get("weights") or {}).items():
        if r not in PVAL:
            if warn: print(f"  ⚠️ '{r}'는 멤버 목록에 없어 무시합니다 (이름 오타·_full·검증 예측 없음·EXCLUDE)")
            continue
        if float(v) > 0: w[r] = float(v)
    c = {"weights": w, "method": cfg.get("method", "prob"), "temp": cfg.get("temp", "none"), "full": cfg.get("full") or "none"}
    assert c["method"] in METHOD_LABEL and c["temp"] in TEMP_LABEL and c["full"] in FULL_LABEL, c
    return c

def member_P(r, split, c):
    if split == "val": P = PVAL[r]                                                 # 검증은 항상 원래 실행
    else:
        P = PTEST[r]
        if c.get("full", "none") != "none" and r in FULL_OF: P = (P + FULL_OF[r]) / 2 if c["full"] == "mix" else FULL_OF[r]
    return temp_scale(P, TT[r]) if c.get("temp") == "val" else P

def _combine(Ps, w, method):       # 멤버 확률 목록 + 가중치 → 조합 점수 (행마다 argmax가 답)
    w = np.asarray(w, dtype=float); w = w / w.sum()
    if method == "prob": return sum(wi * p for wi, p in zip(w, Ps))
    if method == "logprob":
        s = sum(wi * np.log(np.clip(p, 1e-6, 1.0)) for wi, p in zip(w, Ps)); s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    vote = sum(wi * np.eye(4)[p.argmax(1)] for wi, p in zip(w, Ps)); pavg = sum(wi * p for wi, p in zip(w, Ps))
    return vote + 1e-3 * pavg

def ens(cfg, split="val"):
    c = norm_cfg(cfg, warn=False)
    if not c["weights"]: return None
    if split == "test":
        miss = [r for r in c["weights"] if r not in PTEST]
        if miss: raise ValueError(f"test 예측이 없는 실행이 가중치에 있습니다: {miss}")
    runs = list(c["weights"])
    return _combine([member_P(r, split, c) for r in runs], [c["weights"][r] for r in runs], c["method"])

def evaluate(cfg):
    S = ens(cfg, "val")
    if S is None: return None
    a = S.argmax(1); ok = a == Y_VAL; P = S / S.sum(1, keepdims=True)
    return {"acc": float(ok.mean() * 100), "n_err": int((~ok).sum()), "ans": a, "ok": ok,
            "nll": None if norm_cfg(cfg, warn=False)["method"] == "vote" else nll(P, Y_VAL)}

def sign_p(k, n):                  # 양쪽 부호 검정 (고침 k개 · 새로 틀림 n-k개)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0
def boot_ci(ok_ref, ok_new, B=2000, seed=0):   # 같은 문항 쌍으로 뽑는 부트스트랩: 정확도 차이(%p)의 95% 구간
    d = ok_new.astype(float) - ok_ref.astype(float); idx = np.random.RandomState(seed).randint(0, len(d), (B, len(d)))
    m = d[idx].mean(1) * 100; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def ablation(cfg, step=0.5):
    # 멤버 하나만 바꿔 보기: 들어 있는 실행은 빼기·+step·−step, 안 든 실행은 넣기(가중치 1). 순이득 = 고침 − 새로 틀림 (지금 조합 대비)
    c = norm_cfg(cfg, warn=False); base = evaluate(c); rows = []
    for r in sorted(RUNS, key=lambda r: (short_fam(FAMR[r]), -ACC[r])):
        w0 = c["weights"].get(r, 0.0); tries = []
        if w0 > 0:
            if len(c["weights"]) > 1: tries.append((f"빼기 ({w0:g}→0)", 0.0))
            tries.append((f"올리기 ({w0:g}→{w0 + step:g})", w0 + step))
            if w0 - step > 0: tries.append((f"내리기 ({w0:g}→{w0 - step:g})", w0 - step))
        else: tries.append(("넣기 (0→1)" + ("" if r in PTEST else " · test 없음"), 1.0))
        for what, wn in tries:
            w = dict(c["weights"]); w[r] = wn
            e = evaluate({**c, "weights": {k: v for k, v in w.items() if v > 0}}); f, b, n, D = paired(base["ans"], e["ans"], Y_VAL)
            rows.append({"실행": r, "바꾸면": what, "검증(%)": round(e["acc"], 2), "고침": f, "새로 틀림": b, "순이득": n, "바뀐 답": D})
    return pd.DataFrame(rows).sort_values(["순이득", "바뀐 답"], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame()

def sub_answers(path):             # 제출 파일의 답 (TEST_IDS 순서, 0~3). 없거나 문항이 모자라면 None
    p = path if os.path.isabs(path) else os.path.join(OUT_DIR, path)
    if not os.path.exists(p): return None
    s = pd.read_csv(p).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
REF_SUB_ANS = {k: sub_answers(v) for k, v in REF_SUBMISSIONS.items()}
for k, v in REF_SUB_ANS.items():
    if v is None: print(f"  (기존 제출 {k}: 파일 없음 → test 비교에서 뺌)")
REF_EVAL = evaluate(REF_CFG) if REF_CFG["weights"] else None
REF_TEST = ens(REF_CFG, "test").argmax(1) if REF_CFG["weights"] and all(r in PTEST for r in REF_CFG["weights"]) else None
if REF_EVAL: print(f"기준 {REF_NAME}: 검증 {REF_EVAL['acc']:.2f}% (오답 {REF_EVAL['n_err']})")

def report(cfg, name="현재 조합", brief=False):
    c = norm_cfg(cfg)
    if not c["weights"]: print("가중치가 모두 0입니다."); return None
    e = evaluate(c); tot = sum(c["weights"].values()); fams = {}
    for r, w in c["weights"].items(): fams[short_fam(FAMR[r])] = fams.get(short_fam(FAMR[r]), 0) + w / tot
    print(f"▶ {name} | {METHOD_LABEL[c['method']]} · {TEMP_LABEL[c['temp']]} · {FULL_LABEL[c['full']]} | 실행 {len(c['weights'])}개 · 계열 {len(fams)}개")
    print("  가중치: " + " · ".join(f"{r}×{w:g}" for r, w in sorted(c["weights"].items(), key=lambda t: (-t[1], t[0]))))
    print("  계열 비중: " + " · ".join(f"{f} {s * 100:.0f}%" for f, s in sorted(fams.items(), key=lambda t: -t[1])))
    print(f"  검증 {e['acc']:.2f}% (오답 {e['n_err']})" + (f" · NLL {e['nll']:.4f}" if e["nll"] is not None else ""))
    if REF_EVAL is not None:
        f, b, n, D = paired(REF_EVAL["ans"], e["ans"], Y_VAL); lo, hi = boot_ci(REF_EVAL["ok"], e["ok"])
        print(f"  vs 기준 {REF_NAME} ({REF_EVAL['acc']:.2f}%): 고침 {f} · 새로 틀림 {b} · 순이득 {n:+d} · 바뀐 답 {D} | 부호 검정 p={sign_p(f, f + b):.3f} | 차이 95% 구간 [{lo:+.2f}, {hi:+.2f}]%p")
    wrong = np.where(~e["ok"])[0]; nr = np.sum([CORRECT[r] for r in RUNS], 0)[wrong]
    print(f"  오답 {len(wrong)}개 중 맞힌 실행 수: 0개(조합으로 못 고침) {int((nr == 0).sum())} · 1~2개 {int(((nr >= 1) & (nr <= 2)).sum())} · 3개 이상 {int((nr >= 3).sum())}")
    print("  E14 오답 24개 중 맞힘: " + " · ".join(f"{k} {int(sum(e['ok'][_VPOS[i]] for i in v if i in _VPOS))}/{sum(i in _VPOS for i in v)}" for k, v in E14_ERR.items()))
    miss = [r for r in c["weights"] if r not in PTEST]
    if miss: print(f"  ⚠️ test 예측이 없는 실행 {miss} → 이 조합은 저장할 수 없습니다 (검증 탐색만)")
    else:
        at = ens(c, "test").argmax(1)
        print(f"  test 답 분포 {dict(zip(LETTERS, np.bincount(at, minlength=4).tolist()))}")
        if c["full"] != "none":
            used = [r for r in c["weights"] if r in FULL_OF]; at0 = ens({**c, "full": "none"}, "test").argmax(1)
            print(f"  _full 섞은 멤버 {len(used)}개 {used} → _full 때문에 바뀐 test 답 {int((at != at0).sum())}개")
        if REF_TEST is not None: print(f"  test 답이 기준 조합과 다른 문항: {int((at != REF_TEST).sum())}개")
        for k, a in REF_SUB_ANS.items():
            if a is not None:
                d = int((at != a).sum())
                print(f"  test 답이 기존 제출 {k}과 다른 문항: {d}개 → 공개 채점에 약 {d / 2:.0f}개 (다 맞거나 다 틀려도 공개 점수는 ±{d / 2 / N_PUBLIC * 100:.2f}%p 안)")
    ab = ablation(c)
    if len(ab):
        print("  멤버 하나씩 바꿔 보기 (순이득 큰 순" + (", 위 6개)" if brief else ")") + ":")
        print("    " + (ab.head(6) if brief else ab).to_string(index=False).replace("\n", "\n    "))
    if not brief and REF_EVAL is not None:
        ch = np.where(e["ans"] != REF_EVAL["ans"])[0]
        if len(ch):
            print(f"\n  기준 조합과 답이 바뀐 검증 문항 {len(ch)}개 (위 30개):")
            rows = [{"id": VAL_IDS[i], "정답": LETTERS[Y_VAL[i]], "기준": LETTERS[REF_EVAL["ans"][i]], "지금": LETTERS[e["ans"][i]],
                     "결과": "고침" if e["ok"][i] else ("새로 틀림" if REF_EVAL["ok"][i] else "둘 다 틀림"), "맞힌 실행": int(np.sum([CORRECT[r][i] for r in RUNS]))} for i in ch[:30]]
            print("    " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n    "))
    return e

def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def _file_of(run, pre, split):
    load(run, pre, split); return _CACHE[(run, pre, split)][1]

def save(cfg, name="search", note="auto"):
    # note: JSON에 남길 검색 기록 ("auto" = 이름이 검색 후보면 그 기록, None = 안 남김)
    c = norm_cfg(cfg)
    assert c["weights"], "가중치가 모두 0입니다"
    miss = [r for r in c["weights"] if r not in PTEST]
    assert not miss, f"test 예측이 없는 실행이 있어 제출 파일을 만들 수 없습니다: {miss}"
    e = evaluate(c); at = ens(c, "test").argmax(1)
    safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(name)).strip("_") or "search"
    tag = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:6]
    fn = f"submission_{safe}_{tag}.csv"; path = os.path.join(SAVE_DIR, fn)
    sub = sub_df[["id"]].assign(answer=[LETTERS[i] for i in at])
    assert len(sub) == len(sub_df) and set(sub.answer) <= set(LETTERS)
    sub.to_csv(path, index=False)
    mem = {}
    for r in c["weights"]:
        fv, ft = _file_of(r, PRE[r], "val"), _file_of(r, PRE[r], "test")          # 실제로 읽은 파일
        mem[r] = {"family": FAMR[r], "preset": PRE[r], "T": TT[r], "val_acc": ACC[r], "val": os.path.relpath(fv, OUT_DIR), "test": os.path.relpath(ft, OUT_DIR),
                  "val_md5": md5file(fv), "test_md5": md5file(ft)}
        if c["full"] != "none" and r in FULL_FILE: mem[r].update({"full_test": os.path.relpath(FULL_FILE[r], OUT_DIR), "full_test_md5": md5file(FULL_FILE[r])})
    info = {"name": str(name), "file": fn, "cfg": c, "val_acc": e["acc"], "val_err": e["n_err"], "val_nll": e["nll"], "ref": REF_NAME,
            "ref_val_acc": REF_EVAL["acc"] if REF_EVAL else None, "members": mem, "T_GRID": T_GRID, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "test_dist": dict(zip(LETTERS, np.bincount(at, minlength=4).tolist()))}
    if isinstance(note, str) and note == "auto": note = CV_NOTE.get(str(name)) if "CV_NOTE" in globals() else None
    if note: info["search"] = note                                                         # 검색 후보면 그 방법의 중첩 CV 기록
    json.dump(info, open(path[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nfull = sum("full_test" in m for m in mem.values())
    print(f"✅ 저장: {SAVE_SUBDIR}/{fn} (+ 설정 {fn[:-4]}.json) | 검증 {e['acc']:.2f}% · 오답 {e['n_err']} | {FULL_LABEL[c['full']]}"
          + ((f" (섞은 멤버 {nfull}개)" if nfull else " (_full 짝이 있는 멤버가 없어 원래 실행과 같은 답)") if c["full"] != "none" else "") + f" | test 답 분포 {info['test_dist']}")
    return path

def rebuild(json_path):
    # 설정 JSON에 적힌 예측 파일과 온도만으로 test 답을 다시 만들어 저장한 제출과 비교 (지금 멤버 탐색 결과와 무관 → 재현성 점검)
    p = json_path if os.path.isabs(json_path) else os.path.join(OUT_DIR, json_path)
    info = json.load(open(p, encoding="utf-8")); c = info["cfg"]; mem = info["members"]
    def _fp(rel): return os.path.join(OUT_DIR, rel)
    gone = [mem[r][k] for r in c["weights"] for k in ("test", "full_test") if k in mem[r] and not os.path.exists(_fp(mem[r][k]))]
    if gone: print(f"  ⚠️ {info['file']}: 예측 파일이 없어져 다시 만들 수 없습니다: {gone}"); return False
    changed = [r for r in c["weights"] if md5file(_fp(mem[r]["test"])) != mem[r]["test_md5"] or ("full_test" in mem[r] and md5file(_fp(mem[r]["full_test"])) != mem[r]["full_test_md5"])]
    if changed: print(f"  ⚠️ 저장한 뒤 예측 파일이 바뀌었거나 다시 써진 실행: {changed}")
    Ps = []
    for r in c["weights"]:
        P = _probs(_load_preds(_fp(mem[r]["test"]), TEST_IDS))
        if c.get("full", "none") != "none" and "full_test" in mem[r]:
            Pf = _probs(_load_preds(_fp(mem[r]["full_test"]), TEST_IDS)); P = (P + Pf) / 2 if c["full"] == "mix" else Pf
        Ps.append(temp_scale(P, mem[r]["T"]) if c["temp"] == "val" else P)
    at = _combine(Ps, [c["weights"][r] for r in c["weights"]], c["method"]).argmax(1)
    saved = sub_answers(os.path.join(os.path.dirname(p), info["file"]))
    same = saved is not None and bool((saved == at).all())
    print(f"  {info['file']}: 다시 만든 답이 저장한 제출과 " + ("완전히 같음 ✅" if same else f"다름 ⚠️ ({'제출 파일 없음' if saved is None else str(int((saved != at).sum())) + '개'})")
          + f" | 검증 {info['val_acc']:.2f}% · {METHOD_LABEL[c['method']]} · {TEMP_LABEL[c['temp']]}"
          + (f" · {FULL_LABEL[c['full']]} (섞은 멤버 {sum('full_test' in mem[r] for r in c['weights'])}개)" if c.get("full", "none") != "none" else ""))
    return same
print("✅ 조합 함수 준비 완료")
''')

md("""
## 6. 검색 준비 — 검색 풀 · 조합 수 · 중첩 CV 나누기
- 검색 풀의 실행마다 짧은 코드(A, B, C …)를 붙입니다. 아래 표들은 조합을 `A+C+F×2`처럼 코드로 적습니다.
- 전수 검색의 조합 수가 `MAX_SUBSETS`를 넘으면 조합 크기를 줄입니다. 줄인 크기가 출력됩니다.
""")
code(r'''
_VO_ALL = [r for r in RUNS if r not in PTEST]                          # 검증만 있는 실행 (test 추론 전)
def why_missing(r):                                                      # 목록에 넣었는데 검색에 못 넣은 이유
    d = os.path.join(OUT_DIR, r)
    if not os.path.isdir(d): return f"Drive에 폴더가 없습니다: {d} (이름 확인)"
    if r in EXCLUDE: return "EXCLUDE에 들어 있음"
    if is_full(r): return f"이름에 '_{FULL_TOKEN}' 조각 → 전체 데이터 학습으로 보고 뺌"
    if r in _suspect: return f"검증 > {SUSPECT_ACC}% → 검증을 학습에 쓴 실행으로 보고 뺌"
    if r in _vo_low: return f"검증 < {MIN_VAL_ACC}% → 고장으로 보고 뺌"
    for s_ in _skip + _wait:
        if s_.split("@")[0].split(" ")[0] == r: return s_
    fs = sorted(f for f in os.listdir(d) if f.startswith("preds"))
    if not fs: return "폴더에 예측 파일이 없습니다 → preds_final_ep1__up2.0__1000.csv와 train_log.json을 넣으세요"
    return f"검증 예측 파일 이름이 규칙과 다릅니다: {fs[:6]} (필요: preds_final_ep<N>__<해상도>__1000.csv)"
if SEARCH_INCLUDE_VALONLY is True: _VO_ADD = list(_VO_ALL)
elif SEARCH_INCLUDE_VALONLY:
    _VO_ADD = [r for r in dict.fromkeys(SEARCH_INCLUDE_VALONLY) if r in _VO_ALL]
    for r in dict.fromkeys(SEARCH_INCLUDE_VALONLY):
        if r in PTEST: print(f"({r}: test까지 있는 보통 멤버 → 그대로 검색 · 저장 가능)")
        elif r not in _VO_ALL: print(f"⚠️ 검증만 넣기 목록의 {r}: 검색에 못 넣음 → {why_missing(r)}")
else: _VO_ADD = []
if _VO_ADD: print(f"검증만 있는 실행 {len(_VO_ADD)}개를 검색에 넣음: {_VO_ADD} → 이 실행이 든 조합은 '저장: test 없음' (test 예측이 Drive에 오면 이 노트북을 다시 실행)")
SEARCH_POOL = list(SEARCH_ONLY) if SEARCH_ONLY else MEMBERS + _VO_ADD
_bad = [r for r in SEARCH_POOL if r not in PVAL]
if _bad: print(f"⚠️ 멤버 목록에 없어 검색에서 뺌: {_bad}")
SEARCH_POOL = [r for r in dict.fromkeys(SEARCH_POOL) if r in PVAL and r not in SEARCH_EXCLUDE]
assert len(SEARCH_POOL) >= 2, "검색할 실행이 2개 미만입니다 → SEARCH_ONLY / SEARCH_EXCLUDE를 확인하세요"
for s in SEARCH_SETTINGS: assert s[0] in ("prob", "logprob") and s[1] in TEMP_LABEL, f"SEARCH_SETTINGS 항목이 잘못됐습니다: {s} (방법은 prob/logprob, 온도는 none/val)"
M_S, N_V = len(SEARCH_POOL), len(Y_VAL)
_ab = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)] + [f"Z{i}" for i in range(52, 500)]
CODE = dict(zip(SEARCH_POOL, _ab))
def code_of(w):
    order = {r: i for i, r in enumerate(SEARCH_POOL)}
    return "+".join(CODE.get(r, r) + ("" if abs(v - 1) < 1e-9 else f"×{v:g}") for r, v in sorted(w.items(), key=lambda t: (order.get(t[0], 999), t[0])) if v > 0)
print(pd.DataFrame([{"코드": CODE[r], "실행": r, "계열": short_fam(FAMR[r]), "검증(%)": round(ACC[r], 2), "온도 T": round(TT[r], 2), "test": "O" if r in PTEST else "없음"}
                    for r in SEARCH_POOL]).to_string(index=False))
def _n_sub(s): return sum(math.comb(M_S, k) for k in range(1, s + 1))
_want = M_S if not MAX_SIZE else min(int(MAX_SIZE), M_S)
S_MAX = _want
while S_MAX > 1 and _n_sub(S_MAX) > MAX_SUBSETS: S_MAX -= 1
N_SUB = _n_sub(S_MAX)
print(f"\n전수 검색: 실행 {M_S}개 · 조합 크기 1~{S_MAX}" + (f" (조합 수 상한 {MAX_SUBSETS:,} 때문에 {_want}에서 줄임)" if S_MAX < _want else "")
      + f" · 조합 {N_SUB:,}개 × 설정 {len(SEARCH_SETTINGS)}개")
_rng = np.random.RandomState(CV_SEED); FOLD_ID = []
for _ in range(CV_REPEATS):
    _perm = _rng.permutation(N_V); _f = np.empty(N_V, int); _f[_perm] = np.arange(N_V) % CV_K; FOLD_ID.append(_f)
N_F = CV_K * CV_REPEATS
FM = np.zeros((N_V, N_F), np.float32)
for _rp, _f in enumerate(FOLD_ID): FM[np.arange(N_V), _rp * CV_K + _f] = 1.0
HELD = [FM[:, f] > 0 for f in range(N_F)]              # fold f에서 채점에만 쓰는 문항
EPS = 1e-5                                             # 점수 = 맞힌 수 − EPS × NLL 합 (정확도가 같으면 NLL로 가름)
print(f"중첩 CV: 검증 {N_V:,}문항을 {CV_K}겹으로 {CV_REPEATS}번 나눔 → 매번 {N_V - N_V // CV_K}문항으로 고르고 남은 {N_V // CV_K}문항으로 채점")

def set_name(s): return ("확률" if s[0] == "prob" else "로그확률") + ("+온도" if s[1] == "val" else "")
def stack_of(setting, runs=None):
    runs = runs or SEARCH_POOL
    A = np.stack([member_P(r, "val", {"temp": setting[1]}) for r in runs])
    return np.log(np.clip(A, 1e-6, 1.0)) if setting[0] == "logprob" else A
def _scores(Sm, method, mask=None):
    # Sm (..., N, 4): 가중 평균 (prob = 확률 / logprob = 로그확률 평균) → (맞힌 수, NLL 합). mask가 있으면 그 문항만
    ar = np.arange(N_V)
    if method == "logprob":
        Z = Sm - Sm.max(-1, keepdims=True); lp = Z[..., ar, Y_VAL] - np.log(np.exp(Z).sum(-1))
    else: lp = np.log(np.clip(Sm[..., ar, Y_VAL], 1e-12, 1.0))
    ok = Sm.argmax(-1) == Y_VAL
    if mask is not None: ok, lp = ok[..., mask], lp[..., mask]
    return ok.sum(-1).astype(float), -lp.sum(-1)
def w_to_cfg(w, setting):
    return {"weights": {r: float(v) for r, v in zip(SEARCH_POOL, w) if v > 0}, "method": setting[0], "temp": setting[1], "full": "none"}
def subset_cfg(sub, setting):
    return {"weights": {SEARCH_POOL[i]: 1.0 for i in sub}, "method": setting[0], "temp": setting[1], "full": "none"}
def oof_of(chooser):
    # chooser(f) = fold f의 학습 문항으로 고른 조합 → 반복마다 1,000문항 OOF 답 (각 문항은 자기가 채점 문항일 때 고른 조합의 답)
    oof = np.full((CV_REPEATS, N_V), -1); cfgs = []
    for f in range(N_F):
        cfg = chooser(f); cfgs.append(cfg)
        oof[f // CV_K, HELD[f]] = ens(cfg, "val").argmax(1)[HELD[f]]
    assert (oof >= 0).all()
    return oof, cfgs
REF_ANS = REF_EVAL["ans"] if REF_EVAL is not None else None
''')

md("""
## 7. 전수 검색 — 같은 가중치 부분집합 전부
- 설정마다 모든 부분집합을 한 번 훑습니다. 그러면서 두 가지를 같이 구합니다.
  - 검증 전체 점수 상위 목록
  - 중첩 CV fold마다 학습 문항 점수가 가장 높은 조합
- 순위는 **정확도 → NLL** 순입니다. 'NLL' 규칙은 NLL만 봅니다.
- **합의 조합**: 상위 100개 조합에 가장 자주 든 실행을, 그 조합들의 보통 크기(중앙값)만큼 모아 같은 가중치로 만든 조합입니다. 1위 하나를 고르는 것보다 1,000문항의 운에 덜 흔들립니다. 이것도 중첩 CV로 채점합니다.
- **상위 100개 조합에 든 비율**: 실행이 좋은 조합들에 두루 들어가는지 보여 줍니다. 1위 조합 하나보다 이쪽이 덜 흔들립니다.
""")
code(r'''
SCAN_CHUNK, TOP_KEEP = 4096, 300
def scan(setting, verbose=True):
    method = setting[0]; ar = np.arange(N_V)
    A = stack_of(setting).transpose(0, 2, 1).reshape(M_S, -1).astype(np.float32)      # (실행, 보기×문항): 보기마다 연속된 조각 → 조각 연산이 빠름
    top = {"acc": [], "nll": []}; fb = {k: {"s": np.full(N_F, -np.inf), "sub": [None] * N_F} for k in top}
    topf = [[] for _ in range(N_F)]                    # fold마다 학습 문항 점수 상위 CONS_TOP개 (합의 조합용)
    _key = lambda t: (-t[0], len(t[1]), t[1])
    it = itertools.chain.from_iterable(itertools.combinations(range(M_S), k) for k in range(1, S_MAX + 1))
    done, t0, nxt = 0, time.time(), 0.25
    while True:
        blk = list(itertools.islice(it, SCAN_CHUNK))
        if not blk: break
        c = len(blk); I = np.zeros((c, M_S), np.float32)
        for i, t in enumerate(blk): I[i, list(t)] = 1.0
        S, k_ = (I @ A).reshape(c, 4, N_V), I.sum(1)
        if method == "logprob":                                                        # 평균 로그확률은 [log 1e-6, 0] 안이라 exp가 넘치지 않음
            S /= k_[:, None, None]; s0, s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
            lp = S[:, Y_VAL, ar] - np.log(np.exp(s0) + np.exp(s1) + np.exp(s2) + np.exp(s3))
        else:
            s0, s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]; lp = np.log(np.clip(S[:, Y_VAL, ar] / k_[:, None], 1e-12, 1.0))
        mx = np.maximum(np.maximum(s0, s1), np.maximum(s2, s3))
        pred = np.where(s0 == mx, 0, np.where(s1 == mx, 1, np.where(s2 == mx, 2, 3)))    # argmax와 같음 (동점이면 앞 보기)
        C = (pred == Y_VAL).astype(np.float32); L = (-lp).astype(np.float32)
        tc, tl = C.sum(1, dtype=np.float64), L.sum(1, dtype=np.float64)
        hc, hl = (C @ FM).astype(np.float64), (L @ FM).astype(np.float64)
        for rule in top:
            full = tc - EPS * tl if rule == "acc" else -tl
            k = min(TOP_KEEP, c); j = np.argpartition(-full, k - 1)[:k]
            top[rule] += [(float(full[i]), blk[i]) for i in j]
            if len(top[rule]) > 3 * TOP_KEEP: top[rule] = sorted(top[rule], key=lambda t: (-t[0], len(t[1]), t[1]))[:TOP_KEEP]
            tr = (tc[:, None] - hc) - EPS * (tl[:, None] - hl) if rule == "acc" else -(tl[:, None] - hl)
            jb = tr.argmax(0); sb = tr[jb, np.arange(N_F)]
            for f in np.where(sb > fb[rule]["s"])[0]: fb[rule]["s"][f], fb[rule]["sub"][f] = sb[f], blk[jb[f]]
            if rule == "acc":
                kf = min(CONS_TOP, c)
                for f in range(N_F):
                    col = tr[:, f]; jj = np.argpartition(-col, kf - 1)[:kf]; topf[f] += [(float(col[i]), blk[i]) for i in jj]
                    if len(topf[f]) > 3 * CONS_TOP: topf[f] = sorted(topf[f], key=_key)[:CONS_TOP]
        done += c
        if verbose and (done / N_SUB >= nxt or done == N_SUB):
            print(f"    {set_name(setting)}: {done:,}/{N_SUB:,} ({time.time() - t0:.0f}초)", flush=True)
            while nxt <= done / N_SUB: nxt += 0.25
    for rule in top: top[rule] = sorted(top[rule], key=_key)[:TOP_KEEP]
    return top, fb, [sorted(t, key=_key)[:CONS_TOP] for t in topf]
def consensus(tops):
    # 상위 조합들에 가장 자주 든 실행 k개 (k = 그 조합들 크기의 중앙값) → 같은 가중치 조합. 빈도가 같으면 코드 순서
    cnt = np.zeros(M_S)
    for _, sub in tops: cnt[list(sub)] += 1
    k = max(1, int(round(float(np.median([len(sub) for _, sub in tops])))))
    return tuple(sorted(sorted(range(M_S), key=lambda i: (-cnt[i], i))[:k]))

EXH, _t_all = {}, time.time()
for s in SEARCH_SETTINGS:
    top, fb, topf = scan(s); EXH[s] = {"top": top, "fb": fb}
    for rule in ("acc", "nll"):
        EXH[s]["oof_" + rule], EXH[s]["cv_" + rule] = oof_of(lambda f: subset_cfg(fb[rule]["sub"][f], s))
    EXH[s]["cons"] = consensus(top["acc"][:CONS_TOP])
    EXH[s]["oof_cons"], EXH[s]["cv_cons"] = oof_of(lambda f: subset_cfg(consensus(topf[f]), s))
print(f"전수 검색 끝 ({time.time() - _t_all:.0f}초)")

for s in SEARCH_SETTINGS:
    rows = []
    for rank, (_, sub) in enumerate(EXH[s]["top"]["acc"][:TOP_SHOW], 1):
        cfg = subset_cfg(sub, s); e = evaluate(cfg)
        f_, b_, n_, D_ = paired(REF_ANS, e["ans"], Y_VAL) if REF_ANS is not None else (0, 0, 0, 0)
        rows.append({"순위": rank, "조합": code_of(cfg["weights"]), "개수": len(sub), "검증(%)": round(e["acc"], 2), "오답": e["n_err"], "NLL": round(e["nll"], 4),
                     "고침": f_, "새로 틀림": b_, "순이득": n_, "p": round(sign_p(f_, f_ + b_), 3)})
    print(f"\n[{set_name(s)}] 전수 검색 상위 {TOP_SHOW}개 (정확도 → NLL 순, 고침·새로 틀림은 기준 {REF_NAME} 대비)")
    print(pd.DataFrame(rows).to_string(index=False))
    t100 = EXH[s]["top"]["acc"][:100]
    freq = {CODE[r]: sum(i in sub for _, sub in t100) / len(t100) * 100 for i, r in enumerate(SEARCH_POOL)}
    print(f"  상위 {len(t100)}개 조합에 든 비율(%): " + " · ".join(f"{k} {v:.0f}" for k, v in sorted(freq.items(), key=lambda t: -t[1])))
    nsub = EXH[s]["top"]["nll"][0][1]; en = evaluate(subset_cfg(nsub, s))
    print(f"  NLL 1위: {code_of(subset_cfg(nsub, s)['weights'])} | 검증 {en['acc']:.2f}% · NLL {en['nll']:.4f}")
    ec = evaluate(subset_cfg(EXH[s]["cons"], s))
    print(f"  합의 조합 (상위 {min(CONS_TOP, len(EXH[s]['top']['acc']))}개에 가장 자주 든 {len(EXH[s]['cons'])}개 = 그 조합들의 크기 중앙값): {code_of(subset_cfg(EXH[s]['cons'], s)['weights'])} | 검증 {ec['acc']:.2f}% · NLL {ec['nll']:.4f}")
''')

md("""
## 8. 그리디 · 좌표 상승 — 가중치까지 고르기
- **그리디**: 빈 조합에서 시작합니다. 매번 가중치 1을 더했을 때 점수(정확도 → NLL)가 가장 좋은 실행을 더합니다.
  - 같은 실행을 여러 번 더할 수 있어서, 더한 횟수가 가중치가 됩니다.
  - 최대 `GREEDY_STEPS`번 더하고, 그중 점수가 가장 좋았던 시점을 씁니다.
- **좌표 상승**: 기준 조합에서 시작합니다. 실행 하나씩 가중치를 `WEIGHT_GRID` 값으로 바꿔 보고, 좋아지면 받아들입니다. 더 안 좋아질 때까지 반복합니다.
- 두 방법 모두 중첩 CV fold마다 학습 문항으로만 다시 돌려서, 채점 문항 점수를 잽니다.
""")
code(r'''
def greedy_w(A, method, mask=None, steps=GREEDY_STEPS):
    cnt = np.zeros(A.shape[0]); S = np.zeros(A.shape[1:]); best_s, best_w = -np.inf, None
    for _ in range(steps):
        c, l = _scores((S[None] + A) / (cnt.sum() + 1), method, mask); sc = c - EPS * l
        m = int(np.argmax(sc)); cnt[m] += 1; S = S + A[m]                    # 점수가 떨어져도 가장 나은 것을 더하고, 가장 좋았던 시점을 씀
        if sc[m] > best_s + 1e-9: best_s, best_w = sc[m], cnt.copy()
    return best_w
def coord_w(A, method, w0, mask=None, grid=WEIGHT_GRID, max_pass=8):
    w, g = np.array(w0, float), np.array(grid, float)
    def sc_of(Wm):
        tot = Wm.sum(1); Sm = np.einsum("gm,mnk->gnk", Wm, A) / np.where(tot > 0, tot, 1.0)[:, None, None]
        c, l = _scores(Sm, method, mask); s = c - EPS * l; s[tot <= 0] = -np.inf; return s
    cur = sc_of(w[None])[0]
    for _ in range(max_pass):
        moved = False
        for m in range(len(w)):
            Wm = np.repeat(w[None], len(g), 0); Wm[:, m] = g; s = sc_of(Wm); j = int(np.argmax(s))
            if s[j] > cur + 1e-9: w, cur, moved = Wm[j].copy(), s[j], True
        if not moved: break
    return w
W_REF0 = np.array([float(REF_CFG["weights"].get(r, 0.0)) for r in SEARCH_POOL])
if W_REF0.sum() <= 0: W_REF0 = np.ones(M_S); print("기준 조합의 실행이 검색 풀에 없어 좌표 상승은 '모두 1'에서 시작합니다")
GRD, CRD, _t0 = {}, {}, time.time()
for s in SEARCH_SETTINGS:
    A = stack_of(s)
    GRD[s] = {"w": greedy_w(A, s[0])}; CRD[s] = {"w": coord_w(A, s[0], W_REF0)}
    GRD[s]["oof"], GRD[s]["cv"] = oof_of(lambda f: w_to_cfg(greedy_w(A, s[0], ~HELD[f]), s))
    CRD[s]["oof"], CRD[s]["cv"] = oof_of(lambda f: w_to_cfg(coord_w(A, s[0], W_REF0, ~HELD[f]), s))
    for nm, R in (("그리디", GRD[s]), ("좌표 상승", CRD[s])):
        cfg = w_to_cfg(R["w"], s); e = evaluate(cfg)
        print(f"[{set_name(s)}] {nm}: {code_of(cfg['weights'])} | 검증 {e['acc']:.2f}% (오답 {e['n_err']}) · NLL {e['nll']:.4f}")
print(f"그리디·좌표 상승 끝 ({time.time() - _t0:.0f}초)")
''')

md("""
## 9. 결과 정리 — 어떤 검색 결과를 믿을지
**중첩 CV 표** (방법 × 설정마다 한 줄)
- **검증(%)**: 검증 1,000문항 전체로 고른 조합의 검증 정확도입니다. 1,000문항에 맞춰 고른 값이라 부풀려져 있습니다.
- **중첩 CV(%)**: 800문항으로 고른 조합을 남은 200문항으로 채점한 정확도입니다(1,000문항 모두 한 번씩 채점, 4번 평균).
- **과대평가(%p)** = 검증 − 중첩 CV입니다. 클수록 그 방법이 1,000문항의 운에 맞춰 고른 것입니다.
- **CV 고침 / CV 새로 틀림**: CV 답을 기준 조합과 같은 문항으로 비교한 수(4번 평균)입니다.

**판단 요령**
- 중첩 CV가 기준 조합의 검증보다 높고, CV 고침이 새로 틀림보다 뚜렷이 많을 때만 그 방법이 기준보다 낫다고 볼 수 있습니다.
- 비슷하거나 낮으면, 그 방법이 고른 조합의 높은 검증 점수는 운입니다.
  - 이때는 기준 조합을 쓰거나, 상위 100개 조합에 두루 드는 실행으로만 만든 단순한 조합을 쓰는 편이 안전합니다.
- 기준 조합도 검증과 공개 점수를 보고 고른 것이라 조금은 부풀려져 있습니다.
- **후보 표**의 조합은 10번 패널에서 불러와 다듬거나, 11·12번 셀로 보고·저장할 수 있습니다.
- 표 두 개는 `ens_search/` 폴더에 CSV로도 남습니다.

**후보 표의 `_full` 칸** (검증 수치는 모두 `_full` 없이 잰 값)
- 검색 후보는 제출 파일의 test 답을 `FULL_MODE`(기본 `mix`)로 만듭니다. `_full` 칸에 섞는 방식과 짝이 있는 멤버 수가 나옵니다.
- **_full로 바뀐 답**: `_full`을 안 섞었을 때와 test 답이 다른 문항 수입니다. 공개 채점에는 그 절반쯤 들어갑니다.
- **기준** 줄은 지금 제출 파일 그대로(`_full` 없이)입니다. `test≠clean4_qg_hcx`가 0이면 재현이 맞는 것입니다.
- **기준 + _full** 줄은 기준 조합에 `_full`만 섞은 것입니다. `_full` 효과만 따로 보려면 이 줄을 보면 됩니다.
- `test≠…` 칸은 모두 섞은 뒤의 test 답으로 셉니다.
""")
code(r'''
CV_NOTE = {}
def cv_row(name, s, oof, full_cfg):
    accs = (oof == Y_VAL).mean(1) * 100; e = evaluate(full_cfg)
    fx = [paired(REF_ANS, o, Y_VAL) for o in oof] if REF_ANS is not None else [(0, 0, 0, 0)]
    fm, bm = float(np.mean([t[0] for t in fx])), float(np.mean([t[1] for t in fx]))
    return {"방법": name, "설정": set_name(s), "조합(검증 전체로 고름)": code_of(full_cfg["weights"]), "검증(%)": round(e["acc"], 2),
            "중첩 CV(%)": round(float(accs.mean()), 2), "CV 범위": f"{accs.min():.1f}~{accs.max():.1f}", "과대평가(%p)": round(e["acc"] - float(accs.mean()), 2),
            "CV 고침": round(fm, 1), "CV 새로 틀림": round(bm, 1), "CV p": round(sign_p(int(round(fm)), int(round(fm)) + int(round(bm))), 3)}
CV_ROWS, ALLPROC = [], []
for s in SEARCH_SETTINGS:
    for rule, nm in (("acc", "전수(정확도)"), ("nll", "전수(NLL)")):
        CV_ROWS.append(cv_row(nm, s, EXH[s]["oof_" + rule], subset_cfg(EXH[s]["top"][rule][0][1], s))); ALLPROC.append((nm, s, EXH[s]["cv_" + rule], subset_cfg(EXH[s]["top"][rule][0][1], s)))
    CV_ROWS.append(cv_row("전수 합의", s, EXH[s]["oof_cons"], subset_cfg(EXH[s]["cons"], s))); ALLPROC.append(("전수 합의", s, EXH[s]["cv_cons"], subset_cfg(EXH[s]["cons"], s)))
    CV_ROWS.append(cv_row("그리디", s, GRD[s]["oof"], w_to_cfg(GRD[s]["w"], s))); ALLPROC.append(("그리디", s, GRD[s]["cv"], w_to_cfg(GRD[s]["w"], s)))
    CV_ROWS.append(cv_row("좌표 상승", s, CRD[s]["oof"], w_to_cfg(CRD[s]["w"], s))); ALLPROC.append(("좌표 상승", s, CRD[s]["cv"], w_to_cfg(CRD[s]["w"], s)))
def _score_on(cfg, mask):
    S = ens(cfg, "val"); P = S / S.sum(1, keepdims=True)
    return float((S.argmax(1) == Y_VAL)[mask].sum()) - EPS * float(-np.log(np.clip(P[np.arange(N_V), Y_VAL], 1e-12, 1.0))[mask].sum())
# '모든 방법·설정 중 최고 고르기'도 하나의 검색 방법으로 채점 (fold마다 학습 문항 점수가 가장 높은 것)
_oof_all = np.full((CV_REPEATS, N_V), -1)
for f in range(N_F):
    best = max((p[2][f] for p in ALLPROC), key=lambda cfg: _score_on(cfg, ~HELD[f]))
    _oof_all[f // CV_K, HELD[f]] = ens(best, "val").argmax(1)[HELD[f]]
_all_full = max((p[3] for p in ALLPROC), key=lambda cfg: _score_on(cfg, np.ones(N_V, bool)))
CV_ROWS.append({**cv_row("모든 방법 중 최고", ("prob", "none"), _oof_all, _all_full), "설정": set_name((_all_full["method"], _all_full["temp"]))})
CV_TABLE = pd.DataFrame(CV_ROWS)
print(f"기준 {REF_NAME}: 검증 {REF_EVAL['acc']:.2f}% (고정 조합이라 중첩 CV로 재도 같음)\n" if REF_EVAL else "")
print(CV_TABLE.to_string(index=False))
_best_cv = CV_TABLE.iloc[int(CV_TABLE["중첩 CV(%)"].values.argmax())]
if REF_EVAL is not None:
    print(f"\n중첩 CV 최고: {_best_cv['방법']} · {_best_cv['설정']} {_best_cv['중첩 CV(%)']:.2f}% (기준보다 {_best_cv['중첩 CV(%)'] - REF_EVAL['acc']:+.2f}%p · "
          f"CV 고침 {_best_cv['CV 고침']} / 새로 틀림 {_best_cv['CV 새로 틀림']})")

# ── 후보 표 ── (검색 후보의 test 답은 FULL_MODE로 만듦. '기준' 줄은 지금 제출 그대로 = _full 없이)
CANDS, _seen = {}, {}
def _ckey(c): return json.dumps({**c, "full": c["full"] if any(r in FULL_OF for r in c["weights"]) else "none"}, sort_keys=True)   # _full 짝이 없으면 섞어도 같은 답
def _add(name, cfg, note=None, full=None):
    c = norm_cfg({**cfg, "full": FULL_MODE if full is None else full}, warn=False); key = _ckey(c)
    if key in _seen: _seen[key].append(name); return
    _seen[key] = [name]; CANDS[name] = c
    if note: CV_NOTE[name] = note
if REF_CFG["weights"]:
    _add(f"기준 {REF_NAME}", REF_CFG, full="none")
    if FULL_MODE != "none": _add(f"기준 {REF_NAME} + _full {FULL_MODE}", REF_CFG)
_cvmap = {(r["방법"], r["설정"]): r for r in CV_ROWS}
for s in SEARCH_SETTINGS:
    sn = set_name(s)
    for rank, (_, sub) in enumerate(EXH[s]["top"]["acc"][:CAND_TOP], 1): _add(f"전수 {rank}위 · {sn}", subset_cfg(sub, s), {**_cvmap[("전수(정확도)", sn)], "순위": rank})
    _add(f"전수(NLL) 1위 · {sn}", subset_cfg(EXH[s]["top"]["nll"][0][1], s), _cvmap[("전수(NLL)", sn)])
    _add(f"전수 합의 · {sn}", subset_cfg(EXH[s]["cons"], s), _cvmap[("전수 합의", sn)])
    _add(f"그리디 · {sn}", w_to_cfg(GRD[s]["w"], s), _cvmap[("그리디", sn)])
    _add(f"좌표 상승 · {sn}", w_to_cfg(CRD[s]["w"], s), _cvmap[("좌표 상승", sn)])
_rows = []
for name, c in CANDS.items():
    e = evaluate(c); f_, b_, n_, D_ = paired(REF_ANS, e["ans"], Y_VAL) if REF_ANS is not None else (0, 0, 0, 0)
    ok_t = all(r in PTEST for r in c["weights"]); at = ens(c, "test").argmax(1) if ok_t else None
    nf = sum(r in FULL_OF for r in c["weights"]) if c["full"] != "none" else 0
    row = {"후보": name, "같은 조합": ", ".join(_seen[_ckey(c)][1:]), "조합": code_of(c["weights"]), "검증(%)": round(e["acc"], 2), "오답": e["n_err"],
           "NLL": round(e["nll"], 4) if e["nll"] is not None else None, "고침": f_, "새로 틀림": b_, "p": round(sign_p(f_, f_ + b_), 3),
           "_full": f"{c['full']} {nf}개" if nf else "-",
           "_full로 바뀐 답": int((at != ens({**c, "full": "none"}, "test").argmax(1)).sum()) if (at is not None and nf) else "-",
           "test≠기준": int((at != REF_TEST).sum()) if (at is not None and REF_TEST is not None) else "-"}
    for k, a in REF_SUB_ANS.items():
        if a is not None: row["test≠" + k.split(" ")[0]] = int((at != a).sum()) if at is not None else "-"
    row["저장"] = "O" if ok_t else "test 없음"; _rows.append(row)
CAND_TABLE = pd.DataFrame(_rows)
print(f"\n후보 {len(CANDS)}개 — 고침·새로 틀림·p는 기준 대비 검증(_full과 무관) · test 칸은 이 후보로 저장할 제출 파일의 test 답 기준")
print(f"  _full = 섞는 방식과 짝이 있는 멤버 수 (검색 후보는 FULL_MODE='{FULL_MODE}') · _full로 바뀐 답 = _full을 안 섞을 때와 test 답이 다른 문항 수")
print(CAND_TABLE.to_string(index=False))
_stamp = datetime.datetime.now().strftime("%m%d_%H%M")
CV_TABLE.to_csv(os.path.join(SAVE_DIR, f"search_{_stamp}_cv.csv"), index=False); CAND_TABLE.to_csv(os.path.join(SAVE_DIR, f"search_{_stamp}_candidates.csv"), index=False)
json.dump({"pool": {CODE[r]: r for r in SEARCH_POOL}, "settings": [list(s) for s in SEARCH_SETTINGS], "S_MAX": S_MAX, "N_SUB": N_SUB, "cv": [CV_K, CV_REPEATS, CV_SEED],
           "full": {"mode": FULL_MODE, "pairs": FULL_INFO}, "candidates": {k: v for k, v in CANDS.items()}},open(os.path.join(SAVE_DIR, f"search_{_stamp}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n표 저장: {SAVE_SUBDIR}/search_{_stamp}_cv.csv · search_{_stamp}_candidates.csv · search_{_stamp}.json")
''')

md("""
## 10. 슬라이더 패널 — 후보를 불러와 직접 다듬기
- **후보** 메뉴에서 9번 표의 후보를 고르면 슬라이더에 그 가중치가 들어갑니다. 저장 이름도 후보 이름으로 바뀝니다.
- 슬라이더를 놓으면 결과가 갱신됩니다(0 = 뺌, 0.25 단위, 최대 3). 위쪽 메뉴로 결합 방법과 온도 보정을 고릅니다.
- **test** 메뉴: 제출 파일의 test 답에 `_full`을 섞는 방식입니다(원래 실행 / 평균 / 교체). 처음 값은 `FULL_MODE`입니다.
  - 검증 수치는 바뀌지 않습니다. 결과의 '_full 때문에 바뀐 test 답'과 test 비교 줄만 바뀝니다.
  - 후보를 불러오면 그 후보의 설정으로 바뀝니다('기준' 줄은 `_full` 없음). 버튼은 이 메뉴를 건드리지 않습니다.
  - 슬라이더 옆 '_full 짝'이 붙은 실행만 섞입니다.
- 버튼
  - **기준 조합**: `REF_CFG`의 가중치로 되돌립니다.
  - **강한 풀 = 1**: 강한 풀을 모두 1로, 나머지를 0으로 둡니다.
  - **모두 0**
  - **코드로 출력**: 지금 설정을 11번 셀에 붙여 넣을 코드로 출력합니다.
  - **제출 파일 저장**: 지금 설정으로 `ens_search/`에 CSV와 JSON을 씁니다.
- 결과 맨 아래 '멤버 하나씩 바꿔 보기' 표: 들어 있는 실행은 빼기·±0.5, 안 든 실행은 넣기(1)를 했을 때의 고침·새로 틀림입니다.
- 슬라이더로 고친 조합은 중첩 CV를 거치지 않았습니다. 검증 수치가 더 부풀려져 있을 수 있습니다.
""")
code(r'''
try:
    import ipywidgets as W
    from IPython.display import display
except ImportError:
    W = None; print("ipywidgets가 없어 슬라이더를 띄울 수 없습니다 → 11번 셀에서 PICK이나 WEIGHTS를 코드로 정하세요")
START_CFG = dict(REF_CFG) if REF_CFG["weights"] else next(iter(CANDS.values()))
if W is not None:
    SL, _rows_ui, _busy = {}, [], {"on": False}
    for r in sorted(RUNS, key=lambda r: (short_fam(FAMR[r]), -ACC[r])):
        SL[r] = W.FloatSlider(value=float(START_CFG["weights"].get(r, 0.0)), min=0.0, max=3.0, step=0.25, continuous_update=False,
                              readout_format=".2f", layout=W.Layout(width="300px"))
        _rows_ui.append(W.HBox([SL[r], W.HTML(f"<b>{CODE.get(r, '·')}</b> <code>{r}</code> · {short_fam(FAMR[r])} · 검증 {ACC[r]:.2f}% · T {TT[r]:.2f}"
                                               + (" · <b>test 없음</b>" if r not in PTEST else "") + (" · _full 짝" if r in FULL_OF else ""))]))
    _dl, _ds = W.Layout(width="360px"), {"description_width": "40px"}
    DD_C = W.Dropdown(options=["(후보 고르기)"] + list(CANDS), value="(후보 고르기)", description="후보", layout=W.Layout(width="480px"), style=_ds)
    DD_M = W.Dropdown(options=[(v, k) for k, v in METHOD_LABEL.items()], value=START_CFG.get("method", "prob"), description="결합", layout=_dl, style=_ds)
    DD_T = W.Dropdown(options=[(v, k) for k, v in TEMP_LABEL.items()], value=START_CFG.get("temp", "none"), description="온도", layout=_dl, style=_ds)
    DD_F = W.Dropdown(options=[(v, k) for k, v in FULL_LABEL.items()], value=FULL_MODE, description="test", layout=_dl, style=_ds)
    TX = W.Text(value="my_search", description="저장 이름")
    B_REF, B_POOL, B_ZERO, B_CODE, B_SAVE = [W.Button(description=d) for d in ("기준 조합", "강한 풀 = 1", "모두 0", "코드로 출력", "제출 파일 저장")]
    B_SAVE.button_style = "success"
    OUT_UI, MSG_UI = W.Output(), W.Output()
    def ui_cfg(): return {"weights": {r: float(s.value) for r, s in SL.items() if s.value > 0}, "method": DD_M.value, "temp": DD_T.value, "full": DD_F.value}
    def ui_refresh(*_):
        if _busy["on"]: return
        with OUT_UI:
            OUT_UI.clear_output(wait=True); report(ui_cfg(), "슬라이더 조합", brief=True)
    def ui_set(cfg):
        _busy["on"] = True
        try:
            for r, s in SL.items(): s.value = float(cfg["weights"].get(r, 0.0))
            DD_M.value, DD_T.value = cfg.get("method", DD_M.value), cfg.get("temp", DD_T.value)
            if "full" in cfg: DD_F.value = cfg["full"]                                  # 후보를 불러올 때만 (버튼은 test 메뉴를 그대로 둠)
        finally: _busy["on"] = False
        ui_refresh()
    def ui_pick(ch):
        if ch["new"] in CANDS: TX.value = ch["new"]; ui_set(CANDS[ch["new"]])
    def ui_code(_=None):
        c = ui_cfg()
        with MSG_UI:
            MSG_UI.clear_output(); print("# ↓ 11번 셀의 WEIGHTS · METHOD 두 줄을 이것으로 바꾸세요 (PICK = None으로 두고)")
            print("WEIGHTS = {" + ", ".join(f'"{r}": {w:g}' for r, w in sorted(c["weights"].items(), key=lambda t: (-t[1], t[0]))) + "}")
            print(f'METHOD, TEMP, FULL = "{c["method"]}", "{c["temp"]}", "{c["full"]}"')
    def ui_save(_=None):
        with MSG_UI:
            MSG_UI.clear_output()
            try:
                c, nm = ui_cfg(), TX.value
                if nm in CANDS and norm_cfg(c, warn=False) != CANDS[nm]: nm = nm + " 수정"     # 후보를 슬라이더로 고쳤으면 이름을 구분
                save(c, nm)
            except AssertionError as ex: print("⚠️", ex)
    B_REF.on_click(lambda _: ui_set({k: v for k, v in REF_CFG.items() if k != "full"})); B_POOL.on_click(lambda _: ui_set({"weights": {r: 1.0 for r in STRONG}}))
    B_ZERO.on_click(lambda _: ui_set({"weights": {}})); B_CODE.on_click(ui_code); B_SAVE.on_click(ui_save)
    DD_C.observe(ui_pick, names="value")
    for s in SL.values(): s.observe(ui_refresh, names="value")
    for d in (DD_M, DD_T, DD_F): d.observe(ui_refresh, names="value")
    _wrap = W.Layout(flex_flow="row wrap")
    display(W.VBox([DD_C, W.HBox([DD_M, DD_T, DD_F], layout=_wrap),W.VBox(_rows_ui), W.HBox([B_REF, B_POOL, B_ZERO], layout=_wrap), W.HBox([TX, B_CODE, B_SAVE], layout=_wrap), MSG_UI, OUT_UI]))
    ui_refresh()
''')

md("""
## 11. 후보 하나 자세히 보기
- `PICK`에 9번 후보 표의 이름을 넣습니다(예: `"전수 1위 · 확률"`).
- 직접 정한 조합을 보려면 `PICK = None`으로 두고 `WEIGHTS`를 적습니다. 10번 패널의 '코드로 출력' 결과를 붙여 넣어도 됩니다.
- `FULL`: 제출 파일의 test 답에 `_full`을 섞는 방식입니다(기본 = 1번 셀의 `FULL_MODE`). `PICK` 후보에도 이 값을 씁니다.
  - 검증 수치는 바뀌지 않습니다. 결과에 '_full 때문에 바뀐 test 답' 수가 나옵니다.
- 패널보다 더 보여 주는 것
  - 멤버 하나씩 바꿔 보기(전체)
  - 기준 조합과 답이 바뀐 검증 문항 목록
  - 검색 후보라면 그 방법의 중첩 CV 기록
""")
code(r'''
PICK = None                                  # 예: "전수 1위 · 확률" (9번 후보 표의 '후보' 이름)
NAME = "my_search"
FULL = FULL_MODE                             # 제출 파일(test 답)에 _full 섞기: "none" / "mix" / "swap" — 검증 수치에는 영향 없음
WEIGHTS = dict(REF_CFG["weights"])           # PICK = None일 때 씀. 예: {"q35_27b_r16_native": 1, "gemma4_31b_r16_up20": 1.5, ...}
METHOD, TEMP = "prob", "none"
assert FULL in FULL_LABEL, f"FULL은 none/mix/swap 중 하나: {FULL}"
if PICK is not None:
    assert PICK in CANDS, f"후보 이름이 없습니다: {PICK} → 9번 후보 표의 '후보' 칸에서 그대로 복사하세요"
    CFG_NOW = {**CANDS[PICK], "full": FULL}
    if NAME == "my_search": NAME = PICK if CANDS[PICK]["full"] == FULL else f"{PICK} · test {FULL}"      # 후보와 test 설정이 다르면 이름으로 구분
    NOTE_NOW = CV_NOTE.get(PICK)                                                               # 12번 저장 JSON에 남길 검색 기록
    if NOTE_NOW: print(f"검색 기록: {NOTE_NOW['방법']} · {NOTE_NOW['설정']} → 검증 {NOTE_NOW['검증(%)']}% / 중첩 CV {NOTE_NOW['중첩 CV(%)']}% (과대평가 {NOTE_NOW['과대평가(%p)']}%p)\n")
else: CFG_NOW, NOTE_NOW = {"weights": WEIGHTS, "method": METHOD, "temp": TEMP, "full": FULL}, None
_ = report(CFG_NOW, NAME)
''')

md("""
## 12. 11번 셀의 조합을 제출 파일로 저장 (CSV + 설정 JSON → `vqa_runs/ens_search/`)
- `_full`을 섞었으면 JSON의 멤버 칸에 섞은 `_full` 예측 파일과 md5도 적힙니다. 13번 재현 점검은 그 파일로 다시 만듭니다.
- 파일 이름 끝 6글자는 설정의 지문입니다. `_full` 설정이 다르면 지문도 다릅니다(Kaggle은 한글을 지우니 이 지문으로 구분).
""")
code(r'''
SAVED_PATH = save(CFG_NOW, NAME, note=NOTE_NOW)
''')

md("## 13. 재현 점검 — 저장한 설정 JSON마다 제출 파일을 다시 만들어 같은지 확인")
code(r'''
_js = sorted(f for f in os.listdir(SAVE_DIR) if f.startswith("submission_") and f.endswith(".json"))
if not _js: print("저장한 설정이 없습니다")
for f in _js: rebuild(os.path.join(SAVE_DIR, f))
''')

md("""
## 14. 제출 파일끼리 비교 — test 답이 다른 문항 수
최종 2개를 고를 때 씁니다. 두 번째 칸은 첫 번째와 다른 문항에서 틀리는 강한 후보여야 보험이 됩니다.
- `COMPARE`를 비우면 다음 파일을 모두 비교합니다.
  - `ens_search`와 `COMPARE_SUBDIRS`(기본 `manual_ens`)에 저장한 제출
  - `REF_SUBMISSIONS`
- 파일 이름 앞의 번호로 표를 읽습니다.
""")
code(r'''
COMPARE = []          # 비교할 제출 파일 (OUT_DIR 기준 경로). 예: ["ens_search/submission_a_1b2c3d.csv", "manual_ens/submission_b_4e5f6a.csv"]
_dirs = [SAVE_SUBDIR] + [d for d in COMPARE_SUBDIRS if os.path.isdir(os.path.join(OUT_DIR, d))]
_files = COMPARE or ([os.path.join(d, f) for d in _dirs for f in sorted(os.listdir(os.path.join(OUT_DIR, d))) if f.startswith("submission_") and f.endswith(".csv")]
                     + list(REF_SUBMISSIONS.values()))
_A = {f: sub_answers(f) for f in _files}; _A = {f: a for f, a in _A.items() if a is not None}
_names = list(_A)
for i, f in enumerate(_names):
    jp = os.path.join(OUT_DIR, f[:-4] + ".json"); acc = ""
    if os.path.exists(jp):
        _j = json.load(open(jp, encoding="utf-8")); _fm = (_j.get("cfg") or {}).get("full") or "none"
        acc = f" | 검증 {_j['val_acc']:.2f}%" + (f" · test에 _full {_fm}" if _fm != "none" else "")
    print(f"[{i}] {f}{acc} | 답 분포 {dict(zip(LETTERS, np.bincount(_A[f], minlength=4).tolist()))}")
if len(_names) >= 2:
    print("\ntest 답이 다른 문항 수 (행 번호 × 열 번호):")
    print(pd.DataFrame([[int((_A[x] != _A[y]).sum()) for y in _names] for x in _names], index=range(len(_names)), columns=range(len(_names))).to_string())
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
