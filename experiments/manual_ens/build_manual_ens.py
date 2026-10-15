# -*- coding: utf-8 -*-
"""직접 가중치 앙상블 노트북 빌더 (CPU).
- 멤버 자동 탐색·데이터·예측 파일 읽기는 E8/E14 노트북 코드를 그대로 씀 (같은 멤버·같은 계열)
- 슬라이더 패널로 실행별 가중치·결합 방법·온도·_full 섞기를 바꾸면 검증 결과가 바로 갱신됨
- 코드 셀로 같은 조합을 자세히 보고 → 제출 파일 + 설정 JSON 저장 → JSON으로 다시 만들기 · 제출 파일끼리 비교
사용: python3 build_manual_ens.py [출력 경로]"""
import sys, json, uuid

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/manual_ens/out/manual_ensemble.ipynb"
SP = "/tmp/vqa_scratch"
_n14 = {}
_argv = sys.argv; sys.argv = ["build_e14.py", f"{SP}/_e14_tmp.ipynb"]
exec(open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read(), _n14)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

DATA = _n14["_n8"]["DATA"]
DATA = rep(DATA, 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, MAN_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, MANUAL_SUBDIR); os.makedirs(MAN_DIR, exist_ok=True)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 저장 폴더: {MAN_DIR}")')
HELPERS = _n14["_n8"]["HELPERS"]
POOL = _n14["POOL"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# 직접 가중치 앙상블 (CPU)

**하는 일**
- `vqa_runs`에서 멤버를 E14와 같은 규칙으로 자동으로 찾습니다. 조건은 고정 분할로 학습했고 검증 1,000·test 6,714 예측이 모두 있는 실행입니다.
- 계열(기반 모델)·검증 정확도·온도·"혼자 맞힌 문항" 표를 보여 줍니다.
- **슬라이더 패널(6번 셀)**: 실행별 가중치와 결합 방법·온도·`_full` 섞기를 바꾸면 결과가 바로 갱신됩니다.
  - 검증 정확도와 기준 조합 대비 고침/새로 틀림(부호 검정·부트스트랩 구간)
  - 멤버 하나를 빼거나 넣었을 때의 변화
  - 기존 제출 파일과 test 답이 다른 개수
- 정한 조합은 '제출 파일 저장'으로 `vqa_runs/manual_ens/`에 저장합니다. 제출 CSV와 설정 JSON이 같이 저장되어 JSON만 있으면 똑같이 다시 만들 수 있습니다(재현성).

**주의: 검증 1,000문항으로 가중치를 고르면 그 1,000문항에 맞춰집니다.**
- 기준보다 +1~2문항 좋아진 것은 우연 범위입니다. 부호 검정 p와 '바뀐 답' 수를 같이 보세요.
- 바뀐 답이 적고 순이득이 작으면 공개 점수에서 뒤집힐 수 있습니다.

**실행**: 위에서부터 5번 셀까지 실행 → 6번 패널에서 조정 → 저장. 자세한 보고는 7번 셀(코드로 같은 조합), 제출 파일끼리 비교는 10번 셀입니다. GPU가 필요 없습니다.
""")

md("## 1. 설정")
code(r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"      # 기존 제출 파일 폴더 (읽기만 함)
MANUAL_SUBDIR   = "manual_ens"     # 이 노트북이 제출 파일·설정을 쓰는 폴더
SEED, VAL_SIZE = 42, 1000          # 검증 분할 — 팀 공통값
N_PUBLIC = 3357                    # 공개 채점 문항 수 (test 6,714의 절반)

# ── 멤버 자동 탐색 (E14와 같은 규칙) ──
MUST_INCLUDE = []                  # 반드시 있어야 하는 멤버 [(실행, 해상도)] — 없으면 멈춤. 보통 비워 둠
PRESET_PREF = ["up2.0"]            # 실행마다 이 해상도 예측을 우선 사용
INCLUDE = []                       # 비우면 자동 탐색. 넣으면 이 목록만: [("q35_27b_r16_native", "up2.0"), ...]
EXCLUDE = []                       # 자동 탐색에서 뺄 실행 이름
MIN_VAL_ACC = 90.0                 # 이보다 낮은 실행은 고장으로 보고 뺌
POOL_DELTA = 1.5                   # '강한 풀' 버튼: 상위 3개 평균보다 1.5%p 넘게 낮지 않은 실행
T_GRID = "T70"                     # 멤버별 온도 격자 (0.25~8, 검증 NLL 최소)
FULL_SUFFIX = "_full"              # 전체 데이터(검증 포함) 재학습 실행 = <원래 실행>_full → test에만 섞을 수 있음
FULL_MIN_AGREE = 97.0              # _full의 test 답이 원래 실행과 이 비율(%) 이상 같아야 씀 (학습 고장 감지)

# ── 비교 기준 ──
REF_NAME = "clean4_qg_hcx"         # 지금 제출 레시피 (검증 97.7%, 공개 0.97646). 폴더가 없는 실행은 자동으로 빠짐
REF_CFG = {"weights": {"q35_27b_r16_native": 1, "q38_27b_r16_native": 1, "gemma4_31b_r16_up20_dev": 1,
                       "gemma4_31b_r16_up20": 1, "hcx32b_r16_native": 2},
           "method": "prob", "temp": "none", "full": "none"}
REF_SUBMISSIONS = {"majority_vote (공개 0.97676)": "ensembles/submission_majority_vote.csv",     # test 답을 비교할 기존 제출 (OUT_DIR 기준)
                   "clean4_qg_hcx (공개 0.97646)": "ensembles/submission_clean4_qg_hcx.csv"}
START_FROM = "ref"                 # 슬라이더 시작값: "ref"(기준 조합) / "pool"(강한 풀 모두 1) / 저장한 설정 JSON 경로 (예: "manual_ens/submission_a_1b2c3d.json")
''')

md("## 2. 데이터 (CSV만 꺼냄) · 고정 검증 분할")
code(DATA)

md("## 3. 예측 파일 읽기 함수 (E8·E14와 같음)")
code(HELPERS)

md("## 4. 멤버 자동 탐색 (E14와 같은 규칙)")
code(POOL)

md("""
## 5. 멤버 표 · `_full` 짝 · 조합 함수
- **온도 T**: 검증 NLL을 최소로 하는 값입니다. 1보다 크면 과신, 작으면 소심한 멤버입니다. '멤버별 온도 보정'을 켜면 결합 전에 이 값으로 확률을 폅니다.
- **혼자 맞힘**: 다른 어떤 실행도 못 맞혔는데 이 실행만 맞힌 검증 문항 수입니다.
- **`_full` 짝**: test에만 섞을 수 있습니다. 검증을 학습에 써서 검증으로는 잴 수 없기 때문입니다. 원래 실행과 test 답이 `FULL_MIN_AGREE`% 이상 같은 것만 씁니다.
""")
code(r'''
import datetime
TG = T_GRIDS[T_GRID]
def fit_T_P(P):
    ii = np.arange(len(Y_VAL)); return float(TG[int(np.argmin([-np.mean(np.log(temp_scale(P, T)[ii, Y_VAL])) for T in TG]))])
_vo = {r: v for r, v in VALONLY.items() if v is not None and r not in NAMES}                         # test가 아직 없는 실행도 검증 탐색에는 씀
_vo_low = sorted(r for r, v in _vo.items() if (v[1].argmax(1) == Y_VAL).mean() * 100 < MIN_VAL_ACC)
RUNS = list(NAMES) + sorted(r for r in _vo if r not in _vo_low)
if len(RUNS) > len(NAMES): print(f"검증만 있는 실행 {len(RUNS) - len(NAMES)}개도 패널에 넣음 (검증 탐색만 · 들어가면 저장 불가): {RUNS[len(NAMES):]}")
if _vo_low: print(f"검증만 있는 실행 중 검증 < {MIN_VAL_ACC}% 라서 뺀 것: {_vo_low}")
PVAL = {r: PV[i] for i, r in enumerate(NAMES)}; PVAL.update({r: VALONLY[r][1] for r in RUNS if r not in PVAL})
PTEST = {r: PT[i] for i, r in enumerate(NAMES)}
PRE = dict(MEMBERS_ALL); PRE.update({r: VALONLY[r][0] for r in RUNS if r not in PRE})
FAMR = dict(zip(NAMES, FAM)); FAMR.update({r: model_of(r) for r in RUNS if r not in FAMR})
TT = {r: fit_T_P(PVAL[r]) for r in RUNS}
ACC = {r: float((PVAL[r].argmax(1) == Y_VAL).mean() * 100) for r in RUNS}
CORRECT = {r: PVAL[r].argmax(1) == Y_VAL for r in RUNS}

FULL_OF, FULL_FILE = {}, {}
for r in NAMES:
    fr = r + FULL_SUFFIX
    if not os.path.isdir(os.path.join(OUT_DIR, fr)): continue
    ft, why = member_file(fr, PRE[r], "test")
    if not ft: print(f"  _full {fr}: test 예측 없음 ({why}) → 안 씀"); continue
    try: Pf = load(fr, PRE[r], "test")
    except Exception as e: print(f"  _full {fr}: 읽기 실패 ({str(e)[:60]}) → 안 씀"); continue
    agree = float((Pf.argmax(1) == PTEST[r].argmax(1)).mean() * 100)
    full_ok = bool((read_log(fr).get("config") or {}).get("TRAIN_ON_FULL"))
    ok = agree >= FULL_MIN_AGREE and full_ok
    print(f"  _full {r} → {fr}: test 답 일치 {agree:.1f}%" + (" ✅ 섞을 수 있음" if ok else f" ⚠️ 안 씀 ({'일치율 < ' + str(FULL_MIN_AGREE) if agree < FULL_MIN_AGREE else 'train_log에 TRAIN_ON_FULL 없음'})"))
    if ok: FULL_OF[r], FULL_FILE[r] = Pf, ft

_rows = []
for r in sorted(RUNS, key=lambda r: (short_fam(FAMR[r]), -ACC[r])):
    others = np.sum([CORRECT[q] for q in RUNS if q != r], 0) if len(RUNS) > 1 else np.zeros(len(Y_VAL))
    _rows.append({"실행": r, "계열": short_fam(FAMR[r]), "검증(%)": round(ACC[r], 2), "오답": int((~CORRECT[r]).sum()), "NLL": round(nll(PVAL[r], Y_VAL), 4),
                  "온도 T": round(TT[r], 2), "혼자 맞힘": int((CORRECT[r] & (others == 0)).sum()), "test": "O" if r in PTEST else "없음(검증만)", "_full": "O" if r in FULL_OF else ""})
MEM_TABLE = pd.DataFrame(_rows)
print(MEM_TABLE.to_string(index=False))
_acc_sorted = sorted(ACC[r] for r in NAMES); POOL_REF = float(np.mean(_acc_sorted[-3:])) if _acc_sorted else 0.0
STRONG = [r for r in NAMES if ACC[r] >= POOL_REF - POOL_DELTA - 1e-9]
_anyc = np.any([CORRECT[r] for r in RUNS], 0)
print(f"\n강한 풀 {len(STRONG)}개 (상위 3개 평균 {POOL_REF:.2f}% − {POOL_DELTA}%p 이상) | 어떤 실행이라도 맞힌 검증 문항 {int(_anyc.sum())} = 조합으로 가능한 상한 | 전원 오답 {int((~_anyc).sum())}")
_miss_ref = [r for r in REF_CFG["weights"] if r not in PVAL]
if _miss_ref: print(f"⚠️ 기준 조합 {REF_NAME}의 실행 중 찾지 못한 것: {_miss_ref} → 빼고 비교합니다")
REF_CFG = {**REF_CFG, "weights": {r: w for r, w in REF_CFG["weights"].items() if r in PVAL}}

# ── 조합 함수 ──
METHOD_LABEL = {"prob": "확률 가중 평균", "logprob": "로그확률 가중 평균 (기하 평균)", "vote": "가중 다수결 (동률은 확률 평균)"}
TEMP_LABEL = {"none": "온도 보정 없음", "val": "멤버별 온도 보정 (검증으로 맞춘 T)"}
FULL_LABEL = {"none": "test = 원래 실행", "mix": "test = 원래 + _full 평균", "swap": "test = _full로 교체"}
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}          # E14 조합의 검증 오답 24개 (분류는 사람이 본 결과)
E14_ERR = {k: [f"train_{x}.jpg" for x in v.split()] for k, v in E14_ERR.items()}
_VPOS = {i: k for k, i in enumerate(VAL_IDS)}

def norm_cfg(cfg, warn=True):
    w = {}
    for r, v in (cfg.get("weights") or {}).items():
        if r not in PVAL:
            if warn: print(f"  ⚠️ '{r}'는 멤버 목록에 없어 무시합니다 (이름 오타·검증 예측 없음·EXCLUDE)")
            continue
        if float(v) > 0: w[r] = float(v)
    c = {"weights": w, "method": cfg.get("method", "prob"), "temp": cfg.get("temp", "none"), "full": cfg.get("full", "none")}
    assert c["method"] in METHOD_LABEL and c["temp"] in TEMP_LABEL and c["full"] in FULL_LABEL, c
    return c

def member_P(r, split, c):
    if split == "val": P = PVAL[r]
    else:
        P = PTEST[r]
        if c["full"] != "none" and r in FULL_OF: P = (P + FULL_OF[r]) / 2 if c["full"] == "mix" else FULL_OF[r]
    return temp_scale(P, TT[r]) if c["temp"] == "val" else P

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
        print(f"  test 답 분포 {dict(zip(LETTERS, np.bincount(at, minlength=4).tolist()))}" + (f" · _full을 섞은 실행 {sum(r in FULL_OF for r in c['weights'])}개" if c["full"] != "none" else ""))
        if REF_CFG["weights"] and all(r in PTEST for r in REF_CFG["weights"]):
            print(f"  test 답이 기준 조합과 다른 문항: {int((at != ens(REF_CFG, 'test').argmax(1)).sum())}개")
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

def save(cfg, name="manual"):
    c = norm_cfg(cfg)
    assert c["weights"], "가중치가 모두 0입니다"
    miss = [r for r in c["weights"] if r not in PTEST]
    assert not miss, f"test 예측이 없는 실행이 있어 제출 파일을 만들 수 없습니다: {miss}"
    e = evaluate(c); at = ens(c, "test").argmax(1)
    safe = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(name)).strip("_") or "manual"
    tag = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:6]
    fn = f"submission_{safe}_{tag}.csv"; path = os.path.join(MAN_DIR, fn)
    sub = sub_df[["id"]].assign(answer=[LETTERS[i] for i in at])
    assert len(sub) == len(sub_df) and set(sub.answer) <= set(LETTERS)
    sub.to_csv(path, index=False)
    mem = {}
    for r in c["weights"]:
        fv, ft = _file_of(r, PRE[r], "val"), _file_of(r, PRE[r], "test")          # 실제로 읽은 파일 (탐색 뒤에 새 epoch 파일이 생겨도 헷갈리지 않게)
        mem[r] = {"family": FAMR[r], "preset": PRE[r], "T": TT[r], "val_acc": ACC[r], "val": os.path.relpath(fv, OUT_DIR), "test": os.path.relpath(ft, OUT_DIR),
                  "val_md5": md5file(fv), "test_md5": md5file(ft)}
        if c["full"] != "none" and r in FULL_FILE: mem[r].update({"full_test": os.path.relpath(FULL_FILE[r], OUT_DIR), "full_test_md5": md5file(FULL_FILE[r])})
    info = {"name": str(name), "file": fn, "cfg": c, "val_acc": e["acc"], "val_err": e["n_err"], "val_nll": e["nll"], "ref": REF_NAME,
            "ref_val_acc": REF_EVAL["acc"] if REF_EVAL else None, "members": mem, "T_GRID": T_GRID, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "test_dist": dict(zip(LETTERS, np.bincount(at, minlength=4).tolist()))}
    json.dump(info, open(path[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✅ 저장: {MANUAL_SUBDIR}/{fn} (+ 설정 {fn[:-4]}.json) | 검증 {e['acc']:.2f}% · 오답 {e['n_err']} | test 답 분포 {info['test_dist']}")
    return path

def _file_of(run, pre, split):
    load(run, pre, split); return _CACHE[(run, pre, split)][1]

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
        if c["full"] != "none" and "full_test" in mem[r]:
            Pf = _probs(_load_preds(_fp(mem[r]["full_test"]), TEST_IDS)); P = (P + Pf) / 2 if c["full"] == "mix" else Pf
        Ps.append(temp_scale(P, mem[r]["T"]) if c["temp"] == "val" else P)
    at = _combine(Ps, [c["weights"][r] for r in c["weights"]], c["method"]).argmax(1)
    saved = sub_answers(os.path.join(os.path.dirname(p), info["file"]))
    same = saved is not None and bool((saved == at).all())
    print(f"  {info['file']}: 다시 만든 답이 저장한 제출과 " + ("완전히 같음 ✅" if same else f"다름 ⚠️ ({'제출 파일 없음' if saved is None else str(int((saved != at).sum())) + '개'})")
          + f" | 검증 {info['val_acc']:.2f}% · {METHOD_LABEL[c['method']]} · {TEMP_LABEL[c['temp']]} · {FULL_LABEL[c['full']]}")
    return same
print("✅ 조합 함수 준비 완료")
''')

md("""
## 6. 슬라이더 패널 — 가중치를 직접 조정
- 슬라이더를 놓으면 아래 결과가 갱신됩니다(0 = 뺌, 0.25 단위, 최대 3).
- 위쪽 메뉴에서 결합 방법·온도 보정·test의 `_full` 섞기를 고릅니다.
  - `_full` 섞기는 test 답에만 영향을 주고, 검증 수치는 원래 실행 그대로입니다.
- 버튼
  - **기준 조합**: `REF_CFG`로 되돌립니다.
  - **강한 풀 = 1**: 강한 풀을 모두 1로, 나머지를 0으로 둡니다.
  - **모두 0**
  - **코드로 출력**: 지금 설정을 7번 셀에 붙여 넣을 코드로 출력합니다.
  - **제출 파일 저장**: 지금 설정으로 `manual_ens/`에 CSV와 JSON을 씁니다.
- 결과 맨 아래 **'멤버 하나씩 바꿔 보기'** 표: 지금 조합에서 실행 하나만 바꿨을 때의 고침·새로 틀림입니다.
  - 들어 있는 실행은 빼기·+0.5·−0.5, 안 든 실행은 넣기(가중치 1)를 해 봅니다.
  - 순이득이 큰 줄이 다음에 움직여 볼 슬라이더입니다. 다만 순이득 +1~2는 우연 범위입니다.
- 슬라이더 상태는 노트북에 저장되지 않습니다. 쓸 조합은 저장 버튼이나 7·8번 셀로 파일에 남기세요.
""")
code(r'''
try:
    import ipywidgets as W
    from IPython.display import display
except ImportError:
    W = None; print("ipywidgets가 없어 슬라이더를 띄울 수 없습니다 → 7번 셀에서 WEIGHTS를 코드로 정하세요")
if START_FROM == "pool": START_CFG = {"weights": {r: 1.0 for r in STRONG}, "method": "prob", "temp": "none", "full": "none"}
elif START_FROM == "ref" or not START_FROM: START_CFG = dict(REF_CFG)
else: START_CFG = json.load(open(START_FROM if os.path.isabs(START_FROM) else os.path.join(OUT_DIR, START_FROM), encoding="utf-8"))["cfg"]
if W is not None:
    SL, _rows_ui, _busy = {}, [], {"on": False}
    for r in sorted(RUNS, key=lambda r: (short_fam(FAMR[r]), -ACC[r])):
        SL[r] = W.FloatSlider(value=float(START_CFG["weights"].get(r, 0.0)), min=0.0, max=3.0, step=0.25, continuous_update=False,
                              readout_format=".2f", layout=W.Layout(width="300px"))
        _rows_ui.append(W.HBox([SL[r], W.HTML(f"<code>{r}</code> · {short_fam(FAMR[r])} · 검증 {ACC[r]:.2f}% · T {TT[r]:.2f}"
                                               + (" · <b>test 없음</b>" if r not in PTEST else "") + (" · _full 있음" if r in FULL_OF else ""))]))
    _dl, _ds = W.Layout(width="360px"), {"description_width": "40px"}
    DD_M = W.Dropdown(options=[(v, k) for k, v in METHOD_LABEL.items()], value=START_CFG.get("method", "prob"), description="결합", layout=_dl, style=_ds)
    DD_T = W.Dropdown(options=[(v, k) for k, v in TEMP_LABEL.items()], value=START_CFG.get("temp", "none"), description="온도", layout=_dl, style=_ds)
    DD_F = W.Dropdown(options=[(v, k) for k, v in FULL_LABEL.items()], value=START_CFG.get("full", "none"), description="test", layout=_dl, style=_ds)
    TX = W.Text(value="my_ens", description="저장 이름")
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
            DD_M.value, DD_T.value, DD_F.value = cfg.get("method", DD_M.value), cfg.get("temp", DD_T.value), cfg.get("full", DD_F.value)
        finally: _busy["on"] = False
        ui_refresh()
    def ui_code(_=None):
        c = ui_cfg()
        with MSG_UI:
            MSG_UI.clear_output(); print("# ↓ 7번 셀에 그대로 붙여 넣으세요")
            print("WEIGHTS = {" + ", ".join(f'"{r}": {w:g}' for r, w in sorted(c["weights"].items(), key=lambda t: (-t[1], t[0]))) + "}")
            print(f'METHOD, TEMP, FULL_MODE = "{c["method"]}", "{c["temp"]}", "{c["full"]}"')
    def ui_save(_=None):
        with MSG_UI:
            MSG_UI.clear_output()
            try: save(ui_cfg(), TX.value)
            except AssertionError as ex: print("⚠️", ex)
    B_REF.on_click(lambda _: ui_set(REF_CFG)); B_POOL.on_click(lambda _: ui_set({"weights": {r: 1.0 for r in STRONG}}))
    B_ZERO.on_click(lambda _: ui_set({"weights": {}})); B_CODE.on_click(ui_code); B_SAVE.on_click(ui_save)
    for s in SL.values(): s.observe(ui_refresh, names="value")
    for d in (DD_M, DD_T, DD_F): d.observe(ui_refresh, names="value")
    _wrap = W.Layout(flex_flow="row wrap")
    display(W.VBox([W.HBox([DD_M, DD_T, DD_F], layout=_wrap), W.VBox(_rows_ui), W.HBox([B_REF, B_POOL, B_ZERO], layout=_wrap), W.HBox([TX, B_CODE, B_SAVE], layout=_wrap), MSG_UI, OUT_UI]))
    ui_refresh()
''')

md("""
## 7. 코드로 정한 조합 — 자세한 보고
6번 패널의 '코드로 출력' 결과를 붙여 넣거나 직접 적으세요. 없는 실행 이름을 적으면 경고만 하고 무시합니다.

패널보다 더 보여 주는 것:
- 멤버 하나씩 바꿔 보기(전체)
- 기준 조합과 답이 바뀐 검증 문항 목록
""")
code(r'''
NAME = "my_ens"
WEIGHTS = dict(REF_CFG["weights"])          # 예: {"q35_27b_r16_native": 1, "gemma4_31b_r16_up20": 1.5, ...}
METHOD, TEMP, FULL_MODE = "prob", "none", "none"
CFG_NOW = {"weights": WEIGHTS, "method": METHOD, "temp": TEMP, "full": FULL_MODE}
_ = report(CFG_NOW, NAME)
''')

md("## 8. 7번 셀의 조합을 제출 파일로 저장 (CSV + 설정 JSON → `vqa_runs/manual_ens/`)")
code(r'''
SAVED_PATH = save(CFG_NOW, NAME)
''')

md("## 9. 재현 점검 — 저장한 설정 JSON마다 제출 파일을 다시 만들어 같은지 확인")
code(r'''
_js = sorted(f for f in os.listdir(MAN_DIR) if f.startswith("submission_") and f.endswith(".json"))
if not _js: print("저장한 설정이 없습니다")
for f in _js: rebuild(os.path.join(MAN_DIR, f))
''')

md("""
## 10. 제출 파일끼리 비교 — test 답이 다른 문항 수
최종 2개를 고를 때 씁니다. 두 번째 칸은 첫 번째와 다른 문항에서 틀리는 강한 후보여야 보험이 됩니다.
- `COMPARE`를 비우면 `manual_ens`에 저장한 제출 전부와 `REF_SUBMISSIONS`를 비교합니다.
- 파일 이름 앞의 번호로 표를 읽습니다.
""")
code(r'''
COMPARE = []          # 비교할 제출 파일 (OUT_DIR 기준 경로). 예: ["manual_ens/submission_a_1b2c3d.csv", "e14_val_ens/submission_e14_....csv"]
_files = COMPARE or ([os.path.join(MANUAL_SUBDIR, f) for f in sorted(os.listdir(MAN_DIR)) if f.startswith("submission_") and f.endswith(".csv")] + list(REF_SUBMISSIONS.values()))
_A = {f: sub_answers(f) for f in _files}; _A = {f: a for f, a in _A.items() if a is not None}
_names = list(_A)
for i, f in enumerate(_names):
    jp = os.path.join(OUT_DIR, f[:-4] + ".json"); acc = ""
    if os.path.exists(jp): acc = f" | 검증 {json.load(open(jp, encoding='utf-8'))['val_acc']:.2f}%"
    print(f"[{i}] {f}{acc} | 답 분포 {dict(zip(LETTERS, np.bincount(_A[f], minlength=4).tolist()))}")
if len(_names) >= 2:
    print("\ntest 답이 다른 문항 수 (행 번호 × 열 번호):")
    print(pd.DataFrame([[int((_A[x] != _A[y]).sum()) for y in _names] for x in _names], index=range(len(_names)), columns=range(len(_names))).to_string())
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
