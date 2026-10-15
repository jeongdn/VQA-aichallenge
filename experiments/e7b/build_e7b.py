# -*- coding: utf-8 -*-
"""E7-b 노트북 빌더 (CPU): 이미 채점한 E7(Gemma 4 31B 제로샷 예/아니오) 점수를, ens_search·manual_ens가 저장한 조합(설정 JSON) 위에 다시 얹는다.
- 조합은 JSON에 적힌 예측 파일·온도·_full 섞기로 다시 만들고, 저장된 제출 파일과 test 답이 같은지 먼저 확인 (ens_search rebuild와 같은 식)
- E7 점수를 처음 기준 조합(clean4)에 다시 섞어 E7 노트북 기록과 같은지 확인 (점수 파일을 제대로 읽었는지)
- E7은 1차 가설(τ 0.5, α 0.25, 부정형 제외) 그대로. 검증: 같은 문항끼리 고침/새로 틀림, 사전 규칙(+3 채택 후보 / +1~+2 공개 확인 / 0 이하 불채택)
- test: 후보 CSV + 설정 JSON 저장 → JSON만으로 다시 만들어 같은지 확인
- 새 조합에서 불확실한데 E7 점수가 없는 문항은 그대로 두고 목록(e7_missing_ids.csv)만 남김
사용: python3 build_e7b.py [출력 경로]"""
import sys, json, io, contextlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e7b/out/e7_on_best.ipynb"
SP = "/tmp/vqa_scratch"
_n14 = {}
_argv = sys.argv; sys.argv = ["build_e14.py", f"{SP}/_e14_tmp_e7b.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read(), _n14)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

DATA = rep(_n14["_n8"]["DATA"], 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, SAVE_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, SAVE_SUBDIR); os.makedirs(SAVE_DIR, exist_ok=True)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 저장 폴더: {SAVE_DIR}")')
HELPERS = _n14["_n8"]["HELPERS"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# E7-b · 이미 채점한 E7(Gemma 예/아니오 검증)을 지금 조합 위에 다시 얹기 (CPU, 몇 분)

**하는 일**
- E7 점수는 이렇게 만들었습니다.
  - 처음 기준 조합(clean4)에서 1·2위 확률 차가 작은 문항만 골랐습니다.
  - Gemma 4 31B 제로샷에게 보기를 하나씩 "이 후보가 정답입니까? Yes/No"로 물었습니다.
  - 검증 순이득은 +2였고, 공개 점수는 확인하지 않았습니다.
- 이 노트북은 그 점수를 **다시 채점하지 않고** 새 조합(기본: 공개 최고 `ea0e9f`) 위에 같은 규칙으로 섞습니다. GPU가 필요 없습니다.
  1. 조합을 ens_search가 저장한 설정 JSON(예측 파일·온도·`_full` 섞기)으로 다시 만들고, 저장된 제출 파일과 test 답이 같은지 먼저 확인합니다.
  2. E7 점수를 처음 기준 조합(clean4)에 다시 섞어, E7 노트북에 기록된 결과(고침·새로 틀림)가 그대로 나오는지 확인합니다.
  3. 새 조합에서 불확실(1·2위 차 < 0.5)하고 E7 점수가 있는 문항만 (1−α)·p조합 + α·q검증으로 섞습니다. 부정형 질문("~가 아닌 것" 등)은 섞지 않습니다.
- 검증 1,000문항에서 새 조합과 같은 문항끼리 고침·새로 틀림을 세어 판정하고, test 후보 파일을 만듭니다.

**사전 기준 (E7 그대로, 결과를 보기 전에 고정)**
- τ = 0.5, α = 0.25, 부정형 제외입니다. 이 노트북에서 바꾸지 않습니다.
- τ·α 격자 표는 참고로만 보여 줍니다. 표에서 가장 좋은 칸을 고르면 검증셋에 맞춘 것이 됩니다.
- 판정: 순이득 ≥ +3이면 채택 후보, +1~+2면 공개 1회 확인 후보, 0 이하면 불채택입니다.

**E7 점수가 없는 문항**
- E7은 clean4 기준으로 1·2위 차가 0.8보다 작은 문항만 채점했습니다.
- 새 조합에서 불확실한데 점수가 없는 문항은 답을 그대로 둡니다.
- 개수와 목록(`e7_on_best/e7_missing_ids.csv`)만 보여 줍니다.

**결과**: `vqa_runs/e7_on_best/`에만 씁니다. 기존 파일은 읽기만 합니다.

**실행**: 1번 셀의 `BASES`를 확인하고 위에서부터 끝까지 실행하세요. 마지막 셀 출력이 결과 요약입니다.
""")

md("## 1. 설정")
code(r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"          # 기존 제출 파일 폴더 (읽기만 함)
SAVE_SUBDIR  = "e7_on_best"            # 이 노트북이 쓰는 폴더 (제출 후보 CSV + 설정 JSON)
SEED, VAL_SIZE = 42, 1000              # 검증 분할 — 팀 공통값
N_PUBLIC = 3357                        # 공개 채점 문항 수 (test 6,714의 절반)

# ── E7을 얹을 조합: ens_search · manual_ens가 저장한 설정 JSON ──
BASES = ["ea0e9f"]                     # 파일 이름 끝 6자리 해시, 또는 JSON 경로(OUT_DIR 기준). 여러 개면 ["ea0e9f", "b47d7c"]
JSON_DIRS = ["ens_search", "manual_ens"]   # 해시로 찾을 때 뒤지는 폴더

# ── E7 점수 (다시 채점하지 않음) ──
E7_RUN = "e7_yn_gemma_4_31b_it"        # e7_yesno_verify.ipynb가 쓴 폴더 (yn_val__up2.0.csv · yn_test__up2.0.csv · e7_info.json)
PRESET = "up2.0"                       # E7 채점 해상도 (파일 이름에 들어감)

# ── 사전 기준 (E7 1차 가설 그대로 — 바꾸지 않음) ──
TAU, ALPHA = 0.5, 0.25                 # 1·2위 차 < τ인 문항만 (1−α)·p조합 + α·q검증
EXCLUDE_NEG = True                     # 부정형 질문은 섞지 않음
NEG_PAT = r"아닌|않는|없는|틀린|잘못"     # E7·stage2와 같은 부정형 규칙
NET_MIN = 3                            # 순이득 ≥ +3 채택 후보 / +1~+2 공개 1회 확인 / 0 이하 불채택
GRID_TAUS, GRID_ALPHAS = [0.3, 0.5, 0.8], [0.25, 0.5, 1.0]   # 참고 표 (판정에 쓰지 않음)

# ── 재현 확인: E7을 처음 판정한 기준 조합 (clean4_qg_hcx, 가중치 1·1·1·1·2, 확률 평균) ──
E7_REF_MEMBERS = [("q35_27b_r16_native", "up2.0"), ("q38_27b_r16_native", "up2.0"), ("gemma4_31b_r16_up20_dev", "up2.0"),
                  ("gemma4_31b_r16_up20", "up2.0"), ("hcx32b_r16_native", "up2.0")]
E7_REF_WEIGHTS = [1, 1, 1, 1, 2]
''')

md("## 2. 데이터 (CSV만 꺼냄) · 고정 검증 분할")
code(DATA)

md("## 3. 예측 파일 읽기 함수 (E8·E14·ens_search와 같음)")
code(HELPERS)

md("""
## 4. E7 점수 읽기 → 처음 기준 조합(clean4)에서 E7 결과 재현 확인
- 보기마다 s = log P(Yes) − log P(No)이고, q = softmax(s)가 E7의 보기별 확률입니다.
- clean4에 다시 섞은 결과가 E7 노트북 기록(고침·새로 틀림·순이득·바뀐 답)과 같으면 점수 파일을 제대로 읽은 것입니다.
""")
code(r'''
E7_DIR = os.path.join(OUT_DIR, E7_RUN)
assert os.path.isdir(E7_DIR), f"E7 결과 폴더가 없습니다: {E7_DIR} → 1번 셀 E7_RUN을 확인하세요"
_ip = os.path.join(E7_DIR, "e7_info.json")
E7_INFO = json.load(open(_ip, encoding="utf-8")) if os.path.exists(_ip) else {}
def read_yn(tag):
    p = os.path.join(E7_DIR, f"yn_{tag}__{PRESET}.csv")
    assert os.path.exists(p), f"E7 {tag} 점수 파일이 없습니다: {p}"
    return pd.read_csv(p).drop_duplicates(["id", "opt"], keep="last"), p
def q_matrix(yn, ids):
    # 채점 결과 → (문항 수, 4) q = softmax(s). 보기 4개가 다 채점되지 않은 문항은 NaN (E7 노트북과 같은 식)
    pos = {i: n for n, i in enumerate(ids)}; Sm = np.full((len(ids), 4), np.nan)
    for i, k, s in zip(yn.id, yn.opt.astype(int), yn.s):
        if i in pos: Sm[pos[i], k] = s
    ok = ~np.isnan(Sm).any(1); Q = np.full((len(ids), 4), np.nan)
    e = np.exp(Sm[ok] - Sm[ok].max(1, keepdims=True)); Q[ok] = e / e.sum(1, keepdims=True)
    return Q
YN_VAL, YN_VAL_F = read_yn("val"); YN_TEST, YN_TEST_F = read_yn("test")
Q_VAL, Q_TEST = q_matrix(YN_VAL, VAL_IDS), q_matrix(YN_TEST, TEST_IDS)
HAS_VAL, HAS_TEST = ~np.isnan(Q_VAL).any(1), ~np.isnan(Q_TEST).any(1)
NEG_VAL = val_df.question.astype(str).str.contains(NEG_PAT).to_numpy()
NEG_TEST = test_df.question.astype(str).str.contains(NEG_PAT).to_numpy()
YN_MASS = float(pd.concat([YN_VAL, YN_TEST]).yn_mass.mean())
print(f"E7 점수: {E7_INFO.get('model_id', '?')} 제로샷 {E7_INFO.get('precision', '')} · PREFILL={E7_INFO.get('prefill')!r} · 처음 기준 조합 {E7_INFO.get('base', '?')}")
print(f"  채점된 문항: 검증 {int(HAS_VAL.sum())} · test {int(HAS_TEST.sum())} | Yes/No 확률 합 평균 {YN_MASS:.3f}")
assert YN_MASS >= 0.8, "Yes/No 확률 합이 낮습니다 → E7 채점이 잘못됐을 수 있습니다. 위 출력을 확인하세요"

def margin(P):
    s = np.sort(P, axis=1); return s[:, -1] - s[:, -2]
def mix(P, Q, neg, tau=TAU, alpha=ALPHA):
    # 불확실(1·2위 차 < τ)하고 E7 점수가 있는 문항만 (1−α)·P + α·Q로 섞은 뒤 답을 고름. 부정형은 그대로 (E7 노트북과 같은 식)
    sel = (margin(P) < tau) & ~np.isnan(Q).any(1)
    if EXCLUDE_NEG: sel &= ~neg
    Pn = P.copy(); Pn[sel] = (1 - alpha) * P[sel] + alpha * Q[sel]
    return Pn.argmax(1), sel

_rec = (E7_INFO.get(f"val_{PRESET}") or {}).get("primary")
try:
    _P0 = ens_scores(E7_REF_MEMBERS, E7_REF_WEIGHTS, "prob", split="val")
    _a0 = _P0.argmax(1); _a1, _ = mix(_P0, Q_VAL, NEG_VAL); _f, _b, _n, _D = paired(_a0, _a1, Y_VAL)
    _same = _rec is not None and [_rec.get(k) for k in ("fixed", "broken", "net", "D")] == [_f, _b, _n, _D]
    print(f"재현 확인 — clean4({(_a0 == Y_VAL).mean() * 100:.2f}%)에 E7: 고침 {_f} · 새로 틀림 {_b} · 순이득 {_n:+d} · 바뀐 답 {_D}"
          + (" = E7 노트북 기록과 같음 ✅" if _same else (f" ⚠️ E7 노트북 기록({_rec})과 다름 → 위 출력을 확인하세요" if _rec else " (E7 노트북 기록이 없어 비교 못 함)")))
except Exception as ex:
    print(f"재현 확인 생략 — clean4 멤버 예측 파일을 못 읽음: {str(ex)[:100]}")
''')

md("""
## 5. 조합 다시 만들기 (설정 JSON 그대로) → 저장된 제출 파일과 같은지 확인
- 검증은 원래 실행의 예측, test는 JSON에 적힌 `_full` 섞기까지 그대로 씁니다. 온도도 JSON에 적힌 값을 씁니다.
- 검증 정확도가 JSON 기록과 같고 test 답이 저장된 제출 파일과 완전히 같아야 다음으로 넘어갑니다.
- 결합이 '다수결'인 조합은 확률이 아니어서 E7을 섞을 수 없어 건너뜁니다.
""")
code(r'''
def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def sub_answers(path):
    if not os.path.exists(path): return None
    s = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
def _combine(Ps, w, method):           # ens_search · manual_ens와 같은 식
    w = np.asarray(w, dtype=float); w = w / w.sum()
    if method == "prob": return sum(wi * p for wi, p in zip(w, Ps))
    s = sum(wi * np.log(np.clip(p, 1e-6, 1.0)) for wi, p in zip(w, Ps)); s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
def find_json(key):
    key = str(key)
    if key.endswith(".json"):
        p = key if os.path.isabs(key) else os.path.join(OUT_DIR, key)
        assert os.path.exists(p), f"설정 JSON이 없습니다: {p}"; return p
    hits = sorted(os.path.join(OUT_DIR, sd, f) for sd in JSON_DIRS if os.path.isdir(os.path.join(OUT_DIR, sd))
                  for f in os.listdir(os.path.join(OUT_DIR, sd)) if f.startswith("submission_") and f.endswith(f"_{key}.json"))
    assert hits, f"'{key}'로 끝나는 설정 JSON이 {JSON_DIRS} 폴더에 없습니다 → BASES에 JSON 경로(OUT_DIR 기준)를 직접 적으세요"
    assert len(hits) == 1, f"'{key}'로 끝나는 설정 JSON이 여러 개입니다: {hits} → BASES에 경로를 직접 적으세요"
    return hits[0]
def rebuild_base(p):
    # 설정 JSON → (정보, 검증 확률, test 확률, 저장된 test 답)
    info = json.load(open(p, encoding="utf-8")); c, mem = info["cfg"], info["members"]
    def _fp(rel): return os.path.join(OUT_DIR, rel)
    runs = list(c["weights"])
    gone = [mem[r][k] for r in runs for k in ("val", "test", "full_test") if k in mem[r] and not os.path.exists(_fp(mem[r][k]))]
    assert not gone, f"{info['file']}: 설정 JSON에 적힌 예측 파일이 없습니다: {gone}"
    moved = [f"{r}:{k}" for r in runs for k in ("val", "test", "full_test") if k in mem[r] and f"{k}_md5" in mem[r] and md5file(_fp(mem[r][k])) != mem[r][f"{k}_md5"]]
    def P_of(r, split):
        P = _probs(_load_preds(_fp(mem[r][split]), VAL_IDS if split == "val" else TEST_IDS))
        if split == "test" and c.get("full", "none") != "none" and "full_test" in mem[r]:
            Pf = _probs(_load_preds(_fp(mem[r]["full_test"]), TEST_IDS)); P = (P + Pf) / 2 if c["full"] == "mix" else Pf
        return temp_scale(P, mem[r]["T"]) if c.get("temp") == "val" else P
    w = [c["weights"][r] for r in runs]
    PV = _combine([P_of(r, "val") for r in runs], w, c["method"]); PT = _combine([P_of(r, "test") for r in runs], w, c["method"])
    return info, PV, PT, sub_answers(os.path.join(os.path.dirname(p), info["file"])), moved

BASE = {}
for key in BASES:
    p = find_json(key); info = json.load(open(p, encoding="utf-8")); c = info["cfg"]
    tag = os.path.basename(p)[:-5].rsplit("_", 1)[-1]; name = info.get("name", tag)
    if c.get("method") not in ("prob", "logprob"):
        print(f"⚠️ {name} ({tag}): 결합 방법 '{c.get('method')}'은 확률이 아니어서 E7을 섞을 수 없습니다 → 건너뜀"); continue
    info, PV_B, PT_B, saved, moved = rebuild_base(p)
    acc = float((PV_B.argmax(1) == Y_VAL).mean() * 100)
    same_v = abs(acc - float(info["val_acc"])) < 1e-6
    same_t = saved is not None and bool((saved == PT_B.argmax(1)).all())
    print(f"▶ {name} ({tag}) | {c['method']} · 온도 {c.get('temp', 'none')} · _full {c.get('full', 'none')} | 실행 {len(c['weights'])}개: "
          + " + ".join(f"{r}×{w:g}" for r, w in c["weights"].items()))
    print(f"  검증 {acc:.2f}% (JSON 기록 {float(info['val_acc']):.2f}%) {'✅' if same_v else '⚠️'} | test 답이 저장된 제출 {info['file']}과 "
          + ("완전히 같음 ✅" if same_t else ("⚠️ 제출 파일이 없음" if saved is None else f"⚠️ {int((saved != PT_B.argmax(1)).sum())}개 다름")))
    if moved: print(f"  ⚠️ 저장한 뒤 바뀐 예측 파일: {moved}")
    assert same_v and same_t, f"{name}: 조합을 똑같이 다시 만들지 못했습니다 → 위 출력을 확인하세요"
    BASE[tag] = {"name": name, "json": p, "info": info, "PV": PV_B, "PT": PT_B, "A_TEST": saved}
assert BASE, "E7을 얹을 조합이 없습니다 → 1번 셀 BASES를 확인하세요"
''')

md("""
## 6. 검증 판정 — 새 조합과 같은 문항끼리 비교 (판정은 1차 가설 한 줄로만)
- **분류** 칸은 E14 조합 오답 24개를 사람이 이미지로 본 분류(라벨 오류 / 둘 다 답 / 진짜 실수)입니다.
- 불확실한데 E7 점수가 없는 문항은 섞지 않은 채로 셉니다. test 후보도 같은 규칙으로 만듭니다.
""")
code(r'''
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}          # E14 조합의 검증 오답 24개 (분류는 사람이 본 결과)
CLS = {f"train_{x}.jpg": k for k, v in E14_ERR.items() for x in v.split()}
def verdict_of(n):
    return (f"채택 후보 (순이득 {n:+d} ≥ +{NET_MIN})" if n >= NET_MIN else
            f"공개 1회 확인 후보 (순이득 {n:+d})" if n >= 1 else f"불채택 (순이득 {n:+d})")
LINES_VAL, MISSING_VAL, VAL_RES = [], [], {}
def say(s, to=None): print(s); (LINES_VAL if to is None else to).append(s)
for tag, B in BASE.items():
    PV_B = B["PV"]; a0 = PV_B.argmax(1); a1, sel = mix(PV_B, Q_VAL, NEG_VAL)
    m = margin(PV_B); unc = m < TAU; want = (unc & ~NEG_VAL) if EXCLUDE_NEG else unc
    miss = want & ~HAS_VAL; f, b, n, D = paired(a0, a1, Y_VAL); wrong = a0 != Y_VAL
    VAL_RES[tag] = {"acc": float((a0 == Y_VAL).mean() * 100), "acc_e7": float((a1 == Y_VAL).mean() * 100), "fixed": f, "broken": b, "net": n, "D": D,
                    "mixed": int(sel.sum()), "missing": int(miss.sum()), "verdict": verdict_of(n)}
    say(f"▶ {B['name']} ({tag}) — 검증 {VAL_RES[tag]['acc']:.2f}% → {VAL_RES[tag]['acc_e7']:.2f}% | 고침 {f} · 새로 틀림 {b} · 순이득 {n:+d} · 바뀐 답 {D} | 판정: {VAL_RES[tag]['verdict']}")
    say(f"  불확실(1·2위 차 < {TAU}) {int(unc.sum())}문항: 부정형 {int((unc & NEG_VAL).sum())}{' (안 섞음)' if EXCLUDE_NEG else ''} · 섞은 문항 {int(sel.sum())} · E7 점수 없음 {int(miss.sum())}"
        + f" | 조합 오답 {int(wrong.sum())}개 중 불확실 {int((wrong & unc).sum())} · 확신 오답(이 방법으로 못 고침) {int((wrong & ~unc).sum())}")
    if miss.any(): say(f"  E7 점수 없이 남은 불확실 문항 {int(miss.sum())}개 중 조합 오답 {int((miss & wrong).sum())}개")
    MISSING_VAL += [{"split": "val", "base": tag, "id": VAL_IDS[i], "margin": round(float(m[i]), 4)} for i in np.where(miss)[0]]
    ch = np.where(a0 != a1)[0]
    if len(ch):
        _t = pd.DataFrame([{"id": VAL_IDS[i], "결과": "고침" if a1[i] == Y_VAL[i] else ("새로 틀림" if a0[i] == Y_VAL[i] else "둘 다 틀림"),
                            "정답": LETTERS[Y_VAL[i]], "조합": LETTERS[a0[i]], "E7": LETTERS[a1[i]], "p정답(조합)": round(float(PV_B[i, Y_VAL[i]]), 3),
                            "q정답(E7)": round(float(Q_VAL[i, Y_VAL[i]]), 3), "1·2위 차": round(float(m[i]), 3), "분류": CLS.get(VAL_IDS[i], ""),
                            "질문": str(val_df.question.iloc[i])[:40]} for i in ch])
        print("  답이 바뀐 검증 문항:"); print("    " + _t.to_string(index=False).replace("\n", "\n    "))
    _g = []
    for t in GRID_TAUS:
        for a_ in GRID_ALPHAS:
            x, s_ = mix(PV_B, Q_VAL, NEG_VAL, t, a_); ff, bb, nn, dd = paired(a0, x, Y_VAL)
            _g.append({"τ": t, "α": a_, "섞은 문항": int(s_.sum()), "고침": ff, "새로 틀림": bb, "순이득": nn, "바뀐 답": dd, "1차": "◀" if (t == TAU and a_ == ALPHA) else ""})
    print("  참고 표 (판정에 쓰지 않음):"); print("    " + pd.DataFrame(_g).to_string(index=False).replace("\n", "\n    "))
''')

md("""
## 7. test 후보 만들기 → 저장 → 설정 JSON만으로 다시 만들어 같은지 확인
- 조합의 test 답에서, E7 때문에 바뀐 문항만 바꾼 제출 파일을 만듭니다.
- 파일 이름은 `submission_e7_<조합 해시>_t05_a025.csv`입니다. Kaggle에서도 이름이 그대로 보입니다.
- **공개 점수 읽는 법**
  - 두 제출의 test 답이 D개 다르면, 공개 채점에서는 그중 약 D/2개가 갈립니다.
  - 그래서 운만으로도 ±2√(D/2)문항 정도 오르내립니다. 그보다 작은 차이는 좋다·나쁘다를 가를 수 없습니다.
""")
code(r'''
import datetime
def rebuild_e7(json_path):
    # 후보 JSON → 조합 JSON으로 test 확률을 다시 만들고, 적힌 E7 점수 파일로 섞어서 저장한 후보와 같은지
    meta = json.load(open(json_path, encoding="utf-8"))
    _, _, PT0, _, _ = rebuild_base(os.path.join(OUT_DIR, meta["base_json"]))
    yp = os.path.join(OUT_DIR, meta["yn_test"]); assert md5file(yp) == meta["yn_test_md5"], f"E7 test 점수 파일이 저장 뒤 바뀌었습니다: {yp}"
    Q = q_matrix(pd.read_csv(yp).drop_duplicates(["id", "opt"], keep="last"), TEST_IDS)
    assert meta.get("exclude_neg", True) == EXCLUDE_NEG, "EXCLUDE_NEG 설정이 저장할 때와 다릅니다"
    x, _ = mix(PT0, Q, NEG_TEST, meta["tau"], meta["alpha"])
    saved = sub_answers(os.path.join(os.path.dirname(json_path), meta["file"]))
    return saved is not None and bool((saved == x).all())
CANDS, LINES_TEST, MISSING = [], [], list(MISSING_VAL)
for tag, B in BASE.items():
    PT_B, a0 = B["PT"], B["A_TEST"]; a1, sel = mix(PT_B, Q_TEST, NEG_TEST)
    m = margin(PT_B); unc = m < TAU; want = (unc & ~NEG_TEST) if EXCLUDE_NEG else unc; miss = want & ~HAS_TEST
    D = int((a1 != a0).sum())
    MISSING += [{"split": "test", "base": tag, "id": TEST_IDS[i], "margin": round(float(m[i]), 4)} for i in np.where(miss)[0]]
    fn = (f"submission_e7_{tag}_t{TAU:g}_a{ALPHA:g}".replace(".", "")) + ".csv"; path = os.path.join(SAVE_DIR, fn)
    sub = sub_df[["id"]].assign(answer=[LETTERS[i] for i in a1])
    assert len(sub) == len(sub_df) and set(sub.answer) <= set(LETTERS)
    sub.to_csv(path, index=False)
    meta = {"name": f"{B['name']} + E7", "file": fn, "base_json": os.path.relpath(B["json"], OUT_DIR), "base_file": B["info"]["file"],
            "base_val_acc": VAL_RES[tag]["acc"], "e7_run": E7_RUN, "e7_model": E7_INFO.get("model_id"), "e7_prefill": E7_INFO.get("prefill"),
            "yn_val": os.path.relpath(YN_VAL_F, OUT_DIR), "yn_val_md5": md5file(YN_VAL_F), "yn_test": os.path.relpath(YN_TEST_F, OUT_DIR), "yn_test_md5": md5file(YN_TEST_F),
            "tau": TAU, "alpha": ALPHA, "exclude_neg": EXCLUDE_NEG, "val": VAL_RES[tag],
            "test": {"changed": D, "mixed": int(sel.sum()), "uncertain": int(unc.sum()), "missing": int(miss.sum()),
                     "dist": dict(zip(LETTERS, np.bincount(a1, minlength=4).tolist()))},
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(path[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = rebuild_e7(path[:-4] + ".json")
    band = 2 * math.sqrt(D / 2) if D else 0.0
    say(f"✅ 저장: {SAVE_SUBDIR}/{fn} (+ 설정 JSON) | 재현 {'✅' if ok else '⚠️ 다름'} | {B['name']}({tag})과 test 답이 다른 문항 {D}개"
        + f" (불확실 {int(unc.sum())} · 섞은 문항 {int(sel.sum())} · E7 점수 없음 {int(miss.sum())})", LINES_TEST)
    if D: say(f"  → 공개 채점에서 약 {D / 2:.0f}문항이 갈림: 운만으로도 ±{band:.0f}문항. 공개에서 +{math.ceil(band)}문항 이상이어야 E7 덕분이라고 볼 수 있음", LINES_TEST)
    assert ok, "저장한 후보를 설정 JSON만으로 똑같이 다시 만들지 못했습니다 → 위 출력을 확인하세요"
    CANDS.append((fn, D))
if MISSING:
    mp = os.path.join(SAVE_DIR, "e7_missing_ids.csv"); pd.DataFrame(MISSING).to_csv(mp, index=False)
    print(f"불확실한데 E7 점수가 없는 문항 목록: {SAVE_SUBDIR}/e7_missing_ids.csv (검증 {sum(r['split'] == 'val' for r in MISSING)} · test {sum(r['split'] == 'test' for r in MISSING)}행)")
''')

md("## 8. 결과 요약")
code(r'''
print(f"E7-b | E7 점수 {E7_INFO.get('model_id', '?')} ({E7_RUN}) · τ={TAU} · α={ALPHA} · 부정형 {'제외' if EXCLUDE_NEG else '포함'}")
try: print(f"재현 확인(clean4): 고침 {_f} · 새로 틀림 {_b} · 순이득 {_n:+d} · 바뀐 답 {_D}" + (" ✅" if _same else f" (기록 {_rec})"))
except NameError: print("재현 확인(clean4): 생략")
for _l in LINES_VAL + globals().get("LINES_TEST", []): print(_l)
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
import uuid
for c_ in nb["cells"]: c_["id"] = uuid.uuid4().hex[:8]
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells))
