# -*- coding: utf-8 -*-
"""E13 노트북 빌더 (CPU): 4bit(QLoRA)로 학습한 멤버를 같은 레시피의 bf16 재학습 멤버로 바꿨을 때 좋아지는지 판정.
1) 멤버별: 검증 1,000문항 전체에서 4bit vs bf16 (같은 문항끼리 고침/새로 틀림, 부호 검정 p, NLL)
2) 레시피별: 지금 제출 레시피(clean4 · Q38+G · Q38+GDEV 온도 · 다수결)에서 준비된 bf16 멤버를 모두 바꿔 넣음 → 검증 비교 → 제출 후보
사전 기준(E9까지의 결과를 반영해 bf16 결과를 보기 전에 고정): 검증 순이득 + 공개 순이득(clean4 후보) ≥ +3 → 채택. 검증 순이득 ≤ −3 → 불채택.
  공개 순이득 ≥ 0 조건은 뺌: 지금 최고 점수는 여러 제출 중 고른 값이라 부풀려져 있어, 그 조건은 좋은 변화도 거의 항상 떨어뜨림.
데이터·예측 파일 읽기 함수는 E8 노트북 코드를 그대로 씀.
"""
import sys, uuid
import nbformat

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e13/out/e13_bf16_swap.ipynb"
SP = "/tmp/vqa_scratch"
_n8 = {}
_argv = sys.argv; sys.argv = ["build_e8_ens.py", f"{SP}/_e8_tmp.ipynb"]
exec(open("/home/user/vqa/e8/build_e8_ens.py", encoding="utf-8").read(), _n8)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)

CFG = r"""
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"       # 지금 제출 파일 폴더 (읽기만 함)
E13_SUBDIR   = "e13_bf16"           # 새 제출 후보를 쓰는 폴더
SEED, VAL_SIZE = 42, 1000
PRESET = "up2.0"

# ── bf16 실행 찾기: 4bit 멤버마다 폴더 이름이 "<4bit 멤버>_bf16" 또는 "<4bit 멤버>_bf16_s<시드>"인 실행을 자동으로 모음 ──
#    (폴더가 없거나 test 예측이 아직 없으면 자동으로 빼고 진행. 시드를 바꾼 추가 실행도 이름만 맞추면 들어감)
BASE_MEMBERS = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "hcx32b_r16_native"]
# 같은 자리에 넣을 레시피가 조금 다른 bf16 실행 (예: dev 의사 라벨을 더해 학습한 Q38). 학습 설정 차이는 표에만 적고 경고하지 않음(모델이 다를 때만 경고)
EXTRA_RUNS = {"q38_27b_r16_native": ["q38_27b_r16_dev_bf16"], "q35_27b_r16_native": ["q35_27b_r16_dev_bf16"]}

# ── 지금 제출 레시피 (E8·E9와 같음) ──
Q35, Q38 = ("q35_27b_r16_native", PRESET), ("q38_27b_r16_native", PRESET)
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", PRESET), ("gemma4_31b_r16_up20", PRESET), ("hcx32b_r16_native", PRESET)
RECIPES = {
    "clean4_qg_hcx":    dict(file="submission_clean4_qg_hcx.csv",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], how="prob", cal=None),
    "ens_q38_gemma4":   dict(file="submission_ens_q38_gemma4.csv",   members=[Q38, G],                 weights=[1, 1],          how="prob", cal=None),
    "ens_q38_gdev_cal": dict(file="submission_ens_q38_gdev_cal.csv", members=[Q38, GDEV],              weights=[1, 1],          how="prob", cal="T70"),
}
VOTE = dict(name="majority_vote", file="submission_majority_vote.csv", voters=["ens_q38_gemma4", "clean4_qg_hcx", "ens_q38_gdev_cal"])
PUBLIC = {"clean4_qg_hcx": 0.97646, "majority_vote": 0.97676}; N_PUBLIC = 3357

# ── 사전 기준 (bf16 결과를 보기 전에 고정) ──
SWAP_GATE = 0      # '교체' 방식에 넣을 bf16 실행: 멤버 검증 순이득 > 0 (딱 0이면 NLL이 좋아졌을 때만)
MIX_GATE  = -5     # '평균' 방식에 넣을 bf16 실행: 멤버 검증 순이득 ≥ −5 (4bit와 확률 평균 = 같은 레시피를 두 번 학습한 시드 앙상블)
NET_MIN   = 3      # 두 방식 중 clean4 검증 순이득이 큰 쪽(동률이면 평균) 하나로 판정:
                   #   검증 순이득 + 공개 순이득(clean4 후보 − 3,278) ≥ +3 → 채택 / 검증 순이득 ≤ −3 → 불채택 (공개 ≥ 0 조건 없음)
MASS_MIN  = 0.9    # bf16 실행의 평균 letter_mass가 이 값보다 낮고 원래 멤버보다도 0.02 넘게 낮으면 경고 (프롬프트·토큰 처리 이상 감지)
"""

DATA = rep(_n8["DATA"], 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, E13_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E13_SUBDIR); os.makedirs(E13_DIR, exist_ok=True)')
HELPERS = _n8["HELPERS"]

MEMBERS = r"""
# ── 1. 멤버별 비교: 검증 1,000문항 전체에서 4bit vs bf16 실행 (같은 문항끼리) ──
from math import comb
def sign_p(f, b):
    # 두 멤버 답이 갈린 문항 중 한쪽만 맞힌 f·b개로 보는 양측 부호 검정 (차이가 우연일 확률)
    n = f + b
    if n == 0: return 1.0
    k = min(f, b); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def short(run): return run.split("_")[0] + ("dev" if run.endswith("_dev") else "")
def _mass(run):
    d = pd.read_csv(member_file(run, PRESET, "val")[0]); return float(d.letter_mass.mean()) if "letter_mass" in d.columns else float("nan")
_pat = {o: re.compile(rf"^{re.escape(o)}_bf16(_s\d+)?$") for o in BASE_MEMBERS}
CAND = {o: sorted(d for d in os.listdir(OUT_DIR) if _pat[o].match(d) and os.path.isdir(os.path.join(OUT_DIR, d))) or [o + "_bf16"] for o in BASE_MEMBERS}
for o in BASE_MEMBERS: CAND[o] += [r for r in EXTRA_RUNS.get(o, []) if r not in CAND[o]]
AVAIL, STATS, ROWS, HEALTH = {}, {}, [], []
for old in BASE_MEMBERS:
    for new in CAND[old]:
        row = {"4bit 멤버": old, "bf16 실행": new}
        cfg = read_log(new).get("config") or {}
        if not os.path.isdir(os.path.join(OUT_DIR, new)): row["상태"] = "폴더 없음 (학습 전)"; ROWS.append(row); continue
        fv, why_v = member_file(new, PRESET, "val"); ft, why_t = member_file(new, PRESET, "test")
        if cfg.get("TRAIN_ON_FULL"): row["상태"] = "⚠️ 전체 데이터 학습 → 비교 불가"; ROWS.append(row); continue
        if fv is None: row["상태"] = f"검증 예측 없음 ({why_v}) → 학습·평가 중"; ROWS.append(row); continue
        if ft is None: row["상태"] = f"test 예측 없음 ({why_t}) → MAKE_SUBMISSION=True로 12번 셀까지"; ROWS.append(row); continue
        try: Pn, Ptn = load(new, PRESET, "val"), load(new, PRESET, "test")
        except Exception as e:
            row["상태"] = ("test 예측 진행 중 (일부 문항만 있음) → 12번 셀이 끝나면 다시 실행" if "없는 문항" in str(e) and "_test__" in str(e)
                          else f"⚠️ 읽기 실패: {str(e)[:60]}"); ROWS.append(row); continue
        try: Po, Pto = load(old, PRESET, "val"), load(old, PRESET, "test")
        except Exception as e: row["상태"] = f"⚠️ 원래 멤버 읽기 실패: {str(e)[:60]}"; ROWS.append(row); continue
        ao, an = Po.argmax(1), Pn.argmax(1); f, b, n, D = paired(ao, an, Y_VAL); lo, ln = nll(Po, Y_VAL), nll(Pn, Y_VAL)
        mass, mass_o = _mass(new), _mass(old); cfg_o = read_log(old).get("config") or {}
        diff_cfg = [k for k in ("MODEL_ID", "TRAIN_PRESET", "EPOCHS", "LR", "LORA_R", "LORA_ALPHA", "EXTRA_TRAIN_CSV") if cfg_val(cfg, k) != cfg_val(cfg_o, k)]
        g_sw = n > SWAP_GATE or (n == SWAP_GATE and ln < lo); g_mx = n >= MIX_GATE
        row.update({"상태": "✅", "4bit 검증(%)": round(float((ao == Y_VAL).mean() * 100), 2), "bf16 검증(%)": round(float((an == Y_VAL).mean() * 100), 2),
                    "고침": f, "새로 틀림": b, "순이득": n, "바뀐 답": D, "부호 검정 p": (lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")(sign_p(f, b)), "NLL 4bit→bf16": f"{lo:.4f}→{ln:.4f}",
                    "test 바뀐 답": int((Ptn.argmax(1) != Pto.argmax(1)).sum()), "letter_mass 4bit→bf16": f"{mass_o:.3f}→{mass:.3f}",
                    "시드": f"{cfg_val(cfg_o, 'TRAIN_SEED')}→{cfg_val(cfg, 'TRAIN_SEED')}", "레시피 차이": ", ".join(diff_cfg) or "없음",
                    "교체 방식": "넣음" if g_sw else "뺌", "평균 방식": "넣음" if g_mx else "뺌"})
        if mass == mass and mass < MASS_MIN and not (mass_o == mass_o and mass >= mass_o - 0.02):
            HEALTH.append(f"{new}: letter_mass {mass_o:.3f}→{mass:.3f} (보기 글자 밖으로 확률이 샘)"); row["상태"] = "⚠️"
        if D == 0: HEALTH.append(f"{new}: 검증 답이 원래 멤버와 1,000개 모두 같음 (같은 어댑터를 평가했을 가능성)"); row["상태"] = "⚠️"
        warn_cfg = [k for k in diff_cfg if k == "MODEL_ID"] if new in EXTRA_RUNS.get(old, []) else diff_cfg
        if warn_cfg: HEALTH.append(f"{new}: 원래 멤버와 학습 설정이 다름 ({', '.join(warn_cfg)})"); row["상태"] = "⚠️"
        AVAIL.setdefault(old, []).append(new); STATS[new] = dict(old=old, net=n, sw=g_sw, mx=g_mx)
        ROWS.append(row)
MEM_TABLE = pd.DataFrame(ROWS)
for c in ("고침", "새로 틀림", "순이득", "바뀐 답", "test 바뀐 답"):
    if c in MEM_TABLE: MEM_TABLE[c] = MEM_TABLE[c].map(lambda v: "" if pd.isna(v) else int(v))
MEM_TABLE = MEM_TABLE.fillna("")
def show_members():
    ready = MEM_TABLE[MEM_TABLE["bf16 실행"].isin([n for v in AVAIL.values() for n in v])]
    if len(ready): print(ready.drop(columns=["상태"]).to_string(index=False))
    for _, r in MEM_TABLE[~MEM_TABLE.index.isin(ready.index)].iterrows(): print(f"  - {r['bf16 실행']}: {r['상태']}")
    for _, r in ready[ready["상태"] != "✅"].iterrows(): print(f"  - {r['bf16 실행']}: ⚠️ 아래 '확인할 것' 참고")
show_members()
print(f"\n쓸 수 있는 bf16 실행: {sum(len(v) for v in AVAIL.values())}개 {[n for v in AVAIL.values() for n in v]}")
"""

JUDGE = r"""
# ── 2. 레시피별: 두 방식(교체 / 평균)으로 bf16 실행을 넣은 같은 레시피 → 검증 비교 → 판정 → 제출 후보 ──
#    교체: 4bit 멤버 확률 대신 (교체 기준을 넘은) bf16 실행들의 확률 평균
#    평균: 4bit 멤버 확률과 (평균 기준을 넘은) bf16 실행들의 확률을 똑같은 비중으로 평균
#    레시피의 가중치·평균 방식은 그대로, 온도 보정 레시피는 바뀐 멤버 확률로 온도를 다시 맞춤(원래 레시피와 같은 규칙)
V = {"lines": [], "cands": [], "variants": {}}
def say(s): print(s); V["lines"].append(s)
_PC, _TC = {}, {}
def mem_P(run, pre, split, var):
    kind, m = var; key = (run, pre, split, kind, tuple(m.get(run, ())))
    if key not in _PC:
        if run in m and kind == "swap": P = np.mean([load(n, pre, split) for n in m[run]], 0)
        elif run in m and kind == "mix": P = np.mean([load(run, pre, split)] + [load(n, pre, split) for n in m[run]], 0)
        else: P = load(run, pre, split)
        _PC[key] = P
    return _PC[key]
def T_of(run, pre, var, grid):
    kind, m = var; key = (run, pre, kind, tuple(m.get(run, ())), grid)
    if key not in _TC:
        p, i = mem_P(run, pre, "val", var), np.arange(len(Y_VAL))
        _TC[key] = float(T_GRIDS[grid][int(np.argmin([-np.mean(np.log(temp_scale(p, T)[i, Y_VAL])) for T in T_GRIDS[grid]]))])
    return _TC[key]
def recipe_answers(var):
    A = {}
    for nm, r in RECIPES.items():
        w = np.asarray(r["weights"], dtype=float); w = w / w.sum(); A[nm] = {}
        for s in ("val", "test"):
            out = 0.0
            for wi, (run, pre) in zip(w, r["members"]):
                P = mem_P(run, pre, s, var)
                if r["cal"]: P = temp_scale(P, T_of(run, pre, var, r["cal"]))
                out = out + wi * (P if r["how"] == "prob" else np.log(np.clip(P, 1e-12, 1.0)))
            A[nm][s] = out.argmax(1)
    A[VOTE["name"]] = {s: vote_of([A[v][s] for v in VOTE["voters"]]) for s in ("val", "test")}
    return A
def vtag(kind, m): return f"e13_{kind}_" + "_".join(short(o) + (f"x{len(m[o])}" if len(m[o]) > 1 else "") for o in sorted(m))
A0 = recipe_answers(("swap", {}))
for nm in ("clean4_qg_hcx", VOTE["name"]):
    fa = file_answers(RECIPES[nm]["file"] if nm in RECIPES else VOTE["file"])
    print(f"  기준 {nm}: 검증 {(A0[nm]['val'] == Y_VAL).mean() * 100:.2f}%" + ("" if fa is None else f" | 제출 파일 재현 {int((fa == A0[nm]['test']).sum()):,}/{len(TEST_IDS):,}"))
SW = {o: [n for n in v if STATS[n]["sw"]] for o, v in AVAIL.items()}; SW = {o: v for o, v in SW.items() if v}
MX = {o: [n for n in v if STATS[n]["mx"]] for o, v in AVAIL.items()}; MX = {o: v for o, v in MX.items() if v}
RES = {}
if not AVAIL:
    say("판정: 대기 — test 예측까지 끝난 bf16 실행이 없습니다. 학습 노트북 12번 셀(test 추론)이 끝나면 다시 실행하세요")
elif not MX and not SW:
    say(f"판정: 불채택 — 준비된 bf16 실행이 모두 멤버 기준(순이득 ≥ {MIX_GATE})을 넘지 못했습니다. 다른 bf16 실행이 끝나면 다시 실행하세요")
else:
    for kind, m, label in (("swap", SW, "교체"), ("mix", MX, "평균")):
        if not m: say(f"[{label}] 기준(순이득 {'> ' + str(SWAP_GATE) if kind == 'swap' else '≥ ' + str(MIX_GATE)})을 넘은 bf16 실행이 없어 건너뜀"); continue
        A1 = recipe_answers((kind, m)); RES[kind] = {"m": m, "A": A1}
        say(f"[{label}] " + ", ".join(f"{o} → {'+'.join(v)}" for o, v in m.items()))
        for nm in list(RECIPES) + [VOTE["name"]]:
            f, b, n, D = paired(A0[nm]["val"], A1[nm]["val"], Y_VAL); RES[kind][nm] = (f, b, n, D)
            say(f"  {nm}: 검증 {(A0[nm]['val'] == Y_VAL).mean() * 100:.2f}% → {(A1[nm]['val'] == Y_VAL).mean() * 100:.2f}% | 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}"
                + ("  (바뀐 답 10개 미만)" if D < 10 else ""))
        V["variants"][kind] = {"members": m, **{nm: RES[kind][nm] for nm in list(RECIPES) + [VOTE["name"]]}}
    if sum(len(v) for v in AVAIL.values()) > 1:                        # 참고: 한 멤버씩만 넣었을 때 (판정에 쓰지 않음)
        for o in AVAIL:
            parts = []
            for kind, m, label in (("swap", SW, "교체"), ("mix", MX, "평균")):
                if o not in m: parts.append(f"{label} 빠짐"); continue
                As = recipe_answers((kind, {o: m[o]}))
                n_c, D_c = paired(A0["clean4_qg_hcx"]["val"], As["clean4_qg_hcx"]["val"], Y_VAL)[2:]; n_v, D_v = paired(A0[VOTE["name"]]["val"], As[VOTE["name"]]["val"], Y_VAL)[2:]
                parts.append(f"{label}: clean4 {n_c:+d}({D_c}) · 다수결 {n_v:+d}({D_v})")
            say(f"  (참고) {o}만: " + " | ".join(parts))
    best = max(RES, key=lambda k: (RES[k]["clean4_qg_hcx"][2], k == "mix")); m = RES[best]["m"]; A1 = RES[best]["A"]
    f1, b1, n1, D1 = RES[best]["clean4_qg_hcx"]; label = {"swap": "교체", "mix": "평균"}[best]
    if HEALTH: say("  ⚠️ 확인할 것 (판정은 그대로 진행): " + "; ".join(HEALTH))
    if n1 <= -3:
        say(f"판정: 불채택 — 더 나은 쪽({label})도 clean4 검증 순이득 {n1:+d} (≤ −3). 공개로 확인할 필요 없음")
    else:
        b0 = count_of(PUBLIC["clean4_qg_hcx"]); need = NET_MIN - n1                  # 공개에서 필요한 순이득 (음수면 그만큼 떨어져도 채택)
        say(f"판정: 공개 확인 — {label} 방식, clean4 검증 순이득 {n1:+d}(고침 {f1}·새로 틀림 {b1}, 바뀐 답 {D1}). 아래 clean4 후보의 공개 점수로 결정합니다")
        tag = vtag(best, m)
        for nm in ("clean4_qg_hcx", VOTE["name"]):
            fa = file_answers(RECIPES[nm]["file"] if nm in RECIPES else VOTE["file"]); base_ans = fa if fa is not None else A0[nm]["test"]
            cand = np.where(A1[nm]["test"] != A0[nm]["test"], A1[nm]["test"], base_ans)    # 이 실험 때문에 바뀐 답만 기존 제출 파일에서 고침
            fn = f"submission_{nm}__{tag}.csv"; sub_df[["id"]].assign(answer=[LETTERS[k] for k in cand]).to_csv(os.path.join(E13_DIR, fn), index=False)
            Dn = int((cand != base_ans).sum()); V["cands"].append((fn, Dn))
            line = f"  제출 후보: {fn} | 기존 {nm}({PUBLIC[nm]:.5f} = {count_of(PUBLIC[nm]):,}문항)과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개)"
            if nm == "clean4_qg_hcx":
                line += (f" | ◀ 판정용: 공개 {score_of(b0 + need):.5f}({b0 + need:,}문항) 이상이면 채택 (검증 {n1:+d} + 공개 ≥ +{NET_MIN}"
                         + (f", 공개가 {-need}문항까지 떨어져도 채택)" if need < 0 else ")"))
            say(line)
        say("  제출 순서: clean4 후보(판정용) → 채택되면 다수결 후보(실제로 쓸 것). 판정은 clean4 후보 한 줄로만 하고, 다른 방식 파일은 만들지 않습니다")
json.dump({"avail": AVAIL, "table": ROWS, "verdict": V["lines"], "cands": V["cands"], "variants": V["variants"]},
          open(os.path.join(E13_DIR, "e13_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
"""

SUMMARY = r"""
print(f"E13 | 4bit → bf16 | 쓸 수 있는 bf16 실행 {sum(len(v) for v in AVAIL.values())}개")
show_members()
for l in V["lines"]: print(l)
"""

INTRO = r"""# E13 · 4bit 멤버 + bf16 재학습 멤버 판정 (CPU, 파일만 읽음)

**왜**: 지금 27B 이상 멤버는 모두 4bit(QLoRA)로 학습됐고, Qwen3.5 계열은 4bit 학습을 권장하지 않습니다(양자화 오차가 평소보다 큼). 같은 레시피로 정밀도만 bf16으로 바꿔 다시 학습한 실행이 나오면, 이 노트북이 그 효과를 재고 제출 후보를 만듭니다.

**하는 일**
1. 멤버별로 검증 1,000문항 **전체**에서 4bit와 bf16 실행을 같은 문항끼리 비교합니다(고침·새로 틀림, 부호 검정 p, NLL). 모델이 통째로 바뀌므로 바뀐 답이 수십 개라, 애매한 문항 몇 개만 바뀌던 TTA·해상도 실험보다 근거가 강합니다. 폴더 이름이 `<4bit 멤버>_bf16` 또는 `<4bit 멤버>_bf16_s<시드>`면 자동으로 찾고, `EXTRA_RUNS`에 적은 실행(예: dev 의사 라벨을 더한 `q38_27b_r16_dev_bf16`)도 같은 자리 후보로 봅니다.
2. 지금 제출 레시피(clean4 · Q38+Gemma · Q38+GDEV 온도 보정 · 다수결)에 bf16 실행을 **두 방식**으로 넣어 검증을 비교합니다(가중치·방식은 그대로).
   - **교체**: 멤버 검증에서 4bit보다 나은 bf16 실행만 4bit 대신 씀
   - **평균**: 4bit와 bf16 확률을 같은 비중으로 평균 (같은 레시피를 두 번 학습한 시드 앙상블. 크게 나빠진 실행(순이득 < −5)만 뺌)
3. clean4 검증 순이득이 큰 방식 **하나만**(동률이면 평균) 판정하고, 기존 제출 파일에서 그 방식 때문에 바뀐 답만 고친 제출 후보를 만듭니다.

**사전 기준 (bf16 결과를 보기 전에 고정)**: clean4 검증 순이득 + clean4 후보의 공개 순이득(3,278 대비) ≥ +3 → 채택, 검증 순이득 ≤ −3 → 불채택. **공개 순이득 ≥ 0 조건은 두지 않습니다.** 지금 최고 점수는 여러 제출 중 공개 1등을 고른 값이라 실제보다 높게 찍혀 있어, 실력이 같은 새 파일도 공개에서 평균 몇 문항 낮게 나옵니다(E8·E9·E10에서 모두 −2~−7). 채택되면 다수결 후보가 새 최고 후보가 됩니다. 경고(학습 설정 차이·letter_mass 등)는 표시만 하고 판정을 막지 않습니다.

**실행**: GPU 없는 세션에서 위에서부터 끝까지. bf16 학습이 하나 끝날 때마다 다시 실행하면 준비된 실행을 자동으로 반영합니다. `vqa_runs/e13_bf16/`에만 씁니다."""

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 데이터 (CSV만) · 고정 검증 분할"), ("code", DATA),
    ("markdown", "## 3. 예측 파일 읽기 함수 (E8과 같음)"), ("code", HELPERS),
    ("markdown", "## 4. 멤버별: 4bit vs bf16 (검증 1,000문항 전체)"), ("code", MEMBERS),
    ("markdown", "## 5. 레시피별: 두 방식(교체·평균)으로 bf16 넣기 → 판정 → 제출 후보"), ("code", JUDGE),
    ("markdown", "## 6. 결과 요약"), ("code", SUMMARY),
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
