# -*- coding: utf-8 -*-
"""'기준 모델(Anchor) + 보조 모델' 노트북 빌더 (CPU).
- 기준 모델 답은 네 조건을 모두 만족할 때만 뒤집음: ① 보조 모델과 답이 갈림 ② 기준 최고 확률 < CONF ③ 기준 1·2위 차 < MARGIN ④ 보조 모델 AGREE개 이상이 같은 다른 보기
- 검증: 5겹 교차 적합(문턱값은 4구역에서 고르고 남은 1구역에서만 채점) · 부트스트랩 1,000번 · 문항 유형별 성적 · 문턱값 일관성
- 합격선(결과를 보기 전에 고정): 순이득(살림 − 망침) ≥ +10 · 살림 ≥ 망침 × 2 · 부트스트랩 95% 이상에서 순이득 > 0 · 구역별 문턱값이 비슷함
- 확신도 조율: 모델별 온도(T)를 절반에서 맞추고 나머지 절반에서 [확률 평균] vs [온도 보정 평균] vs [순위 투표(Borda)] vs [마진 가중] 비교 + ECE
멤버 탐색·예측 파일 읽기는 E14·ens_search와 같은 코드.
사용: python3 build_anchor.py [출력 경로]"""
import sys, json, uuid, os, io, contextlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/anchor/out/anchor_rescue.ipynb"
SP = "/tmp/vqa_scratch"
_n14 = {}
_argv = sys.argv; sys.argv = ["build_e14.py", f"{SP}/_e14_tmp_anchor.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read(), _n14)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)
DATA = rep(_n14["_n8"]["DATA"], 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)',
           'ENS_DIR, SAVE_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, SAVE_SUBDIR); os.makedirs(SAVE_DIR, exist_ok=True)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 저장 폴더: {SAVE_DIR}")')
HELPERS, POOL = _n14["_n8"]["HELPERS"], _n14["POOL"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# 기준 모델(Anchor) + 보조 모델 — 교체 규칙 · 교차 적합 검증 · 확신도 조율 (CPU)

**기준 모델** = `ANCHOR` (기본: Flash-Next 학습 모델 `q38flash_r16_native_bf16`)
- 기준 모델의 답은 **네 조건을 모두 만족할 때만** 뒤집습니다.
  1. **불일치**: 보조 모델이 기준과 다른 보기를 고름
  2. **확신도**: 기준 모델의 최고 확률 < `CONF`
  3. **마진**: 기준 모델의 1·2위 확률 차 < `MARGIN`
  4. **다수 동의**: 보조 모델 `AGREE`개 이상이 **같은** 다른 보기를 가리킴 → 그 보기로 바꿈
- 기준 모델 확률은 온도 보정(T)을 한 뒤에 문턱값과 비교합니다. T는 문턱값을 고르는 구역에서만 맞춥니다.

**보조 모델** = `RESCUERS`
- 기본 `"auto"`: 계열(기반 모델)마다 검증 최고 실행 하나씩 뽑아 상위 `RESCUER_K`개 계열을 씁니다. 기준 모델 계열은 뺍니다.
- 여기에 zero-shot 모델(122B, RunPod `runpod_zs_any`로 돌린 모델 등, 검증 `ZS_MIN_ACC` 이상)을 모델마다 1개씩 더합니다. 학습 멤버와 같은 계열(예: Gemma 학습 + Gemma zero-shot)이어도 넣습니다. 학습한 모델과 학습하지 않은 대형 모델은 틀리는 문항이 달라 서로 보완합니다.
- **판정 모델**(교차 검증 프롬프트, RunPod `runpod_xcheck_any`): 6-1번 셀이 만든 판정 목록(기준과 보조 모델의 답이 갈린 문항)만 다시 따져 본 모델입니다. 돌린 판정 모델은 모두(`XCHECK_RUNS="auto"`) 한 표씩 더합니다. 목록 밖 문항에서는 표를 내지 않습니다.
- `AGREE` 후보는 표 수에 맞춰 2 ~ 표 수(최대 8)로 넓힙니다.

**검증 (과적합을 막는 절차 — 결과를 보기 전에 고정)**
- 5겹 교차 적합: 4구역에서 문턱값(CONF × MARGIN × AGREE 격자)을 고르고, 남은 1구역에서만 살림/망침을 셉니다. 이것을 5번 반복합니다.
  - 고르는 규칙: 순이득 최대. 단 살림 ≥ 망침 × 2인 칸만 씁니다.
- 부트스트랩 1,000번: 교차 적합 결과에서 순이득 > 0인 비율을 봅니다.
- 문항 유형별(가격·이름·위치·날짜·부정형…) 살림/망침을 봅니다.
- 5구역에서 고른 문턱값이 비슷한지 봅니다.
- **합격선**: 순이득 ≥ +10 · 살림 ≥ 망침 × 2 · 부트스트랩 95% 이상 · 문턱값 일관 → 채택

**확신도 조율 (별도 비교)**
- 모델마다 T를 검증 절반에서 맞추고, 나머지 절반에서 방법 4가지의 정확도를 비교합니다. 절반을 바꿔 두 번 합니다.
  - [확률 평균] · [온도 보정 평균] · [순위 투표(Borda: 1등 3점 … 4등 0점)] · [마진 가중 평균]
- ECE(90%라고 한 문항을 실제로 90% 맞히는지)도 모델별로 보입니다.

**검증셋**: 학습 멤버의 예측이 모두 있는 팀 공통 1,000문항입니다. 결과는 `vqa_runs/anchor_rescue/`에만 씁니다. GPU는 필요 없습니다.
""")

md("## 1. 설정")
code(r'''
# ── 경로 (다른 노트북과 같음) ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"
SAVE_SUBDIR  = "anchor_rescue"
SEED, VAL_SIZE = 42, 1000
N_PUBLIC = 3357

# ── 기준 모델 · 보조 모델 ──
ANCHOR    = "q38flash_r16_native_bf16"   # 기준 모델 = Flash-Next 학습 모델 (검증 96.9 · 공개 0.97110)
RESCUERS  = "auto"                       # "auto" 또는 실행 이름 목록. 예: ["gemma4_31b_r16_up20_ocr_bf16", "q35_27b_r16_e4aux20_native", "q35_122b_a10b_gptq_zs"]
RESCUER_K = 4                            # auto: 학습 멤버 중 계열별 최고을 검증 상위 몇 계열까지
ZS_MIN_ACC = 95.5                        # auto: zero-shot 실행(train_log ZERO_SHOT)은 검증 이 값 이상이면 모델마다 1개씩 추가 (학습 멤버와 같은 계열이어도)
XCHECK_RUNS = "auto"                     # 판정 모델 실행(RunPod runpod_xcheck_any, train_log XCHECK): "auto" = 있는 것 전부 한 표씩 / [] = 안 씀 / 실행 이름 목록
REF_JSON  = "ens_search/submission_my_search_ea0e9f.json"   # 비교용(참고 한 줄): 지금 공개 최고 조합. 없으면 건너뜀

# ── 판정 목록 (교차 검증 프롬프트 대상: 기준과 보조 모델의 답이 갈린 문항) → HF 비공개 저장소 xcheck/xcheck_targets.csv ──
UPLOAD_TARGETS = True                    # RunPod 판정 노트북이 여기서 받아 감. False면 Drive에만 저장
HF_REPO   = ""                           # 비워 두면 "<내 HF 아이디>/ssafy-vqa-private" (브리지 노트북과 같은 규칙)
HF_TOKEN  = ""                           # 비워 두면 Colab 보안 비밀 'HF_TOKEN' → 없으면 입력창

# ── 멤버 탐색 (E14 · ens_search와 같은 규칙) ──
MUST_INCLUDE = []
PRESET_PREF = ["up2.0"]
INCLUDE, EXCLUDE = [], []
MIN_VAL_ACC = 90.0
FULL_TOKEN, SUSPECT_ACC = "full", 99.0   # 전체 데이터 학습(_full)·검증 99% 초과 실행은 뺌 (검증을 학습에 썼음)
T_GRID = "T70"

# ── 교체 규칙: 문턱값 격자 (교차 적합으로 고름) ──
CONF_GRID   = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]        # 기준 최고 확률 < CONF (1.01 = 조건 없음)
MARGIN_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]   # 기준 1·2위 차 < MARGIN
AGREE_GRID  = "auto"                                 # 같은 다른 보기를 고른 보조 모델(판정 모델 포함) 수 ≥ AGREE. "auto" = 2 ~ 보조 모델 수(최대 8)
CALIBRATE_ANCHOR = True                              # 기준 확률에 온도 보정(T) 후 문턱값 적용
K_FOLDS, CV_SEED, N_BOOT, BOOT_SEED = 5, 0, 1000, 0
# ── 합격선 (결과를 보기 전에 고정) ──
MIN_NET, RATIO, BOOT_MIN = 10, 2.0, 0.95
CONSIST_MARGIN, CONSIST_CONF = 0.2, 0.2              # 5구역에서 고른 MARGIN·CONF의 최대−최소가 이 이하 + AGREE가 4구역 이상 같으면 '일관'
''')

md("## 2. 데이터 (CSV만) · 고정 검증 분할")
code(DATA)
md("## 3. 예측 파일 읽기 함수 (E14 · ens_search와 같음)")
code(HELPERS)
md("## 4. 멤버 탐색 (E14와 같은 규칙)")
code(POOL)

md("""
## 5. 기준 모델 · 보조 모델 정하기
- 멤버 표에는 검증 정확도, 온도 T, ECE(보정 전·후)가 나옵니다.
- 보조 모델 각각이 기준 모델이 틀린 문항 중 몇 개를 맞히는지도 보입니다.
""")
code(r'''
import math, datetime, itertools
LINES = []
def say(s): print(s); LINES.append(s)
CATS = {"전화번호": r"전화|번호|연락처", "가격/금액": r"가격|얼마|금액|요금|할인", "이름/상호/제목": r"이름|상호|제목|브랜드|가게|매장|간판|업체",
        "날짜/시간": r"날짜|언제|시간|기간|요일|몇 시|년도|연도", "위치/방향/층": r"어디|위치|방향|몇 층|층에|왼쪽|오른쪽|위쪽|아래",
        "개수/숫자": r"몇 개|몇 명|몇 번|개수|몇 가지|숫자", "색상": r"색|컬러", "부정형": r"아닌|않는|없는|틀린|잘못"}   # stage2와 같은 문항 유형
def sub_answers(path):
    if not os.path.exists(path): return None
    s = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id").reindex(TEST_IDS)["answer"]
    s = s.astype(str).str.strip().str.lower().map({L: i for i, L in enumerate(LETTERS)})
    return None if s.isna().any() else s.to_numpy().astype(int)
def is_full(r): return FULL_TOKEN in str(r).split("_")
ACC = {r: float((PV[i].argmax(1) == Y_VAL).mean() * 100) for i, r in enumerate(NAMES)}
FAMR = dict(zip(NAMES, FAM))
_CFG = {r: (read_log(r).get("config") or {}) for r in NAMES}
XC = {r for r in NAMES if _CFG[r].get("XCHECK")}                                   # 판정 모델 실행 (교차 검증 프롬프트)
ZS = {r for r in NAMES if _CFG[r].get("ZERO_SHOT") and r not in XC}
USABLE = [r for r in NAMES if not is_full(r) and ACC[r] <= SUSPECT_ACC]
assert ANCHOR in USABLE, f"기준 모델 {ANCHOR}를 찾지 못했습니다 (검증·test 예측이 둘 다 있어야 함). 쓸 수 있는 실행: {sorted(USABLE)}"
if RESCUERS == "auto":
    best = {}
    for r in USABLE:
        if r == ANCHOR or FAMR[r] == FAMR[ANCHOR] or r in ZS or r in XC: continue
        if FAMR[r] not in best or ACC[r] > ACC[best[FAMR[r]]]: best[FAMR[r]] = r
    RESC = [r for r in sorted(best.values(), key=lambda r: -ACC[r])][:RESCUER_K]
    zb = {}
    for r in USABLE:
        if r in ZS and r != ANCHOR and ACC[r] >= ZS_MIN_ACC and (FAMR[r] not in zb or ACC[r] > ACC[zb[FAMR[r]]]): zb[FAMR[r]] = r
    RESC += [r for r in sorted(zb.values(), key=lambda r: -ACC[r])]          # 학습 멤버와 같은 계열이어도 넣음 (학습 안 한 모델은 틀리는 문항이 다름)
else:
    miss = [r for r in RESCUERS if r not in USABLE]
    assert not miss, f"보조 모델 중 찾지 못한 실행: {miss} (쓸 수 있는 실행: {sorted(USABLE)})"
    RESC = [r for r in RESCUERS if r != ANCHOR]
RESC_BASE = [r for r in RESC if r not in XC]                                     # 판정 목록은 이 보조 모델들로만 정함 (판정 결과와 무관하게 고정)
assert len(RESC_BASE) >= 2, f"보조 모델(판정 모델 제외)가 2개 이상이어야 합니다: {RESC_BASE}"
if XCHECK_RUNS == "auto":
    XCR = sorted(r for r in USABLE if r in XC)
else:
    XCR = list(XCHECK_RUNS); _m = [r for r in XCR if r not in XC or r not in USABLE]
    assert not _m, f"판정 모델 실행이 아니거나 찾지 못함: {_m} (판정 실행: {sorted(XC)})"
_wrong = [r for r in XCR if _CFG[r].get("XCHECK_ANCHOR") not in (None, ANCHOR)]
if _wrong: print(f"⚠️ 다른 기준 모델로 만든 판정 목록의 판정 실행은 뺍니다: {[(r, _CFG[r].get('XCHECK_ANCHOR')) for r in _wrong]}")
XCR = [r for r in XCR if r not in _wrong]
RESC = RESC_BASE + [r for r in XCR if r not in RESC_BASE]
IDX = {r: NAMES.index(r) for r in [ANCHOR] + RESC}
PA_V, PA_T = PV[IDX[ANCHOR]], PT[IDX[ANCHOR]]
RV = np.stack([PV[IDX[r]] for r in RESC]); RT = np.stack([PT[IDX[r]] for r in RESC])       # (구원 수, 문항, 4)
NV = len(Y_VAL)
def fit_T_on(P, idx):
    TG = T_GRIDS[T_GRID]; y = Y_VAL[idx]; ii = np.arange(len(idx))
    return float(TG[int(np.argmin([-np.mean(np.log(temp_scale(P[idx], T)[ii, y])) for T in TG]))])
def ece(P, y, bins=10):
    conf, pred = P.max(1), P.argmax(1); e = 0.0
    for b in range(bins):
        s = (conf > b / bins) & (conf <= (b + 1) / bins)
        if s.any(): e += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return float(e)
ALL = np.arange(NV)
aw = PA_V.argmax(1) != Y_VAL
rows = []
for r in [ANCHOR] + RESC:
    P = PV[IDX[r]]; T = fit_T_on(P, ALL)
    rows.append({"역할": "기준" if r == ANCHOR else ("판정" if r in XC else "구원"), "실행": r, "계열": short_fam(FAMR[r]), "zero-shot": "O" if r in ZS or r in XC else "", "검증(%)": round(ACC[r], 2),
                 "온도 T": round(T, 2), "ECE 전": round(ece(P, Y_VAL), 4), "ECE 후": round(ece(temp_scale(P, T), Y_VAL), 4),
                 "기준 오답 중 맞힘": "" if r == ANCHOR else f"{int((aw & (P.argmax(1) == Y_VAL)).sum())}/{int(aw.sum())}"})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n기준 {ANCHOR}: 검증 {ACC[ANCHOR]:.2f}% (오답 {int(aw.sum())}) | 보조 모델 {len(RESC_BASE)}개: {RESC_BASE} | 판정 모델 {len(XCR)}개: {XCR or '없음'}")
if XCR: print("  (판정 모델의 검증 %는 판정 대상이 아닌 문항을 기준 답으로 채운 값이라 참고만 하세요. 실력은 6-1번 표)")
''')

md("""
## 6. 교체 규칙 함수 · 전체 검증 격자 (참고)
- 아래 격자는 검증 1,000문항 전체로 본 값이라 참고용입니다. 여기서 가장 좋은 칸을 고르면 과적합입니다. 판정은 7번의 교차 적합으로 합니다.
""")
code(r'''
def rescue_votes(RP):
    # 보조 모델들의 답 → 보기별 표 수 (문항, 4) · 확률 합 (동점 가르기용)
    votes = np.zeros(RP.shape[1:]); psum = RP.sum(0)
    for m in range(RP.shape[0]): votes[np.arange(RP.shape[1]), RP[m].argmax(1)] += 1
    return votes, psum
VOTES_V, PSUM_V = rescue_votes(RV); VOTES_T, PSUM_T = rescue_votes(RT)
def apply_rule(PA, votes, psum, conf, marg, agree):
    # 기준 확률 PA(보정 후) → (새 답, 뒤집은 문항)
    a = PA.argmax(1); s = np.sort(PA, 1); pmax, m = s[:, -1], s[:, -1] - s[:, -2]
    v = votes.copy(); v[np.arange(len(a)), a] = -1                        # 기준 답을 뺀 다른 보기 중
    key = v + 1e-3 * psum / max(psum.max(), 1e-9); alt = key.argmax(1); cnt = v[np.arange(len(a)), alt]
    flip = (cnt >= agree) & (pmax < conf) & (m < marg)
    return np.where(flip, alt, a), flip
def count(a0, a1, flip, y, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    r = int((flip[idx] & (a0[idx] != y[idx]) & (a1[idx] == y[idx])).sum()); h = int((flip[idx] & (a0[idx] == y[idx])).sum())
    return r, h, r - h, int(flip[idx].sum())
AGREE_USE = list(range(2, min(len(RESC), 8) + 1)) if AGREE_GRID == "auto" else list(AGREE_GRID)
GRID = list(itertools.product(CONF_GRID, MARGIN_GRID, AGREE_USE))
def best_cell(PA, idx):
    # 주어진 문항(idx)에서 순이득 최대 (살림 ≥ 망침 × RATIO인 칸만). 없으면 None (= 뒤집지 않음)
    a0 = PA.argmax(1); bestv, bestc = (0, 0, 0), None
    for c in GRID:
        a1, fl = apply_rule(PA, VOTES_V, PSUM_V, *c); r, h, n, _ = count(a0, a1, fl, Y_VAL, idx)
        if r >= RATIO * h and r > 0 and (n, -h, -c[0] - c[1]) > (bestv[0], -bestv[1], -(bestc[0] + bestc[1]) if bestc else -99):
            bestv, bestc = (n, h, r), c
    return bestc
T_ALL = fit_T_on(PA_V, ALL) if CALIBRATE_ANCHOR else 1.0
PAc_V = temp_scale(PA_V, T_ALL)
_g = []
for c in GRID:
    a1, fl = apply_rule(PAc_V, VOTES_V, PSUM_V, *c); r, h, n, k = count(PAc_V.argmax(1), a1, fl, Y_VAL)
    _g.append({"CONF": c[0], "MARGIN": c[1], "AGREE": c[2], "뒤집음": k, "살림": r, "망침": h, "순이득": n, "2배": "O" if r >= RATIO * h and r > 0 else ""})
GT = pd.DataFrame(_g).sort_values(["순이득", "망침"], ascending=[False, True])
print(f"기준 온도 T = {T_ALL:.2f} (검증 전체) | 표 수 {len(RESC)} (구원 {len(RESC_BASE)} + 판정 {len(XCR)}) → AGREE 후보 {AGREE_USE} | 격자 {len(GRID)}칸 중 상위 12칸 (참고용, 판정에 쓰지 않음):")
print(GT.head(12).to_string(index=False))
''')

md("""
## 6-1. 판정 목록 내보내기 (교차 검증 프롬프트 → RunPod `runpod_xcheck_any`) · 판정 모델 성적
- 대상: 보조 모델(판정 모델 제외) 중 1명 이상이 기준과 다른 보기를 고른 문항입니다(검증·test). 정답은 쓰지 않습니다.
- 모델 A/B = 기준 답 / 보조 모델들이 가장 많이 고른 다른 보기입니다. 어느 쪽이 A인지는 판정 노트북이 문항마다 무작위로 정합니다.
- 파일: `anchor_rescue/xcheck_targets.csv` → HF 비공개 저장소 `xcheck/xcheck_targets.csv` (RunPod 판정 노트북이 받아 감)
- 판정 모델 실행이 이미 있으면 대상 문항에서의 성적(판정 다수결 vs 기준)을 보입니다. 이 표를 보고 판정 모델을 빼거나 넣지 않습니다(사후 선택 = 과적합).
""")
code(r'''
RB_V = np.stack([PV[IDX[r]] for r in RESC_BASE]); RB_T = np.stack([PT[IDX[r]] for r in RESC_BASE])
def _targets(PA, RB):
    a = PA.argmax(1); votes, psum = rescue_votes(RB)
    v = votes.copy(); v[np.arange(len(a)), a] = -1
    alt = (v + 1e-3 * psum / max(psum.max(), 1e-9)).argmax(1); n_alt = v[np.arange(len(a)), alt].astype(int)
    return a, alt, n_alt
_av, _altv, _nv = _targets(PA_V, RB_V); _at, _altt, _nt = _targets(PA_T, RB_T)
TGT_V, TGT_T = _nv >= 1, _nt >= 1
TG_TAG = hashlib.md5(json.dumps({"a": ANCHOR, "r": RESC_BASE}).encode()).hexdigest()[:6]
TG_DF = pd.concat([pd.DataFrame({"id": VAL_IDS, "split": "val", "anchor": [LETTERS[i] for i in _av], "alt": [LETTERS[i] for i in _altv], "n_alt": _nv}),
                   pd.DataFrame({"id": TEST_IDS, "split": "test", "anchor": [LETTERS[i] for i in _at], "alt": [LETTERS[i] for i in _altt], "n_alt": _nt})], ignore_index=True)
TG_DF["target"] = (TG_DF.n_alt >= 1).astype(int); TG_DF["n_base"] = len(RESC_BASE); TG_DF["list_tag"] = TG_TAG; TG_DF["anchor_run"] = ANCHOR
TG_PATH = os.path.join(SAVE_DIR, "xcheck_targets.csv"); TG_DF.to_csv(TG_PATH, index=False)
say(f"[판정 목록 {TG_TAG}] 검증 {int(TGT_V.sum())}문항 (기준 오답 {int((aw & TGT_V).sum())}/{int(aw.sum())} 포함, 그중 대안이 정답 {int((aw & TGT_V & (_altv == Y_VAL)).sum())}) · "
    f"test {int(TGT_T.sum())}문항 · 보조 모델 {len(RESC_BASE)}명 기준 → {SAVE_SUBDIR}/xcheck_targets.csv")
if UPLOAD_TARGETS:
    try:
        from huggingface_hub import HfApi
        _tok = HF_TOKEN
        if not _tok:
            try:
                from google.colab import userdata; _tok = userdata.get("HF_TOKEN") or ""
            except Exception: pass
        _tok = _tok or os.environ.get("HF_TOKEN", "")
        if not _tok:
            import getpass; _tok = getpass.getpass("Hugging Face 토큰(Write 권한, hf_로 시작)을 붙여 넣고 Enter: ").strip()
        _api = HfApi(token=_tok); _repo = HF_REPO or f"{_api.whoami()['name']}/ssafy-vqa-private"
        assert _api.repo_info(_repo, repo_type="dataset").private, f"{_repo}가 공개 저장소입니다 → 올리지 않음"
        _api.upload_file(path_or_fileobj=TG_PATH, path_in_repo="xcheck/xcheck_targets.csv", repo_id=_repo, repo_type="dataset", commit_message=f"xcheck targets {TG_TAG}")
        say(f"  ✅ HF 저장소에 올림: {_repo}/xcheck/xcheck_targets.csv → RunPod runpod_xcheck_any가 받아 감")
    except Exception as _e:
        say(f"  ⚠️ HF 저장소에 올리지 못함 ({type(_e).__name__}: {str(_e)[:120]}) → Drive의 {SAVE_SUBDIR}/xcheck_targets.csv를 pod에 올리고 판정 노트북 TARGETS_FILE에 그 경로를 넣으세요")
if XCR:
    _rows, _pre = [], dict(MEMBERS_ALL)
    for r in XCR:
        _jd = {}
        for split, ids in [("val", VAL_IDS), ("test", TEST_IDS)]:
            d = pd.read_csv(_CACHE[(r, _pre[r], split)][1]).drop_duplicates("id", keep="last").set_index("id").reindex(ids)
            _jd[split] = (d["judged"].fillna(0).to_numpy() > 0) if "judged" in d.columns else np.zeros(len(ids), bool)
        jv, s = PV[IDX[r]].argmax(1), TGT_V & _jd["val"]
        _rows.append({"판정 모델": r, "목록": _CFG[r].get("TARGETS_TAG", "?"), "검증 대상 판정": f"{int(s.sum())}/{int(TGT_V.sum())}",
                      "판정 정답": int((s & (jv == Y_VAL)).sum()), "같은 문항 기준 정답": int((s & (_av == Y_VAL)).sum()),
                      "기준과 다른 답": int((s & (jv != _av)).sum()), "그중 정답": int((s & (jv != _av) & (jv == Y_VAL)).sum()),
                      "test 대상 판정": f"{int((TGT_T & _jd['test']).sum())}/{int(TGT_T.sum())}"})
        if _CFG[r].get("TARGETS_TAG") not in (None, TG_TAG):
            say(f"  ⚠️ {r}: 판정할 때의 목록({_CFG[r].get('TARGETS_TAG')})과 지금 목록({TG_TAG})이 다름 → 새 대상은 판정 없음(기준 답에 표). 판정 노트북을 다시 돌리면 새 대상만 이어서 판정")
    print("\n판정 모델 성적 (검증 대상 문항, 참고 — 판정 모델을 고르는 데 쓰지 않음):"); print(pd.DataFrame(_rows).to_string(index=False))
''')

md("""
## 7. 5겹 교차 적합 → 부트스트랩 → 유형별 → 일관성 → 판정 (합격선)
""")
code(r'''
rs = np.random.RandomState(CV_SEED); perm = rs.permutation(NV); folds = [np.sort(perm[f::K_FOLDS]) for f in range(K_FOLDS)]
OOF_A1 = PA_V.argmax(1).copy(); OOF_FLIP = np.zeros(NV, bool); CHOSEN = []
for f, te_idx in enumerate(folds):
    tr_idx = np.setdiff1d(ALL, te_idx)
    T_f = fit_T_on(PA_V, tr_idx) if CALIBRATE_ANCHOR else 1.0
    PAc = temp_scale(PA_V, T_f)
    c = best_cell(PAc, tr_idx)
    CHOSEN.append({"구역": f + 1, "T": round(T_f, 2), "CONF": c[0] if c else None, "MARGIN": c[1] if c else None, "AGREE": c[2] if c else None})
    if c is None: continue
    a1, fl = apply_rule(PAc, VOTES_V, PSUM_V, *c)
    OOF_A1[te_idx], OOF_FLIP[te_idx] = a1[te_idx], fl[te_idx]
A0 = PA_V.argmax(1)
R_, H_, N_, K_ = count(A0, OOF_A1, OOF_FLIP, Y_VAL)
CH = pd.DataFrame(CHOSEN)
for f, te_idx in enumerate(folds):
    r, h, n, k = count(A0, OOF_A1, OOF_FLIP, Y_VAL, te_idx); CH.loc[f, "살림"], CH.loc[f, "망침"], CH.loc[f, "순이득"] = r, h, n
print("구역별로 고른 문턱값 (4구역에서 고름) · 남은 1구역 성적:"); print(CH.to_string(index=False))
say(f"[교차 적합] 기준 {ANCHOR} {(A0 == Y_VAL).mean() * 100:.2f}% → {(OOF_A1 == Y_VAL).mean() * 100:.2f}% | 살림 {R_} · 망침 {H_} · 순이득 {N_:+d} (뒤집은 문항 {K_})")
# 부트스트랩: 교차 적합 결과(문항별 살림/망침)를 다시 뽑아 순이득 > 0 비율
resc = (OOF_FLIP & (A0 != Y_VAL) & (OOF_A1 == Y_VAL)).astype(int); harm = (OOF_FLIP & (A0 == Y_VAL)).astype(int)
bi = np.random.RandomState(BOOT_SEED).randint(0, NV, (N_BOOT, NV)); bnet = resc[bi].sum(1) - harm[bi].sum(1)
BOOT_P = float((bnet > 0).mean()); lo, hi = np.percentile(bnet, [2.5, 97.5])
say(f"[부트스트랩 {N_BOOT}번] 순이득 > 0인 비율 {BOOT_P * 100:.1f}% | 순이득 95% 구간 [{lo:+.0f}, {hi:+.0f}]")
# 유형별
_c = []
for k, pat in CATS.items():
    s = val_df.question.astype(str).str.contains(pat).to_numpy()
    if s.sum() >= 20: _c.append({"유형": k, "문항": int(s.sum()), "살림": int(resc[s].sum()), "망침": int(harm[s].sum()), "순이득": int(resc[s].sum() - harm[s].sum())})
_o = ~np.any([val_df.question.astype(str).str.contains(p).to_numpy() for p in CATS.values()], 0)
_c.append({"유형": "그 밖", "문항": int(_o.sum()), "살림": int(resc[_o].sum()), "망침": int(harm[_o].sum()), "순이득": int(resc[_o].sum() - harm[_o].sum())})
CT = pd.DataFrame(_c); print("\n문항 유형별 (교차 적합 결과):"); print(CT.to_string(index=False))
# 일관성
ok_c = CH.CONF.notna().all()
cons = bool(ok_c and (CH.MARGIN.max() - CH.MARGIN.min() <= CONSIST_MARGIN + 1e-9) and (CH.CONF.max() - CH.CONF.min() <= CONSIST_CONF + 1e-9)
            and CH.AGREE.value_counts().iloc[0] >= K_FOLDS - 1)
say(f"[일관성] 구역별 MARGIN {sorted(set(CH.MARGIN.dropna()))} · CONF {sorted(set(CH.CONF.dropna()))} · AGREE {sorted(set(CH.AGREE.dropna()))} → {'일관 ✅' if cons else '들쭉날쭉 ⚠️'}")
bad_cat = CT[CT.순이득 < -1]
if len(bad_cat): say(f"  ⚠️ 순이득이 −2 이하인 유형: {list(bad_cat.유형)}")
checks = {f"순이득 ≥ +{MIN_NET}": N_ >= MIN_NET, f"살림 ≥ 망침 × {RATIO:g}": R_ >= RATIO * H_ and R_ > 0, f"부트스트랩 ≥ {BOOT_MIN * 100:.0f}%": BOOT_P >= BOOT_MIN, "문턱값 일관": cons}
ADOPT = all(checks.values())
say("[판정] " + ("채택 ✅" if ADOPT else "불채택") + " — " + " · ".join(f"{k} {'O' if v else 'X'}" for k, v in checks.items()))
''')

md("""
## 7-1. 멤버별 기여 — 넣었을 때 vs 뺐을 때 (같은 5겹 교차 적합)
- 멤버를 하나씩 빼고 7번과 똑같이 교차 적합을 다시 합니다. 문턱값도 다시 고릅니다.
- '넣어서 맞게 된 문항'은 넣었을 때만 맞힌 검증 문항이고, '넣어서 틀리게 된 문항'은 넣었을 때만 틀린 문항입니다.
- 부호 검정 p는 그 차이가 우연으로 나올 확률입니다.
  - 달라지는 문항이 적어서 p < 0.05는 거의 나오지 않습니다.
  - 차이가 0~2개면 '효과 없음과 구별되지 않음'으로 봅니다.
- 같은 계열 멤버가 둘이면 하나를 빼도 다른 하나가 같은 표를 내서 0이 나올 수 있습니다. 그래서 '판정 모델 전부'·'zero-shot 전부'를 한꺼번에 뺀 줄도 보입니다.
- 이 표는 멤버를 빼기 위한 것이 아니라 기여를 보는 참고용입니다.
""")
code(r'''
from math import comb
def cv_oof(resc):
    # 7번과 같은 교차 적합을 주어진 보조 모델 목록으로 (문턱값·AGREE 후보도 그 목록 기준으로 다시)
    Vx, Sx = rescue_votes(np.stack([PV[NAMES.index(r)] for r in resc]))
    ag = list(range(2, min(len(resc), 8) + 1)) if AGREE_GRID == "auto" else list(AGREE_GRID)
    grid = list(itertools.product(CONF_GRID, MARGIN_GRID, ag))
    def best(PAc, idx):
        a0 = PAc.argmax(1); bv, bc = (0, 0, 0), None
        for c in grid:
            a1, fl = apply_rule(PAc, Vx, Sx, *c); r, h, n, _ = count(a0, a1, fl, Y_VAL, idx)
            if r >= RATIO * h and r > 0 and (n, -h, -c[0] - c[1]) > (bv[0], -bv[1], -(bc[0] + bc[1]) if bc else -99): bv, bc = (n, h, r), c
        return bc
    a1o, flo = A0.copy(), np.zeros(NV, bool)
    for te_idx in folds:
        tr_idx = np.setdiff1d(ALL, te_idx); PAc = temp_scale(PA_V, fit_T_on(PA_V, tr_idx) if CALIBRATE_ANCHOR else 1.0)
        c = best(PAc, tr_idx)
        if c is None: continue
        a1, fl = apply_rule(PAc, Vx, Sx, *c); a1o[te_idx], flo[te_idx] = a1[te_idx], fl[te_idx]
    return a1o, flo
assert (cv_oof(RESC)[0] == OOF_A1).all(), "7-1 재계산이 7번과 다릅니다 → 위 출력을 확인하세요"
_rows, _cases = [], [(r, "판정" if r in XC else ("zero-shot" if r in ZS else "학습"), [x for x in RESC if x != r]) for r in RESC]
if XCR: _cases.append((f"(판정 모델 전부 {len(XCR)}개)", "판정 묶음", list(RESC_BASE)))
_zs_in = [x for x in RESC if x in ZS]
if len(_zs_in) >= 2: _cases.append((f"(zero-shot 전부 {len(_zs_in)}개)", "zero-shot 묶음", [x for x in RESC if x not in ZS]))
for r, role, rest in _cases:
    if len([x for x in rest if x not in XC]) < 2: continue
    o_a, o_f = cv_oof(rest); ro, ho, no, _ = count(A0, o_a, o_f, Y_VAL)
    gain, loss = int(((OOF_A1 == Y_VAL) & (o_a != Y_VAL)).sum()), int(((OOF_A1 != Y_VAL) & (o_a == Y_VAL)).sum()); n_ = gain + loss
    p = min(1.0, 2 * sum(comb(n_, k) for k in range(max(gain, loss), n_ + 1)) / 2 ** n_) if n_ else 1.0
    _rows.append({"멤버": r, "역할": role, "뺐을 때 순이득": no, "넣었을 때 순이득": N_,
                  "넣어서 맞게 된 문항": gain, "넣어서 틀리게 된 문항": loss, "차이": gain - loss, "부호 검정 p": round(p, 3)})
CONTRIB = pd.DataFrame(_rows)
print(f"지금 구성(7번) 교차 적합 순이득 {N_:+d} | 멤버를 하나씩 뺐을 때 (참고):"); print(CONTRIB.to_string(index=False))
for _, x in CONTRIB.iterrows():
    say(f"[기여] {x['멤버']} ({x['역할']}): 넣어서 맞게 {x['넣어서 맞게 된 문항']} · 틀리게 {x['넣어서 틀리게 된 문항']} → {x['차이']:+d} (p={x['부호 검정 p']}) | 빼면 순이득 {x['뺐을 때 순이득']:+d}")
''')

md("""
## 8. 최종 규칙 → test 제출 파일
- 문턱값은 검증 1,000문항 **전체**에서 같은 기준으로 다시 고릅니다(교차 적합은 판정용).
- 판정과 상관없이 파일은 만들고, 이름 앞에 판정을 붙입니다: `submission_anchor_<채택|불채택>_…`
- 참고 줄: 지금 공개 최고 조합(ea0e9f)과 검증·test를 비교합니다.
""")
code(r'''
FINAL = best_cell(PAc_V, ALL)
if FINAL is None:
    say("[최종] 검증 전체에서 2배 조건을 만족하는 문턱값이 없음 → 기준 모델 답 그대로"); FINAL = (0.0, 0.0, 99)
a1v, flv = apply_rule(PAc_V, VOTES_V, PSUM_V, *FINAL); rv, hv, nv, kv = count(A0, a1v, flv, Y_VAL)
PAc_T = temp_scale(PA_T, T_ALL); a1t, flt = apply_rule(PAc_T, VOTES_T, PSUM_T, *FINAL)
say(f"[최종 규칙] CONF < {FINAL[0]} · MARGIN < {FINAL[1]} · AGREE ≥ {FINAL[2]} · 기준 T {T_ALL:.2f} | 검증(고른 문항 그대로, 참고) 살림 {rv} · 망침 {hv} · 순이득 {nv:+d} → {(a1v == Y_VAL).mean() * 100:.2f}% | test 뒤집음 {int(flt.sum())}")
tag = hashlib.md5(json.dumps({"a": ANCHOR, "r": RESC, "f": list(FINAL), "t": T_ALL, "cal": CALIBRATE_ANCHOR}).encode()).hexdigest()[:6]
fn = f"submission_anchor_{'채택' if ADOPT else '불채택'}_{re.sub(r'[^a-z0-9]+', '_', ANCHOR.lower())}_{tag}.csv"
sub = sub_df[["id"]].assign(answer=[LETTERS[i] for i in a1t]); assert len(sub) == len(sub_df) and set(sub.answer) <= set(LETTERS)
sub.to_csv(os.path.join(SAVE_DIR, fn), index=False)
def _mf(r, split): return os.path.relpath(_CACHE[(r, PRESETS_OF[r], split)][1], OUT_DIR)
PRESETS_OF = dict(MEMBERS_ALL)
meta = {"anchor": ANCHOR, "rescuers": RESC, "rescuers_base": RESC_BASE, "judges": XCR, "targets_tag": TG_TAG, "agree_grid": AGREE_USE, "rule": {"CONF": FINAL[0], "MARGIN": FINAL[1], "AGREE": FINAL[2], "T_anchor": T_ALL, "calibrate": CALIBRATE_ANCHOR},
        "cv": {"rescue": R_, "harm": H_, "net": N_, "boot_p": BOOT_P, "consistent": cons, "chosen": CHOSEN, "adopt": ADOPT},
        "files": {r: {"val": _mf(r, "val"), "test": _mf(r, "test")} for r in [ANCHOR] + RESC}, "test_flips": int(flt.sum()),
        "created": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(meta, open(os.path.join(SAVE_DIR, fn[:-4] + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
say(f"저장: {SAVE_SUBDIR}/{fn} (+ 설정 JSON) | test 답 분포 {dict(zip(LETTERS, np.bincount(a1t, minlength=4).tolist()))}")
_sa = sub_answers(os.path.join(OUT_DIR, ANCHOR, f"submission_{ANCHOR}_final_ep1_up2.0.csv")) if os.path.exists(os.path.join(OUT_DIR, ANCHOR, f"submission_{ANCHOR}_final_ep1_up2.0.csv")) else None
if _sa is not None: say(f"  기준 모델 단독 제출과 test 답이 다른 문항 {int((_sa != a1t).sum())}개 (= 뒤집은 문항 {int(flt.sum())}이어야 정상)")
# 참고: 지금 공개 최고 조합과 비교
_rj = os.path.join(OUT_DIR, REF_JSON)
if os.path.exists(_rj):
    _i = json.load(open(_rj, encoding="utf-8")); _ra = sub_answers(os.path.join(os.path.dirname(_rj), _i["file"]))
    if _ra is not None:
        dd = int((_ra != a1t).sum())
        say(f"  (참고) ea0e9f 검증 {float(_i['val_acc']):.2f}% · 공개 3,281 | 이 파일과 test 답이 다른 문항 {dd}개 → 공개 약 {dd / 2:.0f}문항이 갈림 (운만으로 ±{2 * math.sqrt(dd / 2):.0f})")
''')

md("""
## 9. 확신도 조율 비교 (절반으로 T 맞추고 나머지 절반에서 방법 비교)
- 대상: 기준 + 보조 모델(판정 모델 제외 — 판정 모델은 대상 문항에만 표가 있어 확률 평균에 넣을 수 없음), 같은 가중치입니다.
- 절반 A로 모델별 T를 맞추고 절반 B에서 정확도를 잽니다. 그다음 A·B를 바꿔 한 번 더 합니다. 두 번의 합이 결과입니다.
""")
code(r'''
MODELS = [ANCHOR] + RESC_BASE; PM_V = np.stack([PV[IDX[r]] for r in MODELS])
half = np.random.RandomState(CV_SEED + 1).permutation(NV); HA, HB = np.sort(half[: NV // 2]), np.sort(half[NV // 2:])
def borda(Ps):
    sc = np.zeros(Ps.shape[1:])
    for P in Ps: sc += np.argsort(np.argsort(P, 1), 1)            # 순위 점수: 1등 3 · 2등 2 · 3등 1 · 4등 0
    return sc + 1e-3 * Ps.mean(0)
def margin_w(Ps):
    s = np.sort(Ps, 2); w = (s[:, :, -1] - s[:, :, -2])[:, :, None]
    return (w * Ps).sum(0) / np.clip(w.sum(0), 1e-9, None)
res = {k: 0 for k in ["기준 단독", "확률 평균", "온도 보정 평균", "순위 투표(Borda)", "마진 가중(온도 보정)"]}
for fit_idx, ev_idx in [(HA, HB), (HB, HA)]:
    Ts = [fit_T_on(PM_V[m], fit_idx) for m in range(len(MODELS))]
    Pc = np.stack([temp_scale(PM_V[m], Ts[m]) for m in range(len(MODELS))])
    y = Y_VAL[ev_idx]
    res["기준 단독"] += int((PM_V[0][ev_idx].argmax(1) == y).sum())
    res["확률 평균"] += int((PM_V.mean(0)[ev_idx].argmax(1) == y).sum())
    res["온도 보정 평균"] += int((Pc.mean(0)[ev_idx].argmax(1) == y).sum())
    res["순위 투표(Borda)"] += int((borda(PM_V)[ev_idx].argmax(1) == y).sum())
    res["마진 가중(온도 보정)"] += int((margin_w(Pc)[ev_idx].argmax(1) == y).sum())
res["교체 규칙(7번 교차 적합)"] = int((OOF_A1 == Y_VAL).sum())
CAL = pd.DataFrame([{"방법": k, "맞힘": v, "정확도(%)": round(v / NV * 100, 2)} for k, v in res.items()]).sort_values("맞힘", ascending=False)
print(CAL.to_string(index=False))
say(f"[조율 비교] 1위 {CAL.iloc[0].방법} {CAL.iloc[0]['정확도(%)']}% · 교체 규칙 {res['교체 규칙(7번 교차 적합)'] / NV * 100:.2f}% · 기준 단독 {res['기준 단독'] / NV * 100:.2f}% (모두 고를 때 안 본 절반·구역 기준)")
''')

md("## 10. 결과 요약")
code(r'''
print(f"기준 {ANCHOR} ({ACC[ANCHOR]:.2f}%) | 보조 모델 {RESC_BASE} | 판정 모델 {XCR or '없음'} | AGREE 후보 {AGREE_USE}")
for _l in LINES: print(_l)
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
