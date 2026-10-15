# -*- coding: utf-8 -*-
"""E20 노트북 빌더: '교차 검증' — 조합이 불확실한 문항만, 조합의 1·2위 보기를 '모델 A/B의 답'으로 보여 주고
Qwen3.5-122B(GPTQ, vLLM)에게 단계별로 따져 최종 답을 내게 함 → 표본 K개 투표 → 조합과 섞기 → 검증 판정 → test 후보.
설치·데이터·분할·이미지 크기·모델 받기·vLLM 로드 셀은 zs122b_vllm.ipynb(G4에서 실제로 돌아간 것)를 그대로 가져옴.
조합 다시 만들기는 E7-b(e7_on_best.ipynb)와 같은 식(ens_search 설정 JSON).
사용: python3 build_e20.py [출력 경로]"""
import sys, json, uuid, io, contextlib, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e20/out/e20_xcheck_122b.ipynb"
SPD = "/tmp/vqa_scratch"
_z = {}
_argv = sys.argv; sys.argv = ["build_zs122b.py", f"{SPD}/_zs_tmp_e20.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/zs122b/build_zs122b.py", encoding="utf-8").read(), _z)
sys.argv = _argv
ZC = _z["cells"]
def src(i): return ZC[i]["source"]
def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:90]); return text.replace(old, new)
assert "RUN_NAME   =" in src(2) and "!pip" in src(4) and "drive.mount" in src(6) and "make_split" in src(8)
assert "build_mc_prompt" in src(10) and "def prep_image" in src(12) and "snapshot_download" in src(14) and "LLM(model=MODEL_ID" in src(16)

ENV_TAIL = src(2)[src(2).index("RES_PRESETS = {"):]          # 해상도 표 + torchaudio 차단·캐시·vLLM 환경 변수 (zs122b와 같음)
INSTALL, DATA_C, CSV_C, IMG_C, DL_C = src(4), src(6), src(8), src(12), src(14)
PROMPT_C = rep(src(10), 'TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}',
               'TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}      # 채점용(baseline_v1)과 같음\n'
               'XC_TEMPLATE_KW = {"enable_thinking": bool(THINKING)} if "enable_thinking" in (_tmpl or "") else {}   # 교차 검증용')
LLM_C = src(16)

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# E20 · 불확실 문항 교차 검증 — Qwen3.5-122B GPTQ · vLLM, 학습 없음

**하는 일**
1. 공개 최고 조합(`ea0e9f`)을 설정 JSON으로 다시 만들고, **1·2위 확률 차가 작은 문항만** 고릅니다. 모델 의견이 갈리는 문항과 같은 뜻입니다.
2. 그 문항마다 이미지·질문·보기와 함께 조합의 1·2위 보기를 "모델 A의 답 / 모델 B의 답"으로 보여 줍니다. 어느 쪽이 A인지는 문항마다 무작위입니다.
   - 두 후보 답(모델 A/B)을 보여 주고 근거를 교차 검증하게 하는 프롬프트로 묻습니다: 흔한 착각 패턴을 배제하고 단계별로 결론을 내게 합니다.
   - 마지막 줄 `정답: (x)`를 읽습니다.
3. 문항마다 표본을 `K_SAMPLES`개(기본 5개) 뽑아 투표합니다. 투표 비율 q를 조합 확률 p와 (1−α)·p + α·q로 섞습니다.
   - α = 0.5이면, 투표가 조합보다 더 확실히 한쪽을 가리킬 때만 답이 바뀝니다. 즉 '투표 차 > 조합의 1·2위 차'일 때입니다.
4. 검증에서 조합과 같은 문항끼리 고침·새로 틀림을 세어 판정합니다. 통과하면 test 불확실 문항도 돌려 제출 후보를 만듭니다.

**사전 기준 (결과를 보기 전에 고정)**
- 1차 가설: τ = 0.5, α = 0.5, 표본 5개, 생각 모드 끔, 부정형 질문도 포함합니다.
- 판정: 검증 순이득 ≥ +3이면 채택, +1~+2면 공개 1회 확인 후보, 0 이하면 불채택입니다.
- 검증은 τ < 0.8(참고 표용)까지 채점하고, test는 τ < 0.5만 채점합니다.
- τ·α 격자 표는 참고로만 봅니다.

**런타임**: G4(96GB) **새 런타임**. zs122b 노트북과 같은 설치·모델입니다. 모델 파일은 79GB이고, 런타임을 새로 열면 다시 받습니다.
**결과**: `vqa_runs/e20_xcheck_q35_122b/`에만 씁니다. 끊기면 다시 전체 실행하면 이어서 합니다.
""")

md("## 1. 설정")
code(r'''
RUN_NAME   = "e20_xcheck_q35_122b"              # 결과 폴더 = vqa_runs/<RUN_NAME>
MODEL_ID   = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" # zs122b와 같은 모델 (검증 zero-shot 96.0%)
PRESET     = "up2.0"                            # 다른 멤버와 같은 해상도

# ── 기준 조합 (ens_search · manual_ens가 저장한 설정 JSON) ──
BASE_KEY  = "ea0e9f"                  # 파일 이름 끝 6자리 해시, 또는 JSON 경로(OUT_DIR 기준)
JSON_DIRS = ["ens_search", "manual_ens"]

# ── 사전 기준 (결과를 보기 전에 고정) ──
TAU, ALPHA = 0.5, 0.5                 # 1·2위 차 < τ인 문항만, (1−α)·p조합 + α·q투표
TAU_SCORE_VAL, TAU_SCORE_TEST = 0.8, 0.5   # 채점 범위: 검증은 참고 표까지, test는 1차 가설만
NET_MIN = 3                           # 순이득 ≥ +3 채택 / +1~+2 공개 1회 확인 후보 / 0 이하 불채택
MIN_VALID = 3                         # 답을 읽어 낸 표본이 이보다 적으면 그 문항은 섞지 않음
GRID_TAUS, GRID_ALPHAS = [0.3, 0.5, 0.8], [0.25, 0.5, 1.0]   # 참고 표 (판정에 쓰지 않음)
RUN_TEST = "auto"                     # "auto": 검증 순이득 ≥ +1이면 test 채점 / True: 항상 / False: 안 함

# ── 생성 ──
K_SAMPLES   = 5                       # 문항당 표본 수 (투표)
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.8, 20   # Qwen3.5 권장값 (생각 끔)
MAX_TOKENS  = 1200                    # 단계별 풀이 + 마지막 줄 '정답: (x)'. 첫 묶음 점검에서 '길이 제한에 걸림'이 많으면 늘림
THINKING    = False                   # 생각 모드 끔. True면 <think> 뒤 부분에서 답을 읽음 (훨씬 느림)
GEN_CHUNK   = 12                      # 한 번에 넘기는 문항 수 = 저장 단위

# ── vLLM (zs122b와 같음) ──
QUANTIZATION  = "moe_wna16"
MOE_BACKEND   = "auto"
GPU_MEM_UTIL  = 0.92
MAX_MODEL_LEN = 8192                  # 프롬프트(이미지 약 2,700 + 글자) + 생성 800
MAX_NUM_SEQS  = 16                    # 표본 하나가 한 자리 (16 = 문항 3개 × 표본 5개 정도가 동시에). 시작이 되면 32로 올려도 됨
MAX_BATCHED_TOKENS = 16384
TOPK_LOGPROBS = 64
IMG_WORKERS   = 6
GDN_PREFILL_BACKEND = "triton"
ENGINE_KW     = {}

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
HF_CACHE     = "/content/hf_cache"
SEED, VAL_SIZE = 42, 1000
HF_TOKEN   = ""

''' + ENV_TAIL.rstrip())

md("## 2. 설치 (zs122b와 같음: vLLM 0.30.0 · transformers 5.17.0, 새 런타임에서)")
code(INSTALL)

md("## 3. 데이터 준비 (stage2와 같음 — 이미지 포함 압축 해제)")
code(DATA_C)

md("## 4. CSV 로드 · 고정 검증 분할 (stage2와 같음)")
code(CSV_C)

md("""
## 5. 기준 조합 다시 만들기 → 교차 검증할 문항 고르기 (GPU 불필요)
- 설정 JSON에 적힌 예측 파일·온도·`_full` 섞기로 조합을 다시 만듭니다(E7-b와 같은 식).
- 검증 정확도가 JSON 기록과 같고 test 답이 저장된 제출 파일과 완전히 같아야 넘어갑니다.
- 채점할 문항 수를 먼저 보여 줍니다. 모델을 받기 전에 여기서 확인됩니다.
""")
code(r'''
LETTERS = "abcd"
VAL_IDS, TEST_IDS = val_df.id.tolist(), test_df.id.tolist()
Y_VAL = val_df.answer.map(LETTERS.index).to_numpy()
def _probs(d): return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
def _load_preds(path, ids):
    d = pd.read_csv(path).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in ids if i not in d.index]
    assert not miss, f"{os.path.basename(path)}에 없는 문항 {len(miss)}개 (예: {miss[:3]})"
    return d.loc[ids]
def temp_scale(p, T):
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
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
    assert hits, f"'{key}'로 끝나는 설정 JSON이 {JSON_DIRS} 폴더에 없습니다 → BASE_KEY에 JSON 경로(OUT_DIR 기준)를 직접 적으세요"
    assert len(hits) == 1, f"'{key}'로 끝나는 설정 JSON이 여러 개입니다: {hits} → BASE_KEY에 경로를 직접 적으세요"
    return hits[0]
def rebuild_base(p):
    info = json.load(open(p, encoding="utf-8")); c, mem = info["cfg"], info["members"]
    assert c.get("method") in ("prob", "logprob"), f"결합 방법 '{c.get('method')}'은 확률이 아니어서 섞을 수 없습니다"
    def _fp(rel): return os.path.join(OUT_DIR, rel)
    runs = list(c["weights"])
    gone = [mem[r][k] for r in runs for k in ("val", "test", "full_test") if k in mem[r] and not os.path.exists(_fp(mem[r][k]))]
    assert not gone, f"{info['file']}: 설정 JSON에 적힌 예측 파일이 없습니다: {gone}"
    def P_of(r, split):
        P = _probs(_load_preds(_fp(mem[r][split]), VAL_IDS if split == "val" else TEST_IDS))
        if split == "test" and c.get("full", "none") != "none" and "full_test" in mem[r]:
            Pf = _probs(_load_preds(_fp(mem[r]["full_test"]), TEST_IDS)); P = (P + Pf) / 2 if c["full"] == "mix" else Pf
        return temp_scale(P, mem[r]["T"]) if c.get("temp") == "val" else P
    w = [c["weights"][r] for r in runs]
    PV = _combine([P_of(r, "val") for r in runs], w, c["method"]); PT = _combine([P_of(r, "test") for r in runs], w, c["method"])
    return info, PV, PT, sub_answers(os.path.join(os.path.dirname(p), info["file"]))
def margin(P):
    s = np.sort(P, axis=1); return s[:, -1] - s[:, -2]

BASE_JSON = find_json(BASE_KEY)
BASE_INFO, PB_VAL, PB_TEST, A_SAVED = rebuild_base(BASE_JSON)
BASE_TAG = os.path.basename(BASE_JSON)[:-5].rsplit("_", 1)[-1]
A_VAL, A_TEST = PB_VAL.argmax(1), PB_TEST.argmax(1)
acc_base = float((A_VAL == Y_VAL).mean() * 100)
assert abs(acc_base - float(BASE_INFO["val_acc"])) < 1e-6, f"검증 {acc_base:.2f}%가 JSON 기록 {BASE_INFO['val_acc']:.2f}%와 다릅니다 → 위 출력을 확인하세요"
assert A_SAVED is not None and (A_SAVED == A_TEST).all(), "test 답이 저장된 제출 파일과 다릅니다 → 위 출력을 확인하세요"
print(f"✅ 기준 조합 {BASE_INFO.get('name', BASE_TAG)} ({BASE_TAG}): 검증 {acc_base:.2f}% (JSON 기록과 같음) · test 답 = 저장된 제출 {BASE_INFO['file']}")
M_VAL, M_TEST = margin(PB_VAL), margin(PB_TEST)
TOP2_VAL, TOP2_TEST = np.argsort(-PB_VAL, 1)[:, :2], np.argsort(-PB_TEST, 1)[:, :2]
XC_VAL = val_df[M_VAL < TAU_SCORE_VAL].copy(); XC_TEST = test_df[M_TEST < TAU_SCORE_TEST].copy()
for t in sorted(set(GRID_TAUS + [TAU])):
    g = M_VAL < t
    print(f"  τ={t}: 검증 {int(g.sum())}문항 (조합 오답 {int((g & (A_VAL != Y_VAL)).sum())} · 정답이 1·2위 안 {int((g & ((TOP2_VAL == Y_VAL[:, None]).any(1))).sum())}) | test {int((M_TEST < t).sum())}문항")
print(f"  조합 오답 {int((A_VAL != Y_VAL).sum())}개 중 τ={TAU} 밖(확신 오답, 이 방법으로 못 고침) {int(((M_VAL >= TAU) & (A_VAL != Y_VAL)).sum())}개")
print(f"→ 채점 대상: 검증 {len(XC_VAL)}문항 (τ < {TAU_SCORE_VAL}) · test {len(XC_TEST)}문항 (τ < {TAU_SCORE_TEST}) · 문항당 표본 {K_SAMPLES}개")
''')

md("""
## 6. 프롬프트 · 프로세서
- 채점용 `baseline_v1`(zs122b와 같음)에 더해 교차 검증 프롬프트를 만듭니다.
- 교차 검증 프롬프트(한국어)에 이미지 글자를 먼저 정확히 읽으라는 지시와 마지막 줄 형식만 더했습니다.
""")
code(PROMPT_C.rstrip() + r'''

# ── E20: 교차 검증 프롬프트 (사전에 고정) ──
XC_SYSTEM = "당신은 이미지 속 글자와 내용을 꼼꼼히 확인해 객관식 문제의 정답을 판정하는 검증자입니다."
def xc_user_text(row, A, B):
    opts = "\n".join(f"({L}) {row[L]}" for L in LETTERS)
    return (f"{row['question']}\n{opts}\n\n"
            "[작전 지시: 오답 돌파 프로토콜]\n"
            "1. 두 모델의 판단이 갈렸다.\n"
            f"   - 모델 A의 답: ({LETTERS[A]}) {row[LETTERS[A]]}\n"
            f"   - 모델 B의 답: ({LETTERS[B]}) {row[LETTERS[B]]}\n"
            "2. 이미지에서 질문과 관련된 부분을 찾아 글자와 숫자를 정확히 읽고, 각 모델이 전제로 삼은 근거에서 논리적 모순이나 비약이 있는지 교차 검증하라.\n"
            "3. 이 문제 유형에서 자주 발생하는 대표적 착각 패턴을 배제하고, 최종 결론을 단계별로 도출하라.\n"
            "마지막 줄에는 반드시 \"정답: (a)\"처럼 a, b, c, d 중 하나만 적어라.")
def xc_messages(row, A, B):
    t = xc_user_text(row, A, B)
    if USE_SYSTEM_ROLE:
        return [{"role": "system", "content": [{"type": "text", "text": XC_SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": XC_SYSTEM + "\n\n" + t}]}]
def xc_pair(qid, top2):
    # 조합 1·2위 보기를 모델 A/B에 무작위로 배정 (문항 id로 고정 → 다시 돌려도 같음)
    flip = int(hashlib.md5(f"{qid}|xc".encode()).hexdigest(), 16) % 2
    a, b = int(top2[0]), int(top2[1]); return (b, a) if flip else (a, b)
def render_xc(row, A, B):
    return processor.apply_chat_template(xc_messages(row, A, B), tokenize=False, add_generation_prompt=True, **XC_TEMPLATE_KW)
_ANS_RE = re.compile(r"정답\s*[:：]?\s*[\(\[（]?\s*([abcdABCD])\s*[\)\]）]?(?![A-Za-z])")
def parse_answer(text):
    t = text.split("</think>")[-1]                         # 생각 모드면 생각 뒤 부분만
    m = list(_ANS_RE.finditer(t))
    if m: return m[-1].group(1).lower()
    tail = t.strip().splitlines()[-1].lower() if t.strip() else ""
    m2 = re.findall(r"\(([abcd])\)", tail)
    return m2[-1] if m2 else None
for _s, _e in [("분석 끝.\n정답: (c)", "c"), ("정답은 (a)일 수도 있지만 ... \n정답: b", "b"), ("최종 정답：（d）", "d"), ("답을 모르겠다", None), ("<think>정답: (a)</think>\n정답: (c)", "c")]:
    assert parse_answer(_s) == _e, (_s, parse_answer(_s))
_r0 = val_df.iloc[0].to_dict(); _A0, _B0 = xc_pair(_r0["id"], [0, 1])
_px = render_xc(_r0, _A0, _B0)
assert _px.count(IMAGE_TOKEN) == 1, "교차 검증 프롬프트에 이미지 자리가 정확히 하나여야 합니다"
print(f"교차 검증 템플릿 추가 인자 {XC_TEMPLATE_KW} | 예시 (끝부분):"); print(_px[-700:])
''')

md("## 7. 이미지: transformers 5.17 프로세서와 똑같이 크기 미리 맞추기 + 대조 (zs122b와 같음)")
code(IMG_C)

md("## 8-0. 모델 파일 받기 (79GB · 끊기면 다시 실행하면 이어받기)")
code(DL_C)

md("""
## 8. vLLM으로 모델 로드 (zs122b와 같음)
오류가 나면 zs122b 노트북 7번 셀의 증상별 표대로 1번 셀 설정을 바꾸고 **세션 다시 시작 → 1번 셀부터** 실행하세요.
""")
code(LLM_C)

md("""
## 9. 교차 검증 생성 함수 (끊겨도 이어서)
- 문항마다 표본 `K_SAMPLES`개를 한 요청(`n`)으로 뽑습니다. 이미지·프롬프트 처리는 문항당 한 번입니다.
- 표본 추출에는 문항별 시드를 씁니다(재현용).
- 표본마다 `id, split, sample, letter, finish, ntok, A, B, text`를 JSONL로 저장합니다. 반쯤 쓴 줄은 읽을 때 버립니다.
""")
code(r'''
from concurrent.futures import ThreadPoolExecutor
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
GEN_VAL, GEN_TEST = os.path.join(RUN_DIR, f"xc_val__{PRESET}.jsonl"), os.path.join(RUN_DIR, f"xc_test__{PRESET}.jsonl")
_XPOOL, _XPRE = ThreadPoolExecutor(max_workers=IMG_WORKERS), ThreadPoolExecutor(max_workers=1)
def _seed_of(qid): return int(hashlib.md5(f"{qid}|{SEED}".encode()).hexdigest(), 16) % (2 ** 31)
def _xc_req(item):
    r, A, B = item
    if PRE_RESIZE: im, _ = prep_image(r["path"])
    else: im = load_image(r["path"])
    return {"prompt": render_xc(r, A, B), "multi_modal_data": {"image": im}}
def read_gen(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: rows.append(json.loads(line))
            except Exception: pass                         # 끊겨서 반쯤 쓴 줄
    return pd.DataFrame(rows, columns=["id", "split", "sample", "letter", "finish", "ntok", "A", "B", "text"])
def _fix_tail(path):                   # 저장 도중 끊겨 마지막 줄이 줄바꿈 없이 끝났으면 줄바꿈을 붙여 다음 줄과 섞이지 않게
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n": f.write(b"\n")
def generate_xc(df, top2, ids_all, path, split):
    _fix_tail(path)
    done = read_gen(path).drop_duplicates(["id", "sample"], keep="last"); cnt = done.groupby("id").size() if len(done) else pd.Series(dtype=int)
    pos = {i: n for n, i in enumerate(ids_all)}
    rows = [r for r in df.to_dict("records") if cnt.get(r["id"], 0) < K_SAMPLES]
    print(f"[{split}] {os.path.basename(path)} | 전체 {len(df)}문항 | 완료 {len(df) - len(rows)} | 남음 {len(rows)} | 표본 {K_SAMPLES}개씩", flush=True)
    if not rows: return read_gen(path)
    items = [(r, *xc_pair(r["id"], top2[pos[r["id"]]])) for r in rows]
    chunks = [items[i:i + GEN_CHUNK] for i in range(0, len(items), GEN_CHUNK)]
    fut, t0, n, ntok = _XPRE.submit(lambda c: list(_XPOOL.map(_xc_req, c)), chunks[0]), time.time(), 0, 0
    for k, ch in enumerate(chunks):
        reqs = fut.result()
        if k + 1 < len(chunks): fut = _XPRE.submit(lambda c: list(_XPOOL.map(_xc_req, c)), chunks[k + 1])
        sps = [SamplingParams(n=K_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS, seed=_seed_of(r["id"])) for r, _, _ in ch]
        outs = llm.generate(reqs, sps, use_tqdm=False)
        if len(outs) > len(ch):
            print(f"  (엔진에 남아 있던 이전 요청 {len(outs) - len(ch)}개의 결과는 버림)"); outs = outs[-len(ch):]
        assert len(outs) == len(ch) and all(getattr(o, "prompt", None) in (None, q["prompt"]) for o, q in zip(outs, reqs)), "요청과 결과의 순서가 맞지 않음"
        with open(path, "a", encoding="utf-8") as f:
            for (r, A, B), o in zip(ch, outs):
                have = int(cnt.get(r["id"], 0))
                for j, c in enumerate(o.outputs[have:K_SAMPLES], start=have):
                    f.write(json.dumps({"id": r["id"], "split": split, "sample": j, "letter": parse_answer(c.text), "finish": c.finish_reason,
                                        "ntok": len(c.token_ids), "A": LETTERS[A], "B": LETTERS[B], "text": c.text}, ensure_ascii=False) + "\n")
                    ntok += len(c.token_ids)
        n += len(ch)
        if k == 0:
            _d = read_gen(path); _d = _d[_d.id.isin([r["id"] for r, _, _ in ch])]
            print(f"  첫 묶음 점검: 답을 읽은 표본 {int(_d.letter.notna().sum())}/{len(_d)} | 길이 평균 {_d.ntok.mean():.0f}토큰 (최대 {_d.ntok.max()}) | 길이 제한에 걸림 {int((_d.finish == 'length').sum())}")
            print("  예시 표본 끝부분:", repr(_d.text.iloc[0][-300:]))
            if _d.letter.notna().mean() < 0.5:
                raise RuntimeError("답을 읽어 낸 표본이 절반도 안 됩니다 → 마지막 줄 형식을 안 지킴. 위 예시로 답 형식을 확인하세요 (MAX_TOKENS를 늘리거나 형식 규칙을 고침)")
        if k == 0 or (k + 1) % 5 == 0 or k + 1 == len(chunks):
            el = time.time() - t0
            print(f"  {n:>4}/{len(items)}문항 | {el / n:.1f}s/문항 | 생성 {ntok / max(el, 1e-9):.0f}토큰/s | 남은 시간 ~{(len(items) - n) * el / n / 60:.0f}분", flush=True)
    return read_gen(path)
def vote_matrix(gen, ids_all):
    # 문항별 표본 투표 → q (문항 수 × 4). 답을 읽은 표본이 MIN_VALID보다 적으면 NaN
    pos = {i: n for n, i in enumerate(ids_all)}; Q = np.full((len(ids_all), 4), np.nan); NV = np.zeros(len(ids_all), int)
    g = gen.drop_duplicates(["id", "sample"], keep="last")
    for qid, sub in g.groupby("id"):
        if qid not in pos: continue
        v = sub.letter.dropna(); v = v[v.isin(list(LETTERS))]
        NV[pos[qid]] = len(v)
        if len(v) >= MIN_VALID: Q[pos[qid]] = np.bincount(v.map(LETTERS.index), minlength=4) / len(v)
    return Q, NV
def mix(P, Q, m, tau=TAU, alpha=ALPHA):
    sel = (m < tau) & ~np.isnan(Q).any(1)
    Pn = P.copy(); Pn[sel] = (1 - alpha) * P[sel] + alpha * Q[sel]
    return Pn.argmax(1), sel
def paired(a_old, a_new, y):
    c0, c1 = a_old == y, a_new == y; f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum()); return f, b, f - b, int((a_old != a_new).sum())
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785", "진짜 실수": "2112 0936 5797 5209 1045 1000"}
CLS = {f"train_{x}.jpg": k for k, v in E14_ERR.items() for x in v.split()}
SUMMARY = {"lines": []}
def say(s): print(s); SUMMARY["lines"].append(s)
print("✅ 생성 함수 준비 완료 | 저장:", RUN_DIR)
''')

md("""
## 10. 검증 불확실 문항 교차 검증 → 판정 (1차 가설 한 줄로만)
- 교차 검증 모델 단독 실력도 봅니다. 채점한 문항에서 표본 다수결 답과 조합 답의 정확도를 비교합니다.
- 답이 바뀐 문항마다 표본 하나의 풀이 끝부분을 보여 줍니다(이미지와 대조해 보면 좋습니다).
""")
code(r'''
GEN_V = generate_xc(XC_VAL, TOP2_VAL, VAL_IDS, GEN_VAL, "val")
Q_VAL, NV_VAL = vote_matrix(GEN_V, VAL_IDS)
has = ~np.isnan(Q_VAL).any(1); sc = M_VAL < TAU_SCORE_VAL
maj = np.where(has, np.nan_to_num(Q_VAL).argmax(1), -1)
say(f"[교차 검증 단독] 채점 {int(sc.sum())}문항 중 표본 {MIN_VALID}개 이상 읽음 {int((sc & has).sum())} | 표본 다수결 정답 {int((has & (maj == Y_VAL)).sum())} vs 조합 정답 {int((has & (A_VAL == Y_VAL)).sum())}"
    f" | 조합 1·2위 밖 답을 고른 문항 {int((has & ~(TOP2_VAL == maj[:, None]).any(1)).sum())}")
a1, sel = mix(PB_VAL, Q_VAL, M_VAL); f1, b1, n1, D1 = paired(A_VAL, a1, Y_VAL)
say(f"[1차 가설] τ={TAU}, α={ALPHA}, 표본 {K_SAMPLES}: 검증 {acc_base:.2f}% → {(a1 == Y_VAL).mean() * 100:.2f}% | 고침 {f1} · 새로 틀림 {b1} · 순이득 {n1:+d} · 바뀐 답 {D1} (섞은 문항 {int(sel.sum())})")
VERDICT = (f"채택 (순이득 {n1:+d} ≥ +{NET_MIN})" if n1 >= NET_MIN else f"공개 1회 확인 후보 (순이득 {n1:+d})" if n1 >= 1 else f"불채택 (순이득 {n1:+d})")
say("판정: " + VERDICT)
_g = []
for t in GRID_TAUS:
    for a_ in GRID_ALPHAS:
        x, s_ = mix(PB_VAL, Q_VAL, M_VAL, t, a_); ff, bb, nn, dd = paired(A_VAL, x, Y_VAL)
        _g.append({"τ": t, "α": a_, "섞은 문항": int(s_.sum()), "고침": ff, "새로 틀림": bb, "순이득": nn, "바뀐 답": dd, "1차": "◀" if (t == TAU and a_ == ALPHA) else ""})
print("\n참고 표 (판정에 쓰지 않음):"); print(pd.DataFrame(_g).to_string(index=False))
ch = np.where(A_VAL != a1)[0]
if len(ch):
    _gv = GEN_V.drop_duplicates(["id", "sample"], keep="last")
    rows = []
    for i in ch:
        v = _gv[_gv.id == VAL_IDS[i]]
        rows.append({"id": VAL_IDS[i], "결과": "고침" if a1[i] == Y_VAL[i] else ("새로 틀림" if A_VAL[i] == Y_VAL[i] else "둘 다 틀림"), "정답": LETTERS[Y_VAL[i]],
                     "조합": LETTERS[A_VAL[i]], "E20": LETTERS[a1[i]], "투표": " ".join(f"{L}{int((v.letter == L).sum())}" for L in LETTERS if (v.letter == L).any()),
                     "p정답": round(float(PB_VAL[i, Y_VAL[i]]), 3), "1·2위 차": round(float(M_VAL[i]), 3), "분류": CLS.get(VAL_IDS[i], ""), "질문": str(val_df.question.iloc[i])[:36]})
    print("\n답이 바뀐 검증 문항:"); print(pd.DataFrame(rows).to_string(index=False))
    for i in ch[:6]:
        v = _gv[(_gv.id == VAL_IDS[i]) & (_gv.letter == LETTERS[a1[i]])]
        if len(v): print(f"\n── {VAL_IDS[i]} (정답 {LETTERS[Y_VAL[i]]}, E20 {LETTERS[a1[i]]}) 풀이 끝부분:\n{v.text.iloc[0][-500:]}")
wrong_unc = np.where((A_VAL != Y_VAL) & (M_VAL < TAU))[0]
if len(wrong_unc):
    print(f"\n조합이 틀린 불확실 문항 {len(wrong_unc)}개의 투표: " + " | ".join(
        f"{VAL_IDS[i].replace('train_', '').replace('.jpg', '')} 정답{LETTERS[Y_VAL[i]]}/조합{LETTERS[A_VAL[i]]} → " + ("".join(f"{L}{int(round(Q_VAL[i, k] * NV_VAL[i]))}" for k, L in enumerate(LETTERS) if Q_VAL[i, k] > 0) if not np.isnan(Q_VAL[i]).any() else "표본 부족")
        for i in wrong_unc))
''')

md("""
## 11. test 불확실 문항 교차 검증 → 제출 후보 (검증 순이득 ≥ +1일 때 자동)
- 조합의 test 답에서, 교차 검증 때문에 바뀐 문항만 바꾼 제출 파일을 만듭니다.
- 이름은 `submission_e20_<조합 해시>_t05_a05.csv`입니다.
- 두 제출의 test 답이 D개 다르면 공개에서는 약 D/2개가 갈리고, 운만으로도 ±2√(D/2)문항 오르내립니다.
""")
code(r'''
DO_TEST = (RUN_TEST is True) or (RUN_TEST == "auto" and n1 >= 1)
if not DO_TEST:
    print(f"test 생략 (RUN_TEST={RUN_TEST!r} | {VERDICT}). 그래도 돌리려면 1번 셀에서 RUN_TEST = True로 바꾸고 이 셀만 다시 실행하세요")
else:
    GEN_T = generate_xc(XC_TEST, TOP2_TEST, TEST_IDS, GEN_TEST, "test")
    Q_TEST, NV_TEST = vote_matrix(GEN_T, TEST_IDS)
    at, selt = mix(PB_TEST, Q_TEST, M_TEST); D = int((at != A_TEST).sum())
    fn = (f"submission_e20_{BASE_TAG}_t{TAU:g}_a{ALPHA:g}".replace(".", "")) + ".csv"; path = os.path.join(RUN_DIR, fn)
    sub = sub_df[["id"]].assign(answer=[LETTERS[i] for i in at]); assert len(sub) == len(sub_df) and set(sub.answer) <= set(LETTERS)
    sub.to_csv(path, index=False)
    meta = {"name": f"{BASE_INFO.get('name', BASE_TAG)} + E20", "file": fn, "base_json": os.path.relpath(BASE_JSON, OUT_DIR), "base_file": BASE_INFO["file"],
            "model": MODEL_ID, "tau": TAU, "alpha": ALPHA, "k": K_SAMPLES, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_tokens": MAX_TOKENS,
            "thinking": THINKING, "min_valid": MIN_VALID, "gen_val": os.path.relpath(GEN_VAL, OUT_DIR), "gen_test": os.path.relpath(GEN_TEST, OUT_DIR),
            "gen_test_md5": md5file(GEN_TEST), "val": {"fixed": f1, "broken": b1, "net": n1, "D": D1, "verdict": VERDICT},
            "test": {"changed": D, "mixed": int(selt.sum()), "scored": int(len(XC_TEST)), "dist": dict(zip(LETTERS, np.bincount(at, minlength=4).tolist()))},
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(path[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 재현 점검: 저장한 생성 파일만으로 다시 섞어 같은 답인지
    _Q2, _ = vote_matrix(read_gen(GEN_TEST), TEST_IDS); _a2, _ = mix(PB_TEST, _Q2, M_TEST)
    ok = bool((_a2 == at).all())
    band = 2 * math.sqrt(D / 2) if D else 0.0
    say(f"✅ 저장: {RUN_NAME}/{fn} (+ 설정 JSON) | 재현 {'✅' if ok else '⚠️'} | 조합과 test 답이 다른 문항 {D}개 (채점 {len(XC_TEST)} · 섞은 문항 {int(selt.sum())})")
    if D: say(f"  → 공개 채점에서 약 {D / 2:.0f}문항이 갈림: 운만으로도 ±{band:.0f}문항. 공개에서 +{math.ceil(band)}문항 이상이어야 E20 덕분이라고 볼 수 있음")
    assert ok, "저장한 생성 파일로 다시 만든 답이 다릅니다 → 위 출력을 확인하세요"
''')

md("## 12. 결과 요약")
code(r'''
print(f"E20 | {MODEL_ID} | 기준 {BASE_INFO.get('name', BASE_TAG)} ({BASE_TAG}) {acc_base:.2f}% | τ={TAU} α={ALPHA} K={K_SAMPLES} T={TEMPERATURE} 생각={THINKING} | {GPU_NAME}")
for _l in SUMMARY["lines"]: print(_l)
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "G4", "machine_shape": "hm"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
