# -*- coding: utf-8 -*-
"""RunPod 판정 모델 노트북 빌더 — 답이 갈린 문항 교차 검증.
Colab anchor_rescue(6-1번 셀)가 HF 비공개 저장소에 올린 판정 목록(기준 모델과 보조 모델의 답이 갈린 문항)만,
기준 답과 보조 모델 최다 대안을 '모델 A/B의 답'으로 보여 주고 MODEL_ID 모델에게 단계별로 따져 최종 답을 내게 함
→ 문항마다 표본 K개 다수결 = 이 판정 모델의 한 표 → E14 형식 저장(목록 밖은 기준 답 = 표 없음) → HF 올리기 → anchor_rescue가 한 표로 씀.
설치·데이터·분할·프롬프트·이미지·모델 받기·vLLM 로드·올리기 셀은 runpod_zs_any(B300에서 돌아간 것)를 그대로 쓰고, 생성·답 읽기는 E20과 같다.
사용: python3 build_runpod_xcheck.py [출력 경로]"""
import sys, json, uuid, os, io, contextlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/xcheck/out/runpod_xcheck_any.ipynb"
SPD = "/tmp/vqa_scratch"
_a = {}
_argv = sys.argv; sys.argv = ["build_runpod_zs.py", f"{SPD}/_rpzs_tmp_xc.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/zsany/build_runpod_zs.py", encoding="utf-8").read(), _a)
sys.argv = _argv
A = _a["cells"]
def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:90]); return text.replace(old, new)
def find(marker):
    hits = [c["source"] for c in A if c["cell_type"] == "code" and marker in c["source"]]
    assert len(hits) == 1, (marker, len(hits)); return hits[0]
CFG, INSTALL, DATA_C, CSV_C = find('MODEL_ID  = "google/gemma-4-31B-it"'), find('VLLM_VERSION = "0.30.0"'), find("hf_hub_download"), find("def make_split")
PROMPT_C, IMG_C, DL_C, LLM_C, UP_C = find("def build_mc_prompt"), find('"max_soft_tokens"'), find("snapshot_download(MODEL_ID"), find("llm = LLM(model=MODEL_ID"), find("upload_folder")

# ── 1. 설정: 판정 모델용 ──
s = CFG
i0, i1 = s.index('MODEL_ID  = "google/gemma-4-31B-it"'), s.index("# ── 모델별 설정")
s = s[:i0] + '''MODEL_ID  = "google/gemma-4-31B-it"   # ← 판정 모델. 여기만 바꾸세요. 추천 순서: google/gemma-4-31B-it → Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 → naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
PRESET    = "up2.0"                   # 해상도 (다른 멤버와 같음)
RUN_NAME  = "auto"                    # "auto" → xc_<모델 이름>_<해상도>
TARGETS_FILE = ""                     # 판정 목록. 비워 두면 HF 비공개 저장소의 xcheck/xcheck_targets.csv (Colab anchor_rescue 6-1번 셀이 올림). 직접 올렸으면 그 파일 경로

# ── 교차 검증 프롬프트 생성 (결과를 보기 전에 고정 · 판정 모델 모두 같은 값) ──
K_SAMPLES   = 5                       # 문항당 표본 수 → 다수결 = 이 판정 모델의 한 표
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.8, 20
MAX_TOKENS  = 1200                    # 단계별 풀이 + 마지막 줄 '정답: (x)'. 첫 묶음 점검에서 '길이 제한에 걸림'이 많으면 늘림
MIN_VALID   = 3                       # 답을 읽어 낸 표본이 이보다 적거나 최다 표가 동점이면 기권 (= 표를 내지 않음)
THINKING    = False                   # 생각 모드 끔 (템플릿이 지원할 때만 적용)
GEN_CHUNK   = 32                      # 한 번에 넘기는 문항 수 = 저장 단위
GPU_ID    = None                      # GPU 여러 장 pod에서 노트북 여러 개를 동시에: 노트북마다 0, 1, 2, … (None = 전부 보이게)
VAL_N, PROMPT_VERSION_USE, ANSWER_PREFIX = 1000, "baseline_v1", ""   # 아래 공용 셀용 (바꾸지 마세요)

''' + s[i1:]
j0, j1 = s.index("MODEL_PRESETS = {"), s.index("# ── 공통 기본값")
s = s[:j0] + '''MODEL_PRESETS = {
    "google/gemma-4-31B-it":            {},                                   # 판정 1순위: zero-shot 97.67% (앞 300문항) · Qwen과 다른 계열. 라이선스 동의 + HF 토큰
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4": {"QUANTIZATION": "moe_wna16"},       # 판정 2순위 (zero-shot 검증 96.0%)
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B": {"TRUST_REMOTE_CODE": True},   # 판정 3순위: 네이버·한국어 (95.67%). vLLM 지원이 불확실 → 로드가 안 되면 건너뜀
    "OpenGVLab/InternVL3_5-38B-HF":     {},                                   # 선택 (94.33%)
    "Qwen/Qwen3.5-27B":                 {},                                   # 선택 (96.33%, 122B와 같은 계열)
    "Qwen/Qwen3.6-35B-A3B":             {},                                   # 선택 (96.33%, 122B와 같은 계열)
}
''' + s[j1:]
s = rep(s, "MAX_NUM_SEQS  = 32              # B300이면 64로 올려도 됨 (시작이 실패하면 16)", "MAX_NUM_SEQS  = 64              # 생성은 표본 여러 개를 동시에 돌려야 빠름 (시작이 실패하면 32)")
s = rep(s, 'RUN_NAME = "zs_" + ', 'RUN_NAME = "xc_" + ')
s = rep(s, "| VAL_N = {VAL_N} |", "| 표본 {K_SAMPLES}개 · T {TEMPERATURE} · 최대 {MAX_TOKENS}토큰 |")
CFG_X = s

# ── 5. 프롬프트: 채점용 셀 그대로 + 교차 검증 프롬프트 (E20과 같은 문구) ──
PROMPT_X = PROMPT_C.rstrip() + r'''

# ── 교차 검증 프롬프트 (E20과 같은 문구, 결과를 보기 전에 고정) ──
XC_TEMPLATE_KW = {"enable_thinking": bool(THINKING)} if "enable_thinking" in (_tmpl or "") else {}
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
def xc_pair(qid, anchor, alt):
    # 기준 답·대안을 모델 A/B 자리에 무작위로 (문항 id로 고정 → 다시 돌려도 같음, 판정 모델이 어느 쪽이 기준인지 모름)
    flip = int(hashlib.md5(f"{qid}|xc".encode()).hexdigest(), 16) % 2
    return (alt, anchor) if flip else (anchor, alt)
def render_xc(row, A, B):
    return _apply(xc_messages(row, A, B), tokenize=False, add_generation_prompt=True, **XC_TEMPLATE_KW)
# 마지막 '정답: (x)'를 읽음. 괄호 없는 대문자 A·B는 '모델 A/B'와 헷갈려서 답으로 치지 않음
_ANS_RE = re.compile(r"정답\s*[:：]?\s*\**\s*[\(\[（]\s*([abcdABCD])\s*[\)\]）]|정답\s*[:：]\s*\**\s*([abcdCD])(?![A-Za-z가-힣])")
def parse_answer(text):
    t = text.split("</think>")[-1]                         # 생각 모드면 생각 뒤 부분만
    m = list(_ANS_RE.finditer(t))
    if m: return (m[-1].group(1) or m[-1].group(2)).lower()
    tail = t.strip().splitlines()[-1] if t.strip() else ""
    m2 = re.findall(r"\(([abcd])\)", tail)
    return m2[-1] if m2 else None
for _s, _e in [("분석 끝.\n정답: (c)", "c"), ("정답은 (a)일 수도 있지만 ... \n정답: b", "b"), ("최종 정답：（d）", "d"), ("답을 모르겠다", None),
               ("<think>정답: (a)</think>\n정답: (c)", "c"), ("**정답: (b)**", "b"), ("정답: 모델 A", None), ("정답: B", None), ("정답: d", "d")]:
    assert parse_answer(_s) == _e, (_s, parse_answer(_s))
_r0 = VAL_USE.iloc[0].to_dict(); _A0, _B0 = xc_pair(_r0["id"], 0, 1)
_px = render_xc(_r0, _A0, _B0)
if IMAGE_TOKEN and _px.count(IMAGE_TOKEN) != 1: print(f"ℹ️ 판정 프롬프트에 이미지 토큰 {IMAGE_TOKEN!r}가 {_px.count(IMAGE_TOKEN)}번 보입니다 (첫 묶음 점검에서 확인)")
print(f"판정 템플릿 추가 인자 {XC_TEMPLATE_KW} | 예시 끝부분:"); print(_px[-600:])
'''

TARGETS_X = r'''
from huggingface_hub import hf_hub_download
if TARGETS_FILE:
    _tp = TARGETS_FILE
else:
    _tp = hf_hub_download(HF_REPO, "xcheck/xcheck_targets.csv", repo_type="dataset", token=HF_TOKEN or None, force_download=True, local_dir="/workspace/xcheck")
assert os.path.exists(_tp), f"판정 목록이 없습니다: {_tp}"
TG = pd.read_csv(_tp)
_need = {"id", "split", "anchor", "alt", "n_alt", "target", "list_tag", "anchor_run"}
assert _need <= set(TG.columns), f"판정 목록 열이 부족합니다: {list(TG.columns)} → Colab anchor_rescue 6-1번 셀이 만든 파일인지 확인하세요"
TG_TAG, XC_ANCHOR = str(TG.list_tag.iloc[0]), str(TG.anchor_run.iloc[0])
_tv, _tt = TG[TG.split == "val"].set_index("id"), TG[TG.split == "test"].set_index("id")
assert set(_tv.index) == set(val_df.id), "판정 목록의 검증 문항이 팀 공통 분할과 다릅니다 → 위 출력을 확인하세요"
assert set(_tt.index) == set(test_df.id), "판정 목록의 test 문항이 test.csv와 다릅니다 → 위 출력을 확인하세요"
assert TG.anchor.isin(list(LETTERS)).all() and TG.alt.isin(list(LETTERS)).all() and (TG[TG.target == 1].anchor != TG[TG.target == 1].alt).all(), "판정 목록의 답 글자 이상"
ANCH_V, ANCH_T = _tv.anchor.to_dict(), _tt.anchor.to_dict()
XV = val_df.merge(_tv[_tv.target == 1][["anchor", "alt", "n_alt"]].reset_index(), on="id")       # 검증 대상 (정답 있음 → 판정 모델 성적)
XT = test_df.merge(_tt[_tt.target == 1][["anchor", "alt", "n_alt"]].reset_index(), on="id")
_aw = int((_tv.loc[val_df.id].anchor.to_numpy() != val_df.answer.to_numpy()).sum())
print(f"판정 목록 {TG_TAG} (기준 {XC_ANCHOR}) | 검증 대상 {len(XV)}/{len(val_df)} — 기준 오답 {int((XV.anchor != XV.answer).sum())}/{_aw} 포함, 대안이 정답 {int((XV.alt == XV.answer).sum())}"
      f" | test 대상 {len(XT)}/{len(test_df)} | 표본 {K_SAMPLES}개씩 → 생성 {(len(XV) + len(XT)) * K_SAMPLES:,}개")
'''

GEN_X = r'''
from concurrent.futures import ThreadPoolExecutor
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
GEN_VAL, GEN_TEST = os.path.join(RUN_DIR, f"xc_val__{PRESET}.jsonl"), os.path.join(RUN_DIR, f"xc_test__{PRESET}.jsonl")
_XPOOL, _XPRE = ThreadPoolExecutor(max_workers=IMG_WORKERS), ThreadPoolExecutor(max_workers=1)
GEN_COLS = ["id", "split", "sample", "letter", "finish", "ntok", "A", "B", "text"]
def _seed_of(qid): return int(hashlib.md5(f"{qid}|{SEED}".encode()).hexdigest(), 16) % (2 ** 31)
def _xc_req(item):
    r, A, B = item
    return {"prompt": render_xc(r, A, B), "multi_modal_data": {"image": load_image(r["path"])}}
def read_gen(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: rows.append(json.loads(line))
            except Exception: pass                         # 끊겨서 반쯤 쓴 줄
    return pd.DataFrame(rows, columns=GEN_COLS)
def _fix_tail(path):                   # 저장 도중 끊겨 마지막 줄이 줄바꿈 없이 끝났으면 줄바꿈을 붙여 다음 줄과 섞이지 않게
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n": f.write(b"\n")
def _items(df):
    return [(r, *xc_pair(r["id"], LETTERS.index(r["anchor"]), LETTERS.index(r["alt"]))) for r in df.to_dict("records")]
def _done_counts(path):
    # 같은 문항·같은 A/B 짝으로 이미 뽑은 표본 수 (목록이 바뀌어 짝이 달라진 문항은 새로 판정)
    d = read_gen(path)
    if not len(d): return {}
    return d.drop_duplicates(["id", "A", "B", "sample"], keep="last").groupby(["id", "A", "B"]).size().to_dict()
def generate_xc(df, path, split):
    _fix_tail(path)
    cnt = _done_counts(path)
    items = [(r, A, B) for r, A, B in _items(df) if cnt.get((r["id"], LETTERS[A], LETTERS[B]), 0) < K_SAMPLES]
    print(f"[{split}] {os.path.basename(path)} | 대상 {len(df)}문항 | 완료 {len(df) - len(items)} | 남음 {len(items)} | 표본 {K_SAMPLES}개씩", flush=True)
    if not items: return read_gen(path)
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
                have = int(cnt.get((r["id"], LETTERS[A], LETTERS[B]), 0))
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
                raise RuntimeError("답을 읽어 낸 표본이 절반도 안 됩니다 → 마지막 줄 형식을 안 지키거나 길이 제한에 걸림. "
                                   "'길이 제한에 걸림'이 많으면 1번 셀 MAX_TOKENS를 늘리고, 아니면 위 예시로 답 형식을 확인하세요")
        if k == 0 or (k + 1) % 5 == 0 or k + 1 == len(chunks):
            el = time.time() - t0
            print(f"  {n:>4}/{len(items)}문항 | {el / n:.1f}s/문항 | 생성 {ntok / max(el, 1e-9):.0f}토큰/s | 남은 시간 ~{(len(items) - n) * el / n / 60:.0f}분", flush=True)
    return read_gen(path)
def counts_of(gen, df):
    # 문항별 표본 답 수 (지금 목록의 A/B 짝으로 뽑은 표본만)
    want = {r["id"]: (LETTERS[A], LETTERS[B]) for r, A, B in _items(df)}; out = {}
    if not len(gen): return out
    g = gen.drop_duplicates(["id", "A", "B", "sample"], keep="last")
    for (qid, a_, b_), sub in g.groupby(["id", "A", "B"]):
        if want.get(qid) != (a_, b_): continue
        v = sub.letter.dropna(); v = v[v.isin(list(LETTERS))]
        out[qid] = np.bincount(v.map(LETTERS.index).astype(int).to_numpy(), minlength=4)
    return out
def decide(c):
    # 이 판정 모델의 한 표: 읽은 표본 ≥ MIN_VALID이고 1등이 하나일 때만. 아니면 기권(None)
    if c is None or c.sum() < MIN_VALID: return None
    w = np.flatnonzero(c == c.max()); return int(w[0]) if len(w) == 1 else None
def preds_frame(ids, anchor_map, cnt):
    # E14 형식: 판정한 문항은 표본 비율, 기권·목록 밖 문항은 기준 답(= 다른 보기에 표를 주지 않음)
    rows = []
    for i in ids:
        a = LETTERS.index(anchor_map[i]); c = cnt.get(i); v = decide(c)
        p = (c / c.sum()) if v is not None else np.eye(4)[a]
        rows.append({"id": i, "pred": LETTERS[int(p.argmax())], "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": 1.0,
                     "judged": int(v is not None), "vote": LETTERS[v] if v is not None else "", "n_valid": int(c.sum()) if c is not None else 0, "anchor": anchor_map[i]})
    return pd.DataFrame(rows)
SUMMARY = {"lines": []}
def say(s): print(s); SUMMARY["lines"].append(s)
print("✅ 생성 함수 준비 완료 | 저장:", RUN_DIR)
'''

RUN_X = r'''
GEN_V = generate_xc(XV, GEN_VAL, "val"); CNT_V = counts_of(GEN_V, XV)
GEN_T = generate_xc(XT, GEN_TEST, "test"); CNT_T = counts_of(GEN_T, XT)
_rows = []
for r, A, B in _items(XV):
    c = CNT_V.get(r["id"]); v = decide(c)
    _rows.append({"id": r["id"], "정답": r["answer"], "기준": r["anchor"], "대안": r["alt"], "판정": LETTERS[v] if v is not None else "기권",
                  "A자리": LETTERS[A], "B자리": LETTERS[B], "투표": " ".join(f"{L}{int(c[k])}" for k, L in enumerate(LETTERS) if c[k] > 0) if c is not None else "", "대안 표": r["n_alt"],
                  "질문": str(r["question"])[:34]})
JV = pd.DataFrame(_rows, columns=["id", "정답", "기준", "대안", "판정", "A자리", "B자리", "투표", "대안 표", "질문"]); dec = JV[JV.판정 != "기권"]
_g = pd.concat([GEN_V, GEN_T]) if len(GEN_T) else GEN_V
say(f"[판정 {MODEL_ID}] 검증 대상 {len(JV)} | 판정 {len(dec)} (기권 {len(JV) - len(dec)}) | 판정 정답 {int((dec.판정 == dec.정답).sum())} vs 같은 문항 기준 정답 {int((dec.기준 == dec.정답).sum())}"
    f" | 판정 = 기준 {int((dec.판정 == dec.기준).sum())} · = 대안 {int((dec.판정 == dec.대안).sum())} · 그 밖 {int(((dec.판정 != dec.기준) & (dec.판정 != dec.대안)).sum())}")
_ch = dec[dec.판정 != dec.기준]
say(f"  기준과 다른 답 {len(_ch)}문항: 맞게 바꿈 {int((_ch.판정 == _ch.정답).sum())} · 기준이 맞았는데 바꿈 {int((_ch.기준 == _ch.정답).sum())} (이 모델 혼자 기준. 실제로 뒤집을지는 anchor_rescue가 정함)")
say(f"  표본 {len(_g)}개: 답 읽음 {int(_g.letter.notna().sum())} ({_g.letter.notna().mean() * 100:.1f}%) · 길이 제한 {int((_g.finish == 'length').sum())} · 평균 {_g.ntok.mean():.0f}토큰"
    f" | 판정이 A 자리 보기 {int((dec.판정 == dec.A자리).sum())} · B 자리 {int((dec.판정 == dec.B자리).sum())} · 둘 다 아님 {int(((dec.판정 != dec.A자리) & (dec.판정 != dec.B자리)).sum())} (A·B가 한쪽으로 크게 쏠리면 자리 편향)")
if len(_ch): print("\n기준과 다른 답을 고른 검증 문항:"); print(_ch.drop(columns=["A자리", "B자리"]).to_string(index=False))
_dt = [decide(CNT_T.get(i)) for i in XT.id]
say(f"  test 대상 {len(XT)} | 판정 {sum(v is not None for v in _dt)} · 기준과 다른 답 {sum(v is not None and LETTERS[v] != a for v, a in zip(_dt, XT.anchor))}")
'''

SAVE_X = r'''
VAL_FINAL = os.path.join(RUN_DIR, f"preds_final_ep1__{PRESET}__{len(val_df)}.csv")
TEST_FINAL = os.path.join(RUN_DIR, f"preds_final_ep1_test__{PRESET}__{len(test_df)}.csv")
PVX, PTX = preds_frame(val_df.id.tolist(), ANCH_V, CNT_V), preds_frame(test_df.id.tolist(), ANCH_T, CNT_T)
PVX.to_csv(VAL_FINAL, index=False); PTX.to_csv(TEST_FINAL, index=False)
acc_x = float((PVX.pred.to_numpy() == val_df.answer.to_numpy()).mean())
json.dump({"epoch_acc": {"1": acc_x},
           "config": {"MODEL_ID": "xcheck:" + MODEL_ID, "JUDGE_MODEL_ID": MODEL_ID, "ZERO_SHOT": True, "XCHECK": True, "XCHECK_ANCHOR": XC_ANCHOR, "TARGETS_TAG": TG_TAG,
                      "TRAIN_ON_FULL": False, "SEED": SEED, "EVAL_PRESET": PRESET, "PROMPT_VERSION": "xcheck_v1", "K_SAMPLES": K_SAMPLES, "TEMPERATURE": TEMPERATURE,
                      "TOP_P": TOP_P, "TOP_K": TOP_K, "MAX_TOKENS": MAX_TOKENS, "MIN_VALID": MIN_VALID, "THINKING": THINKING, "TEMPLATE_KW": XC_TEMPLATE_KW,
                      "MM_KW": MM_KW, "ENGINE": f"vllm {vllm.__version__}", "QUANTIZATION": QUANTIZATION, "SPLIT_MD5": split_md5},
           "xcheck": {"val_targets": len(XV), "val_judged": int(PVX.judged.sum()), "test_targets": len(XT), "test_judged": int(PTX.judged.sum()), "lines": SUMMARY["lines"]}},
          open(os.path.join(RUN_DIR, "train_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
# 재현 점검: 저장한 생성 파일만으로 다시 만들어 같은 표인지
_rv = preds_frame(val_df.id.tolist(), ANCH_V, counts_of(read_gen(GEN_VAL), XV)); _rt = preds_frame(test_df.id.tolist(), ANCH_T, counts_of(read_gen(GEN_TEST), XT))
_same = bool((_rv.pred.to_numpy() == PVX.pred.to_numpy()).all() and (_rt.pred.to_numpy() == PTX.pred.to_numpy()).all())
say(f"✅ E14 형식 저장: {os.path.basename(VAL_FINAL)} (판정 {int(PVX.judged.sum())}) · {os.path.basename(TEST_FINAL)} (판정 {int(PTX.judged.sum())}) · train_log.json | 재현 {'✅' if _same else '⚠️'} → {RUN_DIR}")
assert _same, "저장한 생성 파일로 다시 만든 표가 다릅니다 → 위 출력을 확인하세요"
'''

UP_X = rep(UP_C, "→ 9·10번 셀이 끝났는지(VAL_N ≥ 1000, DO_TEST=True) 확인하세요", "→ 11·12번 셀이 끝났는지 확인하세요")
UP_X = rep(UP_X, 'commit_message=f"RunPod zero-shot {RUN_NAME}"', 'commit_message=f"RunPod xcheck {RUN_NAME}"')
UP_X = rep(UP_X, 'print("다음: ① 다른 모델이면 MODEL_ID를 바꾸고 Kernel → Restart Kernel and Run All Cells ② 다 끝나면 Colab 브리지 노트북 B 단계 → Drive 확인 → pod Terminate")',
           'print("다음: ① 다른 판정 모델이면 MODEL_ID를 바꾸고 Kernel → Restart Kernel and Run All Cells ② 다 끝나면 Colab 브리지 노트북 B 단계 → anchor_rescue 다시 실행 → pod Terminate")')

SUMM_X = r'''
print(f"판정 | {MODEL_ID} | {RUN_NAME} | 목록 {TG_TAG} (기준 {XC_ANCHOR}) | 표본 {K_SAMPLES} · T {TEMPERATURE} · 최대 {MAX_TOKENS}토큰 · 생각 {THINKING} | vllm {vllm.__version__} | {GPU_NAME}")
for _l in SUMMARY["lines"]: print(_l)
'''

# ── HyperCLOVA X 대응 (저장소 커밋 고정 · 생각 블록 닫기 · 형식 점검) ──
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hcx_patch.py"), encoding="utf-8").read())

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n") + "\n"})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n") + "\n"})
md("""
# RunPod · 판정 모델 (답이 갈린 문항 교차 검증) — `MODEL_ID`만 바꿔서 실행 · 학습 없음

**하는 일**
1. Colab `anchor_rescue` 6-1번 셀이 HF 비공개 저장소에 올린 **판정 목록**을 받습니다.
   - 판정 목록은 기준 모델(Flash-Next)과 보조 모델의 답이 갈린 문항입니다(검증·test).
2. 문항마다 이미지·질문·보기와 함께 두 답을 "모델 A의 답 / 모델 B의 답"으로 보여 줍니다. 어느 쪽이 기준인지는 문항마다 무작위입니다.
   - 두 후보 답(모델 A/B)을 보여 주고 근거를 교차 검증하게 하는 프롬프트로 묻습니다. 흔한 착각 패턴을 배제하고 단계별로 결론을 내게 합니다.
   - 마지막 줄 `정답: (x)`를 읽습니다.
3. 문항마다 표본 `K_SAMPLES`개(기본 5개)를 뽑고, 다수결로 이 판정 모델의 **한 표**를 정합니다. 표본이 모자라거나 최다 표가 동점이면 기권합니다.
4. 결과를 E14 형식으로 저장하고 HF 저장소 `vqa_runs/xc_<모델>_up20`에 올립니다. 목록 밖 문항은 기준 답으로 채웁니다(표를 내지 않는다는 뜻).
5. 브리지 B 단계를 거쳐 `anchor_rescue`를 다시 돌리면 판정 모델이 한 표씩 들어갑니다. 뒤집을지는 같은 검증(교차 적합·부트스트랩·2배 법칙)이 정합니다.

**실행 순서**
1. B300 pod Jupyter에서 **Run All**
   - 설치·데이터·가중치 캐시는 zero-shot 노트북과 같은 곳을 씁니다. 이미 받은 것은 건너뜁니다.
2. 다음 판정 모델: 1번 셀 `MODEL_ID`만 바꾸고 **Kernel → Restart Kernel and Run All Cells**
   - 추천 순서: `google/gemma-4-31B-it` → `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` → `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B`
3. 다 끝나면 Colab 브리지 B 단계 → `anchor_rescue` 다시 실행 → pod **Terminate**

**사전 기준 (결과를 보기 전에 고정)**
- 판정 모델은 모두 같은 프롬프트·표본 5개·T 0.7·생각 모드 끔으로 돌립니다.
- 돌린 판정 모델은 성적과 상관없이 모두 넣습니다. 성적을 보고 골라 빼면 과적합입니다.
- 채택 여부는 `anchor_rescue`의 검증이 정합니다.

끊기면 다시 Run All 하면 이어서 합니다. B300 오류 대책(Triton 어텐션·Triton MoE·FlashInfer 자동 튜닝 끔)은 zero-shot 노트북과 같습니다.
""")
md("## 1. 설정 — `MODEL_ID`만 바꾸세요"); code(CFG_X)
md("## 2. 설치 (zero-shot 노트북과 같음, 이미 설치돼 있으면 건너뜀)"); code(INSTALL)
md("## 3. 대회 데이터 (HF 비공개 저장소의 zip → /workspace/data)"); code(DATA_C)
md("## 4. CSV · 팀 공통 검증 분할"); code(CSV_C)
md("## 5. 프롬프트 · 교차 검증 프롬프트 · 답 읽기"); code(PROMPT_X)
md("""
## 6. 판정 목록 받기 (Colab `anchor_rescue` 6-1번 셀이 만든 것)
- 모델을 받기 전에 대상 문항 수를 먼저 보여 줍니다.
""")
code(TARGETS_X)
md("## 7. 이미지 해상도 점검 (zero-shot 노트북과 같음)"); code(IMG_C)
md("## 8. 모델 파일 받기 (이미 받은 부분은 다시 안 받음)"); code(DL_C)
md("""
## 9. vLLM으로 모델 로드
오류가 나면 마지막 20줄을 확인하세요. 설정을 바꾼 뒤에는 **Kernel → Restart Kernel and Run All Cells**로 다시 시작하세요.
""")
code(LLM_C)
md("""
## 10. 판정 생성 함수 (끊겨도 이어서)
- 문항마다 표본 `K_SAMPLES`개를 한 요청(`n`)으로 뽑습니다. 문항별 시드를 써서 다시 돌려도 같습니다.
- 표본마다 `id, split, sample, letter, finish, ntok, A, B, text`를 JSONL로 저장합니다. 목록이 바뀌어 A/B 짝이 달라진 문항은 새로 판정합니다.
""")
code(GEN_X)
md("""
## 11. 판정 실행: 검증 대상 → test 대상
- 검증 대상에서는 정답을 알기 때문에 이 판정 모델 혼자의 성적(판정 다수결 vs 기준)을 보여 줍니다. 참고용이고, 이것으로 판정 모델을 고르지 않습니다.
- "A 자리 / B 자리" 수가 한쪽으로 크게 쏠리면 보기 자리 편향입니다.
""")
code(RUN_X)
md("## 12. E14 형식 저장 (anchor_rescue가 한 표로 읽는 파일) · 재현 점검"); code(SAVE_X)
md("## 13. 결과를 HF 비공개 저장소로 올리기 → Colab 브리지 B 단계"); code(UP_X)
md("## 14. 결과 요약"); code(SUMM_X)

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
