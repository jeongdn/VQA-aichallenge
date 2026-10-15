# -*- coding: utf-8 -*-
"""ocr_paddle.ipynb 시험: 작은 무작위 PaddleOCR-VL(실제 구조·실제 프로세서)로 CPU 실행.
시나리오: real(실제 generate, LIMIT) · s0_crash(샤드 0 도중 중단) · s0(이어하기) · s1(샤드 1 → 합치기 완료 → 점검)
fake 모드는 ocr_batch만 설계된 출력(글자+위치 토큰)으로 바꿔 파싱·정렬·중복 제거·합치기·점검 수치를 독립 계산과 대조."""
import sys, os, json, glob, shutil, hashlib, re
import numpy as np, pandas as pd

NB = "/home/user/vqa/ocr/out/ocr_paddle.ipynb"
SP = "/tmp/vqa_scratch"
DATA = f"{SP}/newfam/fx/content"; RUNS = f"{SP}/ocrtest/vqa_runs"; TINY = f"{SP}/ocrtest/tiny_pvl"
scen = sys.argv[1]
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, LIST, MODEL, RUN, MERGE, CHECK, SUMM = cells
TAG = "t_spot" if scen not in ("real", "fb") else "t_real"
if scen in ("real", "fb", "s0_crash"): shutil.rmtree(f"{RUNS}/ocr_paddle/{TAG}", ignore_errors=True)
if scen == "fb":                                                    # 토크나이저에 image_token이 없는 판을 흉내 → 조립 경로
    import transformers, huggingface_hub
    _orig = transformers.AutoProcessor.from_pretrained; _n = {"k": 0}
    def _bad(mid, **kw):
        _n["k"] += 1; raise AttributeError("TokenizersBackend has no attribute image_token")
    transformers.AutoProcessor.from_pretrained = _bad
    huggingface_hub.hf_hub_download = lambda repo, fn, **kw: "/home/user/vqa/ocr/pvl16_chat_template.jinja"
    _orig_tok = transformers.AutoTokenizer.from_pretrained
    def _tok_noattr(mid, **kw):
        t = _orig_tok(mid, **kw)
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_object=t._tokenizer, bos_token=t.bos_token, eos_token=t.eos_token, pad_token=t.pad_token, unk_token=t.unk_token)
    transformers.AutoTokenizer.from_pretrained = _tok_noattr
os.makedirs(RUNS, exist_ok=True)

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)

BOX_TOP, BOX_ANS, BOX_WRONG = (100, 40, 300, 40, 300, 80, 100, 80), (120, 600, 400, 600, 400, 650, 120, 650), (700, 300, 900, 300, 900, 340, 700, 340)
def design(path, row):
    h = int(hashlib.md5(path.encode()).hexdigest()[:8], 16); k = h % 10
    ans = row.get("answer", "a") if isinstance(row.get("answer", None), str) else "a"
    wrong = "abcd"[("abcd".index(ans) + 1) % 4]
    lines = [("영업시간 10:00~22:00", BOX_TOP)]
    if k <= 6: lines.append((str(row[ans]), BOX_ANS))
    elif k <= 8: lines.append((str(row[wrong]), BOX_WRONG))
    else: lines = []
    if k == 3 and lines: lines = lines + [lines[-1]]                     # 같은 줄 반복 생성
    sep = "\n" if h % 2 == 0 else ""
    raw = sep.join(t + "".join(f"<|LOC_{v}|>" for v in b) for t, b in lines)
    if k == 5: raw += sep + "잘린 줄"                                       # 상한에 걸려 위치 없이 끝난 줄
    raw += "</s>" if h % 3 == 0 else ""                                   # 끝에 EOS 문자열이 남는 경우
    exp = []
    for t, b in sorted(dict.fromkeys(lines), key=lambda x: (sum(x[1][1::2]) / 4, sum(x[1][0::2]) / 4)):
        cx, cy = sum(b[0::2]) / 4, sum(b[1::2]) / 4
        v = "위" if cy < 333 else ("아래" if cy > 667 else "중간"); hh = "왼쪽" if cx < 333 else ("오른쪽" if cx > 667 else "가운데")
        tt = re.sub(r"\s+", " ", t).strip()
        if tt: exp.append([f"{v} {hh}", tt])
    if k == 5: exp.append([None, "잘린 줄"])
    return raw, exp

try:
    run(CFG, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "OCR_MODEL_ID": TINY, "OCR_TAG": TAG})
    if scen in ("real", "fb"): g.update({"LIMIT": 10, "MAX_NEW_TOKENS": 6})
    else: g.update({"NUM_SHARDS": 2, "SHARD_ID": 0 if scen.startswith("s0") else 1})
    run(DATA_C, "3 데이터"); run(LIST, "4 목록"); run(MODEL, "5 모델")
    GEN_KW = []
    _orig_gen = g["model"].generate
    def _rec_gen(*a, **kw): GEN_KW.append(dict(kw)); return _orig_gen(*a, **kw)
    g["model"].generate = _rec_gen
    if scen not in ("real", "fb"):
        rows_by_path = {r["path"]: r for d in (g["train_df"], g["test_df"]) for r in d.to_dict("records")}
        calls = {"n": 0}
        def fake_ocr_batch(rows):
            calls["n"] += 1
            if scen == "s0_crash" and calls["n"] == 6: raise RuntimeError("시험용 강제 중단")
            return [(design(r["path"], rows_by_path[r["path"]])[0], 10, False) for r in rows]
        g["ocr_batch"] = fake_ocr_batch
    try:
        run(RUN, "6 OCR 실행")
    except RuntimeError as e:
        if "시험용 강제 중단" not in str(e): raise
        print("[예상된 중단]", e)
        with open(g["SHARD_FILE"], "a", encoding="utf-8") as f: f.write('{"split": "train", "id": "x", "pa')      # 반쯤 쓰인 줄
        raise SystemExit(0)
    run(MERGE, "7 합치기"); run(CHECK, "8 점검"); run(SUMM, "9 요약")
except SystemExit:
    pass

print("\n══════ 검사 ══════"); fails = []
def ok(c, m):
    print(("  ✅ " if c else "  ❌ ") + m)
    if not c: fails.append(m)
ocr_dir = f"{RUNS}/ocr_paddle/{TAG}"
if scen == "fb":
    ok(_n["k"] == 1 and type(g["processor"]).__name__ == "PaddleOCRVLProcessor" and g["tok"].image_token == "<|IMAGE_PLACEHOLDER|>", "기본 로드 실패 → 토크나이저 보강 후 프로세서 조립")
if scen in ("real", "fb"):
    recs = [json.loads(l) for l in open(g["SHARD_FILE"], encoding="utf-8")]
    ok(len(recs) == 10 and all(r["n_tok"] <= 6 for r in recs), f"실제 generate: 기록 10개, 토큰 ≤ 6 ({[r['n_tok'] for r in recs]})")
    ok(GEN_KW and all(k.get("use_cache") is True for k in GEN_KW) and g["model"].generation_config.use_cache is False,
       f"모델 설정은 use_cache=False인데 generate는 use_cache=True로 호출 ({len(GEN_KW)}번)")
    ok(g["TEXT"].endswith("Spotting:\nAssistant:\n") and "<|IMAGE_PLACEHOLDER|>" in g["TEXT"], f"프롬프트 {g['TEXT']!r}")
    ok(g["MINPIX"] == 384 * 384 and g["MAXPIX"] == 2048 * 28 * 28, f"픽셀 범위 {g['MINPIX']}~{g['MAXPIX']}")
    im0 = g["prep_image"](g["MINE"].iloc[0].path); from PIL import Image
    w0, h0 = Image.open(f"{DATA}/{g['MINE'].iloc[0].path}").size
    ok(im0.size == (2 * w0, 2 * h0), f"spotting 전처리 2배 확대 {w0}x{h0} → {im0.size}")
    ok(os.path.exists(f"{ocr_dir}/ocr_text_partial.csv") and not os.path.exists(f"{ocr_dir}/ocr_text.csv"), "일부만 끝나면 partial 파일만")
if scen == "s0_crash":
    n = sum(1 for _ in open(g["SHARD_FILE"], encoding="utf-8"))
    ok(n == 5 * g["BS"] + 1, f"중단 전 5배치 기록 + 반쯤 쓰인 줄 ({n}줄, 배치 {g['BS']})")
if scen == "s0":
    recs = []
    for l in open(g["SHARD_FILE"], encoding="utf-8"):
        try: recs.append(json.loads(l))
        except Exception: pass
    paths = [r["path"] for r in recs]; mine = set(g["MINE"].path)
    ok(len(paths) == len(set(paths)) == len(mine) and set(paths) == mine, f"이어하기: 중복 없이 샤드 0 몫 {len(mine)}장 전부 ({len(paths)})")
    ok(os.path.exists(f"{ocr_dir}/ocr_text_partial.csv") and not os.path.exists(f"{ocr_dir}/ocr_text.csv"), "샤드 1 전이라 partial만")
if scen == "s1":
    ok(os.path.exists(f"{ocr_dir}/ocr_text.csv") and os.path.exists(f"{ocr_dir}/ocr_info.json"), "두 샤드 합쳐 ocr_text.csv 완성")
    df = pd.read_csv(f"{ocr_dir}/ocr_text.csv", keep_default_na=False)
    ok(len(df) == len(g["ALL"]) == df.path.nunique(), f"전체 {len(df)}장, 경로 중복 없음")
    rows_by_path = {r["path"]: r for d in (g["train_df"], g["test_df"]) for r in d.to_dict("records")}
    bad = 0
    for r in df.sample(400, random_state=0).itertuples():
        exp = design(r.path, rows_by_path[r.path])[1]
        if json.loads(r.lines_json) != exp: bad += 1; print("   다름", r.path, json.loads(r.lines_json), exp) if bad <= 3 else None
    ok(bad == 0, f"파싱(위치 태그·위→아래 정렬·반복 제거·잘린 줄·EOS 제거) = 설계값 (400장 중 불일치 {bad})")
    V, S, y = g["V"], g["S"], g["y"]
    def nrm(s): return re.sub(r"[\s\W_]+", "", str(s).lower())
    txt = dict(zip(df.path, df.text))
    ind = np.mean([nrm(r[r["answer"]]) != "" and nrm(r[r["answer"]]) in nrm(txt[r["path"]]) for r in V.to_dict("records")])
    ok(abs((S[np.arange(len(V)), y] == 1.0).mean() - ind) < 1e-12, f"점검: 정답 보기 포함 비율 = 독립 계산 {ind * 100:.2f}%")
    ok(len(V) == 1000, "검증 1,000문항 모두 OCR 있음")
print("\n결과:", "통과" if not fails else f"실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
