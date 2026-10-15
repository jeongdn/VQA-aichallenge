# -*- coding: utf-8 -*-
"""ocr_paddle.ipynb: PaddleOCR-VL로 train·test 이미지의 글자를 뽑아 Drive에 저장 (여러 GPU로 나눠 돌리기·이어하기 지원).
결과 OUT_DIR/ocr_paddle/<OCR_TAG>/ocr_text.csv 를 stage2_ocr_qwen35 / stage2_ocr_gemma4 노트북이 읽는다."""
import json

SRC = "/home/user/vqa/src_stage2.ipynb"
DST = "/home/user/vqa/ocr/out/ocr_paddle.ipynb"
src = json.load(open(SRC, encoding="utf-8"))
DATA_CELL = "".join(src["cells"][6]["source"])          # stage2와 같은 Drive·압축 해제 셀

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# OCR 추출 · PaddleOCR-VL → `ocr_text.csv`

train(검증 1,000 포함)·test 이미지의 글자를 **PaddleOCR-VL**(0.9B, 허깅페이스 공개 모델)로 한 번 뽑아 Drive에 저장합니다. `stage2_ocr_qwen35` / `stage2_ocr_gemma4` 노트북이 이 결과를 프롬프트에 넣어 학습·추론합니다.

- **작업:** 기본은 `Spotting:`입니다(1.5판부터 있는 기능). 사진 속 글자 줄마다 위치(사각형 4점, 0~1000 좌표)까지 내므로, 줄을 위→아래 순서로 정리하고 "(위 왼쪽)" 같은 대략의 위치를 붙입니다.
- **전처리:** 모델 카드의 spotting 공식 전처리를 따릅니다. 가로·세로가 모두 1500px보다 작으면 2배로 키우고(LANCZOS), 최대 픽셀은 2048×28×28입니다. 이미지 방향 보정(EXIF)은 stage2와 같습니다.
- **여러 GPU로 나누기:** `NUM_SHARDS`, `SHARD_ID`를 기기마다 다르게 두고 동시에 돌리면 됩니다(예: 2대면 `(2, 0)`과 `(2, 1)`). 각자 `raw_shard*.jsonl`에 쓰고, 7번 셀이 전부 합칩니다. T4처럼 bf16이 안 되는 GPU는 float32로 돌아갑니다.
- **이어하기:** 끊겨도 다시 실행하면 이미 한 이미지는 건너뜁니다.
- **점검(8번 셀):** OCR만으로 보기를 골랐을 때의 정확도, 정답 보기 문자열이 OCR에 들어 있는 비율, E14 오답 24개(특히 진짜 모델 실수 6개)의 OCR 결과를 보여 줍니다. 학습에 GPU를 쓰기 전에 OCR 품질을 판단하는 용도입니다.

**실행 순서:** 전체 실행 → (나눠 돌렸다면) 모든 샤드가 끝난 뒤 7·8번 셀만 다시 실행.""")
md("## 1. 설정")
code('''OCR_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"   # 2026-05 판. 안 되면 "PaddlePaddle/PaddleOCR-VL-1.5"
OCR_TASK     = "spotting"       # "spotting": 글자 줄 + 위치 (사진 속 흩어진 글자용) / "ocr": 글자만 (문서용)
OCR_TAG      = "pvl16_spotting" # 결과 폴더: OUT_DIR/ocr_paddle/<OCR_TAG>/ — 학습 노트북의 OCR_TAG와 같아야 합니다
SPLITS       = ["train", "test"]  # dev는 학습에 쓰지 않으므로 제외
NUM_SHARDS, SHARD_ID = 1, 0     # 여러 GPU로 나눌 때 기기마다 SHARD_ID만 다르게 (0 ~ NUM_SHARDS-1)
MAX_NEW_TOKENS = 512            # 이미지당 최대 생성 토큰 (모델 카드 기본값. spotting은 줄마다 위치 토큰 8개 → 약 28줄). 넘으면 hit_max로 기록. 배치는 가장 긴 이미지가 끝날 때까지 기다리므로 크게 잡으면 느려짐
UPSCALE_BELOW  = 1500           # spotting 공식 전처리: 가로·세로 모두 이보다 작으면 2배 확대
MAX_PIXELS     = None           # None: 공식 값 (spotting 2048×28×28 ≈ 160만 / ocr 1280×28×28 ≈ 100만)
BATCH_SIZE     = "auto"         # 생성 배치 (OOM이면 자동으로 절반)
LIMIT          = None           # 디버그용: 이 기기 몫의 앞 N개만

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
SEED, VAL_SIZE = 42, 1000       # 검증 분할 (stage2와 같음) — 점검 셀에서만 사용
HF_TOKEN     = ""
''')
md("## 2. 라이브러리 설치 (stage2와 같은 버전)")
code('''!pip -q install "transformers==5.17.0" "accelerate==1.15.0"
''')
md("## 3. 데이터 준비")
code(DATA_CELL)
md("## 4. CSV · 이 기기가 맡을 이미지 목록")
code('''import re, json, glob, time, math, hashlib, datetime
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev.csv")) if os.path.exists(os.path.join(DATA_DIR, "dev.csv")) else None
for _d in [train_df, test_df] + ([dev_df] if dev_df is not None else []):
    for c in ["question", "a", "b", "c", "d"]: _d[c] = _d[c].astype(str)
train_df["answer"] = train_df["answer"].astype(str).str.strip().str.lower()

def img_path(rel): return os.path.join(DATA_DIR, rel)
def load_image(rel):                        # stage2와 같음 (EXIF 방향 보정 → RGB)
    im = Image.open(img_path(rel)); im = ImageOps.exif_transpose(im); return im.convert("RGB")

def make_split(df, val_size=VAL_SIZE, seed=SEED):   # stage2와 같은 고정 검증 분할
    rng = np.random.RandomState(seed); val_idx = []
    for letter in "abcd":
        idx = np.where(df.answer.values == letter)[0]; rng.shuffle(idx); val_idx.extend(idx[: val_size // 4].tolist())
    mask = np.zeros(len(df), dtype=bool); mask[sorted(val_idx)] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
tr_df, val_df = make_split(train_df)
split_md5 = hashlib.md5(",".join(val_df.id.tolist()).encode()).hexdigest()
print(f"검증 분할 md5 {split_md5} ({'✅ 팀 공통' if split_md5 == 'df05e4bc19d27ebceeef2570672149c2' else '⚠️ 다름'})")

_src = {"train": train_df, "test": test_df, "dev": dev_df}
ALL = pd.concat([_src[s].assign(split=s)[["split", "id", "path"]] for s in SPLITS if _src.get(s) is not None], ignore_index=True)
ALL = ALL.drop_duplicates("path").reset_index(drop=True)          # 샤드 나누기는 이 순서 기준 → 기기마다 같은 목록
assert 0 <= SHARD_ID < NUM_SHARDS
MINE = ALL[ALL.index % NUM_SHARDS == SHARD_ID]
if LIMIT: MINE = MINE.head(int(LIMIT))
OCR_DIR = os.path.join(OUT_DIR, "ocr_paddle", OCR_TAG); os.makedirs(OCR_DIR, exist_ok=True)
SHARD_FILE = os.path.join(OCR_DIR, f"raw_shard{SHARD_ID}of{NUM_SHARDS}.jsonl")
print(f"전체 {len(ALL):,}장 ({ALL.split.value_counts().to_dict()}) | 이 기기 몫 {len(MINE):,}장 (샤드 {SHARD_ID}/{NUM_SHARDS}) → {SHARD_FILE}")
''')
md("## 5. PaddleOCR-VL 로드")
code('''from transformers import AutoProcessor, AutoModelForImageTextToText
import transformers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability(0) if DEVICE == "cuda" else (0, 0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE == "cuda" else 0.0
DTYPE = torch.bfloat16 if cap[0] >= 8 else torch.float32      # T4 등 bf16이 안 되는 GPU는 float32 (0.9B라 메모리 충분)
BS = (32 if VRAM_GB >= 38 else 16 if VRAM_GB >= 20 else 8) if BATCH_SIZE == "auto" else int(BATCH_SIZE)

try:
    processor = AutoProcessor.from_pretrained(OCR_MODEL_ID)
except AttributeError as _e:                                     # 토크나이저 설정에 image_token이 없는 판 → 채워 넣고 프로세서를 직접 조립
    print("프로세서 기본 로드 실패 → 토크나이저에 이미지 토큰을 채워 조립합니다:", _e)
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.models.paddleocr_vl import PaddleOCRVLProcessor
    from huggingface_hub import hf_hub_download
    _tk = AutoTokenizer.from_pretrained(OCR_MODEL_ID)
    _tk.image_token = "<|IMAGE_PLACEHOLDER|>"; _tk.image_token_id = _tk.convert_tokens_to_ids(_tk.image_token)
    processor = PaddleOCRVLProcessor(image_processor=AutoImageProcessor.from_pretrained(OCR_MODEL_ID), tokenizer=_tk,
                                     chat_template=open(hf_hub_download(OCR_MODEL_ID, "chat_template.jinja"), encoding="utf-8").read())
processor.tokenizer.padding_side = "left"                       # 배치 생성은 왼쪽 패딩 (모델 문서 권장)
tok = processor.tokenizer
load_kw = dict(dtype=DTYPE, attn_implementation="sdpa")
if DEVICE == "cuda": load_kw["device_map"] = {"": 0}
model = AutoModelForImageTextToText.from_pretrained(OCR_MODEL_ID, **load_kw).eval()
PROMPT = {"ocr": "OCR:", "spotting": "Spotting:"}[OCR_TASK]
if OCR_TASK == "spotting":
    assert "<|LOC_0|>" in tok.get_vocab(), "이 모델에는 위치 토큰이 없습니다(spotting은 1.5판부터) → OCR_TASK = 'ocr'로 바꾸세요"
MAXPIX = int(MAX_PIXELS or (2048 * 28 * 28 if OCR_TASK == "spotting" else 1280 * 28 * 28))
_ip, _sz = processor.image_processor, getattr(processor.image_processor, "size", None)
_sz_get = (lambda k: _sz.get(k) if isinstance(_sz, dict) else getattr(_sz, k, None)) if _sz is not None else (lambda k: None)
MINPIX = int(getattr(_ip, "min_pixels", None) or _sz_get("shortest_edge") or 384 * 384)   # 모델 기본 최소 픽셀 (transformers 5.x는 size에 들어 있음)
TEXT = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                                     tokenize=False, add_generation_prompt=True)
EOS_IDS = {i for i in ([tok.eos_token_id] + list(np.atleast_1d(getattr(model.generation_config, "eos_token_id", None) or []))) if i is not None}
print(f"transformers {transformers.__version__} | {type(model).__name__} | GPU {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'} ({VRAM_GB:.0f}GB) | dtype {DTYPE} | 배치 {BS}")
print(f"작업 {OCR_TASK} | 픽셀 {MINPIX:,} ~ {MAXPIX:,} | 프롬프트 {TEXT!r}")

def prep_image(rel):
    im = load_image(rel)
    if OCR_TASK == "spotting" and im.width < UPSCALE_BELOW and im.height < UPSCALE_BELOW:
        im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)        # 모델 카드의 spotting 전처리
    return im

@torch.inference_mode()
def ocr_batch(rows):
    # → [(raw 문자열, 생성 토큰 수, 상한 도달 여부)] — raw는 특수 토큰(위치 토큰)을 지우지 않고 그대로 디코드
    enc = processor(text=[TEXT] * len(rows), images=[prep_image(r["path"]) for r in rows], padding=True, return_tensors="pt",
                    min_pixels=MINPIX, max_pixels=MAXPIX).to(model.device)
    enc["pixel_values"] = enc["pixel_values"].to(DTYPE)                         # 비전 타워와 같은 정밀도로
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)   # 모델 설정이 use_cache=false라 꼭 켜야 함 (안 켜면 토큰마다 이미지·프롬프트를 처음부터 다시 계산)
    res = []
    for seq in out[:, enc["input_ids"].shape[1]:].tolist():
        n = next((k for k, t in enumerate(seq) if t in EOS_IDS), None)
        hit = n is None
        ids = seq if hit else seq[:n]
        res.append((tok.decode(ids, skip_special_tokens=False), len(ids), hit))
    return res
''')
md("## 6. OCR 실행 (이어하기 지원)")
code('''done = set()
if os.path.exists(SHARD_FILE):
    for line in open(SHARD_FILE, encoding="utf-8"):
        try: done.add(json.loads(line)["path"])
        except Exception: pass                         # 끊기며 반쯤 쓰인 줄은 무시 → 다시 처리
    with open(SHARD_FILE, "rb") as _f:                  # 끊기며 줄바꿈 없이 끝났으면 줄을 바꿔 두어 새 기록이 망가진 줄에 붙지 않게
        _f.seek(0, 2)
        if _f.tell() > 0:
            _f.seek(-1, 2); _need_nl = _f.read(1) != b"\\n"
        else: _need_nl = False
    if _need_nl:
        with open(SHARD_FILE, "a", encoding="utf-8") as _f: _f.write("\\n")
todo = MINE[~MINE.path.isin(done)].to_dict("records")
print(f"이 기기 몫 {len(MINE):,} | 완료 {len(MINE) - len(todo):,} | 남음 {len(todo):,}")
t0, n, bs, nb = time.time(), 0, BS, 0
with open(SHARD_FILE, "a", encoding="utf-8") as fo:
    while n < len(todo):
        batch = todo[n: n + bs]
        try:
            res = ocr_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1: raise RuntimeError("배치 1에서도 GPU 메모리 부족 → MAX_PIXELS를 낮추세요")
            bs = max(1, bs // 2); print(f"  ⚠️ OOM → 배치 {bs}"); continue
        for r, (raw, k, hit) in zip(batch, res):
            fo.write(json.dumps({"split": r["split"], "id": r["id"], "path": r["path"], "raw": raw, "n_tok": int(k), "hit_max": bool(hit)},
                                ensure_ascii=False) + "\\n")
        fo.flush(); n += len(batch); nb += 1
        if nb == 1:                                    # 첫 배치: 속도와 결과 모양을 바로 확인
            print(f"  첫 배치 {len(batch)}장: {time.time() - t0:.1f}s (장당 {(time.time() - t0) / len(batch):.2f}s) | 생성 토큰 {[k for _, k, _ in res][:8]} | 상한 도달 {sum(h for _, _, h in res)}")
            print(f"  첫 이미지 raw: {res[0][0][:300]!r}", flush=True)
        if nb % 10 == 0 or n >= len(todo):
            el = time.time() - t0
            print(f"  {n:>6}/{len(todo)} | {el / n:.2f}s/장 | 남은 시간 ~{(len(todo) - n) * el / n / 60:.0f}분", flush=True)
print("이 기기 몫 끝:", SHARD_FILE)
''')
md("## 7. 모든 샤드 합치기 → `ocr_text.csv`\n여러 기기로 나눠 돌렸다면, 모두 끝난 뒤 **이 셀부터** 다시 실행하세요. 빠진 이미지가 있으면 `ocr_text_partial.csv`만 쓰고, 몇 장이 남았는지 알려 줍니다.")
code('''LOC_RE = re.compile(r"<\\|LOC_(\\d+)\\|>")
SEG_RE = re.compile(r"((?:\\s*<\\|LOC_\\d+\\|>){8})")                 # 글자 줄 = 글자 + 위치 토큰 8개(네 꼭짓점 x,y)
STRIP = [t for t in {tok.eos_token, tok.pad_token, tok.bos_token, getattr(tok, "cls_token", None), "</s>", "<|end_of_sentence|>", "<|begin_of_sentence|>"} if t]

def pos_tag(cx, cy):
    v = "위" if cy < 333 else ("아래" if cy > 667 else "중간")
    h = "왼쪽" if cx < 333 else ("오른쪽" if cx > 667 else "가운데")
    return f"{v} {h}"

def parse_raw(raw):
    # raw → [(위치 태그 또는 None, 글자)] 읽는 순서(위→아래, 같은 줄은 왼→오른)
    s = str(raw)
    for t in STRIP: s = s.replace(t, "")
    if OCR_TASK != "spotting":
        return [(None, re.sub(r"\\s+", " ", l).strip()) for l in s.split("\\n") if l.strip()]
    items, pos = [], 0
    for m in SEG_RE.finditer(s):
        text = re.sub(r"\\s+", " ", LOC_RE.sub("", s[pos:m.start()])).strip(); pos = m.end()
        v = [int(x) for x in LOC_RE.findall(m.group(1))]; xs, ys = v[0::2], v[1::2]
        if text: items.append((sum(xs) / 4, sum(ys) / 4, max(1, max(ys) - min(ys)), text))
    tail = re.sub(r"\\s+", " ", LOC_RE.sub("", s[pos:])).strip()      # 상한에 걸려 위치 없이 잘린 마지막 줄
    if items:
        hmed = float(np.median([it[2] for it in items]))
        items.sort(key=lambda it: (it[1], it[0])); rows = []
        for it in items:                                               # 세로 위치가 비슷하면 같은 줄로 묶고 왼→오른
            if rows and abs(it[1] - rows[-1][-1][1]) < max(8.0, 0.5 * hmed): rows[-1].append(it)
            else: rows.append([it])
        items = [it for r in rows for it in sorted(r, key=lambda it: it[0])]
    out, seen = [], set()
    for cx, cy, _, text in items:
        key = (text, round(cx / 25), round(cy / 25))
        if key in seen: continue                                       # 같은 줄을 반복 생성한 경우 제거
        seen.add(key); out.append((pos_tag(cx, cy), text))
    if tail: out.append((None, tail))
    return out

raws = {}
for fp in sorted(glob.glob(os.path.join(OCR_DIR, "raw_shard*.jsonl"))):
    for line in open(fp, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        raws[d["path"]] = d
rows = []
for r in ALL.itertuples(index=False):
    d = raws.get(r.path)
    if d is None: continue
    lines = parse_raw(d["raw"])
    rows.append({"split": r.split, "id": r.id, "path": r.path, "n_lines": len(lines), "n_chars": sum(len(t) for _, t in lines),
                 "n_tok": d["n_tok"], "hit_max": d["hit_max"], "lines_json": json.dumps(lines, ensure_ascii=False),
                 "text": "\\n".join(t for _, t in lines)})
OCR_DF = pd.DataFrame(rows, columns=["split", "id", "path", "n_lines", "n_chars", "n_tok", "hit_max", "lines_json", "text"])
missing = ALL[~ALL.path.isin(set(OCR_DF.path))]
print("샤드별 기록 수:", {os.path.basename(f): sum(1 for _ in open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(OCR_DIR, "raw_shard*.jsonl")))})
for s, g_ in ALL.groupby("split"):
    got = OCR_DF[OCR_DF.split == s]
    print(f"  {s}: {len(got):,}/{len(g_):,} | 글자 없음 {int((got.n_lines == 0).sum())} | 상한 도달 {int(got.hit_max.sum())} | 줄 수 중앙값 {got.n_lines.median() if len(got) else 0:.0f} | 글자 수 중앙값 {got.n_chars.median() if len(got) else 0:.0f} · 90% {got.n_chars.quantile(0.9) if len(got) else 0:.0f}")
if len(missing) == 0:
    OCR_FILE = os.path.join(OCR_DIR, "ocr_text.csv"); OCR_DF.to_csv(OCR_FILE, index=False)
    json.dump({"OCR_MODEL_ID": OCR_MODEL_ID, "OCR_TASK": OCR_TASK, "MAX_NEW_TOKENS": MAX_NEW_TOKENS, "UPSCALE_BELOW": UPSCALE_BELOW,
               "MAX_PIXELS": MAXPIX, "MIN_PIXELS": MINPIX, "SPLITS": SPLITS, "n": len(OCR_DF),
               "time": datetime.datetime.now().strftime("%m-%d %H:%M")}, open(os.path.join(OCR_DIR, "ocr_info.json"), "w"), indent=1)
    print(f"✅ 전부 완료 → {OCR_FILE} ({len(OCR_DF):,}장). 이제 stage2_ocr_* 노트북을 실행하세요 (OCR_TAG = {OCR_TAG!r})")
else:
    OCR_FILE = os.path.join(OCR_DIR, "ocr_text_partial.csv"); OCR_DF.to_csv(OCR_FILE, index=False)
    print(f"⚠️ 아직 {len(missing):,}장 남음 ({missing.split.value_counts().to_dict()}) → 다른 샤드가 끝난 뒤 이 셀부터 다시 실행하세요. 지금까지 결과: {OCR_FILE}")
''')
md("## 8. 점검: OCR이 정답 쪽 글자를 담고 있나 (검증 1,000문항)\n- **OCR만으로 고르기:** 보기 문자열이 OCR 글자와 얼마나 겹치는지(띄어쓰기·기호 무시, 두 글자 단위)만 보고 가장 많이 겹치는 보기를 고릅니다. 모델 없이 OCR이 담은 정보량을 보는 지표입니다.\n- **E14 오답 24개:** 라벨 오류 10 · 둘 다 답 8 · 진짜 모델 실수 6의 OCR 결과를 그대로 보여 줍니다.")
code('''def _norm(s): return re.sub(r"[\\s\\W_]+", "", str(s).lower())
def _bg(s): return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())
def option_scores(row, text):
    t = _norm(text); tb = _bg(t); sc = []
    for k in "abcd":
        o = _norm(row[k])
        sc.append(0.0 if not o else 1.0 if o in t else min(0.99, len(_bg(o) & tb) / len(_bg(o))))   # 1.0 = 보기 문자열이 통째로 들어 있음
    return sc

E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}
E14_ERR = {k: [f"train_{x}.jpg" for x in v.split()] for k, v in E14_ERR.items()}
_txt = dict(zip(OCR_DF.path, OCR_DF.text)); _lines = dict(zip(OCR_DF.path, OCR_DF.lines_json))
V = val_df[val_df.path.isin(_txt)].copy()
if len(V) == 0:
    print("검증 문항의 OCR 결과가 아직 없습니다 (이 기기 몫에 train이 없거나 아직 진행 중)")
else:
    S = np.array([option_scores(r, _txt[r["path"]]) for r in V.to_dict("records")]); y = V.answer.map("abcd".index).to_numpy()
    top = S.max(1); decided = (top > 0) & ((S == top[:, None]).sum(1) == 1)
    full = S == 1.0
    print(f"검증 {len(V):,}문항 (OCR 있음 {int((V.path.map(lambda p: bool(_txt[p])).sum()))})")
    print(f"  정답 보기가 OCR에 통째로 들어 있음: {full[np.arange(len(V)), y].mean() * 100:.1f}% | 오답 보기가 통째로 들어 있음(오답 3개 중 하나라도): {(full.sum(1) - full[np.arange(len(V)), y] > 0).mean() * 100:.1f}%")
    print(f"  OCR만으로 고르기: 결정 {decided.mean() * 100:.1f}% 문항 | 결정한 문항 정확도 {(S.argmax(1) == y)[decided].mean() * 100:.1f}% | 전체 기준 {((S.argmax(1) == y) & decided).mean() * 100:.1f}%")
    print("  (참고: 부정형 질문 '~가 아닌 것'은 OCR에 없는 보기가 정답이라 이 지표에서 틀린 쪽으로 잡힙니다)")
    for cat, ids in E14_ERR.items():
        print(f"\\n━━ E14 오답 · {cat} ({len(ids)}개) ━━")
        for r in val_df[val_df.id.isin(ids)].to_dict("records"):
            if r["path"] not in _txt: print(f"  {r['id']}: OCR 아직 없음"); continue
            sc = option_scores(r, _txt[r["path"]]); lines = json.loads(_lines[r["path"]])
            print(f"  ▶ {r['id']} | {r['question']}")
            print("     " + " / ".join(f"{k}{'*' if k == r['answer'] else ''} {r[k]} [{s:.2f}]" for k, s in zip("abcd", sc)))
            body = " | ".join((f"({t}) " if t else "") + x for t, x in lines)
            print(f"     OCR {len(lines)}줄: {body[:400]}{' …' if len(body) > 400 else ''}")
''')
md("## 9. 요약")
code('''print(f"OCR: {OCR_MODEL_ID} · {OCR_TASK} · 폴더 {OCR_DIR}")
print(f"완료 {len(OCR_DF):,}/{len(ALL):,}장 | 남음 {len(missing):,} | 결과 파일 {os.path.basename(OCR_FILE)}")
if len(OCR_DF):
    print(f"글자 없음 {(OCR_DF.n_lines == 0).mean() * 100:.1f}% | 상한 도달 {OCR_DF.hit_max.mean() * 100:.1f}% | 이미지당 글자 수 중앙값 {OCR_DF.n_chars.median():.0f}")
    ex = OCR_DF[OCR_DF.n_lines > 0].head(3)
    for r in ex.itertuples(): print(f"  예) {r.id}: {' | '.join((f'({t}) ' if t else '') + x for t, x in json.loads(r.lines_json))[:200]}")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    s = c["source"]; lines = s.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", DST, "| 셀", len(cells))
