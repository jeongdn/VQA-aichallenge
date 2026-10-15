# -*- coding: utf-8 -*-
"""stage2_ocr_{qwen35,gemma4}.ipynb 시험: 작은 Qwen3.5 + 가짜 Qwen 프로세서(CPU)로 노트북 셀을 순서대로 실행.
OCR 파일은 ocr_paddle 시험이 만든 실제 형식의 ocr_text.csv를 쓰고, 일부 문항에 긴 OCR(40줄)·빈 OCR을 넣어 줄이기·빈칸 처리를 검사.
사용: python3 run_s2ocr.py <qwen|gemma> <full|dry|missing>"""
import sys, os, json, glob, shutil, re, math
import numpy as np, pandas as pd

which, scen = sys.argv[1], sys.argv[2]
NB = {"qwen": "/home/user/vqa/ocr/out/stage2_ocr_qwen35.ipynb", "gemma": "/home/user/vqa/ocr/out/stage2_ocr_gemma4.ipynb"}[which]
SP = "/tmp/vqa_scratch"
DATA = f"{SP}/newfam/fx/content"; ROOT = f"{SP}/s2ocr"; RUNS = f"{ROOT}/vqa_runs"; TINY = f"{ROOT}/tiny_qwen35"
SRC_OCR = f"{SP}/ocrtest/vqa_runs/ocr_paddle/t_spot/ocr_text.csv"
sys.path.insert(0, "/home/user/vqa/e1")
import torch, transformers
from tiny_qwen import tiny_config, FakeQwenProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
if not os.path.exists(f"{TINY}/config.json"):
    torch.manual_seed(0); Qwen3_5ForConditionalGeneration(tiny_config()).save_pretrained(TINY)

# ── 가짜 Drive: OCR 파일(긴 OCR·빈 OCR 섞기) + OCR 없는 기준 실행의 검증 예측 ──
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, IMPORTS, SPLIT, PROMPT, MODEL, SCORING, PREP, LOOP, EVAL, CMP, TEST, SUMM = cells
g0 = {}; exec(compile(CFG, "<cfg>", "exec"), g0)
RUN, TAG, BASES = g0["RUN_NAME"], g0["OCR_TAG"], g0["COMPARE_WITH"]
for d in glob.glob(f"{RUNS}/*"):
    if d.endswith("ocr_paddle"): continue
    shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
os.makedirs(f"{RUNS}/ocr_paddle/{TAG}", exist_ok=True)
ocr = pd.read_csv(SRC_OCR, keep_default_na=False)
tr = pd.read_csv(f"{DATA}/train.csv"); tr["answer"] = tr.answer.astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = tr.iloc[sorted(vi)].reset_index(drop=True)
LONG = list(val.id[:3]); EMPTY = list(val.id[3:5])
for vid in LONG:                                                       # 40줄 × 약 40자 = 1,600자 > 1,000 → 정답 보기가 든 줄은 반드시 남아야 함
    r = val[val.id == vid].iloc[0]
    lines = [["위 왼쪽", f"관계없는 안내 문구 {k:02d} 가나다라마바사아자차카타파하 ABCDEFG"] for k in range(40)]
    lines[27] = ["아래 오른쪽", f"여기에 정답: {r[r.answer]}"]
    i = ocr.index[ocr.path == r.path][0]
    ocr.loc[i, "lines_json"] = json.dumps(lines, ensure_ascii=False); ocr.loc[i, "n_lines"] = 40; ocr.loc[i, "n_chars"] = sum(len(t) for _, t in lines)
LOOPED = list(val.id[5:7])                                                # 생성 루프 흔적: 긴 "1000…" 꼬리 + 같은 줄 6번
for vid in LOOPED:
    r = val[val.id == vid].iloc[0]; i = ocr.index[ocr.path == r.path][0]
    lines = json.loads(ocr.loc[i, "lines_json"]) + [["중간 가운데", "모두 갖춘 프리미엄"]] * 6 + [[None, "1" + "0" * 300]]
    ocr.loc[i, "lines_json"] = json.dumps(lines, ensure_ascii=False); ocr.loc[i, "hit_max"] = True
for vid in EMPTY:
    i = ocr.index[ocr.path == val[val.id == vid].iloc[0].path][0]
    ocr.loc[i, "lines_json"] = "[]"; ocr.loc[i, "n_lines"] = 0; ocr.loc[i, "n_chars"] = 0
if scen == "missing": ocr = ocr[~ocr.path.isin(tr.path.head(5))]
ocr.to_csv(f"{RUNS}/ocr_paddle/{TAG}/ocr_text.csv", index=False)
json.dump({"OCR_MODEL_ID": "tiny", "OCR_TASK": "spotting"}, open(f"{RUNS}/ocr_paddle/{TAG}/ocr_info.json", "w"))
R = np.random.RandomState(7)
base_ok = R.rand(len(val)) < 0.9                                        # 기준 실행: 90% 맞힘 (나머지는 다른 글자)
bp = [a if o else "abcd"[("abcd".index(a) + 1) % 4] for a, o in zip(val.answer, base_ok)]
os.makedirs(f"{RUNS}/{BASES[0]}", exist_ok=True)
pd.DataFrame({"id": val.id, "pred": bp}).to_csv(f"{RUNS}/{BASES[0]}/preds_final_ep1__up2.0__1000.csv", index=False)

# ── 작은 모델로 바꿔 끼우기 ──
PROC = FakeQwenProcessor()
transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
def _fp(mid, **kw):
    kw.pop("device_map", None); return Qwen3_5ForConditionalGeneration.from_pretrained(TINY, **kw)
transformers.AutoModelForImageTextToText.from_pretrained = _fp

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
err = None
try:
    run(CFG, "1 설정")
    cfg0 = {k: g[k] for k in ["RUN_NAME", "MODEL_ID", "TRAIN_PRESET", "MICRO_BS", "GRAD_ACCUM", "MAKE_SUBMISSION", "OCR_TAG", "COMPARE_WITH", "EVAL_PRESETS", "USE_4BIT"]}
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "TRAIN_LIMIT": 64, "QUICK_EVAL_N": 12, "EVAL_EVERY": 1000,
              "SAVE_EVERY": 2, "DRY_RUN": scen == "dry", "MICRO_BS": 2, "GRAD_ACCUM": 8})
    run(DATA_C, "3 데이터"); run(IMPORTS, "4 import")
    g["test_df"] = g["test_df"].head(60).copy(); g["sub_df"] = g["sub_df"].head(60).copy()
    run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트"); run(MODEL, "7 모델"); run(SCORING, "8 채점"); run(PREP, "9 학습 준비")
    run(LOOP, "10 학습"); run(EVAL, "11 평가"); run(CMP, "11-2 OCR 효과"); run(TEST, "12 test"); run(SUMM, "13 요약")
except AssertionError as e:
    err = str(e); print("\n[AssertionError]", err[:300])

print("\n══════ 검사 ══════"); fails = []
def ok(c, m):
    print(("  ✅ " if c else "  ❌ ") + m)
    if not c: fails.append(m)
if which == "gemma":
    ok(cfg0["MODEL_ID"] == "google/gemma-4-31B-it" and cfg0["TRAIN_PRESET"] == "up2.0" and cfg0["MICRO_BS"] == 1 and cfg0["GRAD_ACCUM"] == 16
       and cfg0["RUN_NAME"] == "gemma4_31b_r16_up20_ocr_bf16" and cfg0["COMPARE_WITH"] == ["gemma4_31b_r16_up20_bf16", "gemma4_31b_r16_up20"], f"Gemma 레시피 그대로 + OCR 이름: {cfg0}")
else:
    ok(cfg0["MODEL_ID"] == "Qwen/Qwen3.5-27B" and cfg0["TRAIN_PRESET"] == "native" and cfg0["MICRO_BS"] == 2 and cfg0["GRAD_ACCUM"] == 8
       and cfg0["RUN_NAME"] == "q35_27b_r16_native_ocr_bf16", f"Qwen 레시피 그대로 + OCR 이름: {cfg0}")
ok(cfg0["MAKE_SUBMISSION"] is True and cfg0["USE_4BIT"] is False and cfg0["EVAL_PRESETS"] == ["up2.0"], "MAKE_SUBMISSION=True, bf16, 평가 up2.0")
if scen == "missing":
    ok(err is not None and "OCR 결과가 없습니다" in err and "train 이미지 5장" in err, "OCR 빠진 train 이미지 → 학습 전에 멈춤")
else:
    ok(err is None, "끝까지 실행")
    H = g["OCR_HEADER"]; lines_of = {p: json.loads(j) for p, j in zip(ocr.path, ocr.lines_json)}
    rows = {r["id"]: r for r in g["val_df"].to_dict("records")}
    # 일반 문항: 블록 = 머리글 + (위치) 글자 줄 그대로
    normal = [i for i in val.id[7:40] if lines_of[rows[i]["path"]]][:10]
    good = all(g["ocr_block"](rows[i]) == H + "\n" + "\n".join((f"({t}) " if t else "") + x for t, x in lines_of[rows[i]["path"]]) for i in normal)
    ok(good, f"보통 문항 OCR 블록 = 머리글 + 위치 태그 + 글자 줄 ({len(normal)}문항)")
    ok(all(g["ocr_block"](rows[i]) == H + "\n(인식된 글자 없음)" for i in EMPTY), "빈 OCR → '(인식된 글자 없음)'")
    lp_ok = True
    for i in LOOPED:
        b = g["ocr_block"](rows[i]); body = b[len(H) + 1:].split("\n")
        lp_ok &= ("0000000000" not in b and body.count("(중간 가운데) 모두 갖춘 프리미엄") == 3
                  and all(x in body for x in [(f"({t}) " if t else "") + y for t, y in lines_of[rows[i]["path"]][:-7]]))
    ok(lp_ok, "생성 루프 정리: '1000…' 꼬리 제거, 같은 줄 6번 → 3번, 원래 줄은 그대로")
    lg_ok = True
    for i in LONG:
        b = g["ocr_block"](rows[i]); body = b[len(H) + 1:].split("\n")
        r = rows[i]; want = f"(아래 오른쪽) 여기에 정답: {r[r['answer']]}"
        kept = body[:-1]; orig = [(f"({t}) " if t else "") + x for t, x in lines_of[r["path"]]]
        lg_ok &= (body[-1] == "…(일부 줄 생략)" and want in kept and sum(len(x) + 1 for x in kept) <= 1000
                  and [orig.index(x) for x in kept] == sorted(orig.index(x) for x in kept))
    ok(lg_ok, "긴 OCR(1,600자) → 1,000자 안으로 줄이되 정답 보기가 든 줄은 남기고 원래 순서 유지 + '…(일부 줄 생략)'")
    r = rows[normal[0]]
    t1, t2 = g["render"](r, (0, 1, 2, 3)), g["render"](r, (2, 0, 3, 1))
    ok(H in t1 and t1.index(H) < t1.index(str(r["question"])) and t1.split("\n(a)")[0].split(str(r["question"]))[0] == t2.split("\n(a)")[0].split(str(r["question"]))[0],
       "프롬프트: OCR 블록이 질문 앞, 선지를 섞어도 OCR 블록은 같음")
    ok("정답을 반드시 a, b, c, d 중 하나의" in t1.split(H)[1], "답 지시는 맨 끝 그대로")
    cfgl = json.load(open(f"{g['RUN_DIR']}/train_log.json"))["config"]
    ok(cfgl.get("OCR_TAG") == TAG and cfgl.get("PROMPT_VERSION") == f"ocr_v2_{TAG}" and cfgl.get("OCR_MAX_CHARS") == 1000, f"학습 기록에 OCR 설정 ({ {k: cfgl.get(k) for k in ['OCR_TAG', 'PROMPT_VERSION', 'OCR_MAX_CHARS', 'OCR_POS_TAGS']} })")
    if scen == "full":
        rep = open(glob.glob(f"{g['RUN_DIR']}/error_report_final_ep1_up2.0.html")[0], encoding="utf-8").read()
        ok(H in rep, "오답 리포트에 OCR 블록 표시")
        mine = pd.read_csv(f"{g['RUN_DIR']}/preds_final_ep1__up2.0__1000.csv")
        M = val[["id", "answer"]].merge(mine[["id", "pred"]], on="id").merge(pd.DataFrame({"id": val.id, "pb": bp}), on="id")
        fx, br = int(((M.pred == M.answer) & (M.pb != M.answer)).sum()), int(((M.pred != M.answer) & (M.pb == M.answer)).sum())
        c0 = g["OCR_CMP"][0] if g["OCR_CMP"] else {}
        n, k = fx + br, fx
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0
        ok(c0.get("base") == BASES[0] and c0.get("fix") == fx and c0.get("break") == br and abs(c0.get("p", -1) - p) < 1e-12 and len(g["OCR_CMP"]) == 1,
           f"비교: 고침 {fx} · 새로 틀림 {br} · p {p:.3g} = 독립 계산 (없는 두 번째 기준 실행은 건너뜀)")
        sub = glob.glob(f"{g['RUN_DIR']}/submission_*.csv")
        ok(len(sub) == 1 and len(pd.read_csv(sub[0])) == 60, f"제출 파일 {[os.path.basename(x) for x in sub]}")
    else:
        ok(not g["OCR_CMP"], "DRY_RUN → 비교 생략")
print("\n결과:", "통과" if not fails else f"실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
