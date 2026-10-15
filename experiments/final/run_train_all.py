# -*- coding: utf-8 -*-
"""01_train_vlm.ipynb 시험: 작은 Qwen3.5 + 가짜 Qwen 프로세서(CPU)로 코드 셀을 순서대로 실행하고 결과를 검사.
사용: python3 run_train_all.py <scenario>
  dry      : q35_27b_r16_native_bf16 · DRY_RUN (어댑터 재로드·test 경로 점검까지)
  base     : q35_27b_r16_native 본 학습(작게) → 검증·test·제출 파일 (e4의 시작 어댑터가 됨)
  vis      : q35_27b_r16_review1_up20_vis (비전 LoRA) + 학습 기록의 LoRA 층 수 대조
  e4       : q35_27b_r16_e4aux20_native (base 어댑터에서 이어학습 + 보조 학습 + 2번째 epoch 순서 + epoch 끝 평가 생략)
  ocr      : q35_27b_r16_native_ocr_bf16 (OCR 블록)
  sync     : 학습 기록과 다른 값 → 기록대로 맞춤 / 끝난 실행 폴더 보호
  fulldata : FULL_DATA (q35_9b: 통과 · q3vl_8b: epoch 선택 때문에 멈춤)
  dev      : base 레시피 + dev 의사 라벨 (규칙을 독립 계산과 대조)
  resume   : 중간 중단 → 다시 실행하면 이어서 (결과가 끊지 않은 실행과 같은지)"""
import sys, os, json, glob, shutil, re, math, hashlib
import numpy as np, pandas as pd

scen = sys.argv[1]
NB = "/home/user/vqa/final/pkg/01_train_vlm.ipynb"
SP = "/tmp/vqa_scratch"
DATA = f"{SP}/newfam/fx/content"; ROOT = f"{SP}/trainall"; RUNS = f"{ROOT}/vqa_runs"; TINY = f"{SP}/s2ocr/tiny_qwen35"
SRC_OCR = f"{SP}/ocrtest/vqa_runs/ocr_paddle/t_spot/ocr_text.csv"
sys.path.insert(0, "/home/user/vqa/e1")
import torch, transformers
from tiny_qwen import tiny_config, FakeQwenProcessor, FakeTok
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
if not os.path.exists(f"{TINY}/config.json"):
    torch.manual_seed(0); Qwen3_5ForConditionalGeneration(tiny_config()).save_pretrained(TINY)
os.makedirs(RUNS, exist_ok=True)

# 보조 학습(문자열 손실)이 돌도록: 토크나이저 호출 + 뒤쪽 글자를 자르지 않는 프로세서
class _Enc:
    def __init__(self, ids): self.input_ids = ids
def _tok_call(self, s, add_special_tokens=False): return _Enc(self.encode(s))
FakeTok.__call__ = _tok_call
class FullProc(FakeQwenProcessor):
    def __call__(self, text, images, padding=True, return_tensors="pt", **kw):
        from transformers import BatchFeature
        self.calls.append(kw)
        seqs, pvs = [], []
        for t, im in zip(text, images):
            pre, post = t.split("<IMG>")
            seqs.append([20 + (ord(ch) % 250) for ch in pre[-6:]] + [292] + [290] * 4 + [293] + [20 + (ord(ch) % 250) for ch in post[-400:]])
            a = np.asarray(im.resize((8, 8))).astype(np.float32) / 255.0
            rng = np.random.RandomState(int(a.sum() * 1000) % (2**31))
            pvs.append(torch.tensor(rng.randn(16, 3 * 2 * 16 * 16).astype(np.float32)))
        L = max(map(len, seqs))
        ids = torch.tensor([[0] * (L - len(s)) + s for s in seqs]); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
        return BatchFeature({"input_ids": ids, "attention_mask": att, "mm_token_type_ids": (ids == 290).long(), "pixel_values": torch.cat(pvs),
                             "image_grid_thw": torch.tensor([[1, 4, 4]] * len(text))})
PROC = FullProc()
transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
LOADED = []
def _fp(mid, **kw):
    kw.pop("device_map", None); LOADED.append((mid, {k: v for k, v in kw.items() if k != "quantization_config"}))
    return Qwen3_5ForConditionalGeneration.from_pretrained(TINY, **kw)
transformers.AutoModelForImageTextToText.from_pretrained = _fp

cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, RCP, INSTALL, DATA_C, IMPORTS, SPLIT, PROMPT, M1, M2, SCORING, PREP, LOOP, EVAL, TEST, DEVP, SUMM = cells
assert EVAL.count("_pt.letter_mass.mean() > 0.9") == 1
EVAL = EVAL.replace("_pt.letter_mass.mean() > 0.9", "_pt.letter_mass.mean() > 0.0")   # 작은 무작위 모델은 letter_mass가 낮음 (실제 모델 기준 점검은 그대로)

def fresh(run):
    for d in glob.glob(f"{RUNS}/{run}*"): shutil.rmtree(d)

RECIPE = {"dry": "q35_27b_r16_native_bf16", "base": "q35_27b_r16_native", "vis": "q35_27b_r16_review1_up20_vis", "e4": "q35_27b_r16_e4aux20_native",
          "ocr": "q35_27b_r16_native_ocr_bf16", "sync": "q38_27b_r16_native", "fulldata": "q35_9b_r16_native", "dev": "q35_27b_r16_native",
          "resume": "q38_27b_r16_native_bf16"}[scen]
DRY = scen == "dry"
if scen in ("dry", "base", "vis", "ocr", "resume", "e4"): fresh(RECIPE)
if scen == "vis":                        # 원래 실행의 학습 기록 (LoRA 층 수: 작은 모델 기준 언어 4층 · 비전 2블록)
    os.makedirs(f"{RUNS}/{RECIPE}", exist_ok=True)
    json.dump({"config": {"MODEL_ID": "Qwen/Qwen3.5-27B", "TRAIN_PRESET": "up2.0", "STEP_SAMPLES": 16, "VISION_LORA": True, "VIS_LR_SCALE": 0.2,
                          "LORA_TARGET_COUNTS": {"language": 3 * 6 + 7, "vision": 2 * 4 + 2}}, "epoch_acc": {"1": 0.973}},
              open(f"{RUNS}/{RECIPE}/train_log.json", "w"))
if scen == "sync":
    fresh(RECIPE); os.makedirs(f"{RUNS}/{RECIPE}", exist_ok=True)
    json.dump({"config": {"MODEL_ID": "Qwen/Qwen3.8-27B", "TRAIN_PRESET": "native", "EPOCHS": 1, "LR": 0.0002, "STEP_SAMPLES": 32, "TRAIN_SEED": 7,
                          "SEED": 42, "PROMPT_VERSION": "baseline_v1", "LOSS_MODE": "restricted4"}, "epoch_acc": {"1": 0.96}},
              open(f"{RUNS}/{RECIPE}/train_log.json", "w"))
if scen == "ocr":
    os.makedirs(f"{RUNS}/ocr_paddle/pvl16_spotting", exist_ok=True)
    shutil.copy(SRC_OCR, f"{RUNS}/ocr_paddle/pvl16_spotting/ocr_text.csv")
    json.dump({"OCR_MODEL_ID": "tiny", "OCR_TASK": "spotting"}, open(f"{RUNS}/ocr_paddle/pvl16_spotting/ocr_info.json", "w"))
if scen == "e4":
    assert os.path.isfile(f"{RUNS}/q35_27b_r16_native/adapter_ep1/adapter_model.safetensors"), "base 시나리오를 먼저 돌리세요"
    tr = pd.read_csv(f"{DATA}/train.csv")                          # 보조 데이터: 00_data_prep과 같은 형식 (여기서는 간단히 정답 선지 글자)
    tr["answer"] = tr.answer.astype(str).str.strip().str.lower()
    rng = np.random.RandomState(42); vi = []
    for Lt in "abcd":
        idx = np.where(tr.answer.values == Lt)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
    vids = set(tr.id.iloc[sorted(vi)])
    aux = tr[~tr.id.isin(vids)].head(40).copy(); aux["answer_text"] = [str(r[r["answer"]])[:12] for _, r in aux.iterrows()]
    os.makedirs(f"{RUNS}/aux", exist_ok=True); aux[["id", "path", "question", "answer_text"]].to_csv(f"{RUNS}/aux/aux_open_v1.csv", index=False)
if scen == "dev":
    from PIL import Image
    dv = pd.read_csv(f"{DATA}/dev.csv")
    for p in dv.path.head(40):
        os.makedirs(os.path.dirname(f"{DATA}/{p}"), exist_ok=True)
        h = int(hashlib.md5(p.encode()).hexdigest()[:6], 16)
        if not os.path.exists(f"{DATA}/{p}"): Image.new("RGB", (40 + h % 50, 30 + h % 40), (h % 256, (h >> 8) % 256, 90)).save(f"{DATA}/{p}")

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
def setup(recipe, dry, extra_after_recipe=None, full_data=False, suffix="", reuse=False):
    run(CFG, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "DRY_RUN": dry})
    RC = RCP.replace('RECIPE        = "q35_27b_r16_native_bf16"', f'RECIPE        = {recipe!r}', 1)
    RC = RC.replace("FULL_DATA     = False", f"FULL_DATA     = {full_data}", 1).replace('RUN_SUFFIX    = ""', f'RUN_SUFFIX    = {suffix!r}', 1).replace("REUSE_FINISHED = False", f"REUSE_FINISHED = {reuse}", 1)
    run(RC, "1-1 레시피")
    g.update({"TRAIN_LIMIT": 64, "QUICK_EVAL_N": 12, "EVAL_EVERY": 10**9, "SAVE_EVERY": 2, "LOG_EVERY": 1})
    if extra_after_recipe: g.update(extra_after_recipe)
def body(stop_at=None, dev=False):
    run(DATA_C, "3 데이터"); run(IMPORTS, "4 import")
    g["test_df"] = g["test_df"].head(24).copy(); g["sub_df"] = g["sub_df"].head(24).copy()
    if g.get("dev_df") is not None: g["dev_df"] = g["dev_df"].head(40).copy()
    run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트"); run(M1, "7-1 프로세서"); run(M2, "7-2 모델")
    run(SCORING, "8 채점"); run(PREP, "9 학습 준비")
    lp = LOOP if stop_at is None else LOOP.replace("_STOP_AT = None", f"_STOP_AT = {stop_at}", 1)
    run(lp, "10 학습")
    if stop_at is not None: return
    g["val_df"] = g["val_df"]            # (전체 1,000문항 평가 — 작은 모델이라 빠름)
    run(EVAL, "11 평가"); run(TEST, "12 test")
    DV = DEVP.replace("MAKE_DEV_PSEUDO = False", "MAKE_DEV_PSEUDO = True", 1).replace('"up2.0", 0.8, 3, True', '"up2.0", 0.26, 2, True', 1) if dev else DEVP   # 작은 모델은 확신이 낮아 문턱을 낮춰 규칙을 시험
    run(DV, "12-2 dev"); run(SUMM, "13 요약")

fails = []
def ok(c, m):
    print(("  ✅ " if c else "  ❌ ") + m)
    if not c: fails.append(m)
err = None
try:
    if scen == "fulldata":
        os.makedirs(f"{RUNS}/q35_9b_r16_native", exist_ok=True)
        json.dump({"config": {"MODEL_ID": "Qwen/Qwen3.5-9B", "TRAIN_PRESET": "native", "EPOCHS": 2, "STEP_SAMPLES": 16, "SEED": 42}, "epoch_acc": {"1": 0.955, "2": 0.957}},
                  open(f"{RUNS}/q35_9b_r16_native/train_log.json", "w"))
        os.makedirs(f"{RUNS}/q3vl_8b_r16_native", exist_ok=True)
        json.dump({"config": {"MODEL_ID": "Qwen/Qwen3-VL-8B-Instruct", "TRAIN_PRESET": "native", "EPOCHS": 2, "STEP_SAMPLES": 16, "SEED": 42}, "epoch_acc": {"1": 0.957, "2": 0.953}},
                  open(f"{RUNS}/q3vl_8b_r16_native/train_log.json", "w"))
        setup("q35_9b_r16_native", False, full_data=True)
        ok(g["RUN_NAME"] == "q35_9b_r16_native_full" and g["TRAIN_ON_FULL"] is True and g["EPOCHS"] == 2 and g["MAKE_SUBMISSION"] is True, f"FULL_DATA 통과: {g['RUN_NAME']} · EPOCHS {g['EPOCHS']}")
        e2 = None
        try: setup("q3vl_8b_r16_native", False, full_data=True)
        except RuntimeError as e: e2 = str(e)
        ok(e2 is not None and "epoch 1" in e2, f"검증으로 epoch 1을 고른 2 epoch 레시피 → 멈춤: {str(e2)[:80]}")
        e3 = None
        try: setup("q35_27b_r16_native_full", False, full_data=True)
        except AssertionError as e: e3 = str(e)
        ok(e3 is not None and "이미 전체 데이터" in e3, "이미 _full인 레시피에 FULL_DATA → 멈춤")
    elif scen == "sync":
        setup(RECIPE, False)
        ok(g["LR"] == 0.0002 and g["TRAIN_SEED"] == 7 and g["GRAD_ACCUM"] == 16 and g["MICRO_BS"] == 2, f"기록대로 맞춤: LR {g['LR']} · 시드 {g['TRAIN_SEED']} · 배치 {g['MICRO_BS']}x{g['GRAD_ACCUM']}")
        ok(set(g["LOG_SYNCED"]) == {"LR", "TRAIN_SEED", "GRAD_ACCUM"}, f"맞춘 항목 {sorted(g['LOG_SYNCED'])}")
        open(f"{RUNS}/{RECIPE}/submission_x.csv", "w").write("id,answer\n")
        e2 = None
        try: setup(RECIPE, False)
        except RuntimeError as e: e2 = str(e)
        ok(e2 is not None and "이미 끝난 실행" in e2, "제출 파일이 있는 폴더 → 덮어쓰지 않고 멈춤")
        setup(RECIPE, False, suffix="_repro"); ok(g["RUN_NAME"] == RECIPE + "_repro", f"RUN_SUFFIX → {g['RUN_NAME']}")
        setup(RECIPE, True); ok(g["RUN_NAME"] == RECIPE, "DRY_RUN은 _dry 폴더라 막지 않음")
    elif scen == "resume":
        setup(RECIPE, False); body(stop_at=2)
        st1 = g["state"]["global_step"]
        g.clear(); g["__name__"] = "__main__"
        setup(RECIPE, False); body()
        ok(st1 == 2, f"중단 시점 {st1}스텝")
        R1 = f"{RUNS}/{RECIPE}"; a = pd.read_csv(f"{R1}/preds_final_ep1__up2.0__1000.csv")
        shutil.move(R1, R1 + "_resumed_copy")
        g.clear(); g["__name__"] = "__main__"
        setup(RECIPE, False); body()
        b = pd.read_csv(f"{R1}/preds_final_ep1__up2.0__1000.csv")
        dm = float(np.abs(a[["p_a", "p_b", "p_c", "p_d"]].values - b[["p_a", "p_b", "p_c", "p_d"]].values).max())
        ok((a.pred == b.pred).all() and dm < 1e-4, f"끊었다 이어 한 결과 = 한 번에 한 결과 (확률 최대 차 {dm:.2e})")
        shutil.rmtree(R1 + "_resumed_copy")
    else:
        extra = {"MAX_STEPS": 4} if scen == "e4" else None
        if scen == "dev":
            MD0 = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in glob.glob(f"{RUNS}/{RECIPE}/preds_final_*.csv") + [f"{RUNS}/{RECIPE}/train_log.json", f"{RUNS}/{RECIPE}/adapter_ep1/adapter_model.safetensors", f"{RUNS}/{RECIPE}/ckpt_last/adapter/adapter_model.safetensors"]}
        setup(RECIPE, DRY, extra_after_recipe=extra, reuse=(scen == "dev")); body(dev=(scen == "dev"))
except (AssertionError, RuntimeError) as e:
    err = f"{type(e).__name__}: {e}"; print("\n[멈춤]", err[:500])

print("\n══════ 검사 ══════")
RD = g.get("RUN_DIR")
if scen not in ("fulldata", "sync", "resume"):
    ok(err is None, f"끝까지 실행 ({err})")
    if RD and os.path.exists(f"{RD}/train_log.json"):
        J = json.load(open(f"{RD}/train_log.json")); C = J["config"]
        ok(J.get("recipe") == RECIPE and all(k in C for k in ["VISION_LORA", "INIT_ADAPTER", "PROMPT_VERSION", "LORA_TARGET_COUNTS", "PRECISION", "GEN_PROMPT_VERSION"]),
           f"train_log: recipe={J.get('recipe')} · 새 설정 키 기록")
if scen == "dry":
    ok(RD.endswith("_dry") and g["USE_4BIT"] is False and g["PRECISION"] == "float32", f"DRY 폴더 {os.path.basename(RD)} · bf16 레시피(CPU라 float32)")
if scen in ("base", "dev"):
    ok(os.path.exists(f"{RD}/preds_final_ep1__up2.0__1000.csv") and os.path.exists(f"{RD}/preds_final_ep1_test__up2.0__24.csv"), "검증·test 예측 파일 (조합 노트북 이름 규칙)")
    sub = glob.glob(f"{RD}/submission_*.csv"); ok(len(sub) == 1 and len(pd.read_csv(sub[0])) == 24, f"제출 파일 {[os.path.basename(x) for x in sub]}")
    ok(g["USE_4BIT"] == "auto" and g["LORA_TARGET_COUNTS"]["vision"] == 0, "NF4 레시피(auto) · 비전 LoRA 없음")
if scen == "vis":
    C = json.load(open(f"{RD}/train_log.json"))["config"]
    ok(C["VISION_LORA"] is True and g["LORA_TARGET_COUNTS"] == {"language": 25, "vision": 10}, f"비전 LoRA 층 수 {g['LORA_TARGET_COUNTS']} (기록과 대조)")
    ok(len(g["optimizer"].param_groups) == 2 and abs(g["optimizer"].param_groups[1]["initial_lr"] - g["LR"] * 0.2) < 1e-12, "옵티마이저 2그룹 · 비전 LR = LR x 0.2")
    ok(g["TRAIN_PRESET"] == "up2.0" and g["MICRO_BS"] * g["GRAD_ACCUM"] == 16, "학습 해상도 up2.0 · 유효 배치 16")
if scen == "e4":
    C = json.load(open(f"{RD}/train_log.json"))["config"]
    base_sha = hashlib.sha256(open(f"{RUNS}/q35_27b_r16_native/adapter_ep1/adapter_model.safetensors", "rb").read()).hexdigest()[:16]
    ok(C["INIT_ADAPTER_SHA"] == base_sha and C["PLAN_EPOCH_OFFSET"] == 1 and C["GEN_PROMPT_VERSION"] == "v2_ko_sys" and C["AUX_GEN_EVERY"] == 4, "이어학습 설정 기록 (시작 어댑터 지문 = 독립 계산)")
    ok(os.path.exists(f"{RD}/preds_init_check__up2.0__40.csv"), "시작점 재현 점검 파일")
    h = [x for x in g["state"]["history"] if "aux_loss" in x]; ok(len(h) > 0, f"보조 학습 손실 기록 {len(h)}회")
    ok(g["state"]["epoch_acc"] == {} and os.path.exists(f"{RD}/preds_final_ep1__up2.0__1000.csv"), "epoch 끝 평가 생략 → 마지막 어댑터(ep1)로 최종 평가")
    o0, _ = g["epoch_plan"](1); o1 = np.random.RandomState(42 * 1000 + 1).permutation(len(g["train_rows"]))
    ok((o0 == o1).all() and g["total_steps"] == 4, "2번째 epoch 순서(시드 42·epoch 1) · MAX_STEPS")
    ok("선택지는 주어지지 않습니다" in g["GEN_INSTRUCT"], "보조 학습 지시문 한국어(v2)")
if scen == "ocr":
    rows = g["val_df"].to_dict("records"); t = g["render"](rows[0])
    ok(g["OCR_HEADER"] in t and t.index(g["OCR_HEADER"]) < t.index(str(rows[0]["question"])), "OCR 블록이 질문 앞")
    C = json.load(open(f"{RD}/train_log.json"))["config"]
    ok(C["PROMPT_VERSION"] == "ocr_v2_pvl16_spotting" and C["OCR_MAX_CHARS"] == 1000, "학습 기록에 OCR 프롬프트 이름")
    rep = glob.glob(f"{RD}/error_report_*.html"); ok(rep and g["OCR_HEADER"] in open(rep[0], encoding="utf-8").read(), "오답 리포트에 OCR 표시")
if scen == "dev":
    MD1 = {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in MD0}
    ok(MD0 == MD1 and len(MD0) >= 3, f"끝난 실행의 예측 파일·학습 기록이 그대로 ({len(MD0)}개 md5 같음)")
    ok(g.get("REPRO", {}).get("agree") == 1.0 and g["REPRO"]["max_prob_diff"] < 1e-5, f"저장된 어댑터로 새로 채점 = 저장된 예측 ({g.get('REPRO')})")
    ok(g["state"]["global_step"] == g["total_steps"] and len(MD0) >= 5, "학습하지 않음 (어댑터·체크포인트 md5 그대로)")
    f = glob.glob(f"{RUNS}/dev_pseudo__{RECIPE}__up2.0__c0.26_v2_agree.csv"); ok(len(f) == 1, f"dev 의사 라벨 파일 {[os.path.basename(x) for x in f]}")
    if f:
        out = pd.read_csv(f[0]); pr = pd.read_csv(glob.glob(f"{RD}/preds_dev__up2.0__*.csv")[0]).drop_duplicates("id", keep="last").set_index("id")
        dv = g["dev_df"].set_index("id"); want = []
        for i, r in dv.iterrows():
            v = [str(r[c]).strip().lower() for c in ["answer1", "answer2", "answer3", "answer4", "answer5"]]; v = [x for x in v if x in "abcd"]
            if not v: continue
            cnt = {k: v.count(k) for k in set(v)}; top = max(cnt.values()); win = [k for k, n in cnt.items() if n == top]
            p = pr.loc[i]; conf = max(p.p_a, p.p_b, p.p_c, p.p_d)
            if len(win) == 1 and top >= 2 and conf >= 0.26 and p.pred == win[0]: want.append(i)
        ok(sorted(out.id) == sorted(want), f"의사 라벨 문항 {len(out)}개 = 독립 계산 {len(want)}개 (확신 0.26 · 2표 · 다수결 일치)")
print("\n결과:", "통과" if not fails else f"실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
