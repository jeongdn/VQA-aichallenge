# -*- coding: utf-8 -*-
"""stage2_newfam.ipynb 코드 셀을 순서대로 CPU에서 실행하는 시험 도구 (작은 무작위 모델 + 실제 transformers 프로세서).
준비: python3 run_newfam.py prep                 → 가짜 Drive 구조 (실제 CSV + 작은 이미지)
실행: python3 run_newfam.py <fam> <scenario> [key=value ...]
  fam: muse | exaone45
  scenario: zs(ZERO_SHOT_ONLY) · train(zs 결과 재사용 후 학습·평가·제출) · dry · stop(중간 중단) · resume · cap(토큰 상한) · nozs(ZERO_SHOT_N=0)
검사: 프롬프트 끝, 이미지 토큰 수(독립 계산), zero-shot 확률 = 기본 모델 직접 채점(독립 코드), 이어하기, 제출 파일."""
import sys, os, json, shutil, hashlib, math, re, glob, types
import numpy as np, pandas as pd

NB = "/home/user/vqa/newfam/out/stage2_newfam.ipynb"
SP = "/tmp/vqa_scratch/newfam"
ROOT = f"{SP}/fx"; DATA, RUNS = f"{ROOT}/content", f"{ROOT}/vqa_runs"
U = "/tmp/vqa_uploads"

if sys.argv[1] == "prep":
    from PIL import Image, ImageDraw
    shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(DATA); os.makedirs(RUNS)
    for src, dst in [("6247e8ae-train.csv", "train.csv"), ("847ba753-test.csv", "test.csv"),
                     ("eb905257-sample_submission.csv", "sample_submission.csv"), ("359d3794-dev.csv", "dev.csv")]:
        shutil.copy(f"{U}/{src}", f"{DATA}/{dst}")
    tr, te = pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")
    for p in list(tr.path) + list(te.path):
        os.makedirs(os.path.dirname(f"{DATA}/{p}"), exist_ok=True)
        h = int(hashlib.md5(p.encode()).hexdigest()[:8], 16)
        w, hh = 40 + h % 70, 30 + (h >> 7) % 60                  # 크기가 제각각이어야 해상도 맞춤이 검사됨
        im = Image.new("RGB", (w, hh), (h % 256, (h >> 8) % 256, (h >> 16) % 256))
        ImageDraw.Draw(im).rectangle([w // 4, hh // 4, w // 2, hh // 2], fill=((h >> 3) % 256, 30, 200))
        im.save(f"{DATA}/{p}", quality=85)
    print("준비 끝:", len(tr), len(te)); sys.exit()

fam, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, IMPORTS, SPLIT, PROMPT, MODEL, SCORING, PREP, LOOP, EVAL, TEST, SUMM = cells
TINY = {"muse": f"{SP}/tiny_muse", "exaone45": f"{SP}/tiny_exaone45"}[fam]
RUN = f"t_{fam}"
if scen in ("zs", "dry", "nozs", "cap", "stop"):                  # 새 실행: 이전 결과 지움 (train·resume은 앞 단계 결과를 이어 씀)
    for d in glob.glob(f"{RUNS}/{RUN}*"): shutil.rmtree(d)
if scen == "stop": LOOP = LOOP.replace("_STOP_AT = None", "_STOP_AT = 2", 1)

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)

import torch, transformers
from PIL import Image
_orig_proc = transformers.AutoProcessor.from_pretrained
def _proc(mid, **kw):
    p = _orig_proc(mid, **kw)
    if scen == "cap": p.image_processor.max_image_tokens = 10      # 토큰 상한이 픽셀 범위보다 작을 때
    return p
transformers.AutoProcessor.from_pretrained = _proc

exit_msg = None
try:
    run(CFG, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "MODEL_ID": TINY, "RUN_NAME": RUN,
              "TRAIN_LIMIT": 64, "QUICK_EVAL_N": 12, "ZERO_SHOT_N": 24, "EVAL_EVERY": 2, "SAVE_EVERY": 1,
              "RES_PRESETS": {"baseline384": (784 * 2, 784 * 4), "mid": (784 * 3, 784 * 6), "native": (784 * 4, 784 * 8),
                              "up1.5": (784 * 8, 784 * 12), "up2.0": (784 * 12, 784 * 16)},
              "ZERO_SHOT_ONLY": scen == "zs", "DRY_RUN": scen == "dry", "MAKE_SUBMISSION": scen in ("train", "resume", "nozs"),
              "EVAL_PRESETS": ["native", "up2.0"]})
    if scen == "nozs": g["ZERO_SHOT_N"] = 0
    for k, v in over.items(): g[k] = eval(v, g)
    run(DATA_C, "3 데이터"); run(IMPORTS, "4 import")
    g["test_df"] = g["test_df"].head(60).copy(); g["sub_df"] = g["sub_df"].head(60).copy()      # test 추론은 앞 60문항만 (속도)
    run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트")
    run(MODEL, "7 모델"); run(SCORING, "8 채점"); run(PREP, "9 학습 준비")
    run(LOOP, "10 학습 루프"); run(EVAL, "11 평가"); run(TEST, "12 test"); run(SUMM, "13 요약")
except SystemExit as e:
    exit_msg = str(e); print("\n[SystemExit]", exit_msg)
except RuntimeError as e:
    if not (scen == "stop" and "학습이 아직 끝나지 않았습니다" in str(e)): raise
    exit_msg = str(e); print("\n[예상된 중단]", exit_msg)

# ───────────────────────── 독립 검사 ─────────────────────────
print("\n══════ 검사 ══════")
fails = []
def ok(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond: fails.append(msg)
RUN_DIR = g["RUN_DIR"]; proc = g["processor"]; tmpl_kw = g["TEMPLATE_KW"]
probe = g["render"]({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"})
if fam == "muse":
    ok(g["FAMILY"] == "muse" and probe.endswith("<|start|>assistant<|message|>정답: ("), f"Muse 프롬프트 끝 = 답 채널 + '정답: (': {probe[-40:]!r}")
    _e = g["encode_batch"]([g["val_df"].iloc[0].to_dict()], [(0, 1, 2, 3)]); _ids = _e["input_ids"][0].tolist(); _bos = g["tok"].bos_token_id
    ok(g["NO_DOUBLE_BOS"] and _ids[0] == _bos and _ids[1] != _bos, f"BOS 1개 (템플릿 BOS + 토크나이저 자동 BOS 중복 방지): 앞 토큰 {_ids[:3]}")
    ok(g["ZS_TAG"].startswith("zeroshot_") and len(g["ZS_TAG"]) == len("zeroshot_") + 6, f"zero-shot 태그에 형식 해시 {g['ZS_TAG']}")
    ok(tmpl_kw.get("reasoning_strength") == "low" and "Reasoning strength: low." in probe, "추론 강도 low가 시스템 프롬프트에 들어감")
    ok(tmpl_kw.get("current_date") == "2026-09-27" and "Current date: 2026-09-27." in probe, "오늘 날짜 대신 고정 날짜가 들어감")
    ok(g["PIXEL_KW"].get("max_image_tokens") == (10 if scen == "cap" else 4096), f"PIXEL_KW {g['PIXEL_KW']}")
else:
    ok(g["FAMILY"] == "qwen" and g["GEN_SUFFIX"] == "", "EXAONE는 qwen 경로, 답 채널 접미사 없음")
    ok(tmpl_kw == {"enable_thinking": False} and probe.endswith("[|assistant|]<think>\n\n</think>\n\n"), f"EXAONE 프롬프트 끝: {probe[-40:]!r}")

def expect_tokens(path, preset):                                   # 독립 계산: Qwen2-VL smart_resize(28) → (Muse는 토큰 상한 반영)
    mn, mx = g["RES_PRESETS"][preset]
    if fam == "muse": mx = min(mx, (10 if scen == "cap" else 4096) * 784); mn = min(mn, mx)
    w, h = Image.open(f"{DATA}/{path}").size; f = 28
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
    elif hb * wb < mn:
        b = math.sqrt(mn / (h * w)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
        if fam == "muse" and hb * wb > mx:          # 노트북 fit_pixels의 보정 (Qwen 프로세서 자체는 보정 없음)
            b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
    return (hb // 28) * (wb // 28)

val = g["val_df"]
zsf = glob.glob(f"{RUN_DIR}/preds_{g['ZS_TAG']}__*.csv")
if scen in ("nozs", "resume"):
    ok(not zsf or scen == "resume", f"zero-shot 파일 {len(zsf)}개 (nozs면 0)")
if scen in ("zs", "dry", "cap", "stop", "train"):
    ok(len(zsf) == 1, f"zero-shot 예측 파일: {[os.path.basename(x) for x in zsf]}")
    z = pd.read_csv(zsf[0]); zpre = g["ZERO_SHOT_PRESET"]
    n_exp = 24 if not g["DRY_RUN"] else 3 * g["BS"]
    ok(len(z) == n_exp and list(z.id) == list(val.id.head(n_exp)), f"zero-shot 문항 = 검증 앞 {n_exp}개 ({len(z)})")
    et = [expect_tokens(p, zpre) for p in val.path.head(n_exp)]
    ok(z.img_tokens.tolist() == et, f"이미지 토큰 수 = 독립 계산 ({zpre}) 예: {z.img_tokens.tolist()[:6]} vs {et[:6]}")
    if fam == "muse" and scen == "cap": ok(max(z.img_tokens) <= 10, f"토큰 상한 10 지킴 (최대 {max(z.img_tokens)})")
    # zero-shot = 기본 모델 직접 채점 (노트북 함수 안 쓰고 새로 로드)
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(TINY, dtype=torch.float32).eval()
    P = transformers.AutoProcessor.from_pretrained(TINY); P.tokenizer.padding_side = "left"
    if scen == "cap": P.image_processor.max_image_tokens = 10
    tok = P.tokenizer
    lid = [[tok.convert_tokens_to_ids(tok.tokenize(s)[0]) for s in (L, L.upper(), " " + L, " " + L.upper()) if len(tok.tokenize(s)) == 1] for L in "abcd"]
    lid = [list(dict.fromkeys(x)) for x in lid]
    maxd = 0.0
    for r in val.head(6).to_dict("records"):
        msgs = [{"role": "system", "content": [{"type": "text", "text": g["SYSTEM_INSTRUCT"]}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": g["build_mc_prompt"](r["question"], [r[k] for k in "abcd"])}]}]
        kw = {"reasoning_strength": "low", "current_date": "2026-09-27"} if fam == "muse" else {"enable_thinking": False}
        t = P.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + ("<|message|>정답: (" if fam == "muse" else "")
        im = Image.open(f"{DATA}/{r['path']}").convert("RGB")
        mn, mx = g["RES_PRESETS"][zpre]
        if fam == "muse":
            mx = min(mx, P.image_processor.max_image_tokens * 784)
            # 독립 경로: 목표 크기를 직접 계산해 LANCZOS로 맞춘 뒤 넘김
            w, h = im.size; f = 28
            hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
            if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
            elif hb * wb < min(mn, mx):
                b = math.sqrt(min(mn, mx) / (h * w)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
                if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
            im = im.resize((wb, hb), Image.LANCZOS) if (wb, hb) != (w, h) else im
            e = P(text=[t], images=[im], return_tensors="pt", max_image_tokens=P.image_processor.max_image_tokens, add_special_tokens=False)
        else:
            e = P(text=[t], images=[im], return_tensors="pt", min_pixels=mn, max_pixels=mx)
        with torch.no_grad(): lg = base(**e).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1); slot = torch.stack([torch.logsumexp(lp[ids], 0) for ids in lid])
        pr = torch.softmax(slot, 0).numpy(); row = z[z.id == r["id"]].iloc[0]
        maxd = max(maxd, float(np.abs(pr - row[["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)).max()))
    ok(maxd < 1e-4, f"zero-shot 확률 = 기본 모델 직접 채점 (최대 차이 {maxd:.2e})")
    zh0 = [x for x in g["state"]["history"] if "zero_shot_acc" in x]
    if zh0 and zh0[0]["zero_shot_mass"] < 0.5:                                  # 작은 무작위 모델은 letter_mass가 낮음 → 생성 기반 점검이 돌아야 함
        gfp = glob.glob(f"{RUN_DIR}/preds_{g['ZS_TAG']}_gen__*.csv")
        ok(len(gfp) == 1 and len(pd.read_csv(gfp[0])) == n_exp, f"생성 기반 zero-shot 파일 {[os.path.basename(x) for x in gfp]}")
        G = pd.read_csv(gfp[0], keep_default_na=False)
        acc = float((G.gen.map(g["parse_answer_letter"]) == G.answer).mean())
        ok(abs(zh0[0].get("zero_shot_gen_acc", -1) - acc) < 1e-12, f"생성 기반 정확도 기록 = 파일로 다시 계산 ({acc * 100:.1f}%)")
        # 독립 생성: 새로 불러온 기본 모델로 같은 첫 배치(2문항, 왼쪽 패딩)를 그리디 16토큰
        texts, ims = [], []
        for r in val.head(2).to_dict("records"):
            msgs = [{"role": "system", "content": [{"type": "text", "text": g["SYSTEM_INSTRUCT"]}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": g["build_mc_prompt"](r["question"], [r[k] for k in "abcd"])}]}]
            kw = {"reasoning_strength": "low", "current_date": "2026-09-27"} if fam == "muse" else {"enable_thinking": False}
            texts.append(P.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + ("<|message|>정답: (" if fam == "muse" else ""))
            im = Image.open(f"{DATA}/{r['path']}").convert("RGB")
            if fam == "muse":
                mn, mx = g["RES_PRESETS"][zpre]; mx = min(mx, P.image_processor.max_image_tokens * 784); w, h = im.size; f = 28
                hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
                if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
                elif hb * wb < min(mn, mx):
                    b = math.sqrt(min(mn, mx) / (h * w)); hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
                    if hb * wb > mx: b = math.sqrt(h * w / mx); hb, wb = max(f, math.floor(h / b / f) * f), max(f, math.floor(w / b / f) * f)
                im = im.resize((wb, hb), Image.LANCZOS) if (wb, hb) != (w, h) else im
            ims.append(im)
        kwp = {"max_image_tokens": P.image_processor.max_image_tokens, "add_special_tokens": False} if fam == "muse" else {"min_pixels": g["RES_PRESETS"][zpre][0], "max_pixels": g["RES_PRESETS"][zpre][1]}
        e = P(text=texts, images=ims, padding=True, return_tensors="pt", **kwp)
        with torch.no_grad(): out = base.generate(**e, max_new_tokens=16, do_sample=False, use_cache=True)
        gen2 = [P.tokenizer.decode(q[e["input_ids"].shape[1]:], skip_special_tokens=True) for q in out]
        ok(gen2 == list(G.gen.head(2)), f"생성 결과 = 기본 모델 직접 생성 (첫 배치 2문항) {gen2[0][:20]!r}")
    hist = g["state"]["history"]; zh = [x for x in hist if "zero_shot_acc" in x]
    ok(len(zh) == 1, f"history에 zero-shot 기록 1개 ({len(zh)})")
if scen == "zs":
    ok(exit_msg is not None and "ZERO_SHOT_ONLY" in exit_msg, "ZERO_SHOT_ONLY → 학습 전에 멈춤")
    ok(g["state"]["global_step"] == 0 and not os.path.exists(f"{RUN_DIR}/ckpt_last"), "학습 스텝 0, 체크포인트 없음")
if scen == "stop":
    ok(g["state"]["global_step"] == 2 and exit_msg and "학습이 아직" in exit_msg, f"강제 중단 스텝 2 ({g['state']['global_step']}) → 11번 셀이 이어하기 안내")
    ok(os.path.exists(f"{RUN_DIR}/ckpt_last/trainer_state.pt"), "체크포인트 저장됨")
if scen in ("train", "resume", "nozs"):
    ok(exit_msg is None, "끝까지 실행")
    ok(g["state"]["global_step"] == g["total_steps"] == 4, f"학습 {g['state']['global_step']}/{g['total_steps']} 스텝")
    ok(os.path.exists(f"{RUN_DIR}/adapter_ep1/adapter_config.json"), "epoch 어댑터 저장")
    fin = sorted(glob.glob(f"{RUN_DIR}/preds_final_ep1__*__1000.csv"))
    ok([os.path.basename(x) for x in fin] == ["preds_final_ep1__native__1000.csv", "preds_final_ep1__up2.0__1000.csv"], f"최종 검증 예측: {[os.path.basename(x) for x in fin]}")
    f2 = pd.read_csv(f"{RUN_DIR}/preds_final_ep1__up2.0__1000.csv")
    ok(f2.img_tokens.tolist() == [expect_tokens(p, "up2.0") for p in val.path], "최종 평가 이미지 토큰 수 = 독립 계산 (up2.0, 1000문항)")
    sub = glob.glob(f"{RUN_DIR}/submission_*.csv")
    ok(len(sub) == 1 and len(pd.read_csv(sub[0])) == 60 and set(pd.read_csv(sub[0]).answer) <= set("abcd"), f"제출 파일 {[os.path.basename(x) for x in sub]}")
    zh = [x for x in g["state"]["history"] if "zero_shot_acc" in x]
    ok(len(zh) == (0 if scen == "nozs" else 1), f"history의 zero-shot 기록 {len(zh)}개 (이어하기에서 다시 돌지 않음)")
    if scen == "train":
        z0 = pd.read_csv(glob.glob(f"{RUN_DIR}/preds_{g['ZS_TAG']}__*.csv")[0])
        ok(len(z0) == 24, "zs 단계 결과 재사용 (파일 그대로)")
    log = pd.read_csv(f"{RUNS}/results_log_s2.csv"); log = log[log.run.str.startswith(RUN)]
    ok(any(log.run.str.endswith("/" + g["ZS_TAG"])) or scen == "nozs", f"결과 로그에 {g['ZS_TAG']} 행")
print("\n결과:", "통과" if not fails else f"실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
