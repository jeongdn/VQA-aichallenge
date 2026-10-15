# -*- coding: utf-8 -*-
"""3-1번 셀(레시피 복사) 시험: 가짜 train_log.json들로 설정이 정확히 넘어오는지, 막아야 할 경우 멈추는지."""
import json, os, shutil, sys, traceback
NB = "/home/user/vqa/e8/out/e8_full_retrain.ipynb"
ROOT = "/tmp/vqa_scratch/e8train"
cells = [ "".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, RECIPE = cells[0], cells[3]
assert "RECIPE_FROM" in CFG and "E8: RECIPE_FROM" in RECIPE
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(ROOT)
base = {"MODEL_ID": "Qwen/Qwen3.8-27B", "TRAIN_PRESET": "native", "EPOCHS": 1, "MAX_STEPS": None, "LR": 0.0001, "STEP_SAMPLES": 16,
        "LORA_R": 16, "LORA_ALPHA": 32, "SHUFFLE_OPTIONS": True, "TRAIN_ON_FULL": False, "SEED": 42, "TRAIN_SEED": 42,
        "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None}
open(f"{ROOT}/dev_pseudo.csv", "w").write("id\n")
RUNS = {
    "q38_27b_r16_native": (base, {"1": 0.975}),
    "gemma4_31b_r16_up20": (dict(base, MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0"), {"1": 0.976}),
    "gemma4_31b_r16_up20_dev": (dict(base, MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0", EXTRA_TRAIN_CSV=f"{ROOT}/dev_pseudo.csv"), {"1": 0.977}),
    "gdev_missing_csv": (dict(base, MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0", EXTRA_TRAIN_CSV=f"{ROOT}/nope.csv"), {"1": 0.97}),
    "old_cfg": ({k: v for k, v in base.items() if k not in ("TRAIN_SEED", "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "MAX_STEPS")}, {"1": 0.97}),
    "already_full": (dict(base, TRAIN_ON_FULL=True), {"1": 0.99}),
    "ep2_best1": (dict(base, EPOCHS=2), {"1": 0.976, "2": 0.974}),
    "ep2_best2": (dict(base, EPOCHS=2), {"1": 0.974, "2": 0.976}),
    "maxsteps": (dict(base, MAX_STEPS=200), {"1": 0.97}),
    "multires": (dict(base, TRAIN_PRESETS=["native", "up1.5", "up2.0"]), {"1": 0.97}),
    "hcx32b_r16_native": (dict(base, MODEL_ID="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"), {"1": 0.978}),
    "noconfig": ({}, {"1": 0.97}),
}
for r, (cfg, ea) in RUNS.items():
    os.makedirs(f"{ROOT}/{r}"); json.dump({"epoch_acc": ea, "history": [], "config": cfg}, open(f"{ROOT}/{r}/train_log.json", "w"))
os.makedirs(f"{ROOT}/q38_27b_r16_native_full"); open(f"{ROOT}/q38_27b_r16_native_full/preds_final_ep1_test__up2.0__6714.csv", "w").write("id\n")

def run(recipe_from, stage2=True, pre=None, expect_err=None):
    g = {"__name__": "__main__"}
    exec(compile(CFG, "<cfg>", "exec"), g)
    g["OUT_DIR"] = ROOT
    if recipe_from is None: g.pop("RECIPE_FROM")
    else: g["RECIPE_FROM"] = recipe_from
    if not stage2: g.pop("E8_STAGE2_NB"); g["MICRO_BS"], g["GRAD_ACCUM"] = 1, 8
    if pre: g.update(pre)
    try:
        exec(compile(RECIPE, "<recipe>", "exec"), g)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        ok = expect_err is not None and expect_err in msg
        print(f"{'✅' if ok else '❌'} {recipe_from!s:<26} 멈춤 → {msg[:150]}"); return ok
    if expect_err is not None:
        print(f"❌ {recipe_from!s:<26} 멈춰야 하는데 통과"); return False
    return g

ok_all = True
def check(name, g, **want):
    global ok_all
    if g is False or g is True: ok_all &= bool(g); return
    bad = {k: (g.get(k), v) for k, v in want.items() if g.get(k) != v}
    ok_all &= not bad
    print(f"{'✅' if not bad else '❌'} {name:<26} " + ("" if not bad else f"다름: {bad}"))

print("━━ 정상 경우 (출력 일부) ━━")
g = run("q38_27b_r16_native"); check("q38", g, MODEL_ID="Qwen/Qwen3.8-27B", TRAIN_PRESET="native", RUN_NAME="q38_27b_r16_native_full",
                                    TRAIN_ON_FULL=True, MICRO_BS=2, GRAD_ACCUM=8, MAKE_SUBMISSION=True, TEST_PRESET="up2.0", EVAL_PRESETS=["up2.0"], RES_PRESET="native")
g = run("gemma4_31b_r16_up20"); check("gemma", g, MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0", MICRO_BS=1, GRAD_ACCUM=16, RUN_NAME="gemma4_31b_r16_up20_full")
g = run("gemma4_31b_r16_up20_dev"); check("gdev", g, EXTRA_TRAIN_CSV=f"{ROOT}/dev_pseudo.csv", RUN_NAME="gemma4_31b_r16_up20_dev_full")
g = run("old_cfg"); check("old config(기본값)", g, TRAIN_SEED=42, EXTRA_TRAIN_CSV=None, IMG_AUG=False, TRAIN_PRESETS=None, MAX_STEPS=None)
g = run("ep2_best2"); check("EPOCHS=2, ep2 선택", g, EPOCHS=2)
g = run("maxsteps"); check("MAX_STEPS 비례 증가", g, MAX_STEPS=235)
g = run("multires"); check("다해상도 → MICRO_BS 1", g, TRAIN_PRESETS=["native", "up1.5", "up2.0"], MICRO_BS=1, GRAD_ACCUM=16)
g = run(None, stage2=False); check("HCX 노트북에 붙여넣기", g, RECIPE_FROM="hcx32b_r16_native", RUN_NAME="hcx32b_r16_native_full",
                                   MODEL_ID="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", MICRO_BS=1, GRAD_ACCUM=16)
print("━━ 멈춰야 하는 경우 ━━")
check("", run("gdev_missing_csv", expect_err="추가 학습 데이터(EXTRA_TRAIN_CSV)가 없습니다"))
check("", run("already_full", expect_err="이미 전체 데이터로 학습한 실행"))
check("", run("ep2_best1", expect_err="epoch 1를 골랐습니다"))
check("", run("hcx32b_r16_native", expect_err="HCX는 이 노트북이 아니라"))
check("", run("noconfig", expect_err="학습 설정(config)이 없습니다"))
check("", run("q38_typo", expect_err="가 없습니다"))
check("", run("q38_27b_r16_native", pre={"RUN_DIR": f"{ROOT}/q38_27b_r16_native"}, expect_err="결과 폴더(RUN_DIR="))
print("\n전체:", "통과" if ok_all else "실패 있음")
