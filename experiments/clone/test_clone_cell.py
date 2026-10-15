# -*- coding: utf-8 -*-
"""3-1번 셀(레시피 복사 → bf16) 시험: 가짜 train_log.json들로 설정·폴더 이름이 정확히 정해지는지, 막아야 할 경우 멈추는지."""
import json, os, shutil
NB = "/home/user/vqa/clone/out/stage2_bf16_clone.ipynb"
ROOT = "/tmp/vqa_scratch/clonetest"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG = cells[0]; RECIPE = next(c for c in cells if c.startswith("# ── 레시피 복사"))
assert "RECIPE_FROM" in CFG and cells.index(RECIPE) == 3 and "drive.mount" in cells[2]
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(ROOT)
base = {"MODEL_ID": "Qwen/Qwen3.8-27B", "TRAIN_PRESET": "native", "EPOCHS": 1, "MAX_STEPS": None, "LR": 0.0001, "STEP_SAMPLES": 16,
        "LORA_R": 16, "LORA_ALPHA": 32, "SHUFFLE_OPTIONS": True, "TRAIN_ON_FULL": False, "SEED": 42, "TRAIN_SEED": 42,
        "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None}
DEV = f"{ROOT}/dev_pseudo.csv"; open(DEV, "w").write("id\n" + "\n".join(f"dev_{i:04d}" for i in range(2683)) + "\n")
G = dict(base, MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0")
RUNS = {
    "q38_27b_r16_native": base, "q35_27b_r16_native": dict(base, MODEL_ID="Qwen/Qwen3.5-27B"),
    "gemma4_31b_r16_up20": G, "gemma4_31b_r16_up20_dev": dict(G, EXTRA_TRAIN_CSV=DEV),
    "gdev_missing_csv": dict(G, EXTRA_TRAIN_CSV=f"{ROOT}/nope.csv"), "other_extra": dict(base, EXTRA_TRAIN_CSV=f"{ROOT}/other.csv"),
    "old_cfg": {k: v for k, v in base.items() if k not in ("TRAIN_SEED", "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "MAX_STEPS")},
    "already_full": dict(base, TRAIN_ON_FULL=True), "maxsteps_native": dict(base, MAX_STEPS=200),
    "multires": dict(base, TRAIN_PRESETS=["native", "up1.5", "up2.0"]),
    "hcx32b_r16_native": dict(base, MODEL_ID="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"), "noconfig": {},
}
open(f"{ROOT}/other.csv", "w").write("id\n")
for r, cfg in RUNS.items():
    os.makedirs(f"{ROOT}/{r}"); json.dump({"epoch_acc": {"1": 0.97}, "history": [], "config": cfg}, open(f"{ROOT}/{r}/train_log.json", "w"))
os.makedirs(f"{ROOT}/q38_27b_r16_native_bf16"); open(f"{ROOT}/q38_27b_r16_native_bf16/preds_final_ep1_test__up2.0__6714.csv", "w").write("id\n")
def run(recipe_from, pre=None, expect_err=None, **cfg):
    g = {"__name__": "__main__"}; exec(compile(CFG, "<cfg>", "exec"), g)
    g["OUT_DIR"] = ROOT; g["RECIPE_FROM"] = recipe_from; g.update(cfg)
    if pre: g.update(pre)
    try: exec(compile(RECIPE, "<recipe>", "exec"), g)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"; ok = expect_err is not None and expect_err in msg
        print(f"{'✅' if ok else '❌'} {recipe_from!s:<26} {cfg} 멈춤 → {msg[:140]}"); return ok
    if expect_err is not None: print(f"❌ {recipe_from!s:<26} 멈춰야 하는데 통과"); return False
    return g
ok_all = True
def check(name, g, **want):
    global ok_all
    if g is False or g is True: ok_all &= bool(g); return
    want.setdefault("TRAIN_ON_FULL", False); want.setdefault("USE_4BIT", False); want.setdefault("MAKE_SUBMISSION", True); want.setdefault("TEST_PRESET", "up2.0")
    bad = {k: (g.get(k), v) for k, v in want.items() if g.get(k) != v}; ok_all &= not bad
    print(f"{'✅' if not bad else '❌'} {name:<34} RUN_NAME={g.get('RUN_NAME')} " + ("" if not bad else f"다름: {bad}"))
print("━━ 정상 경우 ━━")
g = run("gemma4_31b_r16_up20_dev"); check("GDEV bf16 (기본값)", g, RUN_NAME="gemma4_31b_r16_up20_dev_bf16", MODEL_ID="google/gemma-4-31B-it", TRAIN_PRESET="up2.0",
                                          EXTRA_TRAIN_CSV=DEV, MICRO_BS=1, GRAD_ACCUM=16, TRAIN_SEED=42, RES_PRESET="up2.0", EVAL_PRESETS=["up2.0"])
g = run("q38_27b_r16_native"); check("Q38 bf16 (이미 test 있음 → 안내)", g, RUN_NAME="q38_27b_r16_native_bf16", MODEL_ID="Qwen/Qwen3.8-27B", MICRO_BS=2, GRAD_ACCUM=8, EXTRA_TRAIN_CSV=None)
g = run("q38_27b_r16_native", NEW_SEED=43); check("Q38 시드 43", g, RUN_NAME="q38_27b_r16_native_bf16_s43", TRAIN_SEED=43, EXTRA_TRAIN_CSV=None)
g = run("q38_27b_r16_native", EXTRA_FROM="gemma4_31b_r16_up20_dev"); check("Q38 + dev (EXTRA_FROM)", g, RUN_NAME="q38_27b_r16_dev_bf16", EXTRA_TRAIN_CSV=DEV, TRAIN_SEED=42, MODEL_ID="Qwen/Qwen3.8-27B")
g = run("q35_27b_r16_native", EXTRA_FROM="gemma4_31b_r16_up20_dev", NEW_SEED=7); check("Q35 + dev + 시드 7", g, RUN_NAME="q35_27b_r16_dev_bf16_s7", TRAIN_SEED=7, EXTRA_TRAIN_CSV=DEV)
g = run("gemma4_31b_r16_up20", EXTRA_FROM="gemma4_31b_r16_up20_dev"); check("Gemma + dev = GDEV 이름", g, RUN_NAME="gemma4_31b_r16_up20_dev_bf16", EXTRA_TRAIN_CSV=DEV)
g = run("gemma4_31b_r16_up20_dev", EXTRA_FROM="gemma4_31b_r16_up20_dev"); check("GDEV + 자기 자신 dev (이름 안 겹침)", g, RUN_NAME="gemma4_31b_r16_up20_dev_bf16", EXTRA_TRAIN_CSV=DEV)
g = run("maxsteps_native", EXTRA_FROM="gemma4_31b_r16_up20_dev"); check("MAX_STEPS + dev → 비율만큼 늘림", g, MAX_STEPS=int(round(200 * (5714 + 2683) / 5714)), RUN_NAME="maxsteps_dev_bf16")
g = run("maxsteps_native"); check("MAX_STEPS 그대로 (같은 데이터)", g, MAX_STEPS=200)
g = run("old_cfg"); check("예전 config(기본값)", g, TRAIN_SEED=42, EXTRA_TRAIN_CSV=None, IMG_AUG=False, TRAIN_PRESETS=None, MAX_STEPS=None, RUN_NAME="old_cfg_bf16")
g = run("multires"); check("다해상도 → MICRO_BS 1", g, TRAIN_PRESETS=["native", "up1.5", "up2.0"], MICRO_BS=1, GRAD_ACCUM=16)
print("━━ 멈춰야 하는 경우 ━━")
check("", run("gdev_missing_csv", expect_err="추가 학습 데이터(EXTRA_TRAIN_CSV)가 없습니다"))
check("", run("already_full", expect_err="전체 데이터(검증 포함)로 학습한 실행"))
check("", run("hcx32b_r16_native", expect_err="HCX는 이 노트북이 아니라"))
check("", run("noconfig", expect_err="학습 설정(config)이 없습니다"))
check("", run("q38_typo", expect_err="가 없습니다"))
check("", run("q38_27b_r16_native", EXTRA_FROM="q35_27b_r16_native", expect_err="추가 학습 데이터(EXTRA_TRAIN_CSV) 없이 학습한 실행"))
check("", run("q38_27b_r16_native", EXTRA_FROM="gdev_typo", expect_err="EXTRA_FROM을 확인하세요"))
check("", run("other_extra", EXTRA_FROM="gemma4_31b_r16_up20_dev", expect_err="이미 다른 추가 데이터로 학습했습니다"))
check("", run("q38_27b_r16_native", NEW_SEED=42, expect_err="원래 시드와 같습니다"))
check("", run("q38_27b_r16_native", pre={"RUN_DIR": f"{ROOT}/q38_27b_r16_native"}, expect_err="결과 폴더(RUN_DIR="))
print("\n전체:", "통과" if ok_all else "실패 있음")
