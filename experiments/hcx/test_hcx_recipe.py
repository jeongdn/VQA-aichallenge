import json, os, shutil
NB = "/home/user/vqa/hcx/stage2_hcx_32b_train_full.ipynb"
ROOT = "/tmp/vqa_scratch/hcxtest"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, RECIPE = cells[0], cells[3]
assert "RUN_NAME" in CFG and "E8: RECIPE_FROM" in RECIPE
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
open(f"{ROOT}/mtvqa.csv", "w").write("id\n")
base = {"MODEL_ID": "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", "TRAIN_PRESET": "native", "EPOCHS": 1, "MAX_STEPS": None, "LR": 0.0001, "STEP_SAMPLES": 16,
        "LORA_R": 16, "LORA_ALPHA": 32, "SHUFFLE_OPTIONS": True, "TRAIN_ON_FULL": False, "SEED": 42, "TRAIN_SEED": 42, "EXTRA_TRAIN_CSV": None,
        "IMG_AUG": False, "TRAIN_PRESETS": None, "AUX_GEN_CSV": None, "AUX_GEN_EVERY": 5, "AUX_GEN_WEIGHT": 1.0}
cases = {"plain": base, "aux": dict(base, AUX_GEN_CSV=f"{ROOT}/mtvqa.csv", AUX_GEN_EVERY=4, AUX_GEN_WEIGHT=0.5),
         "aux_missing": dict(base, AUX_GEN_CSV=f"{ROOT}/nope.csv"), "old": {k: v for k, v in base.items() if not k.startswith("AUX") and k not in ("TRAIN_SEED",)}}
ok = True
for name, cfg in cases.items():
    d = f"{ROOT}/{name}"; os.makedirs(d); json.dump({"epoch_acc": {"1": 0.978}, "config": cfg}, open(f"{d}/train_log.json", "w"))
    g = {"__name__": "__main__"}; exec(compile(CFG, "<cfg>", "exec"), g); g["OUT_DIR"] = ROOT; g["RECIPE_FROM"] = name
    try:
        exec(compile(RECIPE, "<recipe>", "exec"), g)
    except Exception as e:
        good = name == "aux_missing" and "AUX_GEN_CSV" in str(e); ok &= good
        print(("✅" if good else "❌"), name, "멈춤:", str(e)[:100]); continue
    want = {"RUN_NAME": name + "_full", "TRAIN_ON_FULL": True, "MICRO_BS": 2, "GRAD_ACCUM": 8, "USE_4BIT": "auto", "TEST_TTA": False, "MAKE_SUBMISSION": True,
            "AUX_GEN_CSV": cfg.get("AUX_GEN_CSV"), "AUX_GEN_EVERY": cfg.get("AUX_GEN_EVERY", 5), "AUX_GEN_WEIGHT": cfg.get("AUX_GEN_WEIGHT", 1.0), "TRAIN_SEED": 42}
    bad = {k: (g.get(k), v) for k, v in want.items() if g.get(k) != v}; ok &= not bad and name != "aux_missing"
    print(("✅" if not bad else "❌"), name, bad or "")
print("전체:", "통과" if ok else "실패")
