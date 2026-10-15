# -*- coding: utf-8 -*-
"""e1_bf16_reeval.ipynb의 코드 셀을 순서대로 실행하는 시험 도구.
mode "tiny": 모델 셀까지 실제로 실행 (작은 Qwen3.5 + 실제 PEFT 어댑터 + 가짜 프로세서, CUDA 함수만 가짜)
mode "mock": 모델 셀 대신 가짜 모델, score_rows를 시나리오별 확률로 교체 → 비교·판정·test 흐름 확인
사용: python3 run_nb.py <mode> <scenario> [key=value ...]"""
import sys, os, json, re, types, contextlib, shutil
import numpy as np

NB = "/home/user/vqa/e1/e1_bf16_reeval.ipynb"
ROOT = "/tmp/vqa_scratch/e1test"
mode, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
FX = json.load(open(f"{ROOT}/fixture.json"))

cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA, IMPORTS, SPLIT, PROMPT, MODEL, SCORING, PRECHECK, VALEVAL, COMPARE, TEST, SUMMARY = cells
assert len(cells) == 13

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True)
    exec(compile(src, f"<{name}>", "exec"), g)

run(CFG, "1 설정")
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": f"{ROOT}/vqa_runs", "DRIVE_ZIP": "/nonexistent.zip",
          "ENSEMBLE_REF": dict(g["ENSEMBLE_REF"], expected_val_acc=round(FX["ens_acc"], 2))})
for k, v in over.items():
    g[k] = eval(v, g)
g["RUN_NAME"] = f"{g['SRC_RUN']}_bf16eval"
if scen == "md5":
    os.makedirs(f"{ROOT}/vqa_runs/{g['RUN_NAME']}", exist_ok=True)
    json.dump({"adapter_md5": "0" * 32}, open(f"{ROOT}/vqa_runs/{g['RUN_NAME']}/e1_info.json", "w"))
run(DATA, "3 Drive·원본 확인·데이터")
run(IMPORTS, "4 import·CSV")
run(SPLIT, "5 분할")
run(PROMPT, "6 프롬프트")

import torch
if mode == "tiny":
    sys.path.insert(0, "/home/user/vqa/e1")
    import transformers
    from tiny_qwen import FakeQwenProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    PROC = FakeQwenProcessor()
    transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
    def _fp(mid, **kw):
        assert mid == g["MODEL_ID"] and kw.get("device_map") == {"": 0} and kw.get("dtype") == torch.bfloat16, kw
        kw.pop("device_map")
        return Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", **kw)
    transformers.AutoModelForImageTextToText.from_pretrained = _fp
    real_avail = torch.cuda.is_available
    vram = float(over.get("_VRAM", "96"))
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda i=0: "FAKE-RTX-PRO-6000"
    torch.cuda.get_device_capability = lambda i=0: (12, 0)
    torch.cuda.get_device_properties = lambda i=0: types.SimpleNamespace(total_memory=int(vram * 2**30))
    torch.cuda.memory_allocated = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda: None
    import peft.peft_model as _ppm; _ppm.infer_device = lambda: "cpu"   # 시험 도구 한정: CUDA를 가짜로 켰으므로 PEFT가 어댑터를 CPU로 읽게
    try:
        run(MODEL, "7 모델 로드")
    finally:
        torch.cuda.is_available = real_avail
    # 단위 시험: 어댑터 대조 함수가 이상을 잡는지
    from peft import PeftModel
    import safetensors.torch as st
    bad = f"{ROOT}/adapter_badkey"; shutil.rmtree(bad, ignore_errors=True); shutil.copytree(g["ADAPTER_DIR"], bad)
    sd = st.load_file(f"{bad}/adapter_model.safetensors"); k0 = sorted(sd)[0]; sd[k0.replace("lora_A", "lora_X")] = sd.pop(k0)
    st.save_file(sd, f"{bad}/adapter_model.safetensors")
    for label, fn in [("키 이름이 다른 파일", lambda: g["check_adapter_weights"](g["model"], bad)),
                      ("lora_B가 0인 어댑터", lambda: g["check_adapter_weights"](
                          PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", dtype=torch.bfloat16),
                                                    f"{ROOT}/adapter_zeroB"), f"{ROOT}/adapter_zeroB"))]:
        try: fn(); print(f"❌ 단위 시험 실패: {label}를 못 잡음")
        except AssertionError as e: print(f"✅ 단위 시험: {label} → 멈춤 ({str(e)[:70]})")
    PRE = PRECHECK.replace("assert m_on.mean() > 0.9", "assert m_on.mean() > 0.0")   # 무작위 작은 모델은 letter_mass가 낮음
else:
    # ── 가짜 모델 + 시나리오별 확률 ──
    ids, y, P0 = FX["val_ids"], np.array(FX["y"]), np.array(FX["P0"])
    W0, S, safe_R, safe_W = FX["W0"], FX["S"], FX["safe_R"], FX["safe_W"]
    R = np.random.RandomState(7)
    def onehotish(g_, conf=0.9):
        p = np.full(4, (1 - conf) / 3); p[g_] = conf; return p
    def wrong(g_):
        p = np.full(4, 0.05); p[(g_ + 2) % 4] = 0.85; return p / p.sum()
    sharpen = {"nll_up": 0.6}.get(scen, 1.3)
    P1 = P0 ** sharpen; P1 /= P1.sum(1, keepdims=True)
    fixes, breaks = {"pass": (safe_W[:6], safe_R[:2]), "strong": (safe_W[:10], safe_R[:1]), "border": (safe_W[:4], safe_R[:2]),
                     "fail": (safe_W[:2], safe_R[:5]), "nll_up": (safe_W[:6], safe_R[:2]), "ens_only": (S, safe_R[:3]),
                     "repro_fail": (S, safe_R[:3]), "resume": (safe_W[:6], safe_R[:2])}.get(scen, (safe_W[:6], safe_R[:2]))
    for i in fixes: P1[i] = onehotish(y[i])
    for i in breaks: P1[i] = wrong(y[i])
    BF = {k: v for k, v in zip(ids, P1)}
    PT0 = np.array(FX["PT0"]); PT1 = PT0.copy()
    for i in range(30): PT1[i] = np.roll(PT1[i], 1)
    BF.update({k: v for k, v in zip(FX["test_ids"], PT1)})
    OFF, CALLS = [False], []
    NT = 5 if scen == "tokmismatch" else 4
    class MockModel:
        training = False; device = "cpu"
        def eval(self): self.training = False; return self
        def train(self, mode=True): self.training = mode; return self
        @contextlib.contextmanager
        def disable_adapter(self):
            OFF[0] = True
            try: yield
            finally: OFF[0] = False
    g.update({"model": MockModel(), "FAMILY": "qwen", "IMAGE_TOKEN_ID": 290, "TEMPLATE_KW": {}, "USE_SYSTEM_ROLE": True, "DEVICE": "cpu",
              "DTYPE": torch.bfloat16, "GPU_NAME": "MOCK", "VRAM_GB": 96.0, "BS": 8, "LETTER_IDS": [[1], [2], [3], [4]], "processor": None})
    PRE = PRECHECK

run(SCORING, "8 채점 함수")
if mode == "mock":
    def mock_score_rows(rows, perms=None):
        CALLS.append(len(rows))
        if OFF[0] and scen != "noop": P = np.full((len(rows), 4), 0.25)
        else: P = np.stack([BF[r["id"]] for r in rows])
        return P, np.full(len(rows), 0.999), [NT] * len(rows)
    g["score_rows"] = mock_score_rows
    if scen == "resume":   # 앞 300문항이 이미 채점된 상태에서 끊겼다고 가정
        import pandas as pd
        rd = g["RUN_DIR"]; P = P1[:300]
        pd.DataFrame({"id": ids[:300], "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3],
                      "letter_mass": 0.999, "img_tokens": 4}).to_csv(f"{rd}/preds_final_ep1__up2.0__1000.csv", index=False)

run(PRE, "9 사전 점검")
run(VALEVAL, "10 검증 채점")
run(COMPARE, "11 비교·판정")
run(TEST, "12 test")
run(SUMMARY, "13 요약")
if mode == "mock":
    print(f"\n[시험 도구] score_rows 호출 문항 수 합계: {sum(CALLS)}")
print("\n[시험 도구] RUN_DIR 파일:", sorted(os.listdir(g["RUN_DIR"])))
