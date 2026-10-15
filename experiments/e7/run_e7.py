# -*- coding: utf-8 -*-
"""e7_yesno_verify.ipynb의 코드 셀을 순서대로 실행하는 시험 도구.
tiny: 모델 셀까지 실제 실행 (작은 Qwen3.5 + 가짜 프로세서, CUDA 함수만 가짜)
mock: 모델 대신 시나리오별 Yes/No 점수 → 판정·부정형 제외·이어하기·후보 파일 확인
사용: python3 run_e7.py <mode> <scenario> [key=value ...]"""
import sys, os, json, types, shutil
import numpy as np

NB = "/home/user/vqa/e7/e7_yesno_verify.ipynb"
E1 = "/tmp/vqa_scratch/e1test"
ROOT = "/tmp/vqa_scratch/e7test"
mode, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
FX = json.load(open(f"{ROOT}/fixture.json"))
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA, IMPORTS, SPLIT, PROMPT, BASE, MODEL, SCORING, PRECHECK, VALRUN, EVAL, TEST, SUMMARY = cells

g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)

run(CFG, "1 설정")
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": f"{ROOT}/vqa_runs", "DRIVE_ZIP": "/nonexistent.zip"})
if mode == "tiny": g["MODEL_ID"] = over.pop("MODEL_ID", "'Qwen/Qwen3.5-27B'").strip("'")
for k, v in over.items():
    if not k.startswith("_"): g[k] = eval(v, g)
g["RUN_NAME"] = "e7_yn_" + g["re"].sub(r"[^a-z0-9]+", "_", g["MODEL_ID"].split("/")[-1].lower()).strip("_")
rd = f"{ROOT}/vqa_runs/{g['RUN_NAME']}"
if scen == "extra":                               # E7-b가 남긴 '점수 없는 문항' 목록 → 그 문항도 채점
    import pandas as pd
    _tg = set(FX["TG"])
    xs = pd.DataFrame([{"split": "val", "base": "x", "id": FX["val_ids"][i]} for i in FX["sets"]["W_conf"][:2]]
                      + [{"split": "val", "base": "x", "id": FX["val_ids"][FX["sets"]["W_small"][0]]}]        # 이미 채점 대상 → 빠져야 함
                      + [{"split": "test", "base": "x", "id": FX["test_ids"][i]} for i in range(6) if i not in _tg][:5])
    os.makedirs(f"{ROOT}/vqa_runs/_extra", exist_ok=True); xs.to_csv(f"{ROOT}/vqa_runs/_extra/missing.csv", index=False)
    g["EXTRA_IDS_CSV"] = "_extra/missing.csv"; EXTRA_X = xs
if scen == "prefill_mismatch":
    os.makedirs(rd, exist_ok=True); json.dump({"prefill": "X"}, open(f"{rd}/e7_info.json", "w"))
run(DATA, "3 Drive·데이터"); run(IMPORTS, "4 import"); run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트"); run(BASE, "7 기준 앙상블")

import torch
sys.path.insert(0, "/home/user/vqa/e1")
from tiny_qwen import FakeQwenProcessor
if mode == "tiny":
    import transformers
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    PROC = FakeQwenProcessor()
    transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
    SEEN = {}
    def _fp(mid, **kw):
        assert mid == g["MODEL_ID"] and kw.get("device_map") == {"": 0} and kw.get("dtype") == torch.bfloat16, kw
        SEEN["quant"] = kw.pop("quantization_config", None) is not None; kw.pop("device_map")
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
    try: run(MODEL, "8 모델 로드")
    finally: torch.cuda.is_available = real_avail
    print("[시험 도구] from_pretrained에 4bit 설정 전달:", SEEN.get("quant"))
    PRE = PRECHECK.replace("assert _mass > 0.5", "assert _mass > 0.0")       # 무작위 작은 모델은 Yes/No 확률이 낮음
    EV = EVAL.replace("if _m >= 0.8 else", "if _m >= 0.0 else")
else:
    class M:
        device = "cpu"
    g.update({"model": M(), "processor": FakeQwenProcessor(), "FAMILY": "qwen", "IMAGE_TOKEN_ID": 290, "TEMPLATE_KW": {}, "USE_SYSTEM_ROLE": True,
              "DEVICE": "cpu", "DTYPE": torch.bfloat16, "GPU_NAME": "MOCK", "VRAM_GB": 96.0, "BS": 8, "use_4bit": False,
              "YES_IDS": [1], "NO_IDS": [2], "tok": types.SimpleNamespace(decode=lambda ids: "Yes" if ids == [1] else "No")})
    PRE, EV = PRECHECK, EVAL

run(SCORING, "9 채점 함수")
if mode == "mock":
    y, yt = np.array(FX["y"]), np.array(FX["yt"]); vid = {i: n for n, i in enumerate(FX["val_ids"])}; tid = {i: n for n, i in enumerate(FX["test_ids"])}
    SET = FX["sets"]; base_v = g["A_VAL"]; base_t = g["A_TEST"]
    strong = {}                                   # (split, index) → 검증 모델이 강하게 고르는 보기
    fixes, breaks = {"pass": (5, 1), "border": (3, 1), "fail": (1, 2), "neg": (5, 1), "lowmass_warn": (5, 1), "lowmass_stop": (5, 1),
                     "resume": (5, 1), "prefill_mismatch": (5, 1), "noexclude": (5, 1), "extra": (3, 1)}.get(scen, (5, 1))
    for i in SET["W_small"][:fixes]: strong[("v", i)] = y[i]
    for i in SET["R_small"][:breaks]: strong[("v", i)] = (y[i] + 1) % 4     # 2위 보기로 끌어당김
    if scen in ("neg", "noexclude"): strong[("v", SET["NEG_W"][0])] = y[SET["NEG_W"][0]]
    for i in FX["TG"][:40]: strong[("t", i)] = yt[i]
    MASS = {"lowmass_warn": 0.7, "lowmass_stop": 0.3}.get(scen, 0.98)
    CALLS = []
    def mock_yn(pairs):
        CALLS.append(len(pairs)); s = []
        for r, k in pairs:
            key = ("v", vid[r["id"]]) if r["id"] in vid else ("t", tid[r["id"]])
            base = base_v[key[1]] if key[0] == "v" else base_t[key[1]]
            tgt = strong.get(key)
            s.append((6.0 if k == tgt else -6.0) if tgt is not None else (1.0 if k == base else -1.0))
        s = np.array(s); sig = 1 / (1 + np.exp(-s))
        return s, MASS * sig, MASS * (1 - sig), [4] * len(pairs)
    g["yn_scores"] = mock_yn
    if scen == "resume":                          # 앞 50건이 채점된 뒤 끊겼다고 가정
        import pandas as pd
        recs = []
        for r in g["VAL_GATED"].to_dict("records"):
            for k in range(4):
                if len(recs) >= 50: break
                s_, py, pn, _ = mock_yn([(r, k)]); CALLS.pop()
                recs.append({"id": r["id"], "opt": k, "s": s_[0], "p_yes": py[0], "p_no": pn[0], "yn_mass": py[0] + pn[0], "img_tokens": 4})
        os.makedirs(g["RUN_DIR"], exist_ok=True); pd.DataFrame(recs).to_csv(os.path.join(g["RUN_DIR"], "yn_val__up2.0.csv"), index=False)

run(PRE, "10 점검"); run(VALRUN, "11 검증 채점"); run(EV, "12 판정"); run(TEST, "13 test"); run(SUMMARY, "14 요약")
if mode == "mock": print(f"\n[시험 도구] yn_scores 호출 건수 합계: {sum(CALLS)}")
print("[시험 도구] RUN_DIR 파일:", sorted(os.listdir(g["RUN_DIR"])))
if scen == "extra":
    import pandas as pd
    _yv = set(pd.read_csv(os.path.join(g["RUN_DIR"], "yn_val__up2.0.csv")).id); _yt = set(pd.read_csv(os.path.join(g["RUN_DIR"], "yn_test__up2.0.csv")).id)
    _ev = set(EXTRA_X.id[EXTRA_X.split == "val"]); _et = set(EXTRA_X.id[EXTRA_X.split == "test"])
    print(f"[시험 도구] 추가 문항 채점됨: 검증 {len(_ev & _yv)}/{len(_ev)} · test {len(_et & _yt)}/{len(_et)}")
    shutil.rmtree(f"{ROOT}/vqa_runs/_extra")
if "cands" in g["V"]:
    import pandas as pd
    for n, d in g["V"]["cands"]:
        c = pd.read_csv(os.path.join(g["RUN_DIR"], n)).answer.map("abcd".index).to_numpy()
        ref = np.array(FX["vote"]) if n.startswith("submission_majority_vote") else np.array(FX["clean"])
        print(f"[시험 도구] 독립 계산: {n} 기존과 다른 답 {int((c != ref).sum())}")
