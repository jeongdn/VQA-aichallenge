# -*- coding: utf-8 -*-
"""e5_hires_gated.ipynb 코드 셀을 순서대로 실행하는 시험 도구 (E9 시험 데이터 재사용).
tiny: 작은 Qwen3.5 + 실제 PEFT 어댑터로 로드·해상도별 채점·이어하기·캐시 삭제까지 실제 실행
mock: 해상도별로 설계한 확률 → 판정·후보 파일을 독립 계산과 대조 (oom4: up4.0에서 메모리 부족 흉내)
사용: python3 run_e5.py <mode> <scenario> [key=value ...]"""
import sys, os, json, types, contextlib, shutil
import numpy as np, pandas as pd
NB = "/home/user/vqa/e5/out/e5_hires_gated.ipynb"
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/e9test"
mode, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA, IMPORTS, SPLIT, PROMPT, GATE, SCORING, LOADER, RUN, EVAL, SUMM = cells
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
shutil.rmtree(f"{ROOT}/vqa_runs/e5_hires", ignore_errors=True)
run(CFG, "1 설정")
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": f"{ROOT}/vqa_runs", "DRIVE_ZIP": "/nonexistent.zip"})
for k, v in over.items(): g[k] = eval(v, g)
run(DATA, "3 Drive·멤버·데이터"); run(IMPORTS, "4 import"); run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트"); run(GATE, "7 애매한 문항"); run(SCORING, "8 채점 함수")
import torch
sys.path.insert(0, "/home/user/vqa/e1")
from tiny_qwen import FakeQwenProcessor
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda i=0: "FAKE-RTX-PRO-6000"
torch.cuda.get_device_capability = lambda i=0: (12, 0)
torch.cuda.get_device_properties = lambda i=0: types.SimpleNamespace(total_memory=int(96 * 2**30))
torch.cuda.memory_allocated = lambda *a, **k: 0
torch.cuda.empty_cache = lambda: None
CALLS = {}
HF = f"{SP}/e5_hfcache"; shutil.rmtree(HF, ignore_errors=True)
for mid in ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B"]: os.makedirs(f"{HF}/models--{mid.replace('/', '--')}/snapshots")
import huggingface_hub.constants as _hc; _hc.HF_HUB_CACHE = HF
T = np.load(f"{ROOT}/truth.npz")
if mode == "tiny":
    import transformers, peft.peft_model as _ppm
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    PROC = FakeQwenProcessor(); LOADED = []
    transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
    def _fp(mid, **kw):
        assert kw.get("device_map") == {"": 0} and kw.get("dtype") == torch.bfloat16 and "quantization_config" not in kw, kw
        LOADED.append(mid); kw.pop("device_map"); return Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", **kw)
    transformers.AutoModelForImageTextToText.from_pretrained = _fp
    _ppm.infer_device = lambda: "cpu"
    run(LOADER, "9 로드 함수"); g["DEVICE"] = "cpu"
    _orig = g["score_rows"]
    def counting(rows, perms=None):
        CALLS[g["RES_PRESET"]] = CALLS.get(g["RES_PRESET"], 0) + len(rows); return _orig(rows, perms)
    g["score_rows"] = counting
    RUNX = RUN.replace("assert m_on.mean() > 0.9", "assert m_on.mean() > 0.0").replace("if n == 0 and preset != PRESET:", "if False:")   # 가짜 프로세서는 격자 고정
else:
    VI = {i: n for n, i in enumerate(g["VAL_IDS"])}; TI = {i: n for n, i in enumerate(g["TEST_IDS"])}
    OFF, CUR = [False], [None]
    class MockModel:
        training = False; device = "cpu"
        def eval(self): return self
        @contextlib.contextmanager
        def disable_adapter(self):
            OFF[0] = True
            try: yield
            finally: OFF[0] = False
    run(LOADER, "9 로드 함수")
    def fake_load(run_):
        CUR[0] = run_; g["model"] = MockModel(); g["MODEL_ID"] = g["MEMBER_INFO"][run_]["model_id"]; g["FAMILY"] = "qwen"; print(f"  (가짜 로드) {run_}")
    g["load_member"] = fake_load
    def designed(run_, rid, preset, off):
        sv = rid in VI; P = (T[f"PV_{run_}"][VI[rid]] if sv else T[f"PT_{run_}"][TI[rid]]).copy()
        if off: return np.full(4, 0.25)
        if preset == "up2.0":
            if scen == "health" and run_ == "q38_27b_r16_native" and (sum(map(ord, rid)) % 3 == 0): return np.roll(P, 1)
            return P
        h = (sum(map(ord, rid)) * 13 + sum(map(ord, run_)) + (7 if preset == "up4.0" else 0)) % 23
        s2 = np.argsort(P)[-2]; d = np.zeros(4); amt = 0.08 * ((h % 7) - 3) * (1.5 if preset == "up4.0" else 1.0)
        d[s2] = amt; d[int(P.argmax())] = -amt
        q = np.clip(P + d, 1e-6, None); return q / q.sum()
    def mock_score_rows(rows, perms=None):
        pr = g["RES_PRESET"]; CALLS[pr] = CALLS.get(pr, 0) + len(rows)
        if scen == "oom4" and pr == "up4.0" and not OFF[0]:
            raise RuntimeError("배치 1에서도 GPU 메모리 부족")
        tok = 4 if (pr == "up2.0" or scen == "nores") else {"up3.0": 9, "up4.0": 16}[pr]        # 기존 파일 img_tokens=4 (up2.0)
        P = np.stack([designed(CUR[0], r["id"], pr, OFF[0]) for r in rows]); return P, np.full(len(rows), 0.996), [tok] * len(rows)
    g["score_rows"] = mock_score_rows
    RUNX = RUN
    if scen == "resume":                                    # q35의 검증 up3.0 절반이 이미 채점된 상태
        r0 = "q35_27b_r16_native"; rows = g["VAL_G"].to_dict("records")[: len(g["VAL_G"]) // 2]; recs = []
        for r in rows:
            p = designed(r0, r["id"], "up3.0", False); recs.append({"id": r["id"], "preset": "up3.0", "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": 0.996, "img_tokens": 4})
        os.makedirs(f"{ROOT}/vqa_runs/e5_hires/{r0}", exist_ok=True); pd.DataFrame(recs).to_csv(f"{ROOT}/vqa_runs/e5_hires/{r0}/res_val.csv", index=False)
        PRE_DONE = len(recs)
if scen == "nores":                                         # 프로세서가 해상도를 무시 → 첫 배치에서 멈추고 아무것도 저장하지 않아야 함
    try: run(RUNX, "10 채점"); print("시험 도구 판정: 실패 (멈추지 않음)"); sys.exit()
    except RuntimeError as e: EMSG = str(e); print("멈춤:", EMSG)
    rp = f"{ROOT}/vqa_runs/e5_hires/q35_27b_r16_native/res_val.csv"; d = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame(columns=["preset"])
    ok = "적용되지 않은" in EMSG and set(d.preset) <= {"up2.0"} and CALLS.get("up3.0") == g["HR_BS"]["up3.0"]
    print("저장된 해상도:", sorted(set(d.preset)), "| up3.0 채점 건수:", CALLS.get("up3.0")); print("시험 도구 판정:", "통과" if ok else "실패"); sys.exit()
run(RUNX, "10 채점")
if scen == "rerun":                                         # 같은 세션에서 10번 셀을 다시 실행 → 아무것도 다시 채점하지 않아야 함
    before = sum(CALLS.values()); run(RUNX, "10 채점 (다시 실행)"); print("다시 실행 추가 채점 건수:", sum(CALLS.values()) - before)
run(EVAL, "11 판정"); run(SUMM, "12 요약")

print("\n━━━━ [시험 도구 확인] ━━━━")
nv, nt = len(g["VAL_G"]), len(g["TEST_G"]); M = len(g["MEMBERS_HR"]); HR = g["HR_PRESETS"]
print("해상도별 채점 건수:", CALLS)
ok = True
if mode == "mock" and scen in ("base", "resume", "oom4", "rerun"):
    exp = {"up2.0": M * (nv + 8)}                           # 검증 재현 + 사전 점검 4문항 × (켬, 끔)
    for pr in HR: exp[pr] = M * (nv + (nt if g["RUN_TEST"] else 0))
    if scen == "resume": exp["up3.0"] -= PRE_DONE
    if scen == "oom4": exp["up4.0"] = M * 1 * g["HR_BS"]["up4.0"]   # 첫 배치에서 실패 → 멤버마다 1번 시도
    print("기대 건수:", exp); ok &= CALLS == exp
    # 독립 계산: 해상도 평균 → 레시피 → 후보 파일
    y = T["y"]; gv = [VI[i] for i in g["VAL_G"].id]; gtt = [TI[i] for i in g["TEST_G"].id]
    use = [p for p in HR if not (scen == "oom4" and p == "up4.0")]
    Mm = {}
    for r in ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]:
        Pv, Pt = T[f"PV_{r}"].copy(), T[f"PT_{r}"].copy()
        if r in g["MEMBERS_HR"]:
            for n, rid in zip(gv, g["VAL_G"].id): Pv[n] = np.mean([T[f"PV_{r}"][n]] + [designed(r, rid, p, False) for p in use], 0)
            for n, rid in zip(gtt, g["TEST_G"].id): Pt[n] = np.mean([T[f"PT_{r}"][n]] + [designed(r, rid, p, False) for p in use], 0)
        Mm[r] = (Pv, Pt)
    w = np.array([1, 1, 1, 1, 2], float) / 6; C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
    a1 = sum(wi * Mm[r][1] for wi, r in zip(w, C4)).argmax(1); a0 = sum(wi * T[f"PT_{r}"] for wi, r in zip(w, C4)).argmax(1)
    av1 = sum(wi * Mm[r][0] for wi, r in zip(w, C4)).argmax(1); av0 = sum(wi * T[f"PV_{r}"] for wi, r in zip(w, C4)).argmax(1)
    c0, c1 = av0 == y, av1 == y
    print(f"독립 계산 clean4 검증: 고침 {int((~c0 & c1).sum())} / 새로 틀림 {int((c0 & ~c1).sum())} | test 바뀐 답 {int((a0 != a1).sum())}")
    line = next(l for l in g["V"]["lines"] if l.strip().startswith("clean4_qg_hcx: 검증"))
    ok &= f"고침 {int((~c0 & c1).sum())} | 새로 틀림 {int((c0 & ~c1).sum())}" in line
    fp = f"{ROOT}/vqa_runs/e5_hires/submission_clean4_qg_hcx__e5_hires.csv"
    if os.path.exists(fp):
        fa = pd.read_csv(f"{ROOT}/vqa_runs/ensembles/submission_clean4_qg_hcx.csv").answer.map("abcd".index).to_numpy()
        f1 = pd.read_csv(fp).answer.map("abcd".index).to_numpy(); e1 = np.where(a1 != a0, a1, fa)
        print(f"clean4 후보 파일 = 독립 계산 {int((f1 == e1).sum())}/{len(f1)}"); ok &= int((f1 == e1).sum()) == len(f1)
    else:
        ok &= any(l.startswith("판정: 불채택") for l in g["V"]["lines"])
    ok &= os.path.exists(f"{ROOT}/vqa_runs/e5_hires/e5_changed_val.html")
if scen == "rerun": ok &= True
if mode == "tiny":
    print("로드된 베이스:", LOADED); left = sorted(os.listdir(HF)); print("남은 HF 캐시:", left); ok &= left == []
verdict = next((l for l in g["V"]["lines"] if l.startswith("판정")), "(판정 줄 없음)")
print("판정 줄:", verdict)
want = {"health": "판정: 확인 필요"}.get(scen)
if want: ok &= verdict.startswith(want)
print("시험 도구 판정:", "통과" if ok else "실패")
