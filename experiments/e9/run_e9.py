# -*- coding: utf-8 -*-
"""e9_tta_gated.ipynb 코드 셀을 순서대로 실행하는 시험 도구.
tiny: 작은 Qwen3.5 + 실제 PEFT 어댑터로 로드·채점·이어하기·캐시 삭제까지 실제 실행 (CUDA 함수만 가짜)
mock: 모델 대신 설계한 확률 → 판정·후보 파일을 독립 계산과 대조
사용: python3 run_e9.py <mode> <scenario> [key=value ...]"""
import sys, os, json, types, contextlib, shutil, re
import numpy as np, pandas as pd
NB = "/home/user/vqa/e9/out/e9_tta_gated.ipynb"
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/e9test"
mode, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA, IMPORTS, SPLIT, PROMPT, GATE, SCORING, LOADER, RUN, EVAL, SUMM = cells
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
shutil.rmtree(f"{ROOT}/vqa_runs/e9_tta", ignore_errors=True)
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
CALLS = []
HF = f"{SP}/e9_hfcache"; shutil.rmtree(HF, ignore_errors=True)
for mid in ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"]: os.makedirs(f"{HF}/models--{mid.replace('/', '--')}/snapshots")
import huggingface_hub.constants as _hc; _hc.HF_HUB_CACHE = HF
if mode == "tiny":
    import transformers, peft.peft_model as _ppm
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    PROC = FakeQwenProcessor(); LOADED = []
    transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
    def _fp(mid, **kw):
        assert kw.get("device_map") == {"": 0} and kw.get("dtype") == torch.bfloat16 and "quantization_config" not in kw, kw
        LOADED.append(mid); kw.pop("device_map")
        return Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", **kw)
    transformers.AutoModelForImageTextToText.from_pretrained = _fp
    _ppm.infer_device = lambda: "cpu"
    run(LOADER, "9 로드 함수")
    g["DEVICE"] = "cpu"                                   # 시험 도구 한정: 실제 계산은 CPU (autocast 끔)
    _orig = g["score_rows"]
    def counting(rows, perms=None): CALLS.append(len(rows)); return _orig(rows, perms)
    g["score_rows"] = counting
    RUNX = RUN.replace("assert m_on.mean() > 0.9", "assert m_on.mean() > 0.0")        # 무작위 작은 모델은 letter_mass가 낮음
else:
    T = np.load(f"{ROOT}/truth.npz")
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
        CUR[0] = run_; g["model"] = MockModel(); g["MODEL_ID"] = g["MEMBER_INFO"][run_]["model_id"]; g["FAMILY"] = "qwen"
        print(f"  (가짜 로드) {run_}")
    g["load_member"] = fake_load
    def designed(run_, rid, perm_k, off):
        # 원래 확률(파일) 기준으로 순서별·LoRA 켬/끔별 확률을 결정적으로 설계
        sv = rid in VI; P = (T[f"PV_{run_}"][VI[rid]] if sv else T[f"PT_{run_}"][TI[rid]]).copy()
        h = (sum(map(ord, rid)) * 7 + sum(map(ord, run_))) % 97
        if off:                                                    # 원본 모델: 절반 문항에서 2위 보기를 확신 있게 (순서 무관하게 일관)
            q = np.full(4, 0.05); alt = np.argsort(P)[-2] if h % 2 == 0 else int(P.argmax()); q[alt] = 0.85; return q / q.sum()
        if perm_k == 0:
            if scen == "health" and run_ == "q38_27b_r16_native" and h % 3 == 0: return np.roll(P, 1)
            return P
        s2 = np.argsort(P)[-2]; d = np.zeros(4); d[s2] = 0.12 * ((h % 5) - 1.5); d[int(P.argmax())] = -d[s2]
        q = np.clip(P + d * (1 + perm_k) / 3, 1e-6, None); return q / q.sum()
    def mock_score_rows(rows, perms=None):
        CALLS.append(len(rows)); perms = perms or [(0, 1, 2, 3)] * len(rows); k = g["TTA_PERMS"].index(tuple(perms[0]))
        P = np.stack([designed(CUR[0], r["id"], k, OFF[0]) for r in rows]); return P, np.full(len(rows), 0.997), [4] * len(rows)
    g["score_rows"] = mock_score_rows
    RUNX = RUN
    if scen == "resume":                                           # q35의 검증·LoRA켬 절반이 이미 채점된 상태
        r0 = "q35_27b_r16_native"; rows = g["VAL_G"].to_dict("records"); recs = []
        CUR[0] = r0
        for k in range(4):
            for r in rows[: len(rows) // 2]:
                p = designed(r0, r["id"], k, False); recs.append({"id": r["id"], "lora": 1, "perm": k, "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": 0.997, "img_tokens": 4})
        os.makedirs(f"{ROOT}/vqa_runs/e9_tta/{r0}", exist_ok=True); pd.DataFrame(recs).to_csv(f"{ROOT}/vqa_runs/e9_tta/{r0}/perm_val__up2.0.csv", index=False)
        PRE_DONE = len(recs)
run(RUNX, "10 채점")
if scen == "incomplete_off":                                       # LoRA를 끈 채점만 일부 빠진 상태 → H1은 진행, H2는 보류
    p = f"{ROOT}/vqa_runs/e9_tta/gemma4_31b_r16_up20/perm_test__up2.0.csv"; d = pd.read_csv(p); d.drop(d[d.lora == 0].index[-7:]).to_csv(p, index=False)
if scen == "incomplete":                                           # gemma4 test의 LoRA 켠 채점 일부가 빠진 상태 (중단된 것처럼)
    p = f"{ROOT}/vqa_runs/e9_tta/gemma4_31b_r16_up20/perm_test__up2.0.csv"; d = pd.read_csv(p)
    drop = d[d.lora == 1].index[-7:]; d.drop(drop).to_csv(p, index=False)
run(EVAL, "11 판정"); run(SUMM, "12 요약")

print("\n━━━━ [시험 도구 확인] ━━━━")
nG = len(g["VAL_G"]) + (len(g["TEST_G"]) if g["RUN_TEST"] else 0); M = len(g["MEMBERS_TTA"]); L = 2 if g["SCORE_BASE"] else 1
expect = M * nG * 4 * L + M * 8 - (PRE_DONE if scen == "resume" else 0)
print(f"score_rows 채점 건수 {sum(CALLS)} (기대 {expect} = 멤버 {M} × 문항 {nG} × 순서 4 × {L} + 사전 점검 8×{M}" + (" − 이미 채점된 것" if scen == "resume" else "") + ")")
ok = sum(CALLS) == expect
if mode == "tiny":
    print("로드된 베이스:", g["LOADED"] if "LOADED" in g else LOADED)
    left = sorted(os.listdir(HF)); print("남은 HF 캐시:", left)
    used = {g["MEMBER_INFO"][r]["model_id"] for r in g["MEMBERS_TTA"]}
    ok &= all(("models--" + m.replace("/", "--")) not in left for m in used) if g["DELETE_HF_CACHE"] else True
if mode == "mock" and scen in ("base", "resume", "nobase"):
    # ── 독립 계산: 노트북과 같은 규칙을 따로 구현해 숫자·파일 대조 ──
    y = T["y"]; VIDS, TIDS = g["VAL_IDS"], g["TEST_IDS"]; gv = [VI[i] for i in g["VAL_G"].id]; gt = [TI[i] for i in g["TEST_G"].id]
    T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
    def temp(p, t): q = np.clip(p, 1e-12, 1) ** (1 / t); return q / q.sum(1, keepdims=True)
    def fitT(p): i = np.arange(len(y)); return T70[int(np.argmin([-np.mean(np.log(temp(p, t)[i, y])) for t in T70]))]
    def members_after(h2):
        out = {}
        for r in ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]:
            Pv, Pt = T[f"PV_{r}"].copy(), T[f"PT_{r}"].copy()
            if r in g["MEMBERS_TTA"]:
                for P, ids, gi in ((Pv, g["VAL_G"].id, gv), (Pt, g["TEST_G"].id, gt)):
                    for rid, n in zip(ids, gi):
                        ft = np.mean([designed(r, rid, k, False) for k in range(4)], 0)
                        if h2:
                            O = np.stack([designed(r, rid, k, True) for k in range(4)]); pb = O.mean(0); bt = pb.argmax(); f_ = ft.argmax(); s = np.sort(ft)
                            if bt != f_ and s[-1] - s[-2] < 0.15 and pb[bt] >= 0.6 and (O.argmax(1) == bt).mean() >= 0.75 and ft[f_] - ft[bt] <= 0.12:
                                ft = 0.5 * ft + 0.5 * pb
                        P[n] = ft
            out[r] = (Pv, Pt)
        return out
    def ensA(Mm, runs, w, cal, split):
        w = np.asarray(w, float) / sum(w); return sum(wi * ((temp(Mm[r][split], fitT(Mm[r][0])) if cal else Mm[r][split])) for wi, r in zip(w, runs)).argmax(1)
    def all_ans(Mm):
        res = {}
        for sp in (0, 1):
            c = ensA(Mm, ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"], [1, 1, 1, 1, 2], False, sp)
            a = ensA(Mm, ["q38_27b_r16_native", "gemma4_31b_r16_up20"], [1, 1], False, sp); b = ensA(Mm, ["q38_27b_r16_native", "gemma4_31b_r16_up20_dev"], [1, 1], True, sp)
            A_ = np.stack([a, c, b], 1); v = np.stack([(A_ == k).sum(1) for k in range(4)], 1).argmax(1); res[sp] = (c, v)
        return res
    base = all_ans(members_after(False) if False else {r: (T[f"PV_{r}"], T[f"PT_{r}"]) for r in ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]})
    h1 = all_ans(members_after(False))
    c0, c1 = base[0][0] == y, h1[0][0] == y
    print(f"독립 계산 H1 clean4 검증: 고침 {int((~c0 & c1).sum())} / 새로 틀림 {int((c0 & ~c1).sum())} / 바뀐 답 {int((base[0][0] != h1[0][0]).sum())} | test 바뀐 답 {int((base[1][0] != h1[1][0]).sum())}")
    line = next(l for l in g["V"]["lines"] if l.strip().startswith("clean4_qg_hcx: 검증"))
    ok &= f"고침 {int((~c0 & c1).sum())} | 새로 틀림 {int((c0 & ~c1).sum())}" in line
    fa = pd.read_csv(f"{ROOT}/vqa_runs/ensembles/submission_clean4_qg_hcx.csv").answer.map("abcd".index).to_numpy()
    f1 = pd.read_csv(f"{ROOT}/vqa_runs/e9_tta/submission_clean4_qg_hcx__e9_tta4.csv").answer.map("abcd".index).to_numpy()
    exp1 = np.where(h1[1][0] != base[1][0], h1[1][0], fa); same = int((f1 == exp1).sum()); ok &= same == len(f1)
    print(f"clean4 H1 후보 파일 = 독립 계산 {same}/{len(f1)}")
    fv = pd.read_csv(f"{ROOT}/vqa_runs/ensembles/submission_majority_vote.csv").answer.map("abcd".index).to_numpy()
    f1v = pd.read_csv(f"{ROOT}/vqa_runs/e9_tta/submission_majority_vote__e9_tta4.csv").answer.map("abcd".index).to_numpy()
    expv = np.where(h1[1][1] != base[1][1], h1[1][1], fv); samev = int((f1v == expv).sum()); ok &= samev == len(f1v)
    print(f"다수결 H1 후보 파일 = 독립 계산 {samev}/{len(f1v)}")
    if g["SCORE_BASE"]:
        h2 = all_ans(members_after(True)); d2 = int((h2[1][0] != h1[1][0]).sum())
        p2 = f"{ROOT}/vqa_runs/e9_tta/submission_clean4_qg_hcx__e9_tta4_baseoff.csv"
        print(f"독립 계산 H2: test에서 H1과 다른 clean4 답 {d2} | H2 후보 파일 {'있음' if os.path.exists(p2) else '없음'}")
        if d2 >= 5:
            f2 = pd.read_csv(p2).answer.map("abcd".index).to_numpy(); exp2 = np.where(h2[1][0] != base[1][0], h2[1][0], fa)
            ok &= int((f2 == exp2).sum()) == len(f2); print(f"clean4 H2 후보 파일 = 독립 계산 {int((f2 == exp2).sum())}/{len(f2)}")
        else: ok &= not os.path.exists(p2)
verdict = next((l for l in g["V"]["lines"] if l.startswith("판정")), "(판정 줄 없음 — 후보 생성)")
print("판정 줄:", verdict)
want = {"health": "판정: 확인 필요", "incomplete": "판정: 대기", "incomplete_off": "판정: 공개 확인"}.get(scen)
if want: ok &= verdict.startswith(want)
if scen == "notest": ok &= not any(f.startswith("submission_") for f in os.listdir(f"{ROOT}/vqa_runs/e9_tta"))
if scen == "incomplete_off": ok &= not any("baseoff" in f for f in os.listdir(f"{ROOT}/vqa_runs/e9_tta")) and any("보류" in l or "H2 후보는 만들지 않음" in l for l in g["V"]["lines"])
print("시험 도구 판정:", "통과" if ok else "실패")
