# -*- coding: utf-8 -*-
"""e9e5_hcx.ipynb 코드 셀을 순서대로 실행하는 시험 도구 (E9 시험 데이터 재사용).
준비: python3 run_hcx.py prep   → E9 시험 데이터 + HCX 어댑터 + E9·E5 mock base 결과(다른 멤버 파일)
mock: HCX 채점을 설계한 확률로 → TTA·고해상도 판정과 후보 파일을 독립 계산과 대조
tiny: 작은 Qwen3.5 + 가짜 HCX 프로세서(생각 블록·thinking 인자·해상도별 격자)로 로드·채점·이어하기를 실제 실행
사용: python3 run_hcx.py <mode> <scenario> [key=value ...]"""
import sys, os, json, types, contextlib, shutil, subprocess, glob
import numpy as np, pandas as pd
NB = "/home/user/vqa/hcx/out/e9e5_hcx.ipynb"
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/e9test"; RUNS = f"{ROOT}/vqa_runs"; HCXR = "hcx32b_r16_native"
OTH9 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev"]; OTH5 = ["q35_27b_r16_native", "q38_27b_r16_native"]
ALL = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", HCXR]

if sys.argv[1] == "prep":
    def sh(c): print("$", c); r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-600:], r.stderr[-300:]); assert r.returncode == 0
    sh("cd /home/user/vqa/e9 && python3 make_fixtures_e9.py")
    import torch
    sys.path.insert(0, "/home/user/vqa/e1")
    from tiny_qwen import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    b = Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", dtype=torch.float32); torch.manual_seed(77)
    pm = get_peft_model(b, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
                                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.5)
    ad = f"{RUNS}/{HCXR}/adapter_ep1"; pm.save_pretrained(ad)
    c = json.load(open(f"{ad}/adapter_config.json")); c["base_model_name_or_path"] = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"; json.dump(c, open(f"{ad}/adapter_config.json", "w"), indent=2)
    sh("cd /home/user/vqa/e9 && python3 run_e9.py mock base | tail -3")
    sh("cd /home/user/vqa/e5 && python3 run_e5.py mock base | tail -3")
    print("준비 끝"); sys.exit()

mode, scen = sys.argv[1], sys.argv[2]
over = dict(a.split("=", 1) for a in sys.argv[3:])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA, IMPORTS, SPLIT, PROMPT, GATE, SCORING, LOADER, RUN, EVAL9, EVAL5, SUMM = cells
for p in [f"{RUNS}/e9_tta/{HCXR}", f"{RUNS}/e5_hires/{HCXR}"]: shutil.rmtree(p, ignore_errors=True)
for p in glob.glob(f"{RUNS}/e9_tta/*_hcx.csv") + glob.glob(f"{RUNS}/e5_hires/*_hcx.csv") + glob.glob(f"{RUNS}/e5_hires/*_hcx.html"): os.remove(p)
if scen == "e9_missing":                                        # E9 노트북이 아직 안 끝난 것처럼: 다른 멤버 TTA 파일을 잠시 치움
    shutil.move(f"{RUNS}/e9_tta", f"{RUNS}/_e9_tta_hidden"); os.makedirs(f"{RUNS}/e9_tta")
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
try:
    run(CFG, "1 설정")
    g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip"})
    for k, v in over.items(): g[k] = eval(v, g)
    run(DATA, "3 Drive·HCX·데이터"); run(IMPORTS, "4 import"); run(SPLIT, "5 분할"); run(PROMPT, "6 프롬프트"); run(GATE, "7 애매한 문항"); run(SCORING, "8 채점 함수")
    import torch
    sys.path.insert(0, "/home/user/vqa/e1")
    from tiny_qwen import FakeQwenProcessor, IMG, VS, VE, PAD
    from transformers import BatchFeature
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda i=0: "FAKE-RTX-PRO-6000"
    torch.cuda.get_device_capability = lambda i=0: (12, 0)
    torch.cuda.get_device_properties = lambda i=0: types.SimpleNamespace(total_memory=int(96 * 2**30))
    torch.cuda.memory_allocated = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda: None
    CALLS, LOADS = {}, []
    T = np.load(f"{ROOT}/truth.npz"); VI = {i: n for n, i in enumerate(g["VAL_IDS"])}; TI = {i: n for n, i in enumerate(g["TEST_IDS"])}
    def P0(run_, rid): return (T[f"PV_{run_}"][VI[rid]] if rid in VI else T[f"PT_{run_}"][TI[rid]]).copy()
    def d_tta(run_, rid, k, off):                               # run_e9.py의 설계와 같음 (HCX도 같은 식)
        P = P0(run_, rid); h = (sum(map(ord, rid)) * 7 + sum(map(ord, run_))) % 97
        if off:
            q = np.full(4, 0.05); alt = np.argsort(P)[-2] if h % 2 == 0 else int(P.argmax()); q[alt] = 0.85; return q / q.sum()
        if k == 0:
            if scen == "health" and run_ == HCXR and h % 3 == 0: return np.roll(P, 1)
            return P
        s2 = np.argsort(P)[-2]; d = np.zeros(4); d[s2] = 0.12 * ((h % 5) - 1.5); d[int(P.argmax())] = -d[s2]
        q = np.clip(P + d * (1 + k) / 3, 1e-6, None); return q / q.sum()
    def d_res(run_, rid, preset):                               # run_e5.py의 설계와 같음
        P = P0(run_, rid)
        if preset == "up2.0": return P
        h = (sum(map(ord, rid)) * 13 + sum(map(ord, run_)) + (7 if preset == "up4.0" else 0)) % 23
        s2 = np.argsort(P)[-2]; d = np.zeros(4); amt = 0.08 * ((h % 7) - 3) * (1.5 if preset == "up4.0" else 1.0)
        d[s2] = amt; d[int(P.argmax())] = -amt; q = np.clip(P + d, 1e-6, None); return q / q.sum()
    if mode == "tiny":
        import transformers, peft.peft_model as _ppm
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        class HyperCLOVAXFakeProcessor(FakeQwenProcessor):
            chat_template = "{%- if thinking is defined and not thinking %}<think>\n{%- endif %}"      # 'thinking' 인자 (enable_thinking 아님)
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
                assert kw == {"thinking": False}, kw
                return super().apply_chat_template(messages, tokenize, add_generation_prompt) + "<think>\n"   # 생각 블록을 연 채 끝남
            def __call__(self, text, images, padding=True, return_tensors="pt", **kw):
                assert all(t.endswith("<think>\n\n</think>\n\n") for t in text), text[0][-40:]       # _fix_prompt가 닫았는지
                self.calls.append(kw); mx = kw.get("max_pixels", 3_400_000); gg = 4 if mx <= 3_400_000 else 6 if mx <= 7_600_000 else 8
                seqs, pvs = [], []
                for t, im in zip(text, images):
                    pre, post = t.split("<IMG>")
                    seqs.append([20 + (ord(ch) % 250) for ch in pre[-6:]] + [VS] + [IMG] * (gg * gg // 4) + [VE] + [20 + (ord(ch) % 250) for ch in post[-40:]])
                    a = np.asarray(im.resize((8, 8))).astype(np.float32) / 255.0; rng = np.random.RandomState(int(a.sum() * 1000) % (2**31))
                    pvs.append(torch.tensor(rng.randn(gg * gg, 3 * 2 * 16 * 16).astype(np.float32)))
                L = max(map(len, seqs))
                ids = torch.tensor([[PAD] * (L - len(s)) + s for s in seqs]); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
                return BatchFeature({"input_ids": ids, "attention_mask": att, "mm_token_type_ids": (ids == IMG).long(), "pixel_values": torch.cat(pvs),
                                     "image_grid_thw": torch.tensor([[1, gg, gg]] * len(text))})
        PROC = HyperCLOVAXFakeProcessor()
        transformers.AutoProcessor.from_pretrained = lambda mid, **kw: PROC
        def _fp(mid, **kw):
            assert kw.get("device_map") == {"": 0} and kw.get("dtype") == torch.bfloat16 and "quantization_config" not in kw, kw
            LOADS.append(mid); kw.pop("device_map"); return Qwen3_5ForConditionalGeneration.from_pretrained(f"{ROOT}/tiny_qwen35", **kw)
        transformers.AutoModelForImageTextToText.from_pretrained = _fp
        _ppm.infer_device = lambda: "cpu"
        run(LOADER, "9 로드 함수"); g["DEVICE"] = "cpu"
        _orig = g["score_rows"]
        def counting(rows, perms=None):
            CALLS[g["RES_PRESET"]] = CALLS.get(g["RES_PRESET"], 0) + len(rows); return _orig(rows, perms)
        g["score_rows"] = counting
        RUNX = RUN.replace("assert m_on.mean() > 0.9", "assert m_on.mean() > 0.0")
    else:
        OFF = [False]
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
            LOADS.append(run_); g["model"] = MockModel(); g["MODEL_ID"] = g["MEMBER_INFO"][run_]["model_id"]; g["FAMILY"] = "qwen"; print(f"  (가짜 로드) {run_}")
        g["load_member"] = fake_load
        def mock_score_rows(rows, perms=None):
            pr = g["RES_PRESET"]; CALLS[pr] = CALLS.get(pr, 0) + len(rows); perms = perms or [(0, 1, 2, 3)] * len(rows); k = g["TTA_PERMS"].index(tuple(perms[0]))
            if scen == "oom4" and pr == "up4.0": raise RuntimeError("배치 1에서도 GPU 메모리 부족")
            tok = 4 if (pr == "up2.0" or scen == "noapply") else {"up3.0": 9, "up4.0": 16}[pr]
            P = np.stack([d_tta(HCXR, r["id"], k, OFF[0]) if pr == "up2.0" else d_res(HCXR, r["id"], pr) for r in rows])
            return P, np.full(len(rows), 0.996), [tok] * len(rows)
        g["score_rows"] = mock_score_rows
        RUNX = RUN
        PRE_DONE = 0
        if scen == "resume":                                    # HCX 검증 TTA(LoRA 켬) 절반이 이미 채점된 상태
            rows = g["VAL_G"].to_dict("records"); recs = []
            for k in range(4):
                for r in rows[: len(rows) // 2]:
                    p = d_tta(HCXR, r["id"], k, False); recs.append({"id": r["id"], "lora": 1, "perm": k, "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": 0.996, "img_tokens": 4})
            os.makedirs(f"{RUNS}/e9_tta/{HCXR}", exist_ok=True); pd.DataFrame(recs).to_csv(f"{RUNS}/e9_tta/{HCXR}/perm_val__up2.0.csv", index=False); PRE_DONE = len(recs)
    run(RUNX, "10 채점")
    if scen == "rerun":
        before = sum(CALLS.values()); run(RUNX, "10 채점 (다시 실행)"); print("다시 실행 추가 채점:", sum(CALLS.values()) - before, "| 모델 로드 횟수:", len(LOADS))
    run(EVAL9, "11 TTA 판정"); run(EVAL5, "12 고해상도 판정"); run(SUMM, "13 요약")
finally:
    if scen == "e9_missing":
        shutil.rmtree(f"{RUNS}/e9_tta"); shutil.move(f"{RUNS}/_e9_tta_hidden", f"{RUNS}/e9_tta")

print("\n━━━━ [시험 도구 확인] ━━━━")
ok = True
nv, nt = len(g["VAL_G"]), len(g["TEST_G"]); DT, DH, SB = g["DO_TTA"], g["DO_HIRES"], g["SCORE_BASE"]; HR = g["HR_PRESETS"]; BSX = g["HR_BS"]
print("해상도별 채점 건수:", CALLS, "| 모델 로드:", LOADS)
if mode == "mock":
    RT = g["RUN_TEST"]; ntt = nt if RT else 0
    exp = {"up2.0": 8 + (nv + ntt) * 4 * (2 if SB else 1) * DT + nv * DH - (PRE_DONE if scen == "resume" else 0)}
    for pr in HR:
        if DH: exp[pr] = nv + ntt
    if scen == "noapply": exp.update({"up3.0": BSX["up3.0"], "up4.0": BSX["up4.0"]})
    if scen == "oom4": exp["up4.0"] = BSX["up4.0"]
    exp = {k: v for k, v in exp.items() if v}
    print("기대 건수:", exp); ok &= CALLS == exp
    ok &= len(LOADS) == 1
    y = T["y"]; gv = [VI[i] for i in g["VAL_G"].id]; gtt = [TI[i] for i in g["TEST_G"].id]
    T70 = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
    def temp(p, t): q = np.clip(p, 1e-12, 1) ** (1 / t); return q / q.sum(1, keepdims=True)
    def fitT(p): i = np.arange(len(y)); return T70[int(np.argmin([-np.mean(np.log(temp(p, t)[i, y])) for t in T70]))]
    def answers(Mm):
        out = {}
        for sp in (0, 1):
            w = np.array([1, 1, 1, 1, 2], float) / 6
            c = sum(wi * Mm[r][sp] for wi, r in zip(w, ALL)).argmax(1)
            a = (0.5 * Mm["q38_27b_r16_native"][sp] + 0.5 * Mm["gemma4_31b_r16_up20"][sp]).argmax(1)
            b = (0.5 * temp(Mm["q38_27b_r16_native"][sp], fitT(Mm["q38_27b_r16_native"][0])) + 0.5 * temp(Mm["gemma4_31b_r16_up20_dev"][sp], fitT(Mm["gemma4_31b_r16_up20_dev"][0]))).argmax(1)
            A_ = np.stack([a, c, b], 1); v = np.stack([(A_ == k).sum(1) for k in range(4)], 1).argmax(1); out[sp] = (c, v)
        return out
    ORIG = {r: (T[f"PV_{r}"].copy(), T[f"PT_{r}"].copy()) for r in ALL}
    base = answers(ORIG)
    def check_files(d, suffix, new, ref):
        good = True
        for j, nm in ((0, "clean4_qg_hcx"), (1, "majority_vote")):
            fp = f"{d}/submission_{nm}__{suffix}.csv"
            if not os.path.exists(fp): print("  없음:", os.path.basename(fp)); return False
            fa = pd.read_csv(f"{RUNS}/ensembles/submission_{nm}.csv").answer.map("abcd".index).to_numpy()
            f1 = pd.read_csv(fp).answer.map("abcd".index).to_numpy(); e1 = np.where(new[1][j] != ref[1][j], new[1][j], fa)
            same = int((f1 == e1).sum()); print(f"  {os.path.basename(fp)} = 독립 계산 {same}/{len(f1)}"); good &= same == len(f1)
        return good
    def paired_line(A_old, A_new, lines, prefix):
        c0, c1 = A_old == y, A_new == y; f, b = int((~c0 & c1).sum()), int((c0 & ~c1).sum())
        want = f"고침 {f} | 새로 틀림 {b}"; hit = any(l.strip().startswith(prefix) and want in l for l in lines)
        print(f"  {prefix} 독립 계산 {want} → 출력에 {'있음' if hit else '없음'}"); return hit
    # ── TTA (E9 4멤버 + HCX) ──
    if DT and scen != "e9_missing":
        def tta_members(use, h2=False):
            Mm = {r: (ORIG[r][0].copy(), ORIG[r][1].copy()) for r in ALL}
            for r in use:
                for P, ids, gi in ((Mm[r][0], g["VAL_G"].id, gv), (Mm[r][1], g["TEST_G"].id, gtt)):
                    for rid, n in zip(ids, gi):
                        ft = np.mean([d_tta(r, rid, k, False) for k in range(4)], 0)
                        if h2:
                            O = np.stack([d_tta(r, rid, k, True) for k in range(4)]); pb = O.mean(0); bt = pb.argmax(); f_ = ft.argmax(); s = np.sort(ft)
                            if bt != f_ and s[-1] - s[-2] < 0.15 and pb[bt] >= 0.6 and (O.argmax(1) == bt).mean() >= 0.75 and ft[f_] - ft[bt] <= 0.12: ft = 0.5 * ft + 0.5 * pb
                        P[n] = ft
            return Mm
        h1 = answers(tta_members(OTH9 + [HCXR])); h1x = answers(tta_members(OTH9))
        L9 = g["V9"]["lines"]; v9 = next((l for l in L9 if l.startswith("판정")), "")
        print("TTA 판정 줄:", v9)
        if scen == "health": ok &= v9.startswith("판정: 확인 필요")
        else:
            ok &= paired_line(base[0][0], h1[0][0], L9, "clean4_qg_hcx: 검증")
            ok &= paired_line(h1x[0][0], h1[0][0], L9, "(HCX 몫) clean4_qg_hcx")
            if not RT: ok &= v9.startswith("판정: 검증만 확인") and not glob.glob(f"{RUNS}/e9_tta/*_hcx.csv")
            elif v9.startswith("판정: 공개 확인"):
                ok &= check_files(f"{RUNS}/e9_tta", "e9_tta4_hcx", h1, base)
                if SB:
                    h2 = answers(tta_members(OTH9 + [HCXR], True)); d2 = int((h2[1][0] != h1[1][0]).sum()); p2 = f"{RUNS}/e9_tta/submission_clean4_qg_hcx__e9_tta4_baseoff_hcx.csv"
                    print(f"  독립 계산 H2: test에서 H1과 다른 clean4 답 {d2} | 파일 {'있음' if os.path.exists(p2) else '없음'}")
                    ok &= check_files(f"{RUNS}/e9_tta", "e9_tta4_baseoff_hcx", h2, base) if d2 >= 5 else not os.path.exists(p2)
                ok &= any("E9(HCX 없음) clean4 후보를 이미 채택했다면" in l for l in L9)
            else: ok &= v9.startswith("판정: 불채택")
    if DT and scen == "e9_missing":
        v9 = next((l for l in g["V9"]["lines"] if l.startswith("판정")), ""); print("TTA 판정 줄:", v9); ok &= v9.startswith("판정: 대기")
    # ── 고해상도 (E5 Q35·Q38 + HCX) ──
    if DH:
        L5 = g["V5"]["lines"]; v5 = next((l for l in L5 if l.startswith("판정")), ""); print("고해상도 판정 줄:", v5)
        if scen == "noapply": ok &= v5.startswith("판정: HCX 고해상도 없음") and not glob.glob(f"{RUNS}/e5_hires/*_hcx.csv")
        elif scen == "health": ok &= v5.startswith("판정: 확인 필요") and "hcx32b_r16_native: up2.0 재현" in v5
        else:
            def res_members(use_hcx):
                Mm = {r: (ORIG[r][0].copy(), ORIG[r][1].copy()) for r in ALL}
                for r in OTH5 + ([HCXR] if use_hcx else []):
                    prs = [p for p in HR if not (r == HCXR and scen == "oom4" and p == "up4.0")]
                    for n, rid in zip(gv, g["VAL_G"].id): Mm[r][0][n] = np.mean([ORIG[r][0][n]] + [d_res(r, rid, p) for p in prs], 0)
                    for n, rid in zip(gtt, g["TEST_G"].id): Mm[r][1][n] = np.mean([ORIG[r][1][n]] + [d_res(r, rid, p) for p in prs], 0)
                return Mm
            r1, r1x = answers(res_members(True)), answers(res_members(False))
            ok &= paired_line(base[0][0], r1[0][0], L5, "clean4_qg_hcx: 검증")
            ok &= paired_line(r1x[0][0], r1[0][0], L5, "(HCX 몫) clean4_qg_hcx")
            if not RT: ok &= v5.startswith("판정: 검증만 확인") and not glob.glob(f"{RUNS}/e5_hires/*_hcx.csv")
            elif v5.startswith("판정: 공개 확인"): ok &= check_files(f"{RUNS}/e5_hires", "e5_hires_hcx", r1, base); ok &= os.path.exists(f"{RUNS}/e5_hires/e5_changed_val_hcx.html")
            else: ok &= v5.startswith("판정: 불채택")
    if not DT: ok &= not glob.glob(f"{RUNS}/e9_tta/{HCXR}/perm_*") and not glob.glob(f"{RUNS}/e9_tta/*_hcx.csv")
    # 다른 멤버 파일·E9·E5 기록은 그대로인지
    ok &= not os.path.exists(f"{RUNS}/e9_tta/{HCXR}/../e9_info.json.bak")
if mode == "tiny":
    ok &= LOADS == ["naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"]
    kws = [c for c in PROC.calls]; mx = sorted({c.get("max_pixels") for c in kws}); print("프로세서가 받은 max_pixels:", mx)
    ok &= mx == [3_400_000, 7_600_000, 12_500_000]
    ok &= os.path.exists(f"{RUNS}/e9_tta/{HCXR}/perm_val__up2.0.csv") and os.path.exists(f"{RUNS}/e5_hires/{HCXR}/res_test.csv")
    d = pd.read_csv(f"{RUNS}/e5_hires/{HCXR}/res_val.csv"); print("검증 해상도별 이미지 토큰:", d.groupby("preset").img_tokens.mean().to_dict())
for nm in ("V9", "V5"):
    print(nm, "판정:", next((l for l in g[nm]["lines"] if l.startswith("판정")), "(없음)"))
print("시험 도구 판정:", "통과" if ok else "실패")
