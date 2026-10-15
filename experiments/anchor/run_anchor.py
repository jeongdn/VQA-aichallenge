# -*- coding: utf-8 -*-
"""anchor_rescue.ipynb 시험: 설계한 가짜 Drive(기준 Flash-Next + 구원 후보 여러 개) → 노트북 코드 셀 실행 → 독립 계산과 대조.
사용: python3 run_anchor.py <nocal|cal|manual|noanchor|zsfam|judge>"""
import sys, os, json, shutil, io, traceback, itertools
import numpy as np, pandas as pd
scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
U = "/tmp/vqa_uploads"
ROOT = f"{SP}/anchortest_{scen}"; DATA = f"{ROOT}/content"; RUNS = f"{ROOT}/vqa_runs"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(DATA); os.makedirs(RUNS)
for a, b in [("6247e8ae-train.csv", "train.csv"), ("847ba753-test.csv", "test.csv"), ("eb905257-sample_submission.csv", "sample_submission.csv")]: shutil.copy(f"{U}/{a}", f"{DATA}/{b}")
tr = pd.read_csv(f"{DATA}/train.csv"); tr["answer"] = tr.answer.astype(str).str.strip().str.lower(); te = pd.read_csv(f"{DATA}/test.csv")
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = tr.iloc[sorted(vi)].reset_index(drop=True); y = val.answer.map("abcd".index).to_numpy(); N, NT = len(val), len(te)
R = np.random.RandomState(3)
def row(top, pt, sec, ps):
    p = np.zeros(4); p[top] = pt; p[sec] = ps; rest = [k for k in range(4) if k not in (top, sec)]; p[rest] = (1 - pt - ps) / 2; return p
def conf(k): return row(k, 0.97, (k + 1) % 4, 0.01)
perm = R.permutation(N); SETS = {"RESC": perm[:25], "CONFERR": perm[25:40], "HARM": perm[40:48], "WEAKONE": perm[48:60]}
JF, JH = perm[-40:-20], perm[-20:-10]
PA = np.stack([conf(y[i]) for i in range(N)])
alt = (y + 1) % 4
for i in SETS["RESC"]: PA[i] = row(alt[i], 0.50, y[i], 0.40)        # 기준 오답 · 확신 낮음 · 차이 0.1 → 구원 3명이 정답
for i in SETS["CONFERR"]: PA[i] = row(alt[i], 0.90, y[i], 0.05)     # 기준 확신 오답 → 확신도 조건 때문에 못 뒤집음
for i in SETS["HARM"]: PA[i] = row(y[i], 0.55, alt[i], 0.40)        # 기준 정답 · 확신 낮음 · 차이 0.15 → 구원 2명이 같은 오답
for i in SETS["WEAKONE"]: PA[i] = row(alt[i], 0.52, y[i], 0.40)     # 기준 오답이지만 구원 1명만 정답 → 다수 동의 부족
if scen == "judge":
    for i in JF: PA[i] = row(alt[i], 0.50, y[i], 0.40)               # 기준 오답 · 보조 모델은 hcx 1명만 정답 → 판정 2명이 보태면 뒤집힘
    for i in JH: PA[i] = row(y[i], 0.55, alt[i], 0.40)               # 기준 정답 · 보조 모델 q38_27b 1명이 대안 → 판정 2명이 3문항에서 대안에 표(망침)
RUNS_DEF = {"q38flash_r16_native_bf16": ("Qwen/Qwen3.8-Flash-Next", False, False), "gemma4_31b_r16_up20_ocr_bf16": ("google/gemma-4-31B-it", False, False),
            "gemma4_31b_r16_up20_bf16": ("google/gemma-4-31B-it", False, False), "q35_27b_r16_e4aux20_native": ("Qwen/Qwen3.5-27B", False, False),
            "hcx32b_r16_native": ("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", False, False), "q38_27b_r16_native": ("Qwen/Qwen3.8-27B", False, False),
            "ivl35_38b_r16_native": ("OpenGVLab/InternVL3_5-38B-HF", False, False), "q35_122b_a10b_gptq_zs": ("Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", True, False),
            "q35_27b_r16_native_full": ("Qwen/Qwen3.5-27B", False, True)}
if scen == "zsfam": RUNS_DEF.update({"zs_gemma_4_31b_it_up20": ("google/gemma-4-31B-it", True, False), "zs_internvl3_5_38b_hf_up20": ("OpenGVLab/InternVL3_5-38B-HF", True, False)})
ACCS = {"gemma4_31b_r16_up20_ocr_bf16": 5, "gemma4_31b_r16_up20_bf16": 12, "q35_27b_r16_e4aux20_native": 1, "hcx32b_r16_native": 3, "q38_27b_r16_native": 4,
        "ivl35_38b_r16_native": 20, "q35_122b_a10b_gptq_zs": 9, "q35_27b_r16_native_full": 0, "zs_gemma_4_31b_it_up20": 2, "zs_internvl3_5_38b_hf_up20": 30}   # 검증이 99%를 넘으면 '검증을 학습에 씀'으로 빠지므로 모두 99% 미만   # 흔들 문항 수 (클수록 검증 낮음)
RES_ORDER = ["gemma4_31b_r16_up20_ocr_bf16", "q35_27b_r16_e4aux20_native", "hcx32b_r16_native", "q38_27b_r16_native", "q35_122b_a10b_gptq_zs"]
def rescuer_val(r, j):
    P = np.stack([conf(y[i]) for i in range(N)])
    free = [i for i in perm[60:] ]
    for i in free[j * 20: j * 20 + ACCS[r]]: P[i] = conf((y[i] + 2) % 4)          # 멤버마다 따로 틀리는 문항 (정확도 차이)
    k = RES_ORDER.index(r) if r in RES_ORDER else -1
    for i in SETS["RESC"]: P[i] = conf(y[i]) if 0 <= k < 3 else conf((y[i] + 3) % 4)            # 구원 3명(gemma OCR·q35·hcx)이 정답, 나머지는 다른 오답
    for i in SETS["CONFERR"]: P[i] = conf(y[i]) if 0 <= k <= 4 else conf(alt[i])               # 기준 확신 오답: 구원 전원 정답 (확신도 문턱이 막는지)
    for i in SETS["HARM"]: P[i] = conf(alt[i]) if k in (0, 1) else conf(y[i])                  # 구원 2명이 같은 오답(기준의 2위 보기)
    for i in SETS["WEAKONE"]: P[i] = conf(y[i]) if k in (0, 4) else conf(alt[i])               # 기준 오답: gemma OCR · 122B 2명만 정답
    if scen == "judge":
        for i in JF: P[i] = conf(y[i]) if k == 2 else conf(alt[i])
        for i in JH: P[i] = conf(alt[i]) if k == 3 else conf(y[i])
    return P
def save_preds(path, ids, P):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in P.argmax(1)], "p_a": P[:, 0], "p_b": P[:, 1], "p_c": P[:, 2], "p_d": P[:, 3], "letter_mass": 0.999}).to_csv(path, index=False)
yt = R.randint(0, 4, NT); PAT = np.stack([conf(k) for k in yt]); TF = R.permutation(NT)[:30]
for i in TF: PAT[i] = row(yt[i], 0.50, (yt[i] + 1) % 4, 0.40)
PV_ALL, PT_ALL = {}, {}
for j, (r, (mid, zs, full)) in enumerate(RUNS_DEF.items()):
    Pv = PA if r == "q38flash_r16_native_bf16" else rescuer_val(r, j)
    Pt = PAT.copy() if r == "q38flash_r16_native_bf16" else np.stack([conf(k) for k in yt])
    if r in RES_ORDER[:3]:
        for i in TF: Pt[i] = conf((yt[i] + 1) % 4)                                     # test: 구원 3명이 기준 2위 보기를 지지 → 30문항 뒤집힘 기대
    save_preds(f"{RUNS}/{r}/preds_final_ep1__up2.0__1000.csv", val.id, Pv); save_preds(f"{RUNS}/{r}/preds_final_ep1_test__up2.0__{NT}.csv", te.id, Pt)
    json.dump({"config": {"MODEL_ID": mid, "ZERO_SHOT": zs, "TRAIN_ON_FULL": full}, "epoch_acc": {"1": float((Pv.argmax(1) == y).mean())}}, open(f"{RUNS}/{r}/train_log.json", "w"))
    PV_ALL[r], PT_ALL[r] = Pv, Pt
XC_DEF = {}
if scen == "judge":
    A0v, A0t = PA.argmax(1), PAT.argmax(1)
    base = ["gemma4_31b_r16_up20_ocr_bf16", "hcx32b_r16_native", "q35_27b_r16_e4aux20_native", "q38_27b_r16_native", "q35_122b_a10b_gptq_zs"]
    TGV = np.any([PV_ALL[r].argmax(1) != A0v for r in base], 0); TGT = np.any([PT_ALL[r].argmax(1) != A0t for r in base], 0)
    def judge_preds(A0, TG, vote_fn):
        P = np.eye(4)[A0].astype(float); jd = np.zeros(len(A0), int)
        for i in np.where(TG)[0]:
            v = vote_fn(i); q = np.full(4, 0.0); q[v] += 0.8; q[A0[i]] += 0.2; P[i] = q / q.sum(); jd[i] = 1
        return P, jd
    jfs, jhs = set(JF.tolist()), set(JH[:3].tolist())
    for r, mid, anc in [("xc_gemma_4_31b_it_up20", "google/gemma-4-31B-it", "q38flash_r16_native_bf16"),
                        ("xc_qwen3_5_122b_a10b_gptq_int4_up20", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", "q38flash_r16_native_bf16"),
                        ("xc_wronganchor_up20", "google/gemma-4-31B-it", "some_other_anchor")]:
        Pv, jv = judge_preds(A0v, TGV, lambda i: y[i] if i in jfs else (alt[i] if i in jhs else A0v[i]))
        Pt, jt = judge_preds(A0t, TGT, lambda i: A0t[i])
        for path, ids, P, jd in [(f"{RUNS}/{r}/preds_final_ep1__up2.0__1000.csv", val.id, Pv, jv), (f"{RUNS}/{r}/preds_final_ep1_test__up2.0__{NT}.csv", te.id, Pt, jt)]:
            save_preds(path, ids, P); d = pd.read_csv(path); d["judged"] = jd; d["n_valid"] = jd * 5; d.to_csv(path, index=False)
        json.dump({"config": {"MODEL_ID": "xcheck:" + mid, "JUDGE_MODEL_ID": mid, "ZERO_SHOT": True, "XCHECK": True, "XCHECK_ANCHOR": anc, "TARGETS_TAG": "old000"},
                   "epoch_acc": {"1": float((Pv.argmax(1) == y).mean())}}, open(f"{RUNS}/{r}/train_log.json", "w"))
        PV_ALL[r], PT_ALL[r] = Pv, Pt; XC_DEF[r] = anc
pd.DataFrame({"id": te.id, "answer": ["abcd"[i] for i in PAT.argmax(1)]}).to_csv(f"{RUNS}/q38flash_r16_native_bf16/submission_q38flash_r16_native_bf16_final_ep1_up2.0.csv", index=False)
if scen == "noanchor": shutil.rmtree(f"{RUNS}/q38flash_r16_native_bf16")
if scen == "integ":                                                              # RunPod 판정 노트북이 만든 실행 폴더를 그대로 넣음 (브리지 B 단계가 하는 일)
    _src = os.environ["XC_RUN_DIR"]; _jr = os.path.basename(_src); shutil.copytree(_src, f"{RUNS}/{_jr}")
    def _rd(f, ids): d = pd.read_csv(f).set_index("id").reindex(ids); return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
    PV_ALL[_jr] = _rd(f"{RUNS}/{_jr}/preds_final_ep1__up2.0__1000.csv", val.id); PT_ALL[_jr] = _rd(f"{RUNS}/{_jr}/preds_final_ep1_test__up2.0__{NT}.csv", te.id)

import types, huggingface_hub
UPL = []
class FakeApi:
    def __init__(self, token=None): assert token == "hf_fake", token
    def whoami(self): return {"name": "tester"}
    def repo_info(self, repo, repo_type=None): return types.SimpleNamespace(private=True)
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message=None): UPL.append((repo_id, path_in_repo, pd.read_csv(path_or_fileobj)))
huggingface_hub.HfApi = FakeApi
NB = "/home/user/vqa/anchor/out/anchor_rescue.ipynb"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
g = {"__name__": "__main__"}; ERR = None
sys.stdout = Tee()
try:
    for k, c in enumerate(cells):
        print(f"\n━━━━ [셀 {k + 1}] ━━━━"); exec(compile(c, f"<c{k}>", "exec"), g)
        if k == 0:
            g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "HF_TOKEN": "hf_fake"})
            if scen == "judge": g["CALIBRATE_ANCHOR"] = False; g["ZS_MIN_ACC"] = 94.0      # JF 20문항 때문에 122B zero-shot이 94.6% → 포함되게
            if scen in ("nocal", "integ"): g["CALIBRATE_ANCHOR"] = False
            if scen == "zsfam": g["ZS_MIN_ACC"] = 94.0          # zero-shot Gemma 94.6% 포함 · InternVL 91.8% 제외
            if scen == "manual": g["RESCUERS"] = ["gemma4_31b_r16_up20_ocr_bf16", "q35_27b_r16_e4aux20_native", "hcx32b_r16_native"]; g["CALIBRATE_ANCHOR"] = False
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
print("\n══════ 판정 ══════")
if scen == "noanchor":
    ok(isinstance(ERR, AssertionError) and "기준 모델 q38flash_r16_native_bf16를 찾지 못했습니다" in str(ERR), f"기준 모델이 없으면 멈춤: {ERR}")
else:
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    RESC = g.get("RESC")
    exp_r = ["gemma4_31b_r16_up20_ocr_bf16", "q35_27b_r16_e4aux20_native", "hcx32b_r16_native"] if scen == "manual" else \
            ["gemma4_31b_r16_up20_ocr_bf16", "hcx32b_r16_native", "q35_27b_r16_e4aux20_native", "q38_27b_r16_native", "q35_122b_a10b_gptq_zs"] + (["zs_gemma_4_31b_it_up20"] if scen == "zsfam" else []) + ([os.path.basename(os.environ["XC_RUN_DIR"])] if scen == "integ" else []) \
            if scen != "judge" else ["hcx32b_r16_native", "gemma4_31b_r16_up20_ocr_bf16", "q35_27b_r16_e4aux20_native", "q38_27b_r16_native", "q35_122b_a10b_gptq_zs",
                                     "xc_gemma_4_31b_it_up20", "xc_qwen3_5_122b_a10b_gptq_int4_up20"]   # judge: JF·JH 때문에 정확도 순서가 hcx > gemma OCR > q35 > q38
    ok(RESC == exp_r, f"보조 모델 {RESC} (기대 {exp_r}: 계열별 최고 상위 4 + zero-shot 122B, _full·같은 계열 2등·5번째 계열 제외)")
    # 독립 계산: 같은 알고리즘을 따로 구현해 교차 적합 결과 대조
    TG = np.exp(np.linspace(np.log(0.25), np.log(8.0), 70))
    def ts(p, T): q = np.clip(p, 1e-12, 1) ** (1 / T); return q / q.sum(1, keepdims=True)
    def fitT(P, idx): return float(TG[int(np.argmin([-np.mean(np.log(ts(P[idx], T)[np.arange(len(idx)), y[idx]])) for T in TG]))])
    RVs = np.stack([PV_ALL[r] for r in RESC]); votes = np.zeros((N, 4))
    for m in range(len(RESC)): votes[np.arange(N), RVs[m].argmax(1)] += 1
    psum = RVs.sum(0)
    def rule(P, c):
        a = P.argmax(1); s = np.sort(P, 1); pm, mg = s[:, -1], s[:, -1] - s[:, -2]
        v = votes.copy(); v[np.arange(N), a] = -1; key = v + 1e-3 * psum / psum.max(); al = key.argmax(1); cn = v[np.arange(N), al]
        fl = (cn >= c[2]) & (pm < c[0]) & (mg < c[1]); return np.where(fl, al, a), fl
    agree_exp = list(range(2, min(len(RESC), 8) + 1))
    ok(g["AGREE_USE"] == agree_exp, f"AGREE 후보 {g['AGREE_USE']} = 2 ~ 표 수 {agree_exp}")
    grid = list(itertools.product([0.5, 0.6, 0.7, 0.8, 0.9, 1.01], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01], agree_exp))
    cal = g["CALIBRATE_ANCHOR"]
    def best(P, idx):
        a0 = P.argmax(1); bv, bc = None, None
        for c in grid:
            a1, fl = rule(P, c); r_ = int((fl[idx] & (a0[idx] != y[idx]) & (a1[idx] == y[idx])).sum()); h_ = int((fl[idx] & (a0[idx] == y[idx])).sum())
            if r_ >= 2 * h_ and r_ > 0:
                key = (r_ - h_, -h_, -c[0] - c[1])
                if bv is None or key > bv: bv, bc = key, c
        return bc
    rs = np.random.RandomState(0); pm_ = rs.permutation(N); folds = [np.sort(pm_[f::5]) for f in range(5)]
    A0 = PA.argmax(1); O1 = A0.copy(); OF = np.zeros(N, bool)
    for te_idx in folds:
        tr_idx = np.setdiff1d(np.arange(N), te_idx); P = ts(PA, fitT(PA, tr_idx)) if cal else PA
        c = best(P, tr_idx)
        if c is None: continue
        a1, fl = rule(P, c); O1[te_idx], OF[te_idx] = a1[te_idx], fl[te_idx]
    rr = int((OF & (A0 != y) & (O1 == y)).sum()); hh = int((OF & (A0 == y)).sum())
    ok(f"살림 {rr} · 망침 {hh} · 순이득 {rr - hh:+d}" in out, f"교차 적합 결과 = 독립 계산 (살림 {rr} · 망침 {hh})")
    c_all = best(ts(PA, fitT(PA, np.arange(N))) if cal else PA, np.arange(N))
    ok(g["FINAL"] == c_all, f"최종 문턱값 {g['FINAL']} = 독립 계산 {c_all}")
    # 7-1 멤버별 기여: 하나 빼고 같은 교차 적합 (독립 구현)
    def oof_for(resc):
        RVx = np.stack([PV_ALL[r] for r in resc]); vx = np.zeros((N, 4))
        for m in range(len(resc)): vx[np.arange(N), RVx[m].argmax(1)] += 1
        px = RVx.sum(0); gx = list(itertools.product([0.5, 0.6, 0.7, 0.8, 0.9, 1.01], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01], list(range(2, min(len(resc), 8) + 1))))
        def rl(P, c):
            a = P.argmax(1); s_ = np.sort(P, 1); pm, mg = s_[:, -1], s_[:, -1] - s_[:, -2]
            v = vx.copy(); v[np.arange(N), a] = -1; al = (v + 1e-3 * px / px.max()).argmax(1); cn = v[np.arange(N), al]
            fl = (cn >= c[2]) & (pm < c[0]) & (mg < c[1]); return np.where(fl, al, a), fl
        def bst(P, idx):
            a0 = P.argmax(1); bv, bc = None, None
            for c in gx:
                a1, fl = rl(P, c); r_ = int((fl[idx] & (a0[idx] != y[idx]) & (a1[idx] == y[idx])).sum()); h_ = int((fl[idx] & (a0[idx] == y[idx])).sum())
                if r_ >= 2 * h_ and r_ > 0 and (bv is None or (r_ - h_, -h_, -c[0] - c[1]) > bv): bv, bc = (r_ - h_, -h_, -c[0] - c[1]), c
            return bc
        O = A0.copy()
        for te_idx in folds:
            tr_idx = np.setdiff1d(np.arange(N), te_idx); P = ts(PA, fitT(PA, tr_idx)) if cal else PA; c = bst(P, tr_idx)
            if c is None: continue
            a1, _ = rl(P, c); O[te_idx] = a1[te_idx]
        return O
    CB = g["CONTRIB"].set_index("멤버")
    for r_chk in [x for x in RESC if x in ("q35_122b_a10b_gptq_zs", "xc_gemma_4_31b_it_up20", "hcx32b_r16_native")]:
        O_wo = oof_for([x for x in RESC if x != r_chk]); gain = int(((O1 == y) & (O_wo != y)).sum()); loss = int(((O1 != y) & (O_wo == y)).sum())
        ok(int(CB.loc[r_chk, "넣어서 맞게 된 문항"]) == gain and int(CB.loc[r_chk, "넣어서 틀리게 된 문항"]) == loss, f"7-1 기여 {r_chk}: 맞게 {gain} · 틀리게 {loss} = 독립 계산")
    if scen == "judge":
        O_wo = oof_for(g["RESC_BASE"]); gain = int(((O1 == y) & (O_wo != y)).sum()); loss = int(((O1 != y) & (O_wo == y)).sum())
        ok(int(CB.loc["(판정 모델 전부 2개)", "넣어서 맞게 된 문항"]) == gain == 20 and int(CB.loc["(판정 모델 전부 2개)", "넣어서 틀리게 된 문항"]) == loss == 3,
           f"7-1 판정 묶음: 맞게 {gain} · 틀리게 {loss} = 독립 계산 = 설계(JF 20 · JH 3)")
    if not cal and scen != "integ":
        a1, fl = rule(PA, c_all); r_all, h_all = int((fl & (A0 != y) & (a1 == y)).sum()), int((fl & (A0 == y)).sum())
        exp = (52, 8) if scen == "nocal" else (72, 11) if scen == "judge" else (40, 0)   # judge: + JF 20 살림 · JH 3 망침        # auto: 확신 없음·마진 없음·2명 동의 → 25 + 15 + 12 살림, 8 망침 / manual(3명): 25 + 15 살림 (1명·2명 오답은 못 넘음)
        ok((r_all, h_all) == exp, f"설계대로 살림 {r_all} · 망침 {h_all} (기대 {exp})")
        ok("[판정] 채택 ✅" in out, "합격선 통과 → 채택")
    # test 파일
    fs = [f for f in os.listdir(f"{RUNS}/anchor_rescue") if f.startswith("submission_anchor_") and f.endswith(".csv")]
    ok(len(fs) == 1, f"제출 파일 1개 {fs}")
    if fs:
        a = pd.read_csv(f"{RUNS}/anchor_rescue/{fs[0]}").answer.map("abcd".index).to_numpy()
        d = int((a != PAT.argmax(1)).sum())
        ok(d == g["meta"]["test_flips"], f"test: 기준 단독과 다른 답 {d} = 뒤집은 수 {g['meta']['test_flips']}")
        if not cal: ok(d == 30 and all(a[i] == (yt[i] + 1) % 4 for i in TF), "test 설계: 30문항만 구원 3명이 지지한 보기로")
        ok(f"기준 모델 단독 제출과 test 답이 다른 문항 {d}개" in out, "기준 단독 제출 파일과 대조 출력")
    ok("[부트스트랩 1000번]" in out and "문항 유형별" in out and "[일관성]" in out and "[조율 비교]" in out, "부트스트랩 · 유형별 · 일관성 · 조율 비교 출력")
    ok("ECE 전" in out and "순위 투표(Borda)" in out and "마진 가중(온도 보정)" in out, "ECE · Borda · 마진 가중")
    # 판정 목록 (보조 모델만으로, 정답 안 봄) · 올리기
    base = g["RESC_BASE"]; A0v_, A0t_ = PA.argmax(1), PAT.argmax(1)
    tv = np.any([PV_ALL[r].argmax(1) != A0v_ for r in base], 0); tt = np.any([PT_ALL[r].argmax(1) != A0t_ for r in base], 0)
    tg = pd.read_csv(f"{RUNS}/anchor_rescue/xcheck_targets.csv")
    ok(len(tg) == N + NT and (tg[tg.split == "val"].target.to_numpy() == tv).all() and (tg[tg.split == "test"].target.to_numpy() == tt).all()
       and (tg[tg.split == "val"].anchor.to_numpy() == np.array(list("abcd"))[A0v_]).all() and (tg.anchor != tg.alt)[tg.target == 1].all(),
       f"판정 목록 = 독립 계산 (검증 {int(tv.sum())} · test {int(tt.sum())}, 기준 답·대안 ≠ 기준)")
    ok(len(UPL) == 1 and UPL[0][:2] == ("tester/ssafy-vqa-private", "xcheck/xcheck_targets.csv") and len(UPL[0][2]) == N + NT, f"HF 올리기 {[u[:2] for u in UPL]}")
    ok(g["MODELS"] == [g["ANCHOR"]] + base, "조율 비교는 판정 모델 제외")
    if scen == "integ":
        _jr = os.path.basename(os.environ["XC_RUN_DIR"])
        ok(g["XCR"] == [_jr] and g["RESC"][-1] == _jr and "판정 모델 성적" in out and "판정할 때의 목록" not in out, f"판정 노트북 결과를 한 표로 잡음: {g['XCR']} (목록 태그 일치)")
    if scen == "judge":
        ok(g["XCR"] == ["xc_gemma_4_31b_it_up20", "xc_qwen3_5_122b_a10b_gptq_int4_up20"] and "다른 기준 모델로 만든 판정 목록" in out, f"판정 모델 {g['XCR']} (다른 기준의 판정은 뺌)")
        ok("판정 모델 성적" in out and "판정할 때의 목록(old000)과 지금 목록" in out, "판정 모델 성적 표 · 목록 바뀜 경고")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
