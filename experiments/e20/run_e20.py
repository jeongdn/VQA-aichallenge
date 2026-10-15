# -*- coding: utf-8 -*-
"""e20_xcheck_122b.ipynb 시험 (CPU): 가짜 vLLM(실제 0.30.0 인자 목록 대조 + 실제 HF 프로세서로 프롬프트·이미지 처리, 생성 모드 추가) +
실제 transformers 5.17 Qwen3VLProcessor(작은 토크나이저)로 노트북 코드 셀을 순서대로 실행. 기준 조합은 설계한 확률.
사용: python3 run_e20.py <main|reject|crash|think|noformat|basebad|noauto>"""
import sys, os, json, shutil, re, math, hashlib, io, traceback
import numpy as np, pandas as pd

scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
ZS = f"{SP}/zs122b"; DATA = f"{SP}/newfam/fx/content"; RUNS = f"{SP}/e20test/vqa_runs_{scen}"
NB = "/home/user/vqa/e20/out/e20_xcheck_122b.ipynb"
os.environ["FAKE_VLLM_API"] = f"{ZS}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{ZS}/tiny_proc"
sys.path.insert(0, "/home/user/vqa/e20/fake_vllm"); os.environ["PYTHONPATH"] = "/home/user/vqa/e20/fake_vllm"
import vllm as FV
import transformers
from transformers import AutoProcessor
_orig_fp = AutoProcessor.from_pretrained
transformers.AutoProcessor.from_pretrained = lambda mid, **kw: _orig_fp(os.environ["FAKE_PROC_DIR"])
import huggingface_hub
def _fake_snapshot(repo_id, **kw):
    d = f"{ZS}/fake_snapshot"; os.makedirs(d, exist_ok=True)
    for i in range(1, 40): open(f"{d}/model.safetensors-{i:05d}-of-00039.safetensors", "wb").write(b"0" * 10)
    open(f"{d}/config.json", "w").write("{}"); return d
huggingface_hub.snapshot_download = _fake_snapshot
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, CSV_C, BASE_C, PROMPT_C, IMG_C, DL_C, LLM_C, GEN_C, VAL_C, TEST_C, SUMM_C = cells

# ── 가짜 Drive: 기준 조합(멤버 3개, 같은 가중치 확률 평균) + ens_search 설정 JSON + 저장된 제출 파일 ──
shutil.rmtree(RUNS, ignore_errors=True); os.makedirs(RUNS)
tr = pd.read_csv(f"{DATA}/train.csv"); tr["answer"] = tr.answer.astype(str).str.strip().str.lower()
te = pd.read_csv(f"{DATA}/test.csv")
for c in ["question", "a", "b", "c", "d"]: tr[c] = tr[c].astype(str); te[c] = te[c].astype(str)
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = tr.iloc[sorted(vi)].reset_index(drop=True); y = val.answer.map("abcd".index).to_numpy()
NT = 200; te200 = te.head(NT)
def row(top, pt, sec, ps):
    p = np.zeros(4); p[top] = pt; p[sec] = ps; rest = [k for k in range(4) if k not in (top, sec)]; p[rest] = (1 - pt - ps) / 2; return p
R = np.random.RandomState(5); perm = [int(i) for i in R.permutation(len(val))]; c0 = 0
def take(n):
    global c0; out = perm[c0:c0 + n]; c0 += n; return out
S = {"U_fix": take(8), "U_keep": take(15), "U_brk": take(2), "U_inv": take(3), "U_far": take(10)}
P = np.stack([row(y[i], 0.97, (y[i] + 1) % 4, 0.01) for i in range(len(val))])
W = {}
for i in S["U_fix"] + S["U_inv"]: w = (y[i] + 1) % 4; W[i] = w; P[i] = row(w, 0.50, y[i], 0.40)       # 조합 오답 · 차이 0.1
for i in S["U_keep"]: w = (y[i] + 2) % 4; W[i] = w; P[i] = row(y[i], 0.55, w, 0.35)                   # 조합 정답 · 차이 0.2
for i in S["U_brk"]: w = (y[i] + 3) % 4; W[i] = w; P[i] = row(y[i], 0.60, w, 0.30)                    # 조합 정답 · 차이 0.3 (교차 검증이 뒤집음)
for i in S["U_far"]: w = (y[i] + 1) % 4; W[i] = w; P[i] = row(y[i], 0.75, w, 0.15)                    # 차이 0.6 (τ 0.5 밖, 0.8 안)
yt = R.randint(0, 4, len(te)); PT = np.stack([row(k, 0.97, (k + 1) % 4, 0.01) for k in yt])
TP = [int(i) for i in R.permutation(NT)]; T = {"T_flip": TP[:12], "T_keep": TP[12:20]}
for i in T["T_flip"] + T["T_keep"]: PT[i] = row(yt[i], 0.50, (yt[i] + 1) % 4, 0.40)
def save_preds(path, ids, Pm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in Pm.argmax(1)], "p_a": Pm[:, 0], "p_b": Pm[:, 1], "p_c": Pm[:, 2], "p_d": Pm[:, 3], "letter_mass": 0.999}).to_csv(path, index=False)
MEM = {}
for r in ["gemma4_31b_r16_up20_bf16", "ivl35_38b_r16_native", "q38_27b_r16_native"]:
    fv, ft = f"{RUNS}/{r}/preds_final_ep1__up2.0__1000.csv", f"{RUNS}/{r}/preds_final_ep1_test__up2.0__{len(te)}.csv"
    save_preds(fv, val.id, P); save_preds(ft, te.id, PT)
    MEM[r] = {"family": r, "preset": "up2.0", "T": 1.0, "val_acc": float((P.argmax(1) == y).mean() * 100), "val": os.path.relpath(fv, RUNS), "test": os.path.relpath(ft, RUNS),
              "val_md5": hashlib.md5(open(fv, "rb").read()).hexdigest(), "test_md5": hashlib.md5(open(ft, "rb").read()).hexdigest()}
os.makedirs(f"{RUNS}/ens_search")
fn = "submission_my_search_ea0e9f.csv"
saved = PT.argmax(1).copy()
if scen == "basebad": saved[3] = (saved[3] + 1) % 4
pd.DataFrame({"id": te.id, "answer": ["abcd"[i] for i in saved]}).to_csv(f"{RUNS}/ens_search/{fn}", index=False)
json.dump({"name": "my_search", "file": fn, "cfg": {"weights": {r: 1.0 for r in MEM}, "method": "prob", "temp": "none", "full": "none"},
           "val_acc": float((P.argmax(1) == y).mean() * 100), "members": MEM}, open(f"{RUNS}/ens_search/{fn[:-4]}.json", "w", encoding="utf-8"), ensure_ascii=False)

def opt_key(r):                   # 가짜 vLLM이 문항을 알아보는 열쇠 = 교차 검증 프롬프트의 질문 + 보기 부분
    return f"{r['question']}\n" + "\n".join(f"({L}) {r[L]}" for L in "abcd")
vrec = val.to_dict("records"); trec = te200.to_dict("records")
FV.STATE["xc_gold"] = {opt_key(vrec[i]): "abcd"[y[i]] for i in range(len(val))}
if scen == "reject":
    for i in S["U_fix"]: FV.STATE["xc_override"][opt_key(vrec[i])] = "abcd"[W[i]]              # 조합 오답 쪽을 지지 → 못 고침
for i in S["U_brk"]: FV.STATE["xc_override"][opt_key(vrec[i])] = "abcd"[W[i]]                   # 조합 정답을 뒤집음 → 새로 틀림
for i in S["U_inv"]: FV.STATE["xc_invalid"][opt_key(vrec[i])] = 3                               # 5개 중 3개는 답을 못 읽음 → 섞지 않음
for i in T["T_flip"]: FV.STATE["xc_override"][opt_key(trec[i])] = "abcd"[(yt[i] + 1) % 4]
for i in T["T_keep"]: FV.STATE["xc_override"][opt_key(trec[i])] = "abcd"[yt[i]]
FV.STATE["xc_mode"] = {"think": "think", "noformat": "noformat"}.get(scen, "good")

OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
def run(g, src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
def run_all(g):
    run(g, CFG, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "HF_CACHE": f"{ZS}/hf_cache_e20"})
    g["RES_PRESETS"]["up2.0"] = (61_440, 92_160)
    if scen == "think": g["THINKING"] = True
    if scen == "noauto": g["RUN_TEST"] = False
    run(g, DATA_C, "3 데이터"); run(g, CSV_C, "4 CSV·분할")
    g["test_df"] = g["test_df"].head(NT).copy(); g["sub_df"] = g["sub_df"].head(NT).copy()
    for nm, src_, lab in [("BASE", BASE_C, "5 기준 조합"), ("PROMPT", PROMPT_C, "6 프롬프트"), ("IMG", IMG_C, "7 이미지"), ("DL", DL_C, "8-0 받기"),
                          ("LLM", LLM_C, "8 vLLM"), ("GEN", GEN_C, "9 생성 함수"), ("VAL", VAL_C, "10 검증"), ("TEST", TEST_C, "11 test"), ("SUMM", SUMM_C, "12 요약")]:
        run(g, src_, lab)

fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
sys.stdout = Tee(); g = {"__name__": "__main__"}; ERR = None
try:
    if scen == "crash":
        FV.STATE["crash_at"] = 2                                   # 검증 두 번째 묶음 생성 중 중단
        try: run_all(g)
        except RuntimeError as e: print("\n[예상된 중단]", e)
        FV.STATE["crash_at"] = None
        gp = f"{RUNS}/e20_xcheck_q35_122b/xc_val__up2.0.jsonl"
        n_before = sum(1 for _ in open(gp, encoding="utf-8"))
        with open(gp, "a", encoding="utf-8") as f: f.write('{"id": "train_9999.jpg", "split": "val", "sam')      # 저장 도중 끊긴 반쯤 쓴 줄
        g = {"__name__": "__main__"}; run_all(g)                  # 새 세션처럼 처음부터
        print(f"[시험 도구] 중단 전 저장 줄 {n_before}")
    else:
        run_all(g)
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()
RD = f"{RUNS}/e20_xcheck_q35_122b"

print("\n══════ 판정 ══════")
if scen == "basebad":
    ok(isinstance(ERR, AssertionError) and "저장된 제출 파일과 다릅니다" in str(ERR), f"기준 조합 재현 실패 → 모델 받기 전에 멈춤: {ERR}")
    ok(FV.STATE["n_init"] == 0, "vLLM을 띄우지 않음")
elif scen == "noformat":
    ok(isinstance(ERR, RuntimeError) and "절반도 안 됩니다" in str(ERR), f"형식을 안 지키면 첫 묶음에서 멈춤: {ERR}")
else:
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    nval = len(S["U_fix"] + S["U_keep"] + S["U_brk"] + S["U_inv"] + S["U_far"])
    ok(f"→ 채점 대상: 검증 {nval}문항 (τ < 0.8) · test 20문항 (τ < 0.5)" in out, "채점 대상 수 (검증 τ<0.8 · test τ<0.5)")
    _rows = []
    for _l in open(f"{RD}/xc_val__up2.0.jsonl", encoding="utf-8"):
        try: _rows.append(json.loads(_l))
        except Exception: pass
    gv = pd.DataFrame(_rows)
    ok(len(gv) == nval * 5 and not gv.duplicated(["id", "sample"]).any(), f"검증 생성 {len(gv)}줄 = {nval}문항 × 표본 5 · 중복 없음")
    # 가짜 vLLM이 받은 요청 점검: 시드·표본 수·A/B = 조합 1·2위 · 무작위 배정
    log = pd.DataFrame(FV.STATE["gen_log"])
    ok((log.n == 5).all() and (log.max_tokens == 1200).all() and log.seed.notna().all(), "요청마다 표본 5개 · 최대 1200토큰 · 문항별 시드")
    top2 = {opt_key(vrec[i]): {"abcd"[y[i]], "abcd"[W[i]]} for i in W}
    _lv = log[log.key.isin(top2)]
    ok(len(_lv) > 0 and all({a, b} == top2[k] for k, a, b in zip(_lv.key, _lv.A, _lv.B)), "모델 A/B = 조합의 1·2위 보기")
    ok(0.2 < (_lv.A == [("abcd"[y[i]] if "abcd"[y[i]] != "abcd"[W[i]] else "") for i in [next(j for j in W if opt_key(vrec[j]) == k) for k in _lv.key]]).mean() < 0.8,
       "A/B 배정이 한쪽으로 쏠리지 않음 (문항별 무작위)")
    ok(log.thinking.all() if scen == "think" else not log.thinking.any(), f"생각 모드 = {scen == 'think'}")
    # 독립 계산: 저장된 생성 → 투표 → 섞기 → 고침·새로 틀림
    Pm = P; mg = np.sort(Pm, 1); mg = mg[:, -1] - mg[:, -2]
    vid = {i: n for n, i in enumerate(val.id)}
    Q = np.full((len(val), 4), np.nan)
    for qid, sub in gv.groupby("id"):
        v = [x for x in sub.letter if isinstance(x, str) and x in "abcd"]
        if len(v) >= 3: Q[vid[qid]] = np.bincount(["abcd".index(x) for x in v], minlength=4) / len(v)
    sel = (mg < 0.5) & ~np.isnan(Q).any(1); Pn = Pm.copy(); Pn[sel] = 0.5 * Pm[sel] + 0.5 * Q[sel]
    a0, a1 = Pm.argmax(1), Pn.argmax(1); fx, bk = int(((a0 != y) & (a1 == y)).sum()), int(((a0 == y) & (a1 != y)).sum())
    exp_f, exp_b = (0, 2) if scen == "reject" else (8, 2)
    ok((fx, bk) == (exp_f, exp_b), f"독립 계산 고침 {fx} · 새로 틀림 {bk} (설계 {exp_f} · {exp_b}: U_inv 3개는 표본 부족으로 안 섞임)")
    ok(f"고침 {fx} · 새로 틀림 {bk} · 순이득 {fx - bk:+d} · 바뀐 답 {fx + bk}" in out, "노트북 출력 = 독립 계산")
    if scen == "reject":
        ok("판정: 불채택 (순이득 -2)" in out and "test 생략" in out and not os.path.exists(f"{RD}/xc_test__up2.0.jsonl"), "불채택 → test 생략")
    elif scen == "noauto":
        ok("판정: 채택 (순이득 +6 ≥ +3)" in out and "test 생략 (RUN_TEST=False" in out, "RUN_TEST=False면 채택이어도 test 생략")
    else:
        ok("판정: 채택 (순이득 +6 ≥ +3)" in out, "판정 문구 (채택)")
        cp = f"{RD}/submission_e20_ea0e9f_t05_a05.csv"
        ok(os.path.exists(cp), "후보 파일 submission_e20_ea0e9f_t05_a05.csv")
        if os.path.exists(cp):
            ans = pd.read_csv(cp).answer.map("abcd".index).to_numpy()
            exp = PT.argmax(1)[:NT].copy()
            for i in T["T_flip"]: exp[i] = (yt[i] + 1) % 4
            ok(list(pd.read_csv(cp).id) == list(te200.id) and (ans == exp).all(), "후보 test 답 = 설계 (T_flip 12문항만 2위 보기로)")
            mj = json.load(open(cp[:-4] + ".json", encoding="utf-8"))
            ok(mj["test"]["changed"] == 12 and mj["val"]["net"] == 6 and mj["k"] == 5 and mj["base_json"] == "ens_search/submission_my_search_ea0e9f.json", f"후보 JSON {mj['test']} {mj['val']}")
        ok("재현 ✅" in out and "+5문항 이상" in out, "재현 점검 · 공개 우연 폭 (D=12 → ±5)")
    if scen == "think": ok(gv.letter.eq(gv.letter).mean() > 0.9 and not any("처음에는" in (t or "")[-40:] for t in gv.text), "생각 모드: </think> 뒤의 답만 읽음")
    if scen == "crash":
        ok("완료 " in out and "남음 " in out and FV.STATE["n_init"] == 2, "중단 뒤 새 세션에서 이어하기")
        raw = [l for l in open(f"{RD}/xc_val__up2.0.jsonl", encoding="utf-8")]
        ok(sum(1 for l in raw if l.startswith('{"id": "train_9999.jpg", "split": "val", "sam') and not l.rstrip().endswith("}")) == 1, "반쯤 쓴 줄은 한 줄로 남고 다음 줄과 섞이지 않음")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
