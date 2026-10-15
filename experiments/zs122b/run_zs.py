# -*- coding: utf-8 -*-
"""zs122b_vllm.ipynb 시험 (CPU): 가짜 vLLM(실제 0.30.0 인자 목록으로 검사 + 실제 HF 프로세서로 프롬프트·이미지 처리) +
실제 transformers 5.17 Qwen3VLProcessor(작은 토크나이저)로 노트북 코드 셀을 순서대로 실행.
사용: python3 run_zs.py <auto|crash|crash2|gate|quick|lowmass|mismatch|nopre>"""
import sys, os, json, glob, shutil, re, math, hashlib
import numpy as np, pandas as pd

scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/zs122b"; RUNS = f"{ROOT}/vqa_runs_{scen}"; DATA = f"{SP}/newfam/fx/content"; REF = f"{ROOT}/ref_auto"
NB = "/home/user/vqa/zs122b/out/zs122b_vllm.ipynb"
os.environ["FAKE_VLLM_API"] = f"{ROOT}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{ROOT}/tiny_proc"
sys.path.insert(0, "/home/user/vqa/zs122b/fake_vllm")
os.environ["PYTHONPATH"] = "/home/user/vqa/zs122b/fake_vllm"      # 설치된 vLLM처럼 하위 프로세스(모델 검사)에서도 보이게
if scen in ("badta", "badta_inspect"):   # import하면 CUDA 불일치 RuntimeError를 내는 torchaudio가 깔려 있음 → 노트북 1번 셀(차단)이 가장 먼저 실행되는 것과 같게
    sys.path.insert(0, f"{SP}/badta")
    if scen == "badta_inspect":          # 설치 셀이 torchaudio를 지우지 못한 상황: vLLM의 모델 검사 프로세스에도 깨진 torchaudio가 보임
        os.environ["PYTHONPATH"] = f"{SP}/badta:/home/user/vqa/zs122b/fake_vllm"
    exec(compile(["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"][0], "<cfg0>", "exec"), {"__name__": "__main__"})
import vllm as FV
import transformers
from transformers import AutoProcessor
_orig_fp = AutoProcessor.from_pretrained
CALLED = []
def _fp(mid, **kw):
    CALLED.append(mid); return _orig_fp(os.environ["FAKE_PROC_DIR"])
transformers.AutoProcessor.from_pretrained = _fp

cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, CSV, PROMPT, IMG, DL, LLMC, SCORE, REPORT, QUICK, FULL, SUMM = cells
assert "snapshot_download" in DL
import huggingface_hub
DL_CALLS = []
def _fake_snapshot(repo_id, **kw):                          # 가짜 다운로드: 실제 저장소처럼 파일 이름만 흉내 낸 작은 폴더
    DL_CALLS.append((repo_id, kw)); d = f"{ROOT}/fake_snapshot"; os.makedirs(d, exist_ok=True)
    for i in range(1, 40): open(f"{d}/model.safetensors-{i:05d}-of-00039.safetensors", "wb").write(b"0" * 10)
    open(f"{d}/config.json", "w").write("{}"); return d
huggingface_hub.snapshot_download = _fake_snapshot
assert "!pip" in INSTALL and "LLM(model=MODEL_ID" in LLMC

# ── 가짜 Drive: 다른 실행들의 예측 파일 ──
shutil.rmtree(RUNS, ignore_errors=True); os.makedirs(RUNS)
tr = pd.read_csv(f"{DATA}/train.csv"); tr["answer"] = tr.answer.astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = tr.iloc[sorted(vi)].reset_index(drop=True)
def fake_preds(ids, ans, acc, seed):
    R = np.random.RandomState(seed); ok = R.rand(len(ids)) < acc
    return pd.DataFrame({"id": ids, "pred": [a if o else "abcd"[("abcd".index(a) + 1) % 4] for a, o in zip(ans, ok)]})
REFS = {"q35_27b_r16_e5lr2e5_native/preds_zeroshot_up2.0__up2.0__1000.csv": (1000, 0.961, 1),
        "zeroshot_flashnext_iq3xxs/preds_zeroshot__IQ3_XXS__val.csv": (300, 0.953, 2),
        "q36_35ba3b_r16_native_bf16/preds_zeroshot__up2.0__300.csv": (300, 0.963, 3),
        "q35_27b_r16_native/preds_final_ep1__up2.0__1000.csv": (1000, 0.972, 4),
        "gemma4_31b_r16_up20/preds_final_ep1__up2.0__1000.csv": (1000, 0.981, 5),
        "q35_27b_r16_native_dry/preds_final_ep1__up2.0__1000.csv": (1000, 0.5, 6),     # _dry → 비교에서 빠져야 함
        "hcx_partial/preds_final_ep1__up2.0__1000.csv": (500, 0.97, 7)}               # 문항 일부만 → 비교에서 빠져야 함
for rel, (n, acc, s) in REFS.items():
    os.makedirs(os.path.dirname(f"{RUNS}/{rel}"), exist_ok=True)
    fake_preds(val.id[:n], val.answer[:n], acc, s).to_csv(f"{RUNS}/{rel}", index=False)

class _ImgProxy:                                          # nopre: 노트북의 Image.fromarray만 1픽셀 바꿔 점검 셀이 불일치를 잡는지 확인
    def __init__(self, mod): self._m = mod
    def __getattr__(self, k): return getattr(self._m, k)
    def fromarray(self, a):
        a = a.copy(); a[0, 0, 0] = (int(a[0, 0, 0]) + 7) % 256; return self._m.fromarray(a)

def run(g, src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)

def run_all(g, stage, first=True):
    run(g, CFG, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "STAGE": stage, "HF_CACHE": f"{ROOT}/hf_cache", "CHUNK": 64})
    g["RES_PRESETS"]["up2.0"] = (61_440, 92_160)          # 시험 속도를 위해 해상도만 작게 (계산 경로는 같음)
    run(g, DATA_C, "3 데이터"); run(g, CSV, "4 CSV·분할")
    g["test_df"] = g["test_df"].head(200).copy(); g["sub_df"] = g["sub_df"].head(200).copy()
    run(g, PROMPT, "5 프롬프트")
    FV.STATE["answers"] = {g["build_mc_prompt"](r.question, [r.a, r.b, r.c, r.d]): "abcd".index(r.answer) for r in g["train_df"].itertuples()}
    if scen == "nopre": g["Image"] = _ImgProxy(g["Image"])
    run(g, IMG, "6 이미지"); run(g, DL, "7-0 다운로드"); run(g, LLMC, "7 vLLM"); run(g, SCORE, "8 채점"); run(g, REPORT, "9 비교 함수")
    run(g, QUICK, "10 ① 앞 300")
    if scen == "leftover":                                  # ① 뒤 사용자가 셀을 중간에 멈춰 요청 5개가 엔진에 남은 상황
        FV.STATE["leftover"] = [g["_make_req"](r)[0] for r in g["val_df"].tail(5).to_dict("records")]
    run(g, FULL, "11 ② 전체"); run(g, SUMM, "12 요약")

FV.STATE.update({"acc": 0.90 if scen == "gate" else 0.985, "mode": {"lowmass": "lowmass", "mismatch": "mismatch"}.get(scen, "normal")})
stage = {"auto": "auto", "leftover": "auto", "nofileno": "quick", "badta": "quick", "badta_inspect": "quick", "crash": "full", "crash2": "full", "gate": "auto", "quick": "quick", "lowmass": "auto", "mismatch": "quick", "nopre": "quick"}[scen]
import io
class _NoFilenoStream(io.TextIOBase):                   # ipykernel OutStream처럼 fileno가 없음 (Colab)
    def __init__(self, s): self._s = s
    def write(self, x): return self._s.write(x)
    def flush(self): return self._s.flush()
    def fileno(self): raise io.UnsupportedOperation("fileno")
if scen == "nofileno": sys.stdout = _NoFilenoStream(sys.stdout)
g = {"__name__": "__main__"}
err = None
if scen in ("crash", "crash2"):
    FV.STATE["crash_at"] = 18 if scen == "crash" else 8          # crash: test 두 번째 묶음 / crash2: 검증 나머지 도중
    try: run_all(g, stage)
    except RuntimeError as e: print("\n[예상된 중단]", e)
    else: raise AssertionError("중단이 일어나지 않음")
    RD = f"{RUNS}/q35_122b_a10b_gptq_zs"
    assert not os.path.exists(f"{RD}/preds_final_ep1__up2.0__1000.csv"), "중단됐는데 E14 파일이 생김"
    assert not os.path.exists(f"{RD}/preds_final_ep1_test__up2.0__200.csv"), "중단됐는데 E14 파일이 생김"
    FV.STATE["crash_at"] = None
    work = f"{RD}/preds_zeroshot_test__up2.0__200.csv" if scen == "crash" else f"{RD}/preds_zeroshot__up2.0__1000.csv"
    done_ids = set(pd.read_csv(work).id)
    pend = [i for i in (g["test_df"].id if scen == "crash" else g["val_df"].id) if i not in done_ids][0]
    with open(work, "a") as f: f.write(f"{pend},b,0.1,0.2,0.3,0.4,0.99")     # 저장 도중 끊긴 반쯤 쓴 줄 (아직 안 한 문항 · 줄바꿈 없음 · 뒤 열 없음)
    PARTIAL = pend
    if scen == "crash":
        run_all(g, stage)                                         # 같은 세션에서 다시 전체 실행 (모델 재사용)
        assert FV.STATE["n_init"] == 1, FV.STATE["n_init"]
    else:
        g = {"__name__": "__main__"}; run_all(g, stage)           # 세션이 죽은 경우: 새 프로세스처럼 처음부터
        assert FV.STATE["n_init"] == 2, FV.STATE["n_init"]
else:
    try: run_all(g, stage)
    except RuntimeError as e:
        err = str(e); print("\n[RuntimeError]", err[:300])
    except ValueError as e:
        err = "ValueError: " + str(e); print("\n[ValueError]", err[:300])

# ── 검사 ──
RD = f"{RUNS}/q35_122b_a10b_gptq_zs"
VW, TW = f"{RD}/preds_zeroshot__up2.0__1000.csv", f"{RD}/preds_zeroshot_test__up2.0__200.csv"
VF, TF = f"{RD}/preds_final_ep1__up2.0__1000.csv", f"{RD}/preds_final_ep1_test__up2.0__200.csv"
kw = FV.STATE["init_kw"]
print("\n━━━━ 검사 ━━━━")
if scen == "badta_inspect":
    assert err and err.startswith("ValueError") and "failed to be inspected" in err, err
    print("✅ badta_inspect: 1번 셀 차단만으로는 vLLM의 모델 검사 프로세스를 못 막음 → 실제 실행과 같은 오류 재현 (그래서 설치 셀에서 torchaudio를 지움)"); sys.exit()
if scen == "lowmass":
    assert err and "letter_mass" in err, err
    assert not os.path.exists(VF); print("✅ lowmass: 첫 묶음에서 멈추고 원인 출력"); sys.exit()

_nb_calls = [c for c in CALLED if c != os.environ["FAKE_PROC_DIR"]]          # (가짜 vLLM이 부른 것은 제외)
assert _nb_calls and all(c == "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" for c in _nb_calls), CALLED
assert DL_CALLS and DL_CALLS[0][0] == "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" and "*.safetensors" in DL_CALLS[0][1]["allow_patterns"]
assert kw["model"] == "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" and kw["quantization"] == "moe_wna16" and kw["max_logprobs"] == 64 and kw["limit_mm_per_prompt"] == {"image": 1, "video": 0}
assert kw["enforce_eager"] is True and kw["enable_prefix_caching"] is False and kw["max_num_seqs"] == 16 and kw["dtype"] == "bfloat16"
assert kw["gdn_prefill_backend"] == "triton" and os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
MIN_PX, MAX_PX = g["MIN_PX"], g["MAX_PX"]
exp_mm = {"min_pixels": 1024, "max_pixels": int(MAX_PX * 1.1)} if scen != "nopre" else {"min_pixels": MIN_PX, "max_pixels": MAX_PX}
assert kw["mm_processor_kwargs"] == exp_mm, (kw["mm_processor_kwargs"], exp_mm)
assert g["PRE_RESIZE"] is (scen != "nopre")

V = pd.read_csv(VW, on_bad_lines="skip").dropna(subset=["id", "p_a", "p_d", "prompt_len"]).drop_duplicates("id", keep="last")
nq = 300
_raw = pd.read_csv(VW, on_bad_lines="skip").dropna(subset=["prompt_len"])
assert _raw.id.is_unique, "검증 진행 파일에 중복 (완성된 줄 기준)"
P = V[["p_a", "p_b", "p_c", "p_d"]].to_numpy()
assert np.allclose(P.sum(1), 1) and (V.pred == [("abcd")[i] for i in P.argmax(1)]).all()
vd = g["val_df"]
if scen == "mismatch":
    assert (V.img_tokens != V.img_tokens_exp).all(); print("✅ mismatch: 모든 문항 불일치로 기록 · 경고 출력 · 멈추지 않음")
else:
    assert (V.img_tokens == V.img_tokens_exp).all(), V[V.img_tokens != V.img_tokens_exp].head()
# 확률을 가짜 vLLM 출력에서 독립 계산해 대조 (5문항)
lm = FV.LLM("x", **{k: v for k, v in kw.items() if k != "model"}); FV.STATE["n_init"] -= 1
for r in vd.head(5).itertuples():
    row = dict(r._asdict()); text = g["render"](row)
    _, (lpd,) = lm._answer(text, 64)
    ids = g["LETTER_IDS"]
    s = np.array([np.logaddexp.reduce([lpd[i].logprob for i in v if i in lpd]) if any(i in lpd for i in v) else -np.inf for v in ids])
    p = np.exp(s - s.max()); p /= p.sum()
    got = V.set_index("id").loc[r.id, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float)
    assert np.allclose(p, got, atol=1e-9), (r.id, p, got)
    assert abs(V.set_index("id").loc[r.id, "letter_mass"] - np.exp(s).sum()) < 1e-9
print("✅ 확률·letter_mass = 가짜 vLLM 출력에서 독립 계산한 값")
miss = V[V.n_missing > 0]
assert len(miss) > 0 and (miss.p_d == 0).all(), "글자 누락 경로가 시험되지 않음"
print(f"✅ 상위 k개에 없는 글자 → 확률 0 처리 ({len(miss)}문항)")
# 프롬프트: 노트북 render와 stage2 프롬프트 형식
t0 = g["render"](vd.iloc[0].to_dict())
assert t0.count("<|image_pad|>") == 1 and t0.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "(a) " in t0 and "정답을 반드시" in t0
assert t0.startswith("<|im_start|>system\nYou are a helpful visual question answering assistant.")
print("✅ 프롬프트 = stage2 baseline_v1 (시스템 지시 + 이미지 + 질문·보기 + 생각 끔)")

if scen == "badta":
    assert sys.modules.get("torchaudio", 0) is None and __import__("importlib").util.find_spec("torchaudio") is None
    print("✅ badta: 깨진 torchaudio가 있어도 차단 덕에 AutoProcessor·vLLM(멀티모달 모듈) 정상")
if scen in ("gate", "quick", "mismatch", "nopre", "nofileno", "badta"):
    assert len(V) == nq and list(V.id) == list(vd.id[:nq]), len(V)
    assert not os.path.exists(VF) and not os.path.exists(TF) and not os.path.exists(f"{RD}/train_log.json")
    assert g["GO_FULL"] is False and "val" not in g["SUMMARY"]
    acc = (V.merge(vd[["id", "answer"]], on="id").pipe(lambda d: (d.pred == d.answer).mean()))
    print(f"✅ {scen}: 앞 {nq}문항만 ({acc * 100:.2f}%) · E14 파일 없음 · GO_FULL=False"); sys.exit()

# 전체 (auto / crash / crash2)
T = pd.read_csv(TW, on_bad_lines="skip").dropna(subset=["id", "p_a", "p_d", "prompt_len"]).drop_duplicates("id", keep="last")
assert len(V) == 1000 and len(T) == 200
vf, tf = pd.read_csv(VF), pd.read_csv(TF)
assert list(vf.id) == list(vd.id) and list(tf.id) == list(g["test_df"].id)
for c in ["id", "pred", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens"]: assert c in vf.columns and c in tf.columns, c
log = json.load(open(f"{RD}/train_log.json"))
acc_v = float((vf.pred == vd.answer.values).mean())
assert log["config"]["MODEL_ID"] == "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" and log["config"]["TRAIN_ON_FULL"] is False and abs(log["epoch_acc"]["1"] - acc_v) < 1e-12
sub = pd.read_csv(f"{RD}/submission_q35_122b_a10b_gptq_zs_final_ep1_up2.0.csv")
assert list(sub.columns) == ["id", "answer"] and list(sub.id) == list(g["sub_df"].id) and (sub.answer.values == tf.pred.values).all()
rl = pd.read_csv(f"{RUNS}/results_log_s2.csv")
assert set(rl.run) >= {"q35_122b_a10b_gptq_zs/zeroshot_300", "q35_122b_a10b_gptq_zs/zeroshot_1000"}, rl.run.tolist()
print(f"✅ E14 파일·train_log·제출 파일 (검증 {acc_v * 100:.2f}%)")

# E14가 이 실행을 멤버로 잡는지: E8/E14 노트북의 읽기 함수를 그대로 실행
_n8 = {}; _argv = sys.argv; sys.argv = ["build_e8_ens.py", f"{SP}/_e8_tmp.ipynb"]
exec(open("/home/user/vqa/e8/build_e8_ens.py", encoding="utf-8").read(), _n8); sys.argv = _argv
e14src = open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read()
pool_fns = e14src[e14src.index("def _presets(run):"): e14src.index("MEMBERS_ALL, _skip")]
eg = {"os": os, "re": re, "json": json, "np": np, "pd": pd, "OUT_DIR": RUNS, "VAL_IDS": list(vd.id), "TEST_IDS": list(g["test_df"].id),
      "LETTERS": "abcd", "ENS_DIR": f"{RUNS}/ensembles", "Y_VAL": np.array(["abcd".index(a) for a in vd.answer])}
exec(_n8["HELPERS"], eg); exec(pool_fns, eg)
run_name = "q35_122b_a10b_gptq_zs"
assert eg["_presets"](run_name) == ["up2.0"], eg["_presets"](run_name)
assert eg["model_of"](run_name) == "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
fv, _ = eg["member_file"](run_name, "up2.0", "val"); ft, _ = eg["member_file"](run_name, "up2.0", "test")
PV, PT = eg["load"](run_name, "up2.0", "val"), eg["load"](run_name, "up2.0", "test")
assert os.path.basename(fv) == "preds_final_ep1__up2.0__1000.csv" and os.path.basename(ft) == "preds_final_ep1_test__up2.0__200.csv"
assert PV.shape == (1000, 4) and PT.shape == (200, 4) and np.allclose(PV, vf[["p_a", "p_b", "p_c", "p_d"]].to_numpy())
print("✅ E14 읽기 함수가 이 실행을 멤버로 찾음 (계열 = MODEL_ID, 검증 1000×4 · test 200×4)")

# 비교 셀: _dry·일부 문항 파일 제외, 수치 독립 계산
rows = g["cmp_rows"]; names = {r["run"] for r in rows}
assert "q35_27b_r16_native_dry" not in names and "hcx_partial" not in names and "zeroshot_flashnext_iq3xxs" not in names   # (300문항 파일은 1,000문항 비교에서 빠짐)
assert {"q35_27b_r16_native", "gemma4_31b_r16_up20", "q35_27b_r16_e5lr2e5_native"} <= names, names
o = pd.read_csv(f"{RUNS}/gemma4_31b_r16_up20/preds_final_ep1__up2.0__1000.csv").set_index("id").pred.loc[vd.id].to_numpy()
me, ans = vf.pred.to_numpy(), vd.answer.to_numpy()
fix, brk = int(((me == ans) & (o != ans)).sum()), int(((me != ans) & (o == ans)).sum())
rg = [r for r in rows if r["run"] == "gemma4_31b_r16_up20"][0]
assert (rg["fix"], rg["break"]) == (fix, brk) and abs(rg["acc"] - (o == ans).mean() * 100) < 1e-9
print(f"✅ 같은 문항 비교 수치 = 독립 계산 (Gemma: 고침 {fix} · 새로 틀림 {brk})")

if scen == "auto":
    shutil.rmtree(REF, ignore_errors=True); os.makedirs(REF)
    shutil.copy(VF, REF); shutil.copy(TF, REF); print("기준 결과 저장:", REF)
else:
    for f in [VF, TF]:
        a, b = pd.read_csv(f), pd.read_csv(f"{REF}/{os.path.basename(f)}")
        assert a.drop(columns=[]).equals(b), f"{os.path.basename(f)}: 중단 없이 돌린 결과와 다름"
    print(f"✅ {scen}: 중단 후 이어서 = 중단 없이 돌린 결과와 완전히 같음 (반쯤 쓴 줄 처리 포함)")
print(f"\n✅✅ 시나리오 {scen} 통과")
