# -*- coding: utf-8 -*-
"""zs_any_vllm.ipynb 시험 (CPU): 가짜 vLLM(실제 0.30.0 인자 목록 대조 + 실제 HF 프로세서) + 실제 transformers 5.17 Qwen3VLProcessor(작은 토크나이저).
사용: python3 run_zsany.py <v1000|extend|import|lowmass|default|crash|ko|prefix>"""
import sys, os, json, shutil, re, io, traceback
import numpy as np, pandas as pd

scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
ZS = f"{SP}/zs122b"; DATA = f"{SP}/newfam/fx/content"; RUNS = f"{SP}/zsanytest/vqa_runs_{scen}"
NB = "/home/user/vqa/zsany/out/zs_any_vllm.ipynb"
os.environ["FAKE_VLLM_API"] = f"{ZS}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{ZS}/tiny_proc"
sys.path.insert(0, "/home/user/vqa/e20/fake_vllm"); os.environ["PYTHONPATH"] = "/home/user/vqa/e20/fake_vllm"
import vllm as FV
import transformers
from transformers import AutoProcessor
_orig_fp = AutoProcessor.from_pretrained
CALLED = []
def _fp(mid, **kw):
    CALLED.append((mid, kw)); return _orig_fp(os.environ["FAKE_PROC_DIR"])
transformers.AutoProcessor.from_pretrained = _fp
import huggingface_hub
DL = []
def _fake_snapshot(repo_id, **kw):
    DL.append((repo_id, kw)); d = f"{ZS}/fake_snapshot"; os.makedirs(d, exist_ok=True)
    for i in range(1, 4): open(f"{d}/model-{i:05d}-of-00003.safetensors", "wb").write(b"0" * 10)
    open(f"{d}/config.json", "w").write("{}"); return d
huggingface_hub.snapshot_download = _fake_snapshot
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, CSV_C, PROMPT_C, IMG_C, DL_C, LLM_C, SCORE_C, VAL_C, TEST_C, SUMM_C = cells
MID = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" if scen == "import" else "Qwen/Qwen3.6-35B-A3B"
NT = 200

shutil.rmtree(RUNS, ignore_errors=True); os.makedirs(RUNS)
tr = pd.read_csv(f"{DATA}/train.csv"); tr["answer"] = tr.answer.astype(str).str.strip().str.lower()
te = pd.read_csv(f"{DATA}/test.csv")
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = tr.iloc[sorted(vi)].reset_index(drop=True)
if scen == "import":        # zs122b가 남긴 것과 같은 모양의 예전 실행 (검증 1,000 · test 200, 확률은 알아보기 쉬운 값)
    d = f"{RUNS}/q35_122b_a10b_gptq_zs"; os.makedirs(d)
    def fk(ids):
        return pd.DataFrame({"id": ids, "pred": "b", "p_a": 0.1, "p_b": 0.7, "p_c": 0.1, "p_d": 0.1, "letter_mass": 0.99, "img_tokens": 64, "img_tokens_exp": 64,
                             "n_missing": 0, "top_token": 74, "prompt_len": 300})
    fk(val.id).to_csv(f"{d}/preds_final_ep1__up2.0__1000.csv", index=False); fk(te.id[:NT]).to_csv(f"{d}/preds_final_ep1_test__up2.0__{NT}.csv", index=False)
    fk(val.id[:300]).to_csv(f"{d}/preds_zeroshot__up2.0__1000.csv", index=False)                      # 작업 파일(겹침) → 중복 없이 한 번만
    json.dump({"config": {"MODEL_ID": MID, "ZERO_SHOT": True, "EVAL_PRESET": "up2.0", "PROMPT_VERSION": "baseline_v1"}}, open(f"{d}/train_log.json", "w"))
    d2 = f"{RUNS}/q35_122b_other_preset"; os.makedirs(d2)                                                  # 해상도가 다른 실행 → 가져오면 안 됨
    fk(val.id).to_csv(f"{d2}/preds_final_ep1__native__1000.csv", index=False)
    json.dump({"config": {"MODEL_ID": MID, "ZERO_SHOT": True, "EVAL_PRESET": "native", "PROMPT_VERSION": "baseline_v1"}}, open(f"{d2}/train_log.json", "w"))

OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
def run(g, src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
def run_all(g, val_n):
    run(g, CFG.replace('MODEL_ID  = "google/gemma-4-31B-it"', f'MODEL_ID  = "{MID}"'), "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip", "HF_CACHE": f"{ZS}/hf_cache_zsany", "VAL_N": val_n, "CHUNK": 64})
    g["RES_PRESETS"]["up2.0"] = (61_440, 92_160)
    if scen == "default": g["PRESET"] = "default"; g["RUN_NAME"] = "zs_qwen3_6_35b_a3b_default"
    if scen == "ko": g["PROMPT_VERSION_USE"] = "ko_v1"; g["RUN_NAME"] = g["RUN_NAME"] + "_ko_v1"
    if scen == "prefix": g["ANSWER_PREFIX"] = "("
    run(g, DATA_C, "3 데이터"); run(g, CSV_C, "4 CSV·분할")
    g["test_df"] = g["test_df"].head(NT).copy(); g["sub_df"] = g["sub_df"].head(NT).copy()
    FV.STATE["answers"] = {g["build_mc_prompt"](r.question, [r.a, r.b, r.c, r.d]) if "build_mc_prompt" in g else None: 0 for r in []}
    for nm, s_, lab in [("PROMPT", PROMPT_C, "5 프롬프트"), ("IMG", IMG_C, "6 해상도"), ("DL", DL_C, "7-0 받기"), ("LLM", LLM_C, "7 vLLM"),
                        ("SCORE", SCORE_C, "8 채점 함수"), ("VAL", VAL_C, "9 검증"), ("TEST", TEST_C, "10 test"), ("SUMM", SUMM_C, "11 요약")]:
        if nm == "IMG":
            FV.STATE["answers"] = {g["build_mc_prompt"](r.question, [r.a, r.b, r.c, r.d]): "abcd".index(r.answer) for r in g["train_df"].itertuples()}
        run(g, s_, lab)

FV.STATE.update({"acc": 0.95, "mode": "lowmass" if scen == "lowmass" else "normal"})
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
sys.stdout = Tee(); ERR = None; g = {"__name__": "__main__"}
try:
    if scen == "extend":
        run_all(g, 1000); n1 = FV.STATE["n_prompts"]
        g = {"__name__": "__main__"}; run_all(g, 1300); n2 = FV.STATE["n_prompts"] - n1
        print(f"[시험 도구] 1차 채점 {n1} · 2차 채점 {n2}")
    elif scen == "crash":
        FV.STATE["crash_at"] = 6
        try: run_all(g, 1000)
        except RuntimeError as e: print("\n[예상된 중단]", e)
        FV.STATE["crash_at"] = None
        vw = [f for f in os.listdir(f"{RUNS}/zs_qwen3_6_35b_a3b_up20") if f.startswith("zs_trainval")][0]
        with open(f"{RUNS}/zs_qwen3_6_35b_a3b_up20/{vw}", "a") as f: f.write("train_9999.jpg,b,0.1,0.2")      # 반쯤 쓴 줄
        g = {"__name__": "__main__"}; run_all(g, 1000)
    else:
        run_all(g, 1000 if scen != "import" else 1100)
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()

print("\n══════ 판정 ══════")
RN = {"default": "zs_qwen3_6_35b_a3b_default", "ko": "zs_qwen3_6_35b_a3b_up20_ko_v1", "import": "zs_qwen3_5_122b_a10b_gptq_int4_up20"}.get(scen, "zs_qwen3_6_35b_a3b_up20")
RD = f"{RUNS}/{RN}"
if scen == "lowmass":
    ok(isinstance(ERR, RuntimeError) and "ANSWER_PREFIX" in str(ERR), f"letter_mass 낮음 → 멈추고 ANSWER_PREFIX 안내: {ERR}")
    ok("답 자리 최상위 토큰" in out, "1위 토큰 출력")
else:
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok(os.path.isdir(RD), f"결과 폴더 이름 {RN}")
    kw = FV.STATE["init_kw"]
    ok(kw["model"] == MID and kw["trust_remote_code"] is False and kw["max_logprobs"] == 64 and kw["dtype"] == "auto", f"vLLM 인자 (모델·dtype·logprobs) {kw.get('dtype')}")
    exp_mm = {} if scen == "default" else {"min_pixels": 61_440, "max_pixels": 92_160}
    ok(kw["mm_processor_kwargs"] == exp_mm, f"해상도 인자 {kw['mm_processor_kwargs']} (기대 {exp_mm})")
    if scen == "import": ok(kw["quantization"] == "moe_wna16" and kw["max_num_seqs"] == 16, "MODEL_PRESETS 적용 (122B: moe_wna16 · 16)")
    else: ok(kw["quantization"] is None and kw["max_num_seqs"] == 32, "기본값 (양자화 없음 · 32)")
    ok(DL and DL[0][0] == MID and "*.gguf" in DL[0][1]["ignore_patterns"], "모델 받기 (gguf 등 제외)")
    vw = pd.read_csv(f"{RD}/zs_trainval__{'default' if scen == 'default' else 'up2.0'}.csv", on_bad_lines="skip").dropna(subset=["src"])
    ok(vw.id.is_unique, "검증 작업 파일 중복 없음")
    pre = "default" if scen == "default" else "up2.0"
    vf, tf = f"{RD}/preds_final_ep1__{pre}__1000.csv", f"{RD}/preds_final_ep1_test__{pre}__{NT}.csv"
    ok(os.path.exists(vf) and os.path.exists(tf), "E14 형식 파일 (검증 1,000 · test)")
    if os.path.exists(vf):
        v = pd.read_csv(vf); ok(list(v.id) == list(val.id), "E14 검증 파일 = 팀 공통 1,000문항 (순서 포함)")
        acc = float((v.pred.values == val.answer.values).mean() * 100)
        ok(f"팀 공통 1,000문항 {acc:.2f}%" in out, f"출력의 1,000문항 정확도 = 파일로 계산한 값 ({acc:.2f}%)")
    log = json.load(open(f"{RD}/train_log.json"))
    ok(log["config"]["ZERO_SHOT"] is True and log["config"]["MODEL_ID"] == MID and log["config"]["EVAL_PRESET"] == pre and log["config"]["TRAIN_ON_FULL"] is False, "train_log (ZERO_SHOT · MODEL_ID · 해상도)")
    sub = pd.read_csv(f"{RD}/submission_{RN}.csv"); ok(len(sub) == NT and set(sub.answer) <= set("abcd"), "단독 제출 파일")
    # ens_search · E14의 멤버 탐색 함수(HELPERS)가 이 실행을 찾는지
    _n = {}
    exec(open("/home/user/vqa/e8/build_e8_ens.py", encoding="utf-8").read().split("HELPERS = r'''")[1].split("'''")[0], _n.update({"os": os, "re": re, "json": json, "np": np, "pd": pd,
          "OUT_DIR": RUNS, "TEST_IDS": list(te.id[:NT]), "VAL_IDS": list(val.id), "Y_VAL": val.answer.map("abcd".index).to_numpy(), "math": __import__("math"), "N_PUBLIC": 3357}) or _n)
    f1, w1 = _n["member_file"](RN, pre, "val"); f2, w2 = _n["member_file"](RN, pre, "test")
    ok(f1 == vf and f2 == tf, f"E14·ens_search 멤버 탐색이 이 실행을 찾음 ({w1}, {w2})")
    if scen == "extend":
        ok("[시험 도구] 1차 채점 1200 · 2차 채점 300" in out, "VAL_N 1000 → 1300: 새 300문항만 채점 (test는 다시 안 함)")
        ok(len(vw) == 1300, f"검증 작업 파일 {len(vw)}행")
    if scen == "import":
        ok(int(vw.src.str.startswith("import").sum()) == 1000 and len(vw) == 1100, f"검증: 예전 1,000 가져옴 + 새 100만 채점 ({len(vw)}행)")
        tw = pd.read_csv(f"{RD}/zs_test__up2.0.csv"); ok(int(tw.src.str.startswith("import").sum()) == NT, "test: 예전 결과 전부 가져옴 (채점 0)")
        ok("q35_122b_other_preset" not in out.split("가져옴")[-1] and out.count("가져옴: q35_122b_a10b_gptq_zs") == 2 + 0, f"해상도가 다른 실행은 안 가져옴 · 같은 실행 파일 2개만 (작업 파일 겹침은 0문항)")
        ok(FV.STATE["n_prompts"] == 100, f"실제 채점 {FV.STATE['n_prompts']}문항 (= 새 검증 100)")
    if scen == "crash":
        ok("완료 " in out and FV.STATE["n_init"] == 2, "중단 뒤 새 세션에서 이어하기")
        ok(len(vw) == 1000, f"반쯤 쓴 줄 무시 · 검증 {len(vw)}행")
    if scen == "ko":
        ok("당신은 이미지를 보고 객관식 질문에 답하는 도우미입니다" in out and log["config"]["PROMPT_VERSION"] == "ko_v1", "ko_v1: system 한 줄 한국어 · 기록")
    if scen == "prefix":
        ok(log["config"]["ANSWER_PREFIX"] == "(" and "assistant\\n<think>\\n\\n</think>\\n\\n('" in out, "ANSWER_PREFIX가 프롬프트 끝에 붙음 · 기록")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
