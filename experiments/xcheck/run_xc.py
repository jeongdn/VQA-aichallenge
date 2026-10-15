# -*- coding: utf-8 -*-
"""runpod_xcheck_any.ipynb 시험 (CPU): 가짜 vLLM(생성 모드) + 실제 5.17 프로세서(작은 토크나이저) + 가짜 HF(판정 목록 받기·결과 올리기).
사용: python3 run_xc.py <main|resume|listchange|nolist>"""
import sys, os, json, shutil, io, traceback, types, hashlib
import numpy as np, pandas as pd
scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
ZS = f"{SP}/zs122b"; DATA = f"{SP}/newfam/fx/content"; ROOT = f"{SP}/xctest_{scen}"; RUNS = f"{ROOT}/vqa_runs"; REMOTE = f"{ROOT}/remote"
NB = "/home/user/vqa/xcheck/out/runpod_xcheck_any.ipynb"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(RUNS); os.makedirs(REMOTE)
os.makedirs("/workspace", exist_ok=True)
os.environ["FAKE_VLLM_API"] = f"{ZS}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{ZS}/tiny_proc"
sys.path.insert(0, "/home/user/vqa/e20/fake_vllm"); os.environ["PYTHONPATH"] = "/home/user/vqa/e20/fake_vllm"
import vllm as FV
import transformers
from transformers import AutoProcessor
_orig_fp = AutoProcessor.from_pretrained
HCX_ID, HCX_REV = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", "a6cdfd3464d1b767259cad23e164eaf39d3e3960"
PROC_CALLS = []
def _fake_fp(mid, **kw):
    PROC_CALLS.append((mid, kw.get("revision"), kw.get("trust_remote_code")))
    p = _orig_fp(os.environ["FAKE_PROC_DIR"])
    if scen.startswith("hcx") and mid == HCX_ID:
        if kw.get("revision") is None:   # 실제 HF main: transformers 네이티브 형식 → 5.17이 모름
            raise ValueError("The checkpoint you are trying to load has model type `hyperclovax_vision_v2` but Transformers does not recognize this architecture.")
        assert kw.get("revision") == HCX_REV and kw.get("trust_remote_code") is True, kw
        p.chat_template = open(f"{SP}/hcx/chat_template.jinja", encoding="utf-8").read()   # HF에서 받은 HyperCLOVA X 템플릿
    return p
transformers.AutoProcessor.from_pretrained = _fake_fp
import huggingface_hub
UP, DL, INFO_KW = [], [], []
LIST_PATH = {"p": f"{ROOT}/list_v1.csv"}
SNAP_KW = []
def _fake_snapshot(repo_id, **kw):
    SNAP_KW.append((repo_id, kw.get("revision")))
    d = f"{ZS}/fake_snapshot"; os.makedirs(d, exist_ok=True)
    for i in range(1, 4): open(f"{d}/model-{i:05d}-of-00003.safetensors", "wb").write(b"0" * 10)
    open(f"{d}/config.json", "w").write("{}"); return d
def _fake_hf_dl(repo_id, filename, repo_type=None, token=None, force_download=False, local_dir=None, **kw):
    DL.append((repo_id, filename, repo_type, force_download))
    if scen == "nolist": raise FileNotFoundError("404 Client Error: Entry Not Found for url: .../xcheck/xcheck_targets.csv")
    return LIST_PATH["p"]
class FakeApi:
    def __init__(self, token=None): self.token = token
    def model_info(self, mid, files_metadata=False, **kw):
        INFO_KW.append((mid, kw.get("revision")))
        return types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=f"model-{i}.safetensors", size=20_000_000_000) for i in range(3)])
    def create_repo(self, repo, repo_type=None, private=None, exist_ok=None): assert repo_type == "dataset" and private is True
    def repo_info(self, repo, repo_type=None): return types.SimpleNamespace(private=True)
    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message=None, **kw):
        UP.append((repo_id, path_in_repo, commit_message)); shutil.copytree(folder_path, os.path.join(REMOTE, path_in_repo), dirs_exist_ok=True)
    def list_repo_files(self, repo, repo_type=None):
        return [os.path.relpath(os.path.join(d, f), REMOTE).replace(os.sep, "/") for d, _, fs in os.walk(REMOTE) for f in fs]
huggingface_hub.snapshot_download = _fake_snapshot; huggingface_hub.HfApi = FakeApi; huggingface_hub.hf_hub_download = _fake_hf_dl
huggingface_hub.login = lambda **kw: None; huggingface_hub.whoami = lambda token=None: {"name": "tester"}
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, CSV_C, PROMPT_C, TARGETS_C, IMG_C, DL_C, LLM_C, GEN_C, RUN_C, SAVE_C, UP_C, SUMM_C = cells
OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
def run(g, src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
L4 = "abcd"
def make_list(val_df, test_df, path, change=None):
    # 설계: 검증 대상 80 = G1(기준 오답·대안=정답) 30 + G2(기준 정답) 40 + G3(기준 오답·대안≠정답) 10 · test 대상 50
    rs = np.random.RandomState(7); vi = rs.permutation(len(val_df)); y = val_df.answer.map(L4.index).to_numpy()
    anc = y.copy(); alt = (y + 1) % 4; tgt = np.zeros(len(val_df), int)
    G = {"G1": vi[:30], "G2": vi[30:70], "G3": vi[70:80]}
    for i in G["G1"]: anc[i] = (y[i] + 2) % 4; alt[i] = y[i]; tgt[i] = 1
    for i in G["G2"]: alt[i] = (y[i] + 3) % 4; tgt[i] = 1
    for i in G["G3"]: anc[i] = (y[i] + 1) % 4; alt[i] = (y[i] + 2) % 4; tgt[i] = 1
    for i in vi[80:100]: anc[i] = (y[i] + 1) % 4                                   # 목록 밖 기준 오답 (판정 없음)
    ti = rs.permutation(len(test_df)); at = rs.randint(0, 4, len(test_df)); altt = (at + 1) % 4; tt = np.zeros(len(test_df), int); tt[ti[:50]] = 1
    extra = []
    if change:                                                                    # 목록 바뀜: G2 앞 5문항의 대안 변경 + 새 대상 3문항
        for i in G["G2"][:5]: alt[i] = (y[i] + 2) % 4
        for i in vi[100:103]: tgt[i] = 1; extra.append(i)
    d = pd.concat([pd.DataFrame({"id": val_df.id, "split": "val", "anchor": [L4[i] for i in anc], "alt": [L4[i] for i in alt], "n_alt": tgt * 2, "target": tgt}),
                   pd.DataFrame({"id": test_df.id, "split": "test", "anchor": [L4[i] for i in at], "alt": [L4[i] for i in altt], "n_alt": tt, "target": tt})], ignore_index=True)
    d["n_base"] = 5; d["list_tag"] = "lst002" if change else "lst001"; d["anchor_run"] = "q38flash_r16_native_bf16"
    d.to_csv(path, index=False)
    return G, y, anc, alt, extra, ti[:50], at
def key_of(r): return f"{r['question']}\n" + "\n".join(f"({L}) {r[L]}" for L in L4)
g = {"__name__": "__main__"}; ERR = None
sys.stdout = Tee()
try:
    _cfg = CFG.replace('MODEL_ID  = "google/gemma-4-31B-it"', 'MODEL_ID  = "' + (HCX_ID if scen.startswith("hcx") else "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4") + '"')
    if scen == "hcxfb":                   # 사용자가 MODEL_PRESETS의 커밋 고정을 지운 경우 → 5번 셀이 스스로 고정해야 함
        assert _cfg.count(', "REVISION": "' + HCX_REV + '"') == 1; _cfg = _cfg.replace(', "REVISION": "' + HCX_REV + '"', "")
    run(g, _cfg, "1 설정")
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "HF_TOKEN": "hf_fake", "GEN_CHUNK": 16})
    g["RES_PRESETS"]["up2.0"] = (61_440, 92_160)
    run(g, DATA_C, "3 데이터"); run(g, CSV_C, "4 CSV")
    if scen == "integ":
        LIST_PATH["p"] = os.environ["XC_LIST"]; G = {"G1": [], "G2": [], "G3": []}
    else:
        G, y, anc, alt, _, TT, at = make_list(g["val_df"], g["test_df"], LIST_PATH["p"])
    vd = g["val_df"]
    for i in G["G1"]: FV.STATE["xc_gold"][key_of(vd.iloc[i])] = L4[y[i]]
    for i in G["G2"][:30]: FV.STATE["xc_gold"][key_of(vd.iloc[i])] = L4[y[i]]
    for i in G["G2"][30:35]: FV.STATE["xc_override"][key_of(vd.iloc[i])] = L4[alt[i]]
    for i in G["G2"][35:40]: FV.STATE["xc_invalid"][key_of(vd.iloc[i])] = 3
    for i in G["G3"]: FV.STATE["xc_gold"][key_of(vd.iloc[i])] = L4[y[i]]
    if scen == "resume": FV.STATE["crash_at"] = 4
    for s_, lab in [(PROMPT_C, "5 프롬프트"), (TARGETS_C, "6 목록"), (IMG_C, "7"), (DL_C, "8"), (LLM_C, "9"), (GEN_C, "10"), (RUN_C, "11 판정"), (SAVE_C, "12 저장"), (UP_C, "13 올리기"), (SUMM_C, "14")]:
        try:
            run(g, s_, lab)
        except RuntimeError as e:
            if scen == "resume" and "가짜 중단" in str(e):
                print("→ (시험) 중단됨 → 10번 셀부터 다시 실행"); FV.STATE["crash_at"] = None
                for s2, lab2 in [(GEN_C, "10 재실행"), (RUN_C, "11 재실행")]: run(g, s2, lab2)
            else: raise
    if scen == "listchange":
        n_before = len(FV.STATE["gen_log"])
        LIST_PATH["p"] = f"{ROOT}/list_v2.csv"; _, _, _, alt2, extra, _, _ = make_list(g["val_df"], g["test_df"], LIST_PATH["p"], change=True)
        for s_, lab in [(TARGETS_C, "6 목록 v2"), (GEN_C, "10"), (RUN_C, "11 판정"), (SAVE_C, "12 저장")]: run(g, s_, lab)
        NEWGEN = FV.STATE["gen_log"][n_before:]
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()
print("\n══════ 판정 ══════")
RUN = "xc_hyperclovax_seed_think_32b_up20" if scen.startswith("hcx") else "xc_qwen3_5_122b_a10b_gptq_int4_up20"
if scen == "nolist":
    ok(isinstance(ERR, FileNotFoundError) and "Entry Not Found" in str(ERR), f"판정 목록이 없으면 모델 받기 전에 멈춤: {ERR!r}")
    ok(FV.STATE["n_init"] == 0, "모델 로드 전")
elif scen.startswith("hcx"):
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok(g.get("REVISION") == HCX_REV, f"저장소 커밋 고정 {g.get('REVISION')}")
    hc = [c for c in PROC_CALLS if c[0] == HCX_ID]
    ok(hc == ([(HCX_ID, None, True), (HCX_ID, HCX_REV, True)] if scen == "hcxfb" else [(HCX_ID, HCX_REV, True)]), f"프로세서 로드 {hc}")
    if scen == "hcxfb": ok("main 프로세서 로드 실패" in out and "a6cdfd3로 고정해 다시 시도" in out, "고정을 지워도 5번 셀이 원격 코드판으로 고정")
    ok(INFO_KW == [(HCX_ID, HCX_REV)] and SNAP_KW == [(HCX_ID, HCX_REV)], f"8번 셀 받기도 같은 커밋 {INFO_KW} {SNAP_KW}")
    ik = FV.STATE["init_kw"] or {}
    ok(ik.get("revision") == ik.get("tokenizer_revision") == ik.get("code_revision") == HCX_REV and ik.get("trust_remote_code") is True and ik.get("model") == HCX_ID,
       f"vLLM도 같은 커밋·원격 코드: {ik.get('revision')} / {ik.get('code_revision')} / {ik.get('trust_remote_code')}")
    ok(g["TEMPLATE_KW"] == {"skip_reasoning": True} and g["XC_TEMPLATE_KW"] == {"skip_reasoning": True}, f"생각 끔 인자 {g['XC_TEMPLATE_KW']}")
    px = g["_px"]
    ok(px.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and px.count("<|IMAGE_PAD|>") == 1 and px.count("<think>") == 1, "판정 프롬프트 끝 = 빈 생각 블록 · 이미지 자리 1개")
    ok(px.startswith("<|im_start|>system\n" + g["XC_SYSTEM"] + "\n<|im_end|>\n<|im_start|>user\n<|mime_start|>") and "'type'" not in px and '"type": "text"' not in px,
       "system 지시문이 글자로 들어감 (목록 그대로 찍히지 않음)")
    ok("[작전 지시: 오답 돌파 프로토콜]" in px and "정답: (a)" in px, "교차 검증 프롬프트 문구")
    ok(g["render"](g["VAL_USE"].iloc[0].to_dict()).endswith("</think>\n\n"), "채점용 render()도 생각 블록 닫힘")
    P = FV.STATE.get("prompts", [])
    ok(len(P) == 130 and all(p.endswith("</think>\n\n") and p.count("<|IMAGE_PAD|>") == 1 for p in P), f"vLLM에 들어간 프롬프트 {len(P)}개 모두 생각 닫힘·이미지 자리 1개")
    ok(all(x["n"] == 5 and x["max_tokens"] == 1200 for x in FV.STATE["gen_log"]) and not any(x["thinking"] for x in FV.STATE["gen_log"]), "표본 5개 · 1200토큰 · 생각 없음")
    V = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1__up2.0__1000.csv"); T = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1_test__up2.0__6714.csv")
    L = json.load(open(f"{RUNS}/{RUN}/train_log.json", encoding="utf-8"))["config"]
    ok(int(V.judged.sum()) == 75 and int(T.judged.sum()) == 50 and L["REVISION"] == HCX_REV and L["TEMPLATE_KW"] == {"skip_reasoning": True} and L["SYSTEM_AS_STRING"] is False,
       f"E14 저장 · train_log에 커밋·생각 끔 기록 (판정 검증 {int(V.judged.sum())} · test {int(T.judged.sum())})")
    ok("재현 ✅" in out and UP and UP[-1][1] == f"vqa_runs/{RUN}", "재현 점검 · 올리기")
    # 7번 셀: HCX 프로세서도 Qwen식 해상도 인자를 받음 (image_processor.merge_size)
    ok(g["FAMILY"] == "qwen" and g["MM_KW"] == {"min_pixels": 61_440, "max_pixels": 92_160} and ik.get("mm_processor_kwargs") == g["MM_KW"], f"해상도 인자 {g['MM_KW']}")
elif scen == "integ":
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    TGL = pd.read_csv(os.environ["XC_LIST"])
    V = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1__up2.0__1000.csv"); T = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1_test__up2.0__6714.csv")
    L = json.load(open(f"{RUNS}/{RUN}/train_log.json", encoding="utf-8"))["config"]
    ok(int(V.judged.sum()) == int(TGL[(TGL.split == "val")].target.sum()) and int(T.judged.sum()) == int(TGL[(TGL.split == "test")].target.sum())
       and L["TARGETS_TAG"] == str(TGL.list_tag.iloc[0]), f"anchor_rescue가 만든 목록으로 판정: 검증 {int(V.judged.sum())} · test {int(T.judged.sum())} · 태그 {L['TARGETS_TAG']}")
else:
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok(DL and DL[0] == ("tester/ssafy-vqa-private", "xcheck/xcheck_targets.csv", "dataset", True), f"판정 목록 받기 {DL[:1]}")
    ik = FV.STATE["init_kw"] or {}
    ok(ik.get("quantization") == "moe_wna16" and ik.get("attention_config") == {"backend": "TRITON_ATTN"} and ik.get("enable_flashinfer_autotune") is False and ik.get("max_num_seqs") == 64,
       f"122B 설정 + B300 우회 + 동시 64: q={ik.get('quantization')} attn={ik.get('attention_config')} seqs={ik.get('max_num_seqs')}")
    GL = FV.STATE["gen_log"]; vd = g["val_df"]; td = g["test_df"]
    tv_keys = {key_of(vd.iloc[i]) for i in np.concatenate([G["G1"], G["G2"], G["G3"]])}; tt_keys = {key_of(td.iloc[i]) for i in TT}
    first = GL if scen != "listchange" else GL[:len(GL) - len(NEWGEN)]
    keys = [x["key"] for x in first]
    if scen == "resume":
        ok(set(keys) == tv_keys | tt_keys, f"중단 후 이어서: 대상 {len(tv_keys | tt_keys)}문항 모두 판정 (요청 {len(keys)} — 중단 전 묶음은 다시 요청해도 저장은 한 번)")
    else:
        ok(sorted(keys) == sorted(tv_keys | tt_keys), f"대상 문항만 판정: 요청 {len(keys)} = 검증 80 + test 50")
    ok(all(x["n"] == 5 and x["max_tokens"] == 1200 for x in GL) and len({x["seed"] for x in first}) == len(set(keys)), "표본 5개 · 최대 1200토큰 · 문항별 시드")
    # A/B 자리: 기준·대안이 모두 들어가고, 자리는 문항마다 무작위
    pos = {key_of(vd.iloc[i]): (L4[anc[i]], L4[alt[i]]) for i in np.concatenate([G["G1"], G["G2"], G["G3"]])}
    pairs_ok = all({x["A"], x["B"]} == set(pos[x["key"]]) for x in first if x["key"] in pos)
    a_is_anchor = sum(x["A"] == pos[x["key"]][0] for x in first if x["key"] in pos)
    ok(pairs_ok and 20 <= a_is_anchor <= 60, f"A/B = {{기준, 대안}} · 기준이 A 자리인 문항 {a_is_anchor}/80 (무작위)")
    J = pd.read_json(f"{RUNS}/{RUN}/xc_val__up2.0.jsonl", lines=True)
    ok(len(J) == 80 * 5 + (8 * 5 if scen == "listchange" else 0) and not J.duplicated(["id", "A", "B", "sample"]).any(), f"검증 표본 {len(J)}개 · 중복 없음")
    V = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1__up2.0__1000.csv").set_index("id"); T = pd.read_csv(f"{RUNS}/{RUN}/preds_final_ep1_test__up2.0__6714.csv").set_index("id")
    ok(len(V) == 1000 and len(T) == 6714 and {"p_a", "p_b", "p_c", "p_d", "judged", "vote", "n_valid"} <= set(V.columns), "E14 파일: 검증 1,000 · test 6,714 (판정 열 포함)")
    ids = vd.id.to_numpy()
    non = np.setdiff1d(np.arange(1000), np.concatenate([G["G1"], G["G2"], G["G3"], np.array(extra if scen == "listchange" else [], int)]))
    ok((V.loc[ids[non]].pred.to_numpy() == np.array(list(L4))[anc[non]]).all() and (V.loc[ids[non]].judged == 0).all(), "목록 밖 문항 = 기준 답 · 판정 없음")
    vote = V.loc[ids].vote.fillna("").to_numpy()
    if scen != "listchange":
        exp = {"G1 → 정답(=대안)": (G["G1"], y), "G2 30 → 기준": (G["G2"][:30], anc), "G2 5 → 대안": (G["G2"][30:35], alt), "G3 → 정답(그 밖)": (G["G3"], y)}
        for k_, (idx, want) in exp.items(): ok(all(vote[i] == L4[want[i]] for i in idx), f"표: {k_} ({len(idx)}문항)")
        ok(all(vote[i] == "" for i in G["G2"][35:40]) and (V.loc[ids[G["G2"][35:40]]].n_valid == 2).all(), "표본 2개만 읽힌 5문항 → 기권 (기준 답)")
        ok("판정 정답 70 vs 같은 문항 기준 정답 35" in out and "기준과 다른 답 45문항: 맞게 바꿈 40 · 기준이 맞았는데 바꿈 5" in out, "검증 성적 줄 (설계값)")
        ok(g["decide"](np.array([2, 2, 0, 1])) is None and g["decide"](np.array([0, 0, 2, 0])) is None and g["decide"](np.array([0, 3, 1, 1])) == 1, "동점·표본 부족 → 기권")
        tv = T.loc[td.id.to_numpy()[TT]]
        ok((tv.judged == 1).all() and ((T.judged == 1).sum() == 50), "test 대상 50문항 판정 (목록 밖은 판정 없음)")
        L = json.load(open(f"{RUNS}/{RUN}/train_log.json", encoding="utf-8"))["config"]
        ok(L["XCHECK"] is True and L["MODEL_ID"] == "xcheck:Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" and L["XCHECK_ANCHOR"] == "q38flash_r16_native_bf16" and L["TARGETS_TAG"] == "lst001",
           f"train_log: XCHECK · 기준 · 목록 {L.get('TARGETS_TAG')}")
        ok("재현 ✅" in out, "재현 점검")
        ok(UP and UP[-1][1] == f"vqa_runs/{RUN}" and "xcheck" in UP[-1][2], f"올리기 {UP[-1][:2] if UP else None}")
        rf = set(FakeApi().list_repo_files("x"))
        ok({f"vqa_runs/{RUN}/{f}" for f in ["preds_final_ep1__up2.0__1000.csv", "preds_final_ep1_test__up2.0__6714.csv", "train_log.json"]} <= rf, "브리지 B 단계가 받는 파일")
    else:
        nk = [x["key"] for x in NEWGEN]
        want = {key_of(vd.iloc[i]) for i in list(G["G2"][:5]) + extra}
        ok(sorted(nk) == sorted(want), f"목록이 바뀐 5문항 + 새 대상 3문항만 새로 판정 (요청 {len(nk)})")
        ok(all(vote[i] == L4[y[i]] for i in G["G1"]), "바뀌지 않은 문항의 표는 그대로")
        L = json.load(open(f"{RUNS}/{RUN}/train_log.json", encoding="utf-8"))["config"]
        ok(L["TARGETS_TAG"] == "lst002", f"train_log 목록 태그 갱신 {L['TARGETS_TAG']}")
        ok((V.loc[ids[extra]].judged == 1).all(), "새 대상 판정됨")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
