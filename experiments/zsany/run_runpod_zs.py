# -*- coding: utf-8 -*-
"""runpod_zs_any.ipynb 시험 (CPU): 가짜 vLLM + 실제 5.17 프로세서(작은 토크나이저) + 가짜 HF 저장소(업로드 기록).
사용: python3 run_runpod_zs.py <main|notest>"""
import sys, os, json, shutil, re, io, traceback, types
import numpy as np, pandas as pd
scen = sys.argv[1]
SP = "/tmp/vqa_scratch"
ZS = f"{SP}/zs122b"; DATA = f"{SP}/newfam/fx/content"; RUNS = f"{SP}/rpzstest/vqa_runs_{scen}"; REMOTE = f"{SP}/rpzstest/remote_{scen}"
NB = "/home/user/vqa/zsany/out/runpod_zs_any.ipynb"
os.makedirs("/workspace", exist_ok=True)
os.environ["FAKE_VLLM_API"] = f"{ZS}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{ZS}/tiny_proc"
sys.path.insert(0, "/home/user/vqa/e20/fake_vllm"); os.environ["PYTHONPATH"] = "/home/user/vqa/e20/fake_vllm"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface/"          # RunPod pod에 미리 정해진 값 흉내 → 노트북이 /workspace/hf로 바꿔야 함
import vllm as FV
import transformers
from transformers import AutoProcessor
_orig_fp = AutoProcessor.from_pretrained
transformers.AutoProcessor.from_pretrained = lambda mid, **kw: _orig_fp(os.environ["FAKE_PROC_DIR"])
import huggingface_hub
UP = []
def _fake_snapshot(repo_id, **kw):
    d = f"{ZS}/fake_snapshot"; os.makedirs(d, exist_ok=True)
    for i in range(1, 4): open(f"{d}/model-{i:05d}-of-00003.safetensors", "wb").write(b"0" * 10)
    open(f"{d}/config.json", "w").write("{}"); return d
class FakeApi:
    def __init__(self, token=None): self.token = token
    def model_info(self, mid, files_metadata=False):
        return types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=f"model-{i}.safetensors", size=20_000_000_000) for i in range(3)])
    def create_repo(self, repo, repo_type=None, private=None, exist_ok=None): assert repo_type == "dataset" and private is True
    def repo_info(self, repo, repo_type=None): return types.SimpleNamespace(private=True)
    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message=None, **kw):
        UP.append((repo_id, path_in_repo)); shutil.copytree(folder_path, os.path.join(REMOTE, path_in_repo), dirs_exist_ok=True)
    def list_repo_files(self, repo, repo_type=None):
        return [os.path.relpath(os.path.join(d, f), REMOTE).replace(os.sep, "/") for d, _, fs in os.walk(REMOTE) for f in fs]
huggingface_hub.snapshot_download = _fake_snapshot; huggingface_hub.HfApi = FakeApi
huggingface_hub.login = lambda **kw: None; huggingface_hub.whoami = lambda token=None: {"name": "tester"}
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL, DATA_C, CSV_C, PROMPT_C, IMG_C, DL_C, LLM_C, SCORE_C, VAL_C, TEST_C, UP_C, SUMM_C = cells
assert "hf_hub_download" in DATA_C and "upload_folder" in UP_C
shutil.rmtree(RUNS, ignore_errors=True); shutil.rmtree(REMOTE, ignore_errors=True); os.makedirs(RUNS); os.makedirs(REMOTE)
OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
def run(g, src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)
g = {"__name__": "__main__"}; ERR = None
sys.stdout = Tee()
try:
    run(g, CFG.replace('MODEL_ID  = "google/gemma-4-31B-it"', 'MODEL_ID  = "Qwen/Qwen3.6-35B-A3B"'), "1 설정")
    hf_env = (os.environ["HF_HOME"], os.environ["HF_HUB_CACHE"], os.environ["HF_XET_CACHE"])
    g.update({"DATA_DIR": DATA, "OUT_DIR": RUNS, "VAL_N": 1000, "CHUNK": 64, "HF_TOKEN": "hf_fake"})
    if scen == "notest": g["DO_TEST"] = False
    g["RES_PRESETS"]["up2.0"] = (61_440, 92_160)
    run(g, DATA_C, "3 데이터"); run(g, CSV_C, "4 CSV")
    g["test_df"] = g["test_df"].head(200).copy(); g["sub_df"] = g["sub_df"].head(200).copy()
    FV.STATE["answers"] = {}
    run(g, PROMPT_C, "5 프롬프트")
    FV.STATE["answers"] = {g["build_mc_prompt"](r.question, [r.a, r.b, r.c, r.d]): "abcd".index(r.answer) for r in g["train_df"].itertuples()}
    for s_, lab in [(IMG_C, "6"), (DL_C, "7-0"), (LLM_C, "7"), (SCORE_C, "8"), (VAL_C, "9"), (TEST_C, "10"), (UP_C, "11 올리기"), (SUMM_C, "12")]: run(g, s_, lab)
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()
print("\n══════ 판정 ══════")
ok(hf_env == ("/workspace/hf", "/workspace/hf/hub", "/workspace/hf/xet") and "pod에 미리 정해진 HF 캐시 경로" in out, f"HF 캐시 고정 {hf_env}")
if scen == "notest":
    ok(isinstance(ERR, AssertionError) and "test 예측 파일이 없습니다" in str(ERR), f"test 없이 올리려 하면 멈춤: {ERR}")
else:
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok("Hugging Face 로그인: tester | 비공개 저장소: tester/ssafy-vqa-private" in out, "HF 로그인 · 저장소 이름 규칙")
    ok("모델 가중치 60GB" in out, "받기 전 모델 크기 조회")
    ok(UP == [("tester/ssafy-vqa-private", "vqa_runs/zs_qwen3_6_35b_a3b_up20")], f"올리기 {UP}")
    rf = set(FakeApi().list_repo_files("x"))
    need = {f"vqa_runs/zs_qwen3_6_35b_a3b_up20/{f}" for f in ["preds_final_ep1__up2.0__1000.csv", "preds_final_ep1_test__up2.0__200.csv", "train_log.json", "zs_trainval__up2.0.csv"]}
    ok(need <= rf, f"저장소에 올라간 파일 (브리지 B 단계가 읽는 이름 포함)")
    ok("✅ 올리기 완료" in out, "올리기 완료 출력")
    ik = FV.STATE.get("init_kw", {})
    ok(ik.get("attention_config") == {"backend": "TRITON_ATTN"} and ik.get("enable_flashinfer_autotune") is False and ik.get("moe_backend") == "triton",
       f"B300 우회 설정이 LLM에 들어감: attention_config={ik.get('attention_config')} · autotune={ik.get('enable_flashinfer_autotune')} · moe={ik.get('moe_backend')}")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
