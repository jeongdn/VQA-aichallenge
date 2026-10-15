# -*- coding: utf-8 -*-
"""RunPod 노트북(Flash-Next / 122B 겸용) + Colab 브리지 노트북 시험 (CPU, 작은 qwen4_exp·qwen3_5_moe + 실제 Qwen3VLProcessor, 가짜 HF 허브).
흐름: 브리지 A(zip 올리기) → RunPod DRY_RUN → 게이트(zero-shot 기준 미달이면 멈춤) → 본 실행 중단(2스텝) → 이어서 끝까지 + 결과 올리기
      → 브리지 B(Drive로 받기·점검) → ens_search 1~5번 셀이 새 멤버로 찾는지
따로: 0번 셀(환경 점검) 시나리오 · 2번 셀(설치) 시나리오 — 가짜 nvidia-smi/pip
실제 /workspace 경로를 그대로 씀 (이 기계에 /workspace를 만들어 둠).
사용: python3 run_runpod2.py flash|122b [시나리오 ...]   (flow / env / install)"""
import sys, os, json, shutil, glob, re, io, contextlib, subprocess, fnmatch, types, time
import numpy as np, pandas as pd

SP = "/tmp/vqa_scratch/runpod"
VAR = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("flash", "122b", "glm", "397b") else "flash"
NB = {"flash": "/home/user/vqa/runpod/out/runpod_flashnext_bf16.ipynb", "122b": "/home/user/vqa/runpod/out/runpod_q35_122b_v2.ipynb",
      "glm": "/home/user/vqa/runpod/out/runpod_glm53flash_fp8.ipynb", "397b": "/home/user/vqa/runpod/out/runpod_q35_397b_fp8.ipynb"}[VAR]
BR = "/home/user/vqa/runpod/out/runpod_bridge_colab.ipynb"
ES = "/home/user/vqa/ens_search/out/ens_search.ipynb"
TINY = {"flash": f"{SP}/tinyflash", "122b": f"{SP}/tiny122b", "glm": f"{SP}/tinyglm/fp8", "397b": f"{SP}/tiny397b_fp8"}[VAR]
FX = "/tmp/vqa_scratch/newfam/fx/content"     # 실제 CSV + 작은 이미지
E14 = "/tmp/vqa_scratch/e14test"
WS = "/workspace"; FAKEHUB = f"{SP}/fakehub"; DRIVE = f"{SP}/drive"
MID = {"flash": "Qwen/Qwen3.8-Flash-Next", "122b": "Qwen/Qwen3.5-122B-A10B", "glm": "zai-org/GLM-5.3-Flash", "397b": "Qwen/Qwen3.5-397B-A17B-FP8"}[VAR]
RUN = {"flash": "q38flash_r16_native_bf16", "122b": "q35_122b_a10b_r16_native_bf16", "glm": "glm53flash_r16_native_fp8w", "397b": "q35_397b_a17b_r16_native_fp8w"}[VAR]
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
assert len(cells) == 17, len(cells)
ENV, CFG, INSTALL, DATA_C, WEIGHTS, IMPORTS, SPLIT, PROMPT, PATCH, MODEL, SCORING, PREP, LOOP, EVAL, TEST, SUMM, UPLOAD = cells
_gs = {}; exec(ENV[ENV.index("_SPECS = {"):ENV.index("assert MODEL_ID in _SPECS")], _gs); SPEC = _gs["_SPECS"][MID]
bcells = ["".join(c["source"]) for c in json.load(open(BR, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
B0, BA, BB, BC = bcells
fails = []
def ok(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg, flush=True)
    if not cond: fails.append(msg)

# ───────────────────── 가짜 HF 허브 ─────────────────────
import huggingface_hub as H
CALLS = []
def _rdir(repo_id, repo_type): return os.path.join(FAKEHUB, (repo_type or "model") + "s", repo_id)
class _Info(types.SimpleNamespace): pass
class FakeApi:
    def __init__(self, token=None): self.token = token
    def whoami(self, token=None): CALLS.append(("whoami",)); return {"name": "tester"}
    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False, **kw):
        d = _rdir(repo_id, repo_type); meta = os.path.join(d, ".meta.json")
        if os.path.exists(meta):
            if not exist_ok: raise RuntimeError("repo exists")
        else:
            os.makedirs(d, exist_ok=True); json.dump({"private": bool(private)}, open(meta, "w"))
        CALLS.append(("create_repo", repo_id, repo_type, private))
    def repo_info(self, repo_id, repo_type=None, **kw):
        return _Info(private=json.load(open(os.path.join(_rdir(repo_id, repo_type), ".meta.json")))["private"])
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None, **kw):
        dst = os.path.join(_rdir(repo_id, repo_type), path_in_repo); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(path_or_fileobj, dst); CALLS.append(("upload_file", repo_id, path_in_repo))
    def upload_folder(self, repo_id, folder_path, path_in_repo=None, repo_type=None, ignore_patterns=None, commit_message=None, **kw):
        n = 0
        for d, _, fs in os.walk(folder_path):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), folder_path).replace(os.sep, "/")
                if ignore_patterns and any(fnmatch.fnmatch(rel, p) for p in ignore_patterns): continue
                dst = os.path.join(_rdir(repo_id, repo_type), path_in_repo or "", rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(os.path.join(d, f), dst); n += 1
        CALLS.append(("upload_folder", repo_id, path_in_repo, n))
    def list_repo_files(self, repo_id, repo_type=None, **kw):
        root = _rdir(repo_id, repo_type)
        return [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, fs in os.walk(root) for f in fs if f != ".meta.json"]
    def get_paths_info(self, repo_id, paths, repo_type=None, **kw):
        root = _rdir(repo_id, repo_type)
        return [_Info(path=p, size=os.path.getsize(os.path.join(root, p))) for p in paths if os.path.isfile(os.path.join(root, p))]
    def delete_repo(self, repo_id, repo_type=None, **kw): shutil.rmtree(_rdir(repo_id, repo_type)); CALLS.append(("delete_repo", repo_id))
def fake_login(token=None, add_to_git_credential=False, **kw): CALLS.append(("login", token))
def fake_whoami(token=None, **kw): return {"name": "tester"}
def fake_hf_hub_download(repo_id, filename, repo_type=None, token=None, local_dir=None, **kw):
    src = os.path.join(_rdir(repo_id, repo_type), filename)
    if not os.path.isfile(src): raise FileNotFoundError(f"EntryNotFound {repo_id}/{filename}")
    os.makedirs(local_dir, exist_ok=True); dst = os.path.join(local_dir, filename); shutil.copy(src, dst)
    CALLS.append(("hf_hub_download", repo_id, filename)); return dst
def fake_snapshot_download(repo_id, repo_type=None, allow_patterns=None, local_dir=None, token=None, max_workers=None, **kw):
    if (repo_type in (None, "model")) and repo_id == MID:
        CALLS.append(("snapshot_model", repo_id)); return TINY                  # 가중치 = 작은 모델 폴더
    root = _rdir(repo_id, repo_type); n = 0
    for rel in FakeApi().list_repo_files(repo_id, repo_type):
        if allow_patterns and not any(fnmatch.fnmatch(rel, p) for p in allow_patterns): continue
        dst = os.path.join(local_dir, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(os.path.join(root, rel), dst); n += 1
    CALLS.append(("snapshot_dataset", repo_id, n)); return local_dir
H.HfApi, H.login, H.whoami, H.hf_hub_download, H.snapshot_download = FakeApi, fake_login, fake_whoami, fake_hf_hub_download, fake_snapshot_download

# ───────────────────── transformers: 122B 이름 → 작은 모델 폴더 ─────────────────────
import torch, transformers
_op, _om = transformers.AutoProcessor.from_pretrained, transformers.AutoModelForImageTextToText.from_pretrained
transformers.AutoProcessor.from_pretrained = lambda mid, *a, **k: _op(TINY if mid == MID else mid, *a, **k)
def _om_fake_gpu(path, *a, **k):
    _ia, _gc = torch.cuda.is_available, torch.cuda.get_device_capability
    torch.cuda.is_available = lambda: True; torch.cuda.get_device_capability = lambda *x, **y: (9, 0)
    try: return _om(path, *a, **k)
    finally: torch.cuda.is_available, torch.cuda.get_device_capability = _ia, _gc
transformers.AutoModelForImageTextToText.from_pretrained = (lambda mid, *a, **k: _om_fake_gpu(TINY, *a, **k) if mid == MID else _om(mid, *a, **k)) if VAR in ("glm", "397b") else \
    (lambda mid, *a, **k: _om(TINY if mid == MID else mid, *a, **k))

def run(g, src, name, quiet=False):
    buf = io.StringIO(); err = None
    try:
        with contextlib.redirect_stdout(buf): exec(compile(src, f"<{name}>", "exec"), g)
    except BaseException as e:
        err = e
    out = buf.getvalue(); g.setdefault("_OUT", {})[name] = out
    if not quiet: print(f"\n━━━━ [{name}] ━━━━\n" + (out if len(out) < 6000 else out[:2500] + "\n  …(생략)…\n" + out[-2500:]), flush=True)
    return err

TINY_OVER = {"TRAIN_LIMIT": 64, "QUICK_EVAL_N": 12, "ZERO_SHOT_N": 24, "EVAL_EVERY": 2, "SAVE_EVERY": 1, "GRAD_ACCUM": 4, "MICRO_BS": 4,
             "RES_PRESETS": {"baseline384": (1024 * 2, 1024 * 4), "mid": (1024 * 3, 1024 * 6), "native": (1024 * 4, 1024 * 8),
                             "up1.5": (1024 * 8, 1024 * 12), "up2.0": (1024 * 12, 1024 * 16)}}

def notebook_env(extra=None):
    g = {"__name__": "__main__", "MODEL_ID": MID, "SPEC": SPEC}
    assert run(g, CFG, "1 설정", quiet=True) is None
    g.update(TINY_OVER); g.update({"DRIVER_MAJOR": 580}); g.update(extra or {})
    return g

def flow():
    print("\n════════ 흐름: 브리지 A → DRY_RUN → 게이트 → 중단 → 이어서 끝까지 + 올리기 → 브리지 B → ens_search ════════")
    for d in (FAKEHUB, DRIVE, f"{WS}/data", f"{WS}/vqa_runs", f"{SP}/work"): shutil.rmtree(d, ignore_errors=True)
    for f in glob.glob(f"{WS}/*.zip"): os.remove(f)
    os.makedirs(DRIVE); os.makedirs(f"{SP}/work")
    # 가짜 Drive: 대회 zip(비밀번호) + E14 시험용 실행 폴더들
    r = subprocess.run(f"cd {FX} && zip -q -r -P YOUR_ZIP_PASSWORD {DRIVE}/ssafy-16-2-ai.zip train.csv test.csv sample_submission.csv dev.csv train test", shell=True)
    assert r.returncode == 0
    shutil.copytree(f"{E14}/vqa_runs", f"{DRIVE}/vqa_runs")
    os.environ["HF_TOKEN"] = "hf_test_token"
    os.chdir(f"{SP}/work")

    # ── 브리지 A ──
    gb = {"__name__": "__main__"}
    e = run(gb, B0.replace('DRIVE_ZIP   = "/content/drive/MyDrive/ssafy-16-2-ai.zip"', f'DRIVE_ZIP   = "{DRIVE}/ssafy-16-2-ai.zip"', 1)
                  .replace('OUT_DIR     = "/content/drive/MyDrive/vqa_runs"', f'OUT_DIR     = "{DRIVE}/vqa_runs"', 1), "브리지 0")
    ok(e is None and gb["HF_REPO"] == "tester/ssafy-vqa-private" and not gb["IN_COLAB"], f"브리지 0: 로그인·비공개 저장소 {gb.get('HF_REPO')} (오류 {e!r})")
    e = run(gb, BA, "브리지 A")
    zp = os.path.join(_rdir("tester/ssafy-vqa-private", "dataset"), "ssafy-16-2-ai.zip")
    ok(e is None and os.path.getsize(zp) == os.path.getsize(f"{DRIVE}/ssafy-16-2-ai.zip"), "브리지 A: zip이 비공개 저장소에 올라감")
    e = run(gb, BA, "브리지 A (다시)", quiet=True); ok(e is None and "이미 올라가 있음" in gb["_OUT"]["브리지 A (다시)"], "브리지 A 다시 실행 → 크기가 같아 건너뜀")
    # 공개 저장소면 멈춰야 함
    gpub = {"__name__": "__main__"}
    json.dump({"private": False}, open(os.path.join(_rdir("tester/public-x", "dataset") if os.makedirs(_rdir("tester/public-x", "dataset"), exist_ok=True) is None else "", ".meta.json"), "w"))
    e = run(gpub, B0.replace('HF_REPO     = ""', 'HF_REPO     = "tester/public-x"', 1).replace('DRIVE_ZIP   = "/content/drive/MyDrive/ssafy-16-2-ai.zip"', f'DRIVE_ZIP   = "{DRIVE}/ssafy-16-2-ai.zip"', 1), "브리지 0 (공개)", quiet=True)
    ok(isinstance(e, AssertionError) and "공개 저장소" in str(e), "공개 저장소면 멈춤")

    # ── RunPod DRY_RUN ──
    g = notebook_env()
    for nm, c in [("3 데이터", DATA_C), ("3-1 가중치", WEIGHTS), ("4 import", IMPORTS), ("5 분할", SPLIT), ("6 프롬프트", PROMPT), ("7-0 보정", PATCH), ("7 모델", MODEL),
                  ("8 채점", SCORING), ("9 학습 준비", PREP), ("10 학습 루프", LOOP), ("11 평가", EVAL), ("12 test", TEST), ("13 요약", SUMM)]:
        e = run(g, c, "dry " + nm, quiet=nm not in ("3 데이터", "10 학습 루프"))
        if e is not None: print(g["_OUT"]["dry " + nm][-3000:]); raise e
        if nm == "10 학습 루프": ET_AT_LOOP = {k: list(v) for k, v in g["EVAL_TIMES"].items()}   # 예상표를 만든 시점의 속도 기록 (뒤 셀이 더 쌓기 전)
    ok(os.path.exists(f"{WS}/ssafy-16-2-ai.zip") and os.path.exists(f"{WS}/data/train.csv") and g["DATA_DIR"] == f"{WS}/data", "데이터: HF에서 zip 받기 → 비밀번호로 풀기 (/workspace/data)")
    ok(("snapshot_model", MID) in CALLS and re.search(r"safetensors \d+개", g["_OUT"]["dry 3-1 가중치"]), "가중치 셀: snapshot_download → 폴더·크기 출력 (색인 파일 점검)")
    mo = g["_OUT"]["dry 7 모델"]
    if VAR == "flash":
        from transformers.models.qwen4_exp import modeling_qwen4_exp as _q4
        ok(g["IS_FLASH"] and getattr(_q4.Qwen4ExpTextQSAIndexer, "_fast_qsa", False) and "Flash-Next 보정 준비" in g["_OUT"]["dry 7-0 보정"], "7-0: qwen4_exp 감지 → QSA 인덱서 벡터화 설치")
        ok("n-gram 표 → CPU RAM" in mo and "n-gram 표 (20044, 8)" in mo and g["NGRAM_MODE"] == "cpu", "7: GPU 1장 모드(여기서는 CPU) — n-gram 표를 조각으로 채움")
        W = g["base"].model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight
        import flash_patch as _fp
        from safetensors import safe_open
        _sh = _fp.ngram_shards(TINY); _ref = []
        for k, fn, i, pfx in _sh:
            with safe_open(os.path.join(TINY, fn), "pt") as f: _ref.append(f.get_tensor(k))
        ok(torch.equal(W.detach().cpu(), torch.cat(_ref).to(W.dtype)), "n-gram 표 = 체크포인트 조각을 이어 붙인 것 (값 일치)")
        ok(re.search(r"LoRA가 붙은 층: (\d+)개 \(비전 타워 0개", mo) is not None, "LoRA: 비전 타워 0개")
    elif VAR == "glm":
        from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
        ok(g["IS_GLM"] and not g["IS_FLASH"] and "GLM-5.3-Flash 보정 준비" in g["_OUT"]["dry 7-0 보정"], "7-0: glm5_next 감지")
        ok(g["FAMILY"] == "glm5" and g["TEMPLATE_KW"] == {"reasoning_effort": "low"} and g["NO_DOUBLE_BOS"], f"7: 계열 glm5 · 템플릿 인자 {g['TEMPLATE_KW']} · [gMASK]<sop> 중복 방지 {g['NO_DOUBLE_BOS']}")
        m_ = re.search(r"GLM FP8 보정: FP8 선형 (\d+)개 · 전문가 (\d+)개", mo)
        ok(m_ is not None and int(m_.group(2)) == 3 and int(m_.group(1)) > 0, f"7: FP8 그대로 불러와 보정 설치 ({m_.group(0) if m_ else mo[-400:]})")
        ok(sum(p.dtype == torch.float8_e4m3fn for p in g["base"].parameters()) > 0, "기반 가중치가 FP8로 남음")
        ok("FP8 층 위 LoRA" in mo and re.search(r"LoRA가 붙은 층: (\d+)개 \(비전 타워 0개", mo) is not None, "LoRA: FP8 층 위 LoRA 다시 초기화 · 비전 0개")
        so = g["_OUT"]["dry 8 채점"]
        ok("✅ GLM 답 자리" in so and so.count("답 자리 후보") == 3 and g["ZS_TAG"].startswith("zeroshot_"), f"8: 답 자리 후보 3개 채점 → {g['GEN_STRIP']!r}+{g['GEN_SUFFIX']!r} · zero-shot 태그 {g['ZS_TAG']}")
        enc_ = g["encode_batch"](g["val_df"].head(2).to_dict("records"), [g["IDENTITY"]] * 2)
        ids0 = [i for i in enc_["input_ids"][0].tolist() if i != g["tok"].pad_token_id]
        gm = g["tok"].convert_tokens_to_ids("[gMASK]")
        ok(ids0[0] == gm and ids0.count(gm) == 1, "실제 입력: [gMASK]<sop>가 한 번만")
        im_ = g["load_image"](g["val_df"].path.iloc[0]); fp_ = g["fit_pixels"](im_, *g["GLM_PIX"])
        ok(int((enc_["input_ids"][0] == g["IMAGE_TOKEN_ID"]).sum()) == fp_.size[0] * fp_.size[1] // 784, f"이미지 토큰 = 28 배수로 맞춘 크기 {fp_.size} / 784")
    elif VAR == "397b":
        from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
        ok(g["IS_FP8"] and not g["IS_GLM"] and not g["IS_FLASH"] and "FP8 체크포인트" in g["_OUT"]["dry 7-0 보정"], "7-0: qwen3_5_moe FP8 체크포인트 감지")
        m_ = re.search(r"(?<!GLM )FP8 보정: FP8 선형 (\d+)개 · 전문가 (\d+)개", mo)
        ok(m_ is not None and (int(m_.group(1)), int(m_.group(2))) == (16, 4), f"7: FP8 그대로 불러와 보정 설치 ({m_.group(0) if m_ else mo[-400:]})")
        ok(sum(p.dtype == torch.float8_e4m3fn for p in g["base"].parameters()) > 0 and g["FAMILY"] == "qwen" and g["TEMPLATE_KW"] == {"enable_thinking": False}, "FP8 그대로 · Qwen 템플릿(enable_thinking=False)")
        ok("FP8 층 위 LoRA 16개" in mo and re.search(r"LoRA가 붙은 층: (\d+)개 \(비전 타워 0개", mo) is not None, "LoRA: FP8 층 위 16개 다시 초기화 · 비전 0개")
    else:
        ok(not g["IS_FLASH"] and not g["IS_FP8"] and "보정 없이" in g["_OUT"]["dry 7-0 보정"], "7-0: 122B는 보정 없음")
    rd = g["RUN_DIR"]; ok(rd.endswith("_dry") and g["split_md5"] == "df05e4bc19d27ebceeef2570672149c2", f"DRY 폴더 {os.path.basename(rd)} · 팀 공통 분할")
    lo = g["_OUT"]["dry 10 학습 루프"]
    ok("✅ DRY RUN 통과" in lo and "학습 357스텝" in lo and "test 6714문항 (up2.0)" in lo and "비용 약 $" in lo, "DRY_RUN 끝 예상표: 학습 357스텝(5,714문항 기준) · test 6714 · 비용")
    ok(len(g["STEP_TIMES"]) == 3 and ("up2.0", False) in g["EVAL_TIMES"] and ("native", False) in g["EVAL_TIMES"], f"속도 기록: 스텝 {len(g['STEP_TIMES'])}개 · 해상도 {sorted(g['EVAL_TIMES'])}")
    # 예상표 숫자 = 기록으로 다시 계산
    et = ET_AT_LOOP; spq_up, spq_nat = et[("up2.0", False)][0] / et[("up2.0", False)][1], et[("native", False)][0] / et[("native", False)][1]
    per = float(np.median(g["STEP_TIMES"][1:]))
    nq = len([s_ for s_ in range(1, 358) if s_ % g["EVAL_EVERY"] == 0 and s_ % 357 != 0])      # 시험 설정 EVAL_EVERY=2 → 178번 (실제 설정 100이면 3번)
    exp_tot = g["MODEL_LOAD_S"] + g["ZERO_SHOT_N"] * spq_up + 357 * per + nq * g["QUICK_EVAL_N"] * spq_nat + 1000 * spq_nat + 1000 * spq_up + 6714 * spq_up
    mt = re.search(r"▶ 합계 ([\d.]+)분", lo); ok(mt and abs(float(mt.group(1)) - exp_tot / 60) < 0.06, f"예상 합계 {mt.group(1) if mt else '?'}분 = 독립 계산 {exp_tot / 60:.2f}분 (빠른 검증 {nq}번)")
    ok(f"학습 중 빠른 검증 {nq}번" in lo, f"빠른 검증 횟수 {nq}번 (본 실행의 epoch 길이 357 기준)")
    ok(len([s_ for s_ in range(1, 358) if s_ % 100 == 0 and s_ % 357 != 0]) == 3, "실제 설정(EVAL_EVERY 100)이면 100·200·300스텝 3번")
    pl = open(os.path.join(rd, "progress.log"), encoding="utf-8").read()
    ok(pl.count("step ") >= 3 and "▶ zeroshot" in pl, "progress.log에 스텝·평가 기록")
    e = run(g, UPLOAD, "dry 14 올리기", quiet=True); ok(isinstance(e, AssertionError) and "DRY_RUN 결과는 올리지 않습니다" in str(e), "DRY_RUN 결과는 올리기 거부")
    n_zip_dl = sum(1 for c in CALLS if c[0] == "hf_hub_download")

    # ── 게이트: 본 실행에서 zero-shot < 기준 → 학습 전에 멈춤 ──
    g = notebook_env({"DRY_RUN": False, "ZERO_SHOT_MIN_ACC": 0.99})
    for nm, c in [("3 데이터", DATA_C), ("3-1 가중치", WEIGHTS), ("4 import", IMPORTS), ("5 분할", SPLIT), ("6 프롬프트", PROMPT), ("7-0 보정", PATCH), ("7 모델", MODEL),
                  ("8 채점", SCORING), ("9 학습 준비", PREP)]:
        e = run(g, c, "gate " + nm, quiet=True); assert e is None, (nm, e)
    ok(sum(1 for c in CALLS if c[0] == "hf_hub_download") == n_zip_dl and "데이터 OK" in g["_OUT"]["gate 3 데이터"], "두 번째 실행: 이미 풀린 데이터 → 다시 받지 않음")
    e = run(g, LOOP, "gate 10 학습 루프", quiet=True)
    ok(isinstance(e, SystemExit) and "학습 전에 멈춥니다" in str(e) and g["state"]["global_step"] == 0 and not os.path.exists(os.path.join(g["RUN_DIR"], "ckpt_last")),
       f"zero-shot {g['state']['history'][-1]['zero_shot_acc'] * 100:.1f}% < 99% → 학습 전에 멈춤, 체크포인트 없음")

    # ── 본 실행: 2스텝에서 중단 → 다시 실행하면 이어서 끝까지 ──
    g = notebook_env({"DRY_RUN": False, "ZERO_SHOT_MIN_ACC": 0.0})
    for nm, c in [("3 데이터", DATA_C), ("3-1 가중치", WEIGHTS), ("4 import", IMPORTS), ("5 분할", SPLIT), ("6 프롬프트", PROMPT), ("7-0 보정", PATCH), ("7 모델", MODEL),
                  ("8 채점", SCORING), ("9 학습 준비", PREP), ("10 학습 루프", LOOP.replace("_STOP_AT = None", "_STOP_AT = 2", 1)), ("11 평가", EVAL)]:
        e = run(g, c, "stop " + nm, quiet=True)
        if e is not None: break
    ok(isinstance(e, RuntimeError) and "학습이 아직 끝나지 않았습니다" in str(e) and g["state"]["global_step"] == 2, "2스텝에서 중단 → 11번 셀이 이어하기 안내")
    zs_first = pd.read_csv(glob.glob(os.path.join(g["RUN_DIR"], f"preds_{g['ZS_TAG']}__up2.0__24.csv"))[0])
    g = notebook_env({"DRY_RUN": False, "ZERO_SHOT_MIN_ACC": 0.0})
    t0 = time.time()
    for nm, c in [("3 데이터", DATA_C), ("3-1 가중치", WEIGHTS), ("4 import", IMPORTS), ("5 분할", SPLIT), ("6 프롬프트", PROMPT), ("7-0 보정", PATCH), ("7 모델", MODEL),
                  ("8 채점", SCORING), ("9 학습 준비", PREP), ("10 학습 루프", LOOP), ("11 평가", EVAL), ("12 test", TEST), ("13 요약", SUMM), ("14 올리기", UPLOAD)]:
        e = run(g, c, "full " + nm, quiet=nm not in ("10 학습 루프", "14 올리기"))
        if e is not None: print(g["_OUT"]["full " + nm][-3000:]); raise e
    print(f"(본 실행 {time.time() - t0:.0f}초)")
    rd = g["RUN_DIR"]
    ok(g["RESUMING"] and g["state"]["global_step"] == g["total_steps"] == 4, f"이어서 학습: {g['state']['global_step']}/{g['total_steps']} 스텝")
    ok(len(pd.read_csv(os.path.join(rd, f"preds_{g['ZS_TAG']}__up2.0__24.csv"))) == 24, "zero-shot은 다시 돌지 않음 (이어하기)")
    V = pd.read_csv(os.path.join(rd, "preds_final_ep1__up2.0__1000.csv")); T = pd.read_csv(os.path.join(rd, "preds_final_ep1_test__up2.0__6714.csv"))
    ok(V.id.nunique() == 1000 and T.id.nunique() == 6714 and set(V.id) == set(g["val_df"].id), "검증 1,000 · test 6,714 예측 (E14 파일 이름)")
    rep = f"tester/ssafy-vqa-private"; remote = set(FakeApi().list_repo_files(rep, "dataset"))
    pre = f"vqa_runs/{RUN}/"
    ok({pre + "preds_final_ep1__up2.0__1000.csv", pre + "preds_final_ep1_test__up2.0__6714.csv", pre + "train_log.json",
        pre + "adapter_ep1/adapter_config.json", pre + "results_log_s2_runpod.csv", pre + "progress.log"} <= remote, "올리기: 예측·로그·어댑터·결과 로그")
    ok(not any(f.startswith(pre + "ckpt_last") for f in remote), "체크포인트(ckpt_last)는 올리지 않음")
    ok(json.load(open(os.path.join(rd, "train_log.json")))["config"]["MODEL_ID"] == MID, f"train_log의 MODEL_ID = {MID} (E14 계열 이름)")
    # 업로드한 파일 = 로컬 파일
    same = all(open(os.path.join(rd, f), "rb").read() == open(os.path.join(_rdir(rep, "dataset"), pre + f), "rb").read()
               for f in ("preds_final_ep1__up2.0__1000.csv", "preds_final_ep1_test__up2.0__6714.csv"))
    ok(same, "저장소의 예측 파일 = pod의 파일 (바이트 단위)")

    # ── 브리지 B ──
    shutil.copy(f"{WS}/vqa_runs/valid_ids.csv", f"{DRIVE}/vqa_runs/valid_ids.csv")      # Drive에 팀 공통 분할 기록이 있을 때 대조되는지
    e = run(gb, BB, "브리지 B")
    dst = f"{DRIVE}/vqa_runs/{RUN}"
    ok(e is None and "검증 문항 = 팀 공통 분할 ✅" in gb["_OUT"]["브리지 B"] and os.path.exists(f"{dst}/preds_final_ep1_test__up2.0__6714.csv")
       and os.path.exists(f"{dst}/adapter_ep1/adapter_model.safetensors"), "브리지 B: Drive vqa_runs/<실행 이름>/ 에 복사 + 분할 대조")
    # 분할이 다르면 멈춰야 함
    pd.DataFrame({"id": list(pd.read_csv(f"{DRIVE}/vqa_runs/valid_ids.csv").id[:-1]) + ["train_9999"]}).to_csv(f"{DRIVE}/vqa_runs/valid_ids.csv", index=False)
    e = run(gb, BB, "브리지 B (분할 다름)", quiet=True); ok(isinstance(e, AssertionError) and "팀 공통 분할" in str(e), "분할이 다르면 멈춤")
    shutil.copy(f"{WS}/vqa_runs/valid_ids.csv", f"{DRIVE}/vqa_runs/valid_ids.csv")
    e = run(gb, BC, "브리지 C", quiet=True); ok(e is None and "아무것도 지우지 않았습니다" in gb["_OUT"]["브리지 C"] and os.path.isdir(_rdir(rep, "dataset")), "브리지 C: 기본값은 지우지 않음")

    # ── ens_search가 새 멤버로 찾는지 (1~5번 셀) ──
    es = ["".join(c["source"]) for c in json.load(open(ES, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ge = {"__name__": "__main__"}
    assert run(ge, es[0], "es 1", quiet=True) is None
    ge.update({"DATA_DIR": f"{E14}/content", "OUT_DIR": f"{DRIVE}/vqa_runs", "DRIVE_ZIP": "/nonexistent.zip", "MIN_VAL_ACC": 0.0})
    for i, nm in [(1, "es 2"), (2, "es 3"), (3, "es 4"), (4, "es 5")]:
        e = run(ge, es[i], nm, quiet=True)
        if e is not None: print(ge["_OUT"][nm][-3000:]); raise e
    ok(RUN in ge["MEMBERS"] and ge["FAMR"][RUN] == MID and ge["PRE"][RUN] == "up2.0", f"ens_search가 새 멤버로 찾음: 계열 {MID} · 해상도 up2.0")
    ok(np.allclose(ge["PTEST"][RUN], T.set_index("id").loc[ge["TEST_IDS"]][["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)), "ens_search의 test 확률 = pod가 만든 파일")

def env():
    print(f"\n════════ 0번 셀 (모델 선택 + 환경 점검) — {MID} ════════")
    real_run, real_du = subprocess.run, shutil.disk_usage
    DU = types.SimpleNamespace
    B300 = "NVIDIA B300 SXM6 AC, 275040, 580.95.05, 10.3"            # 268.6GiB
    B200 = "NVIDIA B200, 183359, 580.65.06, 10.0"
    MEM = lambda gb: f"MemTotal:       {int(2e12 / 1024)} kB\nMemAvailable:   {int(1.9e12 / 1024)} kB\n"   # 호스트 전체(2TB) — 컨테이너 한도는 cgroup
    def CG(limit_b, cur_b=3e9, file_b=1e9, v=2):
        if v == 2: return {"/sys/fs/cgroup/memory.max": f"{int(limit_b)}\n", "/sys/fs/cgroup/memory.current": f"{int(cur_b)}\n", "/sys/fs/cgroup/memory.stat": f"anon {int(cur_b - file_b)}\nfile {int(file_b)}\n"}
        return {"/sys/fs/cgroup/memory/memory.limit_in_bytes": f"{int(limit_b)}\n", "/sys/fs/cgroup/memory/memory.usage_in_bytes": f"{int(cur_b)}\n", "/sys/fs/cgroup/memory/memory.stat": f"cache {int(file_b)}\ntotal_cache {int(file_b)}\n"}
    blobs = f"{WS}/hf/hub/models--" + MID.replace("/", "--") + "/blobs"
    if VAR == "flash":
        cases = [  # 이름, nvidia-smi, 볼륨 남은, 컨테이너 남은, cgroup, 미리 받은 GB, 기대(None=통과: NGRAM_MODE 값 / 문자열=멈춤 문구)
            ("B300 1장 · RAM 188GB", B300, 700e9, 60e9, CG(188e9), 0, ("cpu", None)),
            ("B300 1장 · RAM 188GB · cgroup v1", B300, 700e9, 60e9, CG(188e9, v=1), 0, ("cpu", None)),
            ("B300 1장 · RAM 188GB · 다운로드 캐시 150GB", B300, 700e9, 60e9, CG(188e9, cur_b=155e9, file_b=150e9), 0, ("cpu", None)),
            ("B300 1장 · RAM 120GB (빠듯)", B300, 700e9, 60e9, CG(120e9), 0, ("cpu", "RAM이 빠듯")),
            ("B300 1장 · RAM 100GB", B300, 700e9, 60e9, CG(100e9), 0, "RAM 한도"),
            ("B300 2장", B300 + "\n" + B300, 900e9, 60e9, CG(376e9), 0, ("gpu", "GPU 2장 모드")),
            ("B200 2장 (358GiB)", B200 + "\n" + B200, 900e9, 60e9, CG(376e9), 0, "안 들어갑니다"),
            ("H100 1장", "NVIDIA H100 80GB HBM3, 81559, 580.65.06, 9.0", 700e9, 60e9, CG(188e9), 0, "안 들어갑니다"),
            ("드라이버 낮음", B300.replace("580.95.05", "570.124.06"), 700e9, 60e9, CG(188e9), 0, "드라이버"),
            ("볼륨 작음 (200GB)", B300, 200e9, 60e9, CG(188e9), 0, "Volume Disk"),
            ("볼륨 200GB · 가중치 300GB 이미 받음", B300, 200e9, 60e9, CG(188e9), 300, ("cpu", None)),
            ("컨테이너 작음", B300, 700e9, 5e9, CG(188e9), 0, "Container Disk")]
    elif VAR == "glm":
        H200 = "NVIDIA H200, 143771, 570.172.08, 9.0"
        cases = [("B300 2장", B300 + "\n" + B300, 700e9, 60e9, CG(1006e9), 0, (None, "FP8 그대로")),
                 ("H200 6장 · 드라이버 570(CUDA 12.8)", "\n".join([H200] * 6), 700e9, 60e9, CG(1506e9), 0, (None, "GPU 6장")),
                 ("B200 3장", "\n".join([B200] * 3), 700e9, 60e9, CG(849e9), 0, (None, "GPU 3장")),
                 ("B300 1장", B300, 700e9, 60e9, CG(188e9), 0, "안 들어갑니다"),
                 ("B200 2장 (358GiB)", B200 + "\n" + B200, 700e9, 60e9, CG(376e9), 0, "안 들어갑니다"),
                 ("B300 2장 · 드라이버 570", (B300 + "\n" + B300).replace("580.95.05", "570.172.08"), 700e9, 60e9, CG(1006e9), 0, "드라이버"),
                 ("볼륨 작음 (300GB)", B300 + "\n" + B300, 300e9, 60e9, CG(1006e9), 0, "Volume Disk")]
    elif VAR == "397b":
        H200 = "NVIDIA H200, 143771, 570.172.08, 9.0"
        cases = [("B300 2장", B300 + "\n" + B300, 700e9, 60e9, CG(1006e9), 0, (None, "GPU 2장")),
                 ("B200 3장", "\n".join([B200] * 3), 700e9, 60e9, CG(849e9), 0, (None, "GPU 3장")),
                 ("H200 4장", "\n".join([H200] * 4), 700e9, 60e9, CG(1006e9), 0, (None, "GPU 4장")),
                 ("B300 1장", B300, 700e9, 60e9, CG(188e9), 0, "안 들어갑니다"),
                 ("H200 3장 (421GiB)", "\n".join([H200] * 3), 700e9, 60e9, CG(1006e9), 0, "안 들어갑니다"),
                 ("볼륨 작음 (400GB)", B300 + "\n" + B300, 400e9, 60e9, CG(1006e9), 0, "Volume Disk")]
    else:
        cases = [("B300 1장", B300, 700e9, 60e9, CG(188e9), 0, (None, None)),
                 ("B200 2장", B200 + "\n" + B200, 700e9, 60e9, CG(376e9), 0, (None, "GPU 2장")),
                 ("H100 1장", "NVIDIA H100 80GB HBM3, 81559, 580.65.06, 9.0", 700e9, 60e9, CG(188e9), 0, "안 들어갑니다"),
                 ("볼륨 작음", B300, 100e9, 60e9, CG(188e9), 0, "Volume Disk")]
    real_open = open
    for name, smi, vol, cont, cg, pre_gb, expect in cases:
        def fake_run(cmd, *a, **k):
            if isinstance(cmd, list) and cmd[0] == "nvidia-smi": return subprocess.CompletedProcess(cmd, 0, smi + "\n", "")
            return real_run(cmd, *a, **k)
        files = {"/proc/meminfo": MEM(0), **cg}
        def fake_open(path, *a, **k):
            if str(path) in files: return io.StringIO(files[str(path)])
            if str(path).startswith("/sys/fs/cgroup"): raise FileNotFoundError(path)
            return real_open(path, *a, **k)
        shutil.rmtree(f"{WS}/hf/hub/models--" + MID.replace("/", "--"), ignore_errors=True)
        if pre_gb:
            os.makedirs(blobs, exist_ok=True)
            with real_open(f"{blobs}/part.incomplete", "wb") as f: f.truncate(int(pre_gb * 1e9))       # 희소 파일 (디스크를 쓰지 않음)
        subprocess.run = fake_run
        shutil.disk_usage = lambda p, vol=vol, cont=cont: DU(total=vol + 1e9, used=1e9, free=vol) if str(p).startswith("/workspace") else DU(total=cont + 1e9, used=1e9, free=cont)
        g = {"__name__": "__main__", "open": fake_open}
        e = run(g, ENV.replace('MODEL_ID = "Qwen/Qwen3.8-Flash-Next"', f'MODEL_ID = "{MID}"', 1), "0 환경 " + name, quiet=True)
        subprocess.run, shutil.disk_usage = real_run, real_du
        shutil.rmtree(f"{WS}/hf/hub/models--" + MID.replace("/", "--"), ignore_errors=True)
        out = g["_OUT"]["0 환경 " + name]
        if isinstance(expect, tuple):
            mode, phrase = expect
            ok(e is None and "✅ 환경 OK" in out and g.get("NGRAM_MODE") == mode and (phrase is None or phrase in out) and g["DRIVER_MAJOR"] in (570, 580),
               f"{name}: 통과 · NGRAM_MODE={g.get('NGRAM_MODE')} · RAM 한도 {g.get('RAM_LIMIT_GIB', 0):.0f}GiB 남음 {g.get('RAM_FREE_GIB', 0):.0f}GiB" + (f" · '{phrase}'" if phrase else "") + (f" (오류 {e!r})" if e else ""))
        else:
            ok(isinstance(e, SystemExit) and expect in str(e), f"{name}: 멈춤 — {str(e).splitlines()[1].strip()[:110] if isinstance(e, SystemExit) else repr(e)}")
    ok(os.environ.get("HF_HOME") == "/workspace/hf", "HF_HOME = /workspace/hf (가중치 캐시가 볼륨 디스크에)")
    # pod에 HF_HOME이 미리 정해져 있고(예: /workspace/.cache/huggingface) 그 캐시에 같은 모델이 이미 있는 경우 → 무시하고 /workspace/hf 하나만, 중복은 알려 줌
    alt = f"{WS}/.cache_test/huggingface"; ab = f"{alt}/hub/models--" + MID.replace("/", "--") + "/blobs"
    os.makedirs(ab, exist_ok=True)
    with real_open(f"{ab}/dup.bin", "wb") as f: f.truncate(int(337e9))
    smi = {"flash": B300, "glm": "\n".join([B200] * 3)}.get(VAR, B300); vol = 900e9
    def fake_run2(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[0] == "nvidia-smi": return subprocess.CompletedProcess(cmd, 0, smi + "\n", "")
        return real_run(cmd, *a, **k)
    files = {"/proc/meminfo": MEM(0), **CG(1006e9)}
    def fake_open2(path, *a, **k):
        if str(path) in files: return io.StringIO(files[str(path)])
        if str(path).startswith("/sys/fs/cgroup"): raise FileNotFoundError(path)
        return real_open(path, *a, **k)
    saved = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE")}
    os.environ["HF_HOME"] = alt; os.environ["HUGGINGFACE_HUB_CACHE"] = f"{alt}/hub"
    subprocess.run = fake_run2
    shutil.disk_usage = lambda p: DU(total=vol + 1e9, used=1e9, free=vol) if str(p).startswith("/workspace") else DU(total=61e9, used=1e9, free=60e9)
    g = {"__name__": "__main__", "open": fake_open2}
    try: e = run(g, ENV.replace('MODEL_ID = "Qwen/Qwen3.8-Flash-Next"', f'MODEL_ID = "{MID}"', 1), "0 환경 HF_HOME 미리 정해짐", quiet=True)
    finally: subprocess.run, shutil.disk_usage = real_run, real_du
    out = g["_OUT"]["0 환경 HF_HOME 미리 정해짐"]
    ok(e is None and os.environ["HF_HOME"] == "/workspace/hf" and os.environ["HF_HUB_CACHE"] == "/workspace/hf/hub" and os.environ["HF_XET_CACHE"] == "/workspace/hf/xet"
       and "HUGGINGFACE_HUB_CACHE" not in os.environ and "대신 /workspace/hf 하나만" in out and "다른 캐시에도 이 모델이 있습니다" in out and "(337GB)" in out,
       "pod에 HF_HOME·HUGGINGFACE_HUB_CACHE가 미리 정해져 있어도 /workspace/hf 하나만 씀 · 다른 캐시의 같은 모델(337GB)은 알려 줌" + (f" (오류 {e!r})" if e else ""))
    shutil.rmtree(f"{WS}/.cache_test", ignore_errors=True)
    for k, v in saved.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v

def install():
    print("\n════════ 2번 셀 (설치) ════════")
    real_run, real_which = subprocess.run, shutil.which
    cases = [("CUDA 12.8 torch · 드라이버 580", "12.8", 580, "cu130", True, None),
             ("CUDA 13.0 torch 이미 있음", "13.0", 580, None, True, None),
             ("드라이버 575", "12.4", 575, "cu129", False, None),
             ("import 실패", "13.0", 580, None, True, "import 점검 실패"),
             ("USE_FLA = False", "13.0", 580, None, True, None),
             ("H200 · CUDA 12.8 torch · 드라이버 570", "12.8", 570, None, True, None)]
    for name, cuda0, drv, want_idx, has7z, expect in cases:
        use_fla = name != "USE_FLA = False"
        state = {"cuda": cuda0, "cmds": []}
        def fake_run(cmd, *a, **k):
            if isinstance(cmd, list) and cmd[:2] == [sys.executable, "-c"]:
                code_ = cmd[2]
                if "get_device_capability" in code_:
                    return subprocess.CompletedProcess(cmd, 0, json.dumps({"v": "2.13.0+cu130" if state["cuda"] == "13.0" else "2.8.0+cu128", "cuda": state["cuda"], "ok": True, "cap": [9, 0] if name.startswith("H200") else [10, 3], "n": 1}) + "\n", "")
                if "torchvision" in code_ and "__version__" in code_ and "transformers" not in code_:
                    return subprocess.CompletedProcess(cmd, 0, "0.28.0\n", "")
                if expect == "import 점검 실패": return subprocess.CompletedProcess(cmd, 1, "", "ImportError: libcudart.so.12")
                return subprocess.CompletedProcess(cmd, 0, "torch 2.13.0 (CUDA 13.0) | transformers 5.17.0\n", "")
            state["cmds"].append(cmd)
            if isinstance(cmd, str) and "download.pytorch.org/whl/" in cmd: state["cuda"] = "13.0" if "cu130" in cmd else "12.9"
            return subprocess.CompletedProcess(cmd, 0, "", "")
        subprocess.run = fake_run; shutil.which = (lambda n: "/usr/bin/" + n) if has7z else (lambda n: None if n in ("7z", "unzip") else real_which(n))
        g = {"__name__": "__main__", "DRIVER_MAJOR": drv, "USE_FLA": use_fla}
        e = run(g, INSTALL, "2 설치 " + name, quiet=True)
        subprocess.run, shutil.which = real_run, real_which
        cm = [c for c in state["cmds"] if isinstance(c, str)]
        torch_cmd = [c for c in cm if "download.pytorch.org" in c]
        pin = open("/tmp/torch_pin.txt").read() if os.path.exists("/tmp/torch_pin.txt") else ""
        if expect:
            ok(isinstance(e, AssertionError) and expect in str(e), f"{name}: 멈춤 ({expect})")
            continue
        ok(e is None, f"{name}: 오류 없음 ({e!r})")
        ok((want_idx is None and not torch_cmd) or (want_idx and len(torch_cmd) == 1 and f"/whl/{want_idx}" in torch_cmd[0]), f"{name}: torch 설치 {'없음' if not torch_cmd else torch_cmd[0].split('whl/')[1]}")
        tf = [c for c in cm if "transformers==5.17.0" in c]
        ok(len(tf) == 1 and "-c /tmp/torch_pin.txt" in tf[0] and "peft==0.21.0" in tf[0] and (("flash-linear-attention==0.5.2" in tf[0]) == use_fla), f"{name}: 버전 고정 설치 + torch 고정 파일 ({pin.strip().splitlines()[0] if pin else '?'}) · fla {'설치' if use_fla else '빼고'}")
        ok(any("uninstall -y -q flash-linear-attention" in c for c in cm) == (not use_fla), f"{name}: fla {'유지' if use_fla else '제거'}")
        ok(any("uninstall -y -q torchaudio" in c for c in cm), f"{name}: torchaudio 제거")
        ok(any("apt-get install" in c and "p7zip-full" in c for c in cm) == (not has7z), f"{name}: 7z {'있음 → apt 안 씀' if has7z else '없음 → apt로 설치'}")

def nanguard():
    """B300 cuDNN 어텐션 버그 흉내(전부 가려진 질의 행 → 역전파 NaN, cudnn_nan_emu.py)로 DRY_RUN 10번 셀 실행 (flow를 먼저 돌려 데이터가 있어야 함)
    ① 7-0 공통 보정 있음 → NaN 없이 통과 (평가는 흉내 대상 아님, 학습 SDPA 호출은 흉내를 거침)
    ② 보정을 뺌 → 안전장치가 스텝마다 건너뛰고 3번째에 멈춤 · LoRA는 NaN 없이 그대로 (옛 노트북은 LoRA가 NaN이 되어 평가에서 멈췄음)"""
    print("\n════════ NaN: cuDNN 버그 흉내로 10번 셀 ════════")
    sys.path.insert(0, "/home/user/vqa/runpod"); import cudnn_nan_emu as EMU
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]; base_sdpa = getattr(base_sdpa, "_orig", base_sdpa)
    os.makedirs(f"{SP}/work", exist_ok=True); os.chdir(f"{SP}/work")
    for patched in (True, False):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = base_sdpa                      # 깨끗한 상태 → 7-0 셀이 보정 설치
        g = notebook_env()
        for nm, c in [("3 데이터", DATA_C), ("3-1 가중치", WEIGHTS), ("4 import", IMPORTS), ("5 분할", SPLIT), ("6 프롬프트", PROMPT), ("7-0 보정", PATCH), ("7 모델", MODEL),
                      ("8 채점", SCORING), ("9 학습 준비", PREP)]:
            e = run(g, c, f"nan{int(patched)} " + nm, quiet=True); assert e is None, (nm, e, g["_OUT"][f"nan{int(patched)} " + nm][-2000:])
        ok(getattr(ALL_ATTENTION_FUNCTIONS["sdpa"], "_unmask_rows", False) and "공통 보정" in g["_OUT"][f"nan{int(patched)} 7-0 보정"], "7-0: 공통 보정(SDPA 전부 가려진 행 풀기) 설치")
        if not patched: ALL_ATTENTION_FUNCTIONS["sdpa"] = base_sdpa      # ② 보정을 뺀 옛 상태
        EMU.install(); EMU.CALLS.update(n=0, poisoned=0)
        try: e = run(g, LOOP, f"nan{int(patched)} 10 학습 루프", quiet=True)
        finally: EMU.uninstall()
        lo = g["_OUT"][f"nan{int(patched)} 10 학습 루프"]
        lora = [p_ for n_, p_ in g["model"].named_parameters() if "lora_" in n_]
        lora_ok = len(lora) > 0 and all(bool(torch.isfinite(p_).all()) for p_ in lora)
        gn = [float(x) for x in re.findall(r"gnorm ([\w.]+)", lo) if x not in ("nan", "inf")]
        if patched:
            ok(e is None and "✅ DRY RUN 통과" in lo and "건너뜁니다" not in lo and EMU.CALLS["n"] > 0 and EMU.CALLS["poisoned"] == 0 and lora_ok and len(gn) == 3,
               f"① 버그 흉내 + 보정: 3스텝 모두 gnorm 유한 {[round(x, 2) for x in gn]} · 흉내 SDPA 호출 {EMU.CALLS['n']}번 중 전부 가려진 행 0 · DRY RUN 통과 (오류 {e!r})")
        else:
            ok(isinstance(e, FloatingPointError) and lo.count("건너뜁니다") == 3 and g["NAN_SKIPS"] == 3 and EMU.CALLS["poisoned"] > 0 and lora_ok
               and sum(1 for h in g["state"]["history"] if h.get("nan_skip")) == 3,
               f"② 버그 흉내 + 보정 없음: 스텝마다 NaN → 건너뜀 3번 → 멈춤 · LoRA {len(lora)}개 NaN 없음 (전부 가려진 행이 있던 호출 {EMU.CALLS['poisoned']}번, 오류 {type(e).__name__})")
    ALL_ATTENTION_FUNCTIONS["sdpa"] = base_sdpa

if __name__ == "__main__":
    scns = [a for a in sys.argv[1:] if a not in ("flash", "122b", "glm", "397b")] or ["env", "install", "flow"]
    for s_ in scns: {"flow": flow, "env": env, "install": install, "nanguard": nanguard}[s_]()
    print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
    sys.exit(1 if fails else 0)
