# -*- coding: utf-8 -*-
"""RunPod Flash-Next 노트북을 실제 Jupyter 커널로 두 번 실행 (DRY_RUN → 새 커널에서 본 실행 + 올리기).
0번(nvidia-smi)·2번(pip) 셀만 시험용으로 바꾸고, 나머지 셀은 그대로. 가짜 HF 허브·작은 모델은 fakehf.py.
(run_runpod.py flow를 먼저 돌려 /workspace/data·가짜 허브의 zip이 있어야 함 → 여기서는 결과 폴더만 지우고 다시 만듦)"""
import os, sys, json, shutil, glob, nbformat, nbclient
import pandas as pd
KM = os.environ.get("KMODEL", "flash")
NB = {"flash": "/home/user/vqa/runpod/out/runpod_flashnext_bf16.ipynb", "glm": "/home/user/vqa/runpod/out/runpod_glm53flash_fp8.ipynb", "397b": "/home/user/vqa/runpod/out/runpod_q35_397b_fp8.ipynb"}[KM]
KMID = {"flash": "Qwen/Qwen3.8-Flash-Next", "glm": "zai-org/GLM-5.3-Flash", "397b": "Qwen/Qwen3.5-397B-A17B-FP8"}[KM]
KRUN = {"flash": "q38flash_r16_native_bf16", "glm": "glm53flash_r16_native_fp8w", "397b": "q35_397b_a17b_r16_native_fp8w"}[KM]
os.environ["FAKE_MID"] = KMID
SP = "/tmp/vqa_scratch/runpod"
ROOT = f"{SP}/kernel_{KM}"; os.makedirs(ROOT, exist_ok=True)
shutil.rmtree("/workspace/vqa_runs", ignore_errors=True)
shutil.rmtree("/workspace/data", ignore_errors=True)                    # 데이터도 다시 풀어 보기 (zip은 가짜 허브에서 다시 받음)
for f in glob.glob("/workspace/*.zip"): os.remove(f)
OVER = ('import os, sys; os.environ[\"FAKE_MID\"] = \"' + KMID + '\"; sys.path.insert(0, \"/home/user/vqa/runpod\"); import fakehf2; fakehf2.install()\n'
        'TRAIN_LIMIT, QUICK_EVAL_N, ZERO_SHOT_N, EVAL_EVERY, SAVE_EVERY, GRAD_ACCUM, MICRO_BS = 64, 12, 24, 2, 1, 4, 4\n'
        'RES_PRESETS = {"baseline384": (1024 * 2, 1024 * 4), "mid": (1024 * 3, 1024 * 6), "native": (1024 * 4, 1024 * 8), "up1.5": (1024 * 8, 1024 * 12), "up2.0": (1024 * 12, 1024 * 16)}\n'
        'ZERO_SHOT_MIN_ACC = 0.0\n')
if os.environ.get("EMU_CUDNN") == "1":                                   # B300 cuDNN 어텐션 버그 흉내 (전부 가려진 행 → 역전파 NaN) → 7-0 공통 보정으로 통과해야 함
    OVER += 'import cudnn_nan_emu; cudnn_nan_emu.install(); print("(시험) cuDNN 버그 흉내 켬")\n'
def texts(c):
    t = []
    for o in c.get("outputs", []):
        if o.output_type == "stream": t.append(o.text)
        elif o.output_type == "error": t.append("ERROR " + o.ename + ": " + o.evalue)
        elif "text/plain" in o.get("data", {}): t.append(o.data["text/plain"])
    return "".join(t)
def run(dry):
    nb = nbformat.read(NB, as_version=4)
    ci = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    src0 = nb.cells[ci[0]].source
    spec = src0[src0.index("_SPECS = {"):src0.index("assert MODEL_ID in _SPECS")]
    nb.cells[ci[0]].source = (f'MODEL_ID = "{KMID}"\n' + spec + 'SPEC = _SPECS[MODEL_ID]; NGRAM_MODE = None; RAM_LIMIT_GIB = 175.0\n'
                              'DRIVER_MAJOR = 580; import os; os.environ["HF_TOKEN"] = "hf_test"; print("(시험) 0번 셀: 모델 선택만, 점검은 건너뜀")')
    nb.cells[ci[2]].source = 'print("(시험) 2번 셀 건너뜀 — 이 기계에는 이미 설치됨")'
    if not dry: nb.cells[ci[1]].source = nb.cells[ci[1]].source.replace("DRY_RUN    = True ", "DRY_RUN    = False", 1)
    assert ("DRY_RUN    = False" in nb.cells[ci[1]].source) == (not dry)
    nb.cells.insert(ci[1] + 1, nbformat.v4.new_code_cell(OVER))
    try:
        nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
    except nbclient.exceptions.CellExecutionError as e:
        nbformat.write(nb, f"{ROOT}/executed_{'dry' if dry else 'full'}.ipynb")
        if dry: raise
        print("본 실행 오류:", str(e)[:3000]); raise
    nbformat.write(nb, f"{ROOT}/executed_{'dry' if dry else 'full'}.ipynb")
    return [texts(c) for c in nb.cells if c.cell_type == "code"]
# ① DRY_RUN: 14번 셀은 '올리지 않습니다'로 멈춰야 함 → 그 셀만 오류 (마지막 셀)
try:
    run(True); raise SystemExit("DRY_RUN에서 14번 셀이 멈추지 않았습니다")
except nbclient.exceptions.CellExecutionError as e:
    assert "DRY_RUN 결과는 올리지 않습니다" in str(e), str(e)[:2000]
    print("✅ 실제 커널 DRY_RUN: 끝까지 실행 → 14번 셀이 DRY_RUN 결과 올리기를 거부 (의도대로)")
dry = nbformat.read(f"{ROOT}/executed_dry.ipynb", as_version=4)
lo = [texts(c) for c in dry.cells if c.cell_type == "code" and "for step in range(step0, total_steps)" in c.source][0]
assert "✅ DRY RUN 통과" in lo and "▶ 합계" in lo and "건너뜁니다" not in lo and "gnorm nan" not in lo, lo[-1500:]
print(lo[lo.index("✅ DRY RUN 통과"):][:900])
# ② 새 커널에서 본 실행 + 올리기
out = run(False)
up = out[-1]; assert "✅ 올리기 완료" in up, up
print("✅ 실제 커널 본 실행: 학습 → 검증 → test → 올리기\n" + up)
rd = f"/workspace/vqa_runs/{KRUN}"
T = pd.read_csv(f"{rd}/preds_final_ep1_test__up2.0__6714.csv"); V = pd.read_csv(f"{rd}/preds_final_ep1__up2.0__1000.csv")
assert T.id.nunique() == 6714 and V.id.nunique() == 1000
print("✅ 실제 커널 시험 통과")
