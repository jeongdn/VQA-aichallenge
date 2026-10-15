# -*- coding: utf-8 -*-
"""RunPod 122B 노트북을 실제 Jupyter 커널로 두 번 실행 (DRY_RUN → 새 커널에서 본 실행 + 올리기).
0번(nvidia-smi)·2번(pip) 셀만 시험용으로 바꾸고, 나머지 셀은 그대로. 가짜 HF 허브·작은 모델은 fakehf.py.
(run_runpod.py flow를 먼저 돌려 /workspace/data·가짜 허브의 zip이 있어야 함 → 여기서는 결과 폴더만 지우고 다시 만듦)"""
import os, sys, json, shutil, glob, nbformat, nbclient
import pandas as pd
NB = "/home/user/vqa/runpod/out/runpod_q35_122b_bf16.ipynb"
SP = "/tmp/vqa_scratch/runpod"
ROOT = f"{SP}/kernel"; os.makedirs(ROOT, exist_ok=True)
shutil.rmtree("/workspace/vqa_runs", ignore_errors=True)
shutil.rmtree("/workspace/data", ignore_errors=True)                    # 데이터도 다시 풀어 보기 (zip은 가짜 허브에서 다시 받음)
for f in glob.glob("/workspace/*.zip"): os.remove(f)
OVER = ('import sys; sys.path.insert(0, "/home/user/vqa/runpod"); import fakehf; fakehf.install()\n'
        'TRAIN_LIMIT, QUICK_EVAL_N, ZERO_SHOT_N, EVAL_EVERY, SAVE_EVERY, GRAD_ACCUM, MICRO_BS = 64, 12, 24, 2, 1, 4, 4\n'
        'RES_PRESETS = {"baseline384": (1024 * 2, 1024 * 4), "mid": (1024 * 3, 1024 * 6), "native": (1024 * 4, 1024 * 8), "up1.5": (1024 * 8, 1024 * 12), "up2.0": (1024 * 12, 1024 * 16)}\n'
        'ZERO_SHOT_MIN_ACC = 0.0\n')
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
    nb.cells[ci[0]].source = 'DRIVER_MAJOR = 580; import os; os.environ["HF_TOKEN"] = "hf_test"; print("(시험) 0번 셀 건너뜀")'
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
assert "✅ DRY RUN 통과" in lo and "▶ 합계" in lo, lo[-1500:]
print(lo[lo.index("✅ DRY RUN 통과"):][:900])
# ② 새 커널에서 본 실행 + 올리기
out = run(False)
up = out[-1]; assert "✅ 올리기 완료" in up, up
print("✅ 실제 커널 본 실행: 학습 → 검증 → test → 올리기\n" + up)
rd = "/workspace/vqa_runs/q35_122b_a10b_r16_native_bf16"
T = pd.read_csv(f"{rd}/preds_final_ep1_test__up2.0__6714.csv"); V = pd.read_csv(f"{rd}/preds_final_ep1__up2.0__1000.csv")
assert T.id.nunique() == 6714 and V.id.nunique() == 1000
print("✅ 실제 커널 시험 통과")
