# -*- coding: utf-8 -*-
"""manual_ensemble.ipynb를 실제 Jupyter 커널(ipykernel)로 처음부터 끝까지 실행 (Colab과 같은 방식: 셀 순서대로, 위젯은 커널 comm).
6번 셀 뒤에 '사용자 조작' 셀을 끼워 슬라이더 값 바꾸기·버튼 누르기를 하고, 위젯 상태(Output 위젯 안 출력)까지 확인."""
import json, os, shutil, sys, copy, glob
import nbformat, nbclient

SP = "/tmp/vqa_scratch"
SRC = f"{SP}/e14test"; ROOT = f"{SP}/mankernel"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
shutil.copytree(f"{SRC}/vqa_runs", f"{ROOT}/vqa_runs"); os.symlink(f"{SRC}/content", f"{ROOT}/content")
nb = nbformat.read("/home/user/vqa/manual_ens/out/manual_ensemble.ipynb", as_version=4)
code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
c1 = nb.cells[code_idx[0]]
for old, new in [('DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"', 'DRIVE_ZIP    = "/nonexistent.zip"'),
                 ('DATA_DIR     = "/content"', f'DATA_DIR     = "{ROOT}/content"'),
                 ('OUT_DIR      = "/content/drive/MyDrive/vqa_runs"', f'OUT_DIR      = "{ROOT}/vqa_runs"')]:
    assert c1.source.count(old) == 1, old; c1.source = c1.source.replace(old, new)
act = nbformat.v4.new_code_cell(r'''
# (시험) 사용자가 패널을 조작한 것처럼
SL["q3vl_8b_r16_native"].value = 1.5
DD_M.value = "logprob"
B_CODE.click()
TX.value = "kernel test"
B_SAVE.click()
SL["hcx32b_r16_native_bf16"].value = 1.0
B_SAVE.click()                       # 검증만 있는 실행 → 저장 막힘
B_REF.click()
import glob as _g
print("SAVED", sorted(os.path.basename(f) for f in _g.glob(os.path.join(MAN_DIR, "submission_*.csv"))))
print("SLIDER_AFTER_REF", {r: s.value for r, s in SL.items() if s.value > 0}, DD_M.value)
''')
nb.cells.insert(code_idx[5] + 1, act)
client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", store_widget_state=True, resources={"metadata": {"path": ROOT}})
client.execute()
out_nb = f"{ROOT}/executed.ipynb"; nbformat.write(nb, out_nb)

def texts(cell):
    t = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream": t.append(o.text)
        elif o.output_type == "error": t.append("ERROR " + o.ename + ": " + o.evalue)
        elif "text/plain" in o.get("data", {}): t.append(o.data["text/plain"])
    return "".join(t)
errs = [texts(c) for c in nb.cells if c.cell_type == "code" and any(o.output_type == "error" for o in c.get("outputs", []))]
assert not errs, errs
# 위젯 상태: Output 위젯 두 개(MSG_UI, OUT_UI)의 출력
state = nb.metadata["widgets"]["application/vnd.jupyter.widget-state+json"]["state"]
outs = [v for v in state.values() if v["model_name"] == "OutputModel"]
otext = ["".join(o.get("text", "") for o in v["state"].get("outputs", [])) for v in outs]
print("Output 위젯 수:", len(outs))
for t in otext: print("── Output 위젯 내용 (앞 700자) ──\n" + t[:700])
panel_cell = nb.cells[code_idx[5]]
assert any("application/vnd.jupyter.widget-view+json" in o.get("data", {}) for o in panel_cell.outputs), "패널이 위젯으로 표시되지 않음"
assert any("▶ 슬라이더 조합" in t for t in otext), "패널 결과가 Output 위젯에 없음"
assert any("7번 셀에 그대로 붙여 넣으세요" in t or "test 예측이 없는 실행" in t or "✅ 저장" in t for t in otext)
act_txt = texts(nb.cells[code_idx[5] + 1]); print("── 조작 셀 출력 ──\n" + act_txt)
assert "submission_kernel_test_" in act_txt and "SLIDER_AFTER_REF" in act_txt and "'prob'" in act_txt.splitlines()[-1] or act_txt.strip().endswith("prob")
for i in code_idx[6:]:
    print(f"── 셀 {i} 출력 (앞 600자) ──\n" + texts(nb.cells[i])[:600])
print("\n✅ 실제 커널 실행 통과 · 저장 파일:", sorted(os.path.basename(f) for f in glob.glob(f"{ROOT}/vqa_runs/manual_ens/*")))
