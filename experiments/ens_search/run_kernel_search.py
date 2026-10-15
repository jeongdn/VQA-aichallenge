# -*- coding: utf-8 -*-
"""ens_search.ipynb를 실제 Jupyter 커널로 처음부터 끝까지 실행 + 패널 조작 셀 끼워 넣기 + 위젯 Output 내용 확인."""
import os, shutil, glob, nbformat, nbclient
SP = "/tmp/vqa_scratch"; ROOT = f"{SP}/searchkernel"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
shutil.copytree(f"{SP}/e14test/vqa_runs", f"{ROOT}/vqa_runs"); os.symlink(f"{SP}/e14test/content", f"{ROOT}/content")
nb = nbformat.read("/home/user/vqa/ens_search/out/ens_search.ipynb", as_version=4)
ci = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]; c1 = nb.cells[ci[0]]
for old, new in [('DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"', 'DRIVE_ZIP    = "/nonexistent.zip"'),
                 ('DATA_DIR     = "/content"', f'DATA_DIR     = "{ROOT}/content"'), ('OUT_DIR      = "/content/drive/MyDrive/vqa_runs"', f'OUT_DIR      = "{ROOT}/vqa_runs"')]:
    assert c1.source.count(old) == 1; c1.source = c1.source.replace(old, new)
act = nbformat.v4.new_code_cell(r'''
_pk = [k for k in CANDS if k.startswith("좌표 상승")][0]
DD_C.value = _pk
B_SAVE.click()
print("PICKED", _pk, "| TX", TX.value, "| weights ok", all(abs(SL[r].value - CANDS[_pk]["weights"].get(r, 0.0)) < 1e-12 for r in SL), "| DD_F", DD_F.value)
_rf = f"기준 {REF_NAME} + _full {FULL_MODE}"; DD_C.value = _rf; B_SAVE.click(); print("PICKED2", _rf, "| DD_F", DD_F.value)
print("SAVED", sorted(os.path.basename(f) for f in __import__("glob").glob(os.path.join(SAVE_DIR, "submission_*.csv"))))
''')
nb.cells.insert(ci[9] + 1, act)
nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", store_widget_state=True, resources={"metadata": {"path": ROOT}}).execute()
nbformat.write(nb, f"{ROOT}/executed.ipynb")
def texts(c):
    t = []
    for o in c.get("outputs", []):
        if o.output_type == "stream": t.append(o.text)
        elif o.output_type == "error": t.append("ERROR " + o.ename + ": " + o.evalue)
        elif "text/plain" in o.get("data", {}): t.append(o.data["text/plain"])
    return "".join(t)
errs = [texts(c) for c in nb.cells if c.cell_type == "code" and any(o.output_type == "error" for o in c.get("outputs", []))]
assert not errs, errs
state = nb.metadata["widgets"]["application/vnd.jupyter.widget-state+json"]["state"]
otext = ["".join(o.get("text", "") for o in v["state"].get("outputs", [])) for v in state.values() if v["model_name"] == "OutputModel"]
assert any("▶ 슬라이더 조합" in t for t in otext) and any("✅ 저장" in t for t in otext), otext
assert any("_full 섞은 멤버 2개" in t for t in otext) and any("test = 원래 + _full 평균 (섞은 멤버 2개)" in t for t in otext), otext
_act = texts([c for c in nb.cells if c.cell_type == "code" and "PICKED2" in c.source][0]); assert "| DD_F mix" in _act and _act.count("DD_F mix") == 2, _act
_rep = texts([c for c in nb.cells if c.cell_type == "code"][13]); assert _rep.count("완전히 같음 ✅") == 3 and "(섞은 멤버 2개)" in _rep, _rep
cc = [c for c in nb.cells if c.cell_type == "code"]
print(texts(cc[10])); print("── 9번 셀 앞부분 ──"); print(texts(cc[8])[:1800])
print("── 13번 재현 ──"); print(texts(cc[13]))
print("✅ 실제 커널 실행 통과 · 저장:", sorted(os.path.basename(f) for f in glob.glob(f"{ROOT}/vqa_runs/ens_search/*")))
