import json, sys
NB="/home/user/vqa/newfam/out/stage2_newfam.ipynb"
SP="/tmp/vqa_scratch/newfam"
cells=["".join(c["source"]) for c in json.load(open(NB,encoding="utf-8"))["cells"] if c["cell_type"]=="code"]
CFG, INSTALL, DATA_C, IMPORTS, SPLIT, PROMPT, MODEL, SCORING = cells[:8]
g={"__name__":"__main__"}
exec(CFG,g)
g.update({"DATA_DIR":f"{SP}/fx/content","OUT_DIR":f"{SP}/fx/vqa_runs","DRIVE_ZIP":"/x.zip","MODEL_ID":f"{SP}/tiny_muse","RUN_NAME":"t_diag",
          "RES_PRESETS":{"baseline384":(784*2,784*4),"mid":(784*3,784*6),"native":(784*4,784*8),"up1.5":(784*8,784*12),"up2.0":(784*12,784*16)}})
for c in [DATA_C, IMPORTS, SPLIT, PROMPT, MODEL, SCORING]: exec(c,g)
exec(open("/home/user/vqa/newfam/muse_diag_cell.py",encoding="utf-8").read(), g)
print("진단 셀 실행 완료")
