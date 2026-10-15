# -*- coding: utf-8 -*-
"""e8_ensemble.ipynb 코드 셀을 순서대로 실행하는 시험 도구. 사용: python3 run_e8.py <scenario>"""
import sys, os, json, math
import numpy as np, pandas as pd
NB = "/home/user/vqa/e8/out/e8_ensemble.ipynb"
ROOT = "/tmp/vqa_scratch/e8test"
scen = sys.argv[1]
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, DATA, HELP, MEMT, REPRO, ADDC, FULL, NOISE, SUMM = cells
g = {"__name__": "__main__"}
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
run(CFG, "1 설정")
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": f"{ROOT}/vqa_runs", "EXPECTED_VAL": {"clean4_qg_hcx": 97.9}})
for n, src in [("2 데이터", DATA), ("3 함수", HELP), ("4 멤버 표", MEMT), ("5 재현", REPRO), ("5-1 멤버 추가", ADDC), ("6 전체 데이터", FULL), ("7 노이즈 후보", NOISE), ("8 요약", SUMM)]: run(src, n)

# ── 독립 계산으로 확인 ──
T = np.load(f"{ROOT}/truth.npz")
print("\n━━━━ [시험 도구 확인] ━━━━")
ok = True
if scen in ("full", "full_add"):
    w = np.array([1, 1, 1, 1, 2], float) / 6
    C4 = ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]
    use = {"q38_27b_r16_native", "hcx32b_r16_native"}                  # 정상인 전체 데이터 멤버만
    for tag in ("full", "mix"):
        P = sum(wi * ((T[f"PF_{r}"] if tag == "full" else (T[f"PT_{r}"] + T[f"PF_{r}"]) / 2) if r in use else T[f"PT_{r}"]) for wi, r in zip(w, C4))
        a = P.argmax(1)
        f = pd.read_csv(f"{ROOT}/vqa_runs/e8_full/submission_clean4_qg_hcx_{tag}.csv").answer.map("abcd".index).to_numpy()
        same = int((a == f).sum()); ok &= same == len(a)
        print(f"clean4_{tag}: 독립 계산과 같은 답 {same}/{len(a)} | 기존 clean4와 다른 답 {int((a != T['clean']).sum())}")
    exp_ok = use | ({"q3vl_32b_r16_native"} if scen == "full_add" else set())
    print("OK_FULL:", g["OK_FULL"]); ok &= set(g["OK_FULL"]) == exp_ok
    if scen == "full_add":      # clean4 + q3vl(가중치 1) 전체 데이터 버전 독립 계산
        w7 = np.array([1, 1, 1, 1, 2, 1], float) / 7; C7 = C4 + ["q3vl_32b_r16_native"]
        a7 = sum(wi * (T[f"PF_{r}"] if r in exp_ok else T[f"PT_{r}"]) for wi, r in zip(w7, C7)).argmax(1)
        f7 = pd.read_csv(f"{ROOT}/vqa_runs/e8_full/submission_clean4_plus_full.csv").answer.map("abcd".index).to_numpy()
        print(f"clean4_plus_full: 독립 계산과 같은 답 {int((a7 == f7).sum())}/{len(a7)}"); ok &= int((a7 == f7).sum()) == len(a7)
    ok &= os.path.exists(f"{ROOT}/vqa_runs/e8_full/submission_majority_vote_full.csv")
print("시험 도구 판정:", "통과" if ok else "실패")
