# -*- coding: utf-8 -*-
"""e7_on_best.ipynb 코드 셀을 순서대로 실행하는 시험 도구 (fixture 복사본에서).
사용: python3 run_e7b.py <scenario>"""
import sys, os, json, shutil, io, contextlib, traceback
import numpy as np, pandas as pd
NB = "/home/user/vqa/e7b/out/e7_on_best.ipynb"
SRC = "/tmp/vqa_scratch/e7btest"
scen = sys.argv[1]
ROOT = SRC + "_" + scen
if os.path.exists(ROOT): shutil.rmtree(ROOT)
shutil.copytree(SRC, ROOT)
RUNS = f"{ROOT}/vqa_runs"
FX = json.load(open(f"{ROOT}/fixture.json"))
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, DATA, HELPERS, E7, BASE, VAL, TEST, SUMMARY = cells
g = {"__name__": "__main__"}
OUT = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): sys.__stdout__.write(s); OUT.write(s); return len(s)
def run(src, name):
    print(f"\n━━━━ [{name}] ━━━━", flush=True); exec(compile(src, f"<{name}>", "exec"), g)
fails = []
def ok(c, m):
    print(("✅ " if c else "❌ ") + m, flush=True)
    if not c: fails.append(m)

# ── 시나리오별 준비 ──
if scen == "mismatch":
    p = FX["ea_csv"].replace(SRC, ROOT); d = pd.read_csv(p); d.loc[5, "answer"] = "abcd"[("abcd".index(d.loc[5, "answer"]) + 1) % 4]; d.to_csv(p, index=False)
if scen == "dup":
    src = f"{RUNS}/ens_search/submission_전수_1위_확률_ea0e9f.json"; shutil.copy(src, f"{RUNS}/manual_ens/submission_other_ea0e9f.json")
if scen == "recdiff":
    p = f"{RUNS}/e7_yn_gemma_4_31b_it/e7_info.json"; j = json.load(open(p)); j["val_up2.0"]["primary"]["fixed"] = 9; json.dump(j, open(p, "w"))
if scen == "noref":
    shutil.rmtree(f"{RUNS}/hcx32b_r16_native")
if scen == "lowmass":
    for _t in ("val", "test"):
        p = f"{RUNS}/e7_yn_gemma_4_31b_it/yn_{_t}__up2.0.csv"; d = pd.read_csv(p); d["yn_mass"] = 0.3; d.to_csv(p, index=False)
if scen == "moved":
    p = f"{RUNS}/gemma4_31b_r16_up20_bf16/preds_final_ep1__up2.0__1000.csv"; open(p, "a").write("")   # 내용 같음 → md5 같음 (경고 없어야)

sys.stdout = Tee()
try:
    run(CFG, "1 설정")
    g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip"})
    if scen == "multi": g["BASES"] = ["ea0e9f", "c0ffee", "l0gpr0", "v0te00"]
    if scen == "jsonpath": g["BASES"] = ["ens_search/submission_전수_1위_확률_ea0e9f.json"]
    if scen == "badhash": g["BASES"] = ["zzzzzz"]
    if scen == "noe7": g["E7_RUN"] = "e7_nope"
    if scen == "onlyvote": g["BASES"] = ["v0te00"]
    run(DATA, "2 데이터"); run(HELPERS, "3 함수"); run(E7, "4 E7"); run(BASE, "5 조합"); run(VAL, "6 검증"); run(TEST, "7 test")
    if scen == "rerun": run(VAL, "6 검증 (다시)"); run(TEST, "7 test (다시)")
    run(SUMMARY, "8 요약")
    ERR = None
except Exception as e:
    ERR = e; traceback.print_exc(file=sys.stdout)
finally:
    sys.stdout = sys.__stdout__
out = OUT.getvalue()

print("\n══════ 판정 ══════")
S, T = FX["S"], FX["T"]
if scen in ("main", "jsonpath", "rerun", "multi", "moved"):
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok("= E7 노트북 기록과 같음 ✅" in out, "clean4 재현 확인 ✅")
    ok("고침 3 · 새로 틀림 1 · 순이득 +2 · 바뀐 답 4 = E7" in out, "clean4 재현 수치")
    V = g.get("VAL_RES", {}).get("ea0e9f", {})
    ok({k: V.get(k) for k in ("fixed", "broken", "net", "D", "mixed", "missing")} == {"fixed": 3, "broken": 1, "net": 2, "D": 4, "mixed": 14, "missing": 2}, f"ea0e9f 검증 결과 {V}")
    ok("공개 1회 확인 후보 (순이득 +2)" in out, "판정 문구")
    ok("검증 99.40% (JSON 기록 99.40%) ✅" in out and "완전히 같음 ✅" in out, "조합 재현 (검증·test)")
    cp = f"{RUNS}/e7_on_best/submission_e7_ea0e9f_t05_a025.csv"
    ok(os.path.exists(cp), "후보 파일 이름 submission_e7_ea0e9f_t05_a025.csv")
    if os.path.exists(cp):
        a = pd.read_csv(cp); exp = FX["exp_test"]
        ok(list(a.id) == FX["test_ids"] and [("abcd".index(x)) for x in a.answer] == exp, "후보 test 답 = 기대 (TC 30문항만 2위 보기로)")
        mj = json.load(open(cp[:-4] + ".json", encoding="utf-8"))
        ok(mj["test"]["changed"] == 30 and mj["test"]["uncertain"] == 95 and mj["test"]["mixed"] == 85 and mj["test"]["missing"] == 7, f"후보 JSON test 칸 {mj['test']}")
        ok(mj["base_json"] == "ens_search/submission_전수_1위_확률_ea0e9f.json" and mj["yn_test"] == "e7_yn_gemma_4_31b_it/yn_test__up2.0.csv", "후보 JSON 경로(OUT_DIR 기준)")
    ok("재현 ✅" in out, "후보 재현 점검 ✅")
    mp = f"{RUNS}/e7_on_best/e7_missing_ids.csv"
    if os.path.exists(mp):
        mdf = pd.read_csv(mp); mea = mdf[mdf.base == "ea0e9f"]
        ok(sorted(mea[mea.split == "val"].id) == sorted(FX["val_ids"][i] for i in S["Dm"]), "점수 없는 검증 문항 목록 = Dm")
        ok(sorted(mea[mea.split == "test"].id) == sorted(FX["test_ids"][i] for i in T["TM"]), "점수 없는 test 문항 목록 = TM (보기 3개만 채점된 것 포함)")
        ok(not mdf.duplicated().any(), "목록에 중복 행 없음")
    else: ok(False, "e7_missing_ids.csv 없음")
    ok("공개에서 +8문항 이상" in out, "공개 우연 폭 문구 (D=30 → ±8)")
    summ = out.split("━━━━ [8 요약] ━━━━")[-1]
    ok(summ.count("✅ 저장: e7_on_best/submission_e7_ea0e9f") == 1 and summ.count("▶ 전수 1위 · 확률 (ea0e9f)") == 1, "요약에 줄 중복 없음")
    if scen == "multi":
        ok("v0te00): 결합 방법 'vote'은 확률이 아니어서" in out, "다수결 조합 건너뜀")
        for t in ("c0ffee", "l0gpr0"):
            ok(os.path.exists(f"{RUNS}/e7_on_best/submission_e7_{t}_t05_a025.csv"), f"{t} 후보 저장")
            ok(t in g.get("VAL_RES", {}), f"{t} 검증 판정")
        _tsec = out.split("━━━━ [7 test] ━━━━")[-1].split("━━━━ [8 요약] ━━━━")[0]
        ok(_tsec.count("재현 ✅") == 3, "후보 3개 모두 재현 ✅")
        ok(out.count("완전히 같음 ✅") == 3, "조합 3개 모두 재현 (온도·로그확률·_full 포함)")
    if scen == "moved": ok("저장한 뒤 바뀐 예측 파일" not in out, "내용이 같으면 경고 없음")
elif scen == "mismatch":
    ok(isinstance(ERR, AssertionError) and "똑같이 다시 만들지 못했습니다" in str(ERR), f"제출 파일이 다르면 멈춤: {ERR}")
    ok("⚠️ 1개 다름" in out, "다른 문항 수 표시")
    ok(not os.path.exists(f"{RUNS}/e7_on_best/submission_e7_ea0e9f_t05_a025.csv"), "후보 파일 안 만듦")
elif scen == "dup":
    ok(isinstance(ERR, AssertionError) and "여러 개입니다" in str(ERR), f"같은 해시 JSON이 둘이면 멈춤: {ERR}")
elif scen == "badhash":
    ok(isinstance(ERR, AssertionError) and "'zzzzzz'로 끝나는 설정 JSON이" in str(ERR), f"없는 해시: {ERR}")
elif scen == "noe7":
    ok(isinstance(ERR, AssertionError) and "E7 결과 폴더가 없습니다" in str(ERR), f"E7 폴더 없음: {ERR}")
elif scen == "recdiff":
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok("⚠️ E7 노트북 기록(" in out, "기록과 다르면 경고")
elif scen == "noref":
    ok(ERR is None, f"끝까지 실행 (오류: {ERR!r})")
    ok("재현 확인 생략" in out and "재현 확인(clean4): 생략" in out, "clean4 멤버가 없으면 생략하고 계속")
elif scen == "lowmass":
    ok(isinstance(ERR, AssertionError) and "Yes/No 확률 합이 낮습니다" in str(ERR), f"확률 합 낮으면 멈춤: {ERR}")
elif scen == "onlyvote":
    ok(isinstance(ERR, AssertionError) and "E7을 얹을 조합이 없습니다" in str(ERR), f"다수결만 있으면 멈춤: {ERR}")
print("\n결과:", "✅ 모두 통과" if not fails else f"❌ 실패 {len(fails)}개 → {fails}")
sys.exit(1 if fails else 0)
