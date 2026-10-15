# -*- coding: utf-8 -*-
import subprocess, shutil, os, re, glob, sys
ROOT = "/tmp/vqa_scratch/e7test"
def run(mode, scen, *over):
    for d in glob.glob(f"{ROOT}/vqa_runs/e7_yn_*"): shutil.rmtree(d)
    r = subprocess.run([sys.executable, "/home/user/vqa/e7/run_e7.py", mode, scen, *over], capture_output=True, text=True, timeout=1200)
    out = r.stdout + r.stderr
    verdict = next((l for l in out.splitlines() if l.startswith("판정:")), "")
    rd = glob.glob(f"{ROOT}/vqa_runs/e7_yn_*"); files = sorted(os.listdir(rd[0])) if rd else []
    calls = re.search(r"호출 건수 합계: (\d+)", out)
    return out, verdict, files, int(calls.group(1)) if calls else None
def has_test(f): return any(x.startswith("submission_") for x in f) and "yn_test__up2.0.csv" in f
def indep_ok(o):
    a = re.findall(r"다른 답 (\d+)개 / 6714", o); b = re.findall(r"독립 계산: \S+ 기존과 다른 답 (\d+)", o)
    return len(a) == 2 and a == b
CASES = [
    ("채택 (+4)",            "mock", "pass",   [], "판정: 채택", True,  lambda o, f, c: "고침 5 | 새로 틀림 1 | 순이득 +4" in o and indep_ok(o) and "탐색 표에서만" not in o),
    ("보류 (+2)",            "mock", "border", [], "판정: 보류", False, lambda o, f, c: "순이득 +2" in o and "test 채점 생략" in o),
    ("불채택 (−1)",          "mock", "fail",   [], "판정: 불채택", False, lambda o, f, c: "순이득 -1" in o),
    ("부정형 제외",           "mock", "neg",    [], "판정: 채택", True,  lambda o, f, c: "순이득 +4" in o and "(섞지 않음)" in o),
    ("부정형 포함",           "mock", "noexclude", ["EXCLUDE_NEG=False"], "판정: 채택", True, lambda o, f, c: "부정형 포함: " in o and "순이득 +5" in o),
    ("Yes/No 확률 낮음(경고)",  "mock", "lowmass_warn", [], "판정: 확인 필요", False, lambda o, f, c: "Yes/No 확률 합 평균 0.700" in o),
    ("Yes/No 확률 낮음(멈춤)",  "mock", "lowmass_stop", [], "", False, lambda o, f, c: "Yes/No 확률 합이 낮습니다" in o),
    ("이어하기 (50건 완료)",    "mock", "resume", [], "판정: 채택", True, lambda o, f, c: c == 24 + 186 + 1200 and "완료 50 | 남음 186" in o),
    ("PREFILL 다름",          "mock", "prefill_mismatch", [], "", False, lambda o, f, c: "섞이지 않도록 RUN_NAME을 바꾸세요" in o),
    ("기준 재현 실패",          "mock", "base_mismatch", ["BASE=dict(BASE, expected_val_acc=97.6)"], "", False, lambda o, f, c: "재현되지 않습니다" in o),
    ("RUN_TEST=False",       "mock", "pass",   ["RUN_TEST=False"], "판정: 채택", False, lambda o, f, c: "test 채점 생략 (RUN_TEST=False" in o),
    ("추가 채점 목록 (E7-b)",  "mock", "extra",  [], "판정: 보류", True,  lambda o, f, c: "추가 채점 목록 _extra/missing.csv: 검증 2 · test 5문항" in o and "순이득 +2" in o
                                                                          and "추가 문항 채점됨: 검증 3/3 · test 5/5" in o and c == 24 + 236 + 8 + 1200 + 20 and indep_ok(o)),
    ("작은 실모델 전체 경로",     "tiny", "tiny",   ["RUN_TEST=True"], "판정:", True, lambda o, f, c: "4bit 설정 전달: False" in o and "yn_val__up2.0.csv" in f and "전체 1,200건" in o),
    ("A100 40GB → 4bit",     "tiny", "tiny4",  ["MODEL_ID='google/gemma-4-31B-it'", "_VRAM=40"], "판정:", False, lambda o, f, c: "| 4bit(NF4) |" in o and "4bit 설정 전달: True" in o),
    ("20GB → 못 올림",        "tiny", "nofit",  ["MODEL_ID='google/gemma-4-31B-it'", "_VRAM=20"], "", False, lambda o, f, c: "4bit로도" in o),
]
ok_all = True
for name, mode, scen, over, vexp, want_test, extra in CASES:
    out, verdict, files, calls = run(mode, scen, *over)
    ok = (verdict.startswith(vexp) if vexp else verdict == "") and has_test(files) == want_test and bool(extra(out, files, calls))
    ok_all &= ok
    last = out.strip().splitlines()[-1][:100] if out.strip() else ""
    print(f"{'✅' if ok else '❌'} {name:<18} | {verdict[:80] or '(멈춤) ' + last} | test {'O' if has_test(files) else 'X'}")
    if not ok: print(out[-3000:])
print("\n전체:", "통과" if ok_all else "실패 있음")
