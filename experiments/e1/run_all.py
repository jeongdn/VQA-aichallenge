# -*- coding: utf-8 -*-
"""시나리오별로 노트북을 끝까지 돌리고, 판정·test 실행 여부·파일·호출 수가 기대와 맞는지 확인한다."""
import subprocess, shutil, os, re, json, glob, sys

ROOT = "/tmp/vqa_scratch/e1test"
RUNS = f"{ROOT}/vqa_runs"
def clean():
    for d in glob.glob(f"{RUNS}/*_bf16eval"): shutil.rmtree(d)

def run(mode, scen, *over):
    clean()
    r = subprocess.run([sys.executable, "/home/user/vqa/e1/run_nb.py", mode, scen, *over], capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    verdict = next((l for l in out.splitlines() if l.startswith("판정:")), "")
    rd = glob.glob(f"{RUNS}/*_bf16eval")
    files = sorted(os.listdir(rd[0])) if rd else []
    calls = re.search(r"score_rows 호출 문항 수 합계: (\d+)", out)
    return out, verdict, files, int(calls.group(1)) if calls else None

T = "preds_final_ep1_test__up2.0__6714.csv"
CASES = [
    # (이름, mode, scen, over, 기대 판정 시작, test 파일 있어야 함, 추가 확인)
    ("채택(2√D 미만)",        "mock", "pass",       [], "판정: 채택", True, lambda o, f, c: "2√D 미만" in o and "다른 문항: 30개" in o),
    ("채택(강한 근거)",        "mock", "strong",     [], "판정: 채택", True, lambda o, f, c: "2√D 미만" not in o.split("판정:")[-1]),
    ("보류(+2)",             "mock", "border",     [], "판정: 보류", False, lambda o, f, c: "test 추론 생략" in o),
    ("불채택(−3)",            "mock", "fail",       [], "판정: 불채택", False, lambda o, f, c: "순이득 -3" in o),
    ("보류(NLL 증가)",        "mock", "nll_up",     [], "판정: 보류", False, lambda o, f, c: "NLL은 늘어남" in o),
    ("앙상블만 +3 이상",       "mock", "ens_only",   [], "판정: 보류", True, lambda o, f, c: "앙상블 교체 순이득 +4" in o and "순이득 +1" in o),
    ("앙상블 재현 실패",        "mock", "repro_fail", ["ENSEMBLE_REF=dict(ENSEMBLE_REF, expected_val_acc=97.7)"], "판정: 보류", False,
     lambda o, f, c: "재현이 안 됩니다" in o),
    ("이어하기(300 완료)",      "mock", "resume",     [], "판정: 채택", True, lambda o, f, c: c == 16 + 700 + 6714 and "완료 300 | 남음 700" in o),
    ("전체 데이터 실행",        "mock", "full",       ["SRC_RUN='q35_27b_r16_native_full'"], "판정: 검증 생략", True,
     lambda o, f, c: "검증 채점을 건너뜁니다" in o and not any(x.startswith("preds_final_ep1__") for x in f)),
    ("기준 파일 없음",          "mock", "noref",      ["SRC_RUN='q35_noref'"], "판정: 판정 불가", False, lambda o, f, c: "문항 단위 비교는 못 합니다" in o),
    ("RUN_TEST=False",       "mock", "pass",       ["RUN_TEST=False"], "판정: 채택", False, lambda o, f, c: "test 추론 생략 (RUN_TEST=False" in o),
    ("어댑터 효과 없음",        "mock", "noop",       [], "", False, lambda o, f, c: "어댑터를 켜고 꺼도 출력이 같습니다" in o),
    ("이미지 토큰 불일치",      "mock", "tokmismatch", [], "", False, lambda o, f, c: "이미지 토큰 수가 기존 결과와 다릅니다" in o),
    ("다른 어댑터 결과 폴더",    "mock", "md5",        [], "", False, lambda o, f, c: "다른 어댑터로 만든 결과가 있습니다" in o),
    ("MODEL_ID 불일치",       "mock", "modelid",    ["MODEL_ID='Qwen/Qwen3.8-27B'"], "", False, lambda o, f, c: "MODEL_ID를 'Qwen/Qwen3.5-27B'로 바꾸세요" in o),
    ("SRC_RUN 오타",          "mock", "typo",       ["SRC_RUN='q35_27b_r16_nativ'"], "", False, lambda o, f, c: "원본 실행 폴더가 없습니다" in o),
    ("VRAM 40GB",            "tiny", "vram",       ["_VRAM=40"], "", False, lambda o, f, c: "A100 80GB로 바꾸세요" in o),
]
ok_all = True
for name, mode, scen, over, vexp, want_test, extra in CASES:
    out, verdict, files, calls = run(mode, scen, *over)
    ok = verdict.startswith(vexp) if vexp else (verdict == "")
    ok &= (T in files) == want_test
    ok &= bool(extra(out, files, calls))
    ok_all &= ok
    print(f"{'✅' if ok else '❌'} {name:<16} | {verdict[:95] or '(멈춤) ' + (out.strip().splitlines()[-1][:95] if out.strip() else '')} | test {'O' if T in files else 'X'}")
    if not ok: print(out[-2500:])
print("\n전체:", "통과" if ok_all else "실패 있음")
