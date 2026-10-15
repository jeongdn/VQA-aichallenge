# -*- coding: utf-8 -*-
"""설치 셀(2번) 논리 시험: !pip 줄은 기록하고(지우기는 흉내), 깨진 torchaudio·vllm 누락을 흉내 냄.
점검 프로세스는 vLLM의 모델 검사 프로세스처럼 '차단 없는 새 프로세스'에서 vllm.model_executor.models.qwen3_5를 import함.
사용: python3 test_install.py <a|c|d|e>
 a: 깨진 torchaudio(CUDA 불일치) 있음 → 지움 → 점검 OK
 c: torchaudio 없음 → 지우기 없음 → 점검 OK
 d: 점검 프로세스에서 vllm import 실패 → RuntimeError + 오류 출력
 e: 대조군 — 깨진 torchaudio가 지워지지 않고 남음 → 점검 실패 + 다시 지우라는 안내"""
import sys, os, json, re, tempfile
scen = sys.argv[1]
SP = "/tmp/vqa_scratch/zs122b"
NB = "/home/user/vqa/zs122b/out/zs122b_vllm.ipynb"
BAD = "/tmp/vqa_scratch/badta"   # import하면 CUDA 불일치 RuntimeError를 내는 가짜 torchaudio (+ 배포 정보)
FAKE = "/home/user/vqa/zs122b/fake_vllm"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, INSTALL = cells[0], cells[1]
src = "\n".join((m.group(1) + "_PIP.append(" + repr(m.group(2)) + ")") if (m := re.match(r"^(\s*)!(pip .*)$", l)) else l for l in INSTALL.split("\n"))

class PipLog(list):
    def append(self, cmd):
        super().append(cmd)
        if "uninstall" in cmd and "torchaudio" in cmd and scen == "a":     # 지우기 흉내: 이 프로세스·점검 프로세스 모두에서 사라짐
            if BAD in sys.path: sys.path.remove(BAD)
            os.environ["PYTHONPATH"] = FAKE

_bin = tempfile.mkdtemp(); open(f"{_bin}/nvidia-smi", "w").write("#!/bin/sh\necho '| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.0 |'\n")
os.chmod(f"{_bin}/nvidia-smi", 0o755); os.environ["PATH"] = _bin + ":" + os.environ["PATH"]      # 가짜 nvidia-smi (CUDA 13 드라이버)
os.environ["FAKE_VLLM_API"] = f"{SP}/vllm_api.json"; os.environ["FAKE_PROC_DIR"] = f"{SP}/tiny_proc"
if scen in ("a", "e"):
    sys.path.insert(0, BAD)                                   # 설치 셀의 _installed("torchaudio")가 찾도록
    os.environ["PYTHONPATH"] = f"{BAD}:{FAKE}"
else:
    os.environ["PYTHONPATH"] = "" if scen == "d" else FAKE
g = {"__name__": "__main__", "_PIP": PipLog()}
exec(compile(CFG, "<cfg>", "exec"), g)
g["HF_CACHE"] = f"{SP}/hf_cache_t"
try:
    exec(compile(src, "<install>", "exec"), g)
    out = "ok"
except SystemExit as e: out = f"SystemExit: {e}"
except RuntimeError as e: out = f"RuntimeError: {e}"
print("\n결과:", out, "| pip 명령:", list(g["_PIP"]))
ua = [p for p in g["_PIP"] if "uninstall" in p and "torchaudio" in p]
if scen == "a": assert out == "ok" and ua
if scen == "c": assert out == "ok" and not ua
if scen == "d": assert out.startswith("RuntimeError") and "설치 점검 실패" in out
if scen == "e": assert out.startswith("RuntimeError") and ua
print(f"✅ 설치 셀 시나리오 {scen} 통과")
