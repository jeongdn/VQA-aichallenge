# -*- coding: utf-8 -*-
"""stage2_newfam.ipynb → runpod_q35_122b_bf16.ipynb (RunPod B300에서 Qwen3.5-122B-A10B bf16 LoRA)
학습·평가·제출 코드(분할·프롬프트·손실·체크포인트·이어하기·파일 이름)는 stage2 그대로 두고, 다음만 바꾼다.
  0번(새): RunPod 환경 점검 — GPU 메모리 합계·드라이버·볼륨/컨테이너 디스크 (torch를 import하지 않음)
  2번: 설치 — torch가 CUDA 12.9 미만이면 cu130/cu129 휠로 교체(B300 필수), torchaudio 제거, 버전 고정 설치, 7z
  3번: 데이터 — Drive 대신 HF 비공개 저장소에서 zip 받기 → 7z(없으면 unzip)로 풀기
  3-1번(새): 가중치 244GB 미리 받기 (볼륨 디스크의 HF 캐시)
  7번: GPU 2장 이상이면 device_map="auto" · 로드 후 GPU 메모리 출력
  8번: 진행 로그 파일(progress.log) · 해상도별 채점 속도 기록(첫 배치 제외)
  9번: 손실 계산 시 정답 텐서를 로짓과 같은 GPU로 (여러 GPU 대비)
  10번: 본 학습에서 zero-shot이 기준 미만이면 멈춤 · DRY_RUN 끝에 전체 시간·비용 예상표
  14번(새): 결과를 HF 비공개 저장소로 올리기 (점검 포함)
사용: python3 build_runpod122b.py [출력 경로]"""
import json, sys, os, uuid

SRC = "/home/user/vqa/newfam/out/stage2_newfam.ipynb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/runpod/out/runpod_q35_122b_bf16.ipynb"
C = json.load(open(SRC, encoding="utf-8"))["cells"]
def src(i): return "".join(C[i]["source"])
def rep(s, old, new, count=1):
    n = s.count(old); assert n == count, f"'{old[:70]}' 가 {n}번 나옴 (기대 {count})"
    return s.replace(old, new)

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

# ───────────────────────── 제목 ─────────────────────────
md("""
# RunPod · Qwen3.5-122B-A10B bf16 LoRA → 검증 → test → 결과 올리기

stage2와 **학습·평가·제출 방식이 완전히 같습니다**(분할 md5, 프롬프트, 정답 글자 손실, 선지 셔플, LoRA r16, 1 epoch, up2.0 평가).
결과 파일 이름도 같아서, Drive `vqa_runs/q35_122b_a10b_r16_native_bf16/`에 넣으면 ens_search·manual_ensemble이 새 멤버로 바로 읽습니다.

**실행 순서** (자세한 RunPod 사용법은 가이드 문서 참고)
1. `DRY_RUN = True` 그대로 두고 메뉴 **Run → Run All Cells**
   - 처음 한 번은 설치 약 5분, 데이터 받기 몇 분, 가중치 244GB 받기가 걸립니다.
   - 3스텝만 학습한 뒤, 마지막에 **본 학습 예상 시간·비용 표**가 나옵니다.
2. 이상이 없으면 1번 셀에서 `DRY_RUN = False`로 바꾸고 **Kernel → Restart Kernel and Run All Cells**
   - 설치·데이터·가중치는 이미 있어서 바로 넘어갑니다.
   - zero-shot 300문항 → 학습(약 357스텝) → 검증 1,000 → test 6,714 순서로 진행합니다.
3. 끝나면 마지막 14번 셀이 결과를 HF 비공개 저장소로 올립니다 → Colab 브리지 노트북 **B 단계**로 Drive에 받기 → RunPod pod **Terminate**

**중간에 끊겨도 됩니다**
- 브라우저를 닫아도 학습은 계속됩니다. 진행 상황은 `/workspace/vqa_runs/<실행 이름>/progress.log`에 계속 적힙니다.
- pod가 멈췄다 다시 켜졌다면 1번 셀부터 다시 실행하세요. 볼륨 디스크의 체크포인트에서 이어서 학습합니다(25스텝마다 저장).
""")

# ───────────────────────── 0. 환경 점검 (새) ─────────────────────────
md("""
## 0. RunPod 환경 점검 (가장 먼저)
- GPU 메모리 합계, 드라이버, 디스크 공간을 확인합니다.
  - GPU 메모리 합계 250GiB 이상: B300 1장 또는 B200 2장
  - 드라이버 575 이상: CUDA 12.9 이상을 쓰기 위해
  - `/workspace` 남은 공간 300GB 이상: 가중치 244GB + 데이터·결과
  - 컨테이너 디스크 남은 공간 15GB 이상: 라이브러리 설치용
- 여기서 torch를 불러오지 않으므로, 2번 셀이 torch를 새로 설치해도 커널을 다시 시작할 필요가 없습니다.
""")
code(r'''
import os, sys, shutil, subprocess, re
os.environ.setdefault("HF_HOME", "/workspace/hf")               # 가중치 캐시는 반드시 볼륨 디스크(/workspace)에 — 컨테이너 디스크는 작고 pod를 멈추면 지워짐
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
WORK = "/workspace"
assert os.path.isdir(WORK), "/workspace가 없습니다 → pod 설정에서 Volume Disk(마운트 경로 /workspace)를 켰는지 확인하세요"

_q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True)
assert _q.returncode == 0 and _q.stdout.strip(), "nvidia-smi 실패 → GPU pod인지 확인하세요\n" + (_q.stderr or "")
GPUS = []
for _line in _q.stdout.strip().splitlines():
    _name, _mem, _drv, _cap = [x.strip() for x in _line.split(",")[:4]]
    GPUS.append({"name": _name, "mem_gib": float(_mem) / 1024, "driver": _drv, "cap": _cap})
for _i, _g in enumerate(GPUS): print(f"GPU{_i}: {_g['name']} | {_g['mem_gib']:.0f}GiB | 드라이버 {_g['driver']} | compute {_g['cap']}")
TOTAL_VRAM_GIB = sum(_g["mem_gib"] for _g in GPUS)
DRIVER_MAJOR = int(GPUS[0]["driver"].split(".")[0])
_dw, _dc = shutil.disk_usage(WORK), shutil.disk_usage("/")
_ram = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 1e6
print(f"볼륨 디스크 /workspace: 남은 {_dw.free / 1e9:.0f}GB / 전체 {_dw.total / 1e9:.0f}GB | 컨테이너 디스크: 남은 {_dc.free / 1e9:.0f}GB | RAM {_ram:.0f}GB")
_have_w = os.path.isdir(os.path.join(os.environ["HF_HOME"], "hub", "models--Qwen--Qwen3.5-122B-A10B", "snapshots"))
_prob = []
if TOTAL_VRAM_GIB < 250: _prob.append(f"GPU 메모리 합계 {TOTAL_VRAM_GIB:.0f}GiB < 250GiB → 122B bf16(약 228GiB)이 안 들어갑니다. B300 1장 또는 B200 2장으로 pod를 다시 만드세요")
if DRIVER_MAJOR < 575: _prob.append(f"드라이버 {GPUS[0]['driver']} < 575 → CUDA 12.9 이상을 못 씁니다. 다른 pod(다른 서버)를 새로 만드세요")
if _dw.free < 300e9 and not _have_w: _prob.append(f"/workspace 남은 공간 {_dw.free / 1e9:.0f}GB < 300GB → pod의 Volume Disk를 600GB로 늘리세요 (Pods → ⋮ → Edit Pod)")
if _dc.free < 15e9: _prob.append(f"컨테이너 디스크 남은 공간 {_dc.free / 1e9:.0f}GB < 15GB → Container Disk를 50GB로 늘리세요 (Edit Pod)")
if _prob: raise SystemExit("❌ 환경 문제:\n  - " + "\n  - ".join(_prob))
if len(GPUS) > 1: print(f"GPU {len(GPUS)}장 → 모델을 층 단위로 나눠 올립니다 (속도는 한 장과 비슷)")
if TOTAL_VRAM_GIB < 270: print("⚠️ GPU 메모리 합계가 빠듯합니다 (270GiB 미만). 학습 중 OOM이 나면 자동으로 마이크로 배치를 줄입니다")
print("✅ 환경 OK — 1번 셀로 넘어가세요" + (" (가중치는 이미 받아 둠)" if _have_w else ""))
''')

# ───────────────────────── 1. 설정 ─────────────────────────
md("## 1. 실험 설정")
s = src(2)
s = rep(s, s[:s.index("DRY_RUN    = False")],
        'RUN_NAME   = "q35_122b_a10b_r16_native_bf16"   # 저장 폴더 이름 = ens_search가 읽는 멤버 이름. 설정을 바꾸면 이름도 바꾸세요\n'
        'MODEL_ID   = "Qwen/Qwen3.5-122B-A10B"          # bf16 약 244GB → B300(288GB) 1장 또는 B200 2장\n')
s = rep(s, 'DRY_RUN    = False          # True: 3스텝만 학습해 전체 흐름 점검 (본 학습 전 필수) / False: 본 학습\n',
        'DRY_RUN    = True           # ① True: 3스텝만 학습해 메모리·속도·비용 확인 → ② 통과하면 False로 바꾸고 Kernel → Restart Kernel and Run All Cells\n')
s = rep(s, 'ZERO_SHOT_ONLY   = True     # True: zero-shot 점검만 하고 멈춤 (새 모델은 이것부터) / False: 이어서 학습\n',
        'ZERO_SHOT_ONLY   = False    # True면 zero-shot만 하고 멈춤 (122B는 GPTQ 4bit로 이미 96.0% 확인 → 바로 학습)\n')
s = rep(s, 'ZERO_SHOT_PRESET = "up2.0"  # zero-shot 채점 해상도 (기존 기록 Q35 제로샷 96.1%가 up2.0)\n',
        'ZERO_SHOT_PRESET = "up2.0"  # zero-shot 채점 해상도 (기존 기록 Q35 제로샷 96.1%가 up2.0)\n'
        'ZERO_SHOT_MIN_ACC = 0.93    # 본 학습에서 zero-shot이 이보다 낮으면 학습 전에 멈춤 (프롬프트·해상도·커널 고장을 비싼 학습 전에 잡음). 참고: GPTQ 4bit zero-shot 96.0%\n'
        'GPU_PRICE_PER_HOUR = 7.89   # DRY_RUN 끝 예상 비용 계산용 시간당 요금 ($). B300 1장 Secure 7.89 · Community 6.94 · B200 2장 13.58\n'
        'USE_FLA    = True           # 선형 어텐션 고속 커널(flash-linear-attention). DRY_RUN에서 fla/Triton 오류가 나면 False → Kernel → Restart Kernel and Run All Cells (느려지지만 동작)\n')
s = rep(s, 'LOG_EVERY, EVAL_EVERY, SAVE_EVERY = 10, 100, 50    # 옵티마이저 스텝 단위\n',
        'LOG_EVERY, EVAL_EVERY, SAVE_EVERY = 10, 100, 25    # 옵티마이저 스텝 단위 (체크포인트는 LoRA뿐이라 작고 빠름 → 25스텝마다)\n')
s = rep(s, 'MAKE_SUBMISSION = False     # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성\n',
        'MAKE_SUBMISSION = True      # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성 (ens_search에 test 예측이 필요)\n')
s = rep(s, '''DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
USE_4BIT   = False         # 4bit 없이 bf16 LoRA — H100(80GB)·G4(96GB) 전용. A100 40GB에서는 메모리 부족
''', '''# ── RunPod 경로 (바꾸지 마세요. 모두 볼륨 디스크 /workspace 안) ──
ZIP_PATH     = "/workspace/ssafy-16-2-ai.zip"   # 대회 데이터 zip. 없으면 3번 셀이 HF 비공개 저장소에서 받음 (Jupyter로 직접 올려도 됨)
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/workspace/data"
OUT_DIR      = "/workspace/vqa_runs"
HF_REPO      = ""            # 비워 두면 "<내 HF 아이디>/ssafy-vqa-private" (Colab 브리지 노트북과 같은 규칙)
HF_ZIP_NAME  = "ssafy-16-2-ai.zip"
USE_4BIT   = False         # bf16 LoRA (122B는 MoE 전문가 층 때문에 4bit 학습이 안 됨)
''')
s = rep(s, 'HF_TOKEN   = ""            # Gemma처럼 라이선스 승인이 필요한 모델일 때 HF 토큰 (비워 두면 Colab 보안 비밀 \'HF_TOKEN\'을 찾아봄)\n',
        'HF_TOKEN   = ""            # 비워 두면 pod 환경 변수 HF_TOKEN → 그것도 없으면 3번 셀에서 입력창이 뜸 (Write 권한 토큰)\n')
s = rep(s, 'MUSE_REASONING = "low"', '# ── (다른 계열용 설정 — 122B에서는 쓰이지 않음) ──\nMUSE_REASONING = "low"')
code(s)

# ───────────────────────── 2. 설치 ─────────────────────────
md("""
## 2. 라이브러리 설치 (RunPod용)
- **torch**: B300(Blackwell Ultra)은 CUDA 12.9 이상으로 빌드된 torch가 필요합니다.
  - 지금 torch가 그보다 낮거나 GPU 계산이 안 되면 CUDA 13.0용 휠로 바꿉니다(드라이버 580 미만이면 12.9용).
  - 이 경우 2~5분 걸립니다.
- 오래된 torchaudio는 지웁니다. 새 torch와 CUDA 버전이 달라 transformers의 프로세서 import를 깨뜨리고, 오디오는 쓰지 않습니다.
- **버전 고정**: transformers 5.17.0 · accelerate 1.15.0 · peft 0.21.0 · flash-linear-attention 0.5.2
  - Colab stage2와 같은 조합입니다.
  - 설치하는 동안 torch 버전은 고정해 둡니다.
- `7z`가 없으면 apt로 설치합니다(대회 zip 풀기용).
- 마지막에 새 프로세스에서 import를 점검합니다.
""")
code(r'''
import subprocess, sys, json, shutil, os
def _run(cmd, check=True, quiet=False):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if not quiet or r.returncode: print(f"$ {cmd}\n{(r.stdout or '')[-1200:]}{(r.stderr or '')[-1500:]}")
    if check and r.returncode: raise RuntimeError(f"명령 실패 (코드 {r.returncode}): {cmd} → 위 출력을 확인하세요")
    return r
_PROBE = ("import json, torch; ok = torch.cuda.is_available(); "
          "x = float((torch.randn(256, 256, device='cuda', dtype=torch.bfloat16) @ torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)).float().abs().sum()) if ok else None; "
          "print(json.dumps({'v': torch.__version__, 'cuda': torch.version.cuda, 'ok': bool(ok and x == x), "
          "'cap': list(torch.cuda.get_device_capability(0)) if ok else None, 'n': torch.cuda.device_count()}))")
def torch_probe():
    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    try: return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: return {"ok": False, "error": ((r.stderr or "") + (r.stdout or ""))[-600:]}
def cuda_ok(p):
    try: return bool(p.get("ok")) and tuple(int(x) for x in str(p.get("cuda")).split(".")[:2]) >= (12, 9)
    except Exception: return False
_p = torch_probe(); print("지금 torch:", _p)
if not cuda_ok(_p):
    _idx = "cu130" if globals().get("DRIVER_MAJOR", 580) >= 580 else "cu129"
    print(f"→ torch를 CUDA {_idx[2:4]}.{_idx[4:]}용 휠로 새로 설치합니다 (2~5분)")
    _run(f"{sys.executable} -m pip install -q --upgrade torch torchvision --index-url https://download.pytorch.org/whl/{_idx}")
    _p = torch_probe(); print("새 torch:", _p)
    assert cuda_ok(_p), f"torch를 새로 설치한 뒤에도 GPU 점검 실패: {_p} → 위 출력을 확인하세요"
_run(f"{sys.executable} -m pip uninstall -y -q torchaudio", check=False, quiet=True)
_run(f"{sys.executable} -m pip uninstall -y -q torchao", check=False, quiet=True)      # 구버전 torchao가 있으면 peft 0.21이 LoRA를 붙일 때 ImportError
_tv = subprocess.run([sys.executable, "-c", "import torchvision; print(torchvision.__version__)"], capture_output=True, text=True).stdout.strip()
open("/tmp/torch_pin.txt", "w").write(f"torch=={_p['v']}\n" + (f"torchvision=={_tv}\n" if _tv else ""))   # 아래 설치가 torch를 바꾸지 못하게 고정
_fla = globals().get("USE_FLA", True)
_run(f"{sys.executable} -m pip install -q -c /tmp/torch_pin.txt 'transformers==5.17.0' 'accelerate==1.15.0' 'peft==0.21.0' "
     + ("'flash-linear-attention==0.5.2' " if _fla else "") + "'huggingface_hub[hf_xet]' pandas pillow")
if not _fla: _run(f"{sys.executable} -m pip uninstall -y -q flash-linear-attention fla-core", check=False, quiet=True); print("USE_FLA = False → 선형 어텐션 커널을 지웠습니다 (PyTorch 기본 구현, 느림)")
if shutil.which("7z") is None:
    _run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq p7zip-full unzip", check=False, quiet=True)
print("압축 도구:", shutil.which("7z") or shutil.which("unzip") or "없음 → 3번 셀이 파이썬 zipfile로 풉니다(느림)")
_chk = subprocess.run([sys.executable, "-c",
    "import torch, transformers, peft, huggingface_hub, accelerate; " + ("import fla; " if _fla else "fla = None; ") +
    "from transformers import AutoProcessor, AutoModelForImageTextToText; "
    "from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration; "
    "print('torch', torch.__version__, '(CUDA', str(torch.version.cuda) + ')', '| transformers', transformers.__version__, '| peft', peft.__version__, "
    "'| accelerate', accelerate.__version__, '| fla', getattr(fla, '__version__', '?') if fla else '안 씀', '| hub', huggingface_hub.__version__)"],
    capture_output=True, text=True)
print(_chk.stdout.strip())
assert _chk.returncode == 0, "라이브러리 import 점검 실패 → 위 출력을 확인하세요:\n" + (_chk.stderr or "")[-2500:]
print("✅ 설치 완료")
''')

# ───────────────────────── 3. 데이터 ─────────────────────────
md("""
## 3. 데이터 준비 (HF 비공개 저장소 → `/workspace/data`)
- Colab 브리지 노트북 **A 단계**로 올려 둔 대회 zip을 받아 풉니다.
- `/workspace/ssafy-16-2-ai.zip`이 이미 있으면 받지 않습니다. Jupyter 왼쪽 파일 창에 zip을 직접 끌어다 올려도 됩니다.
- HF 토큰은 1번 셀의 `HF_TOKEN`, pod 환경 변수 `HF_TOKEN` 순으로 찾습니다. 둘 다 없으면 입력창이 뜹니다.
""")
code(r'''
import os, subprocess, shutil, getpass, time
from huggingface_hub import login, whoami, hf_hub_download
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN: HF_TOKEN = getpass.getpass("Hugging Face 토큰(Write 권한, hf_로 시작)을 붙여 넣고 Enter: ").strip()
login(token=HF_TOKEN, add_to_git_credential=False)
HF_USER = whoami(token=HF_TOKEN)["name"]
HF_REPO = HF_REPO or f"{HF_USER}/ssafy-vqa-private"
print(f"Hugging Face 로그인: {HF_USER} | 비공개 저장소: {HF_REPO}")
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

def _find_data_dir(root):
    if os.path.exists(os.path.join(root, "train.csv")): return root
    for d, _, fs in os.walk(root):
        if "train.csv" in fs and "test.csv" in fs: return d
    return None
if _find_data_dir(DATA_DIR) is None:
    if not os.path.exists(ZIP_PATH):
        t0 = time.time(); print(f"대회 데이터 받는 중: {HF_REPO}/{HF_ZIP_NAME}")
        try:
            _got = hf_hub_download(HF_REPO, HF_ZIP_NAME, repo_type="dataset", token=HF_TOKEN, local_dir=os.path.dirname(ZIP_PATH))
        except Exception as e:
            raise SystemExit(f"❌ {HF_REPO}에서 {HF_ZIP_NAME}를 받지 못했습니다 ({type(e).__name__}: {str(e)[:200]})\n"
                             "→ Colab 브리지 노트북 A 단계를 먼저 실행했는지 확인하세요. 또는 Jupyter 왼쪽 파일 창에 zip을 /workspace로 직접 올리세요.")
        if os.path.abspath(_got) != os.path.abspath(ZIP_PATH): shutil.move(_got, ZIP_PATH)
        print(f"  받음 {os.path.getsize(ZIP_PATH) / 1e9:.2f}GB ({time.time() - t0:.0f}초)")
    t0 = time.time(); print("압축 해제 중... (몇 분)")
    if shutil.which("7z") or shutil.which("unzip"):
        _cmd = (["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", ZIP_PATH] if shutil.which("7z")
                else ["unzip", "-q", "-o", "-P", ZIP_PASSWORD, ZIP_PATH, "-d", DATA_DIR])
        _r = subprocess.run(_cmd, capture_output=True, text=True)
        assert _r.returncode == 0, f"압축 해제 실패 ({_cmd[0]}) → 비밀번호·파일을 확인하세요:\n" + (_r.stdout or "")[-600:] + (_r.stderr or "")[-800:]
    else:                                               # 7z·unzip 둘 다 없으면 파이썬 zipfile (느림, ZipCrypto 방식만 가능)
        import zipfile
        with zipfile.ZipFile(ZIP_PATH) as _z: _z.extractall(DATA_DIR, pwd=ZIP_PASSWORD.encode())
    print(f"  완료 ({time.time() - t0:.0f}초)")
_dd = _find_data_dir(DATA_DIR)
assert _dd is not None, f"압축을 풀었는데 {DATA_DIR} 아래에 train.csv가 없습니다 → zip 내용을 확인하세요"
if _dd != DATA_DIR: print(f"train.csv가 하위 폴더에 있어 DATA_DIR을 바꿉니다: {_dd}"); DATA_DIR = _dd
for f in ["train.csv", "test.csv", "sample_submission.csv"]:
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"{f} 가 {DATA_DIR} 에 없습니다"
print("데이터 OK:", DATA_DIR, "| 결과 저장:", OUT_DIR)
''')

# ───────────────────────── 3-1. 가중치 (새) ─────────────────────────
md("""
## 3-1. 모델 가중치 받기 (약 244GB, 처음 한 번만)
- 볼륨 디스크의 HF 캐시(`/workspace/hf`)에 받습니다. 이미 받았으면 몇 초 만에 끝납니다.
- 끊기면 이 셀을 다시 실행하세요. 받은 데까지는 그대로 두고 이어 받습니다.
""")
code(r'''
import time, os
from huggingface_hub import snapshot_download
t0 = time.time(); print(f"{MODEL_ID} 가중치 확인·받기 (약 244GB)")
MODEL_DIR = snapshot_download(MODEL_ID, token=HF_TOKEN or None, max_workers=16)
_sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(MODEL_DIR) for f in fs)   # 캐시의 파일은 심볼릭 링크 → 실제 크기를 따라감
_nst = sum(f.endswith(".safetensors") for f in os.listdir(MODEL_DIR))
print(f"✅ {MODEL_DIR} | safetensors {_nst}개 · {_sz / 1e9:.0f}GB | {time.time() - t0:.0f}초")
''')

# ───────────────────────── 4~6 그대로 ─────────────────────────
for i in (7, 8, 9, 10, 11, 12):
    (md if C[i]["cell_type"] == "markdown" else code)(src(i))

# ───────────────────────── 7. 모델 로드 ─────────────────────────
md(src(13).replace("출력의 **'답 바로 앞 프롬프트 끝'**이 모델의 답 자리인지 확인하세요 (Qwen: `</think>` 뒤, Muse: `<|start|>assistant<|message|>`).",
                   "출력의 **'답 바로 앞 프롬프트 끝'**이 `</think>` 뒤인지 확인하세요. 244GB를 GPU에 올리는 데 몇 분 걸립니다.")
   + "GPU가 2장 이상이면 층을 나눠 올립니다(`device_map=\"auto\"`).\n")
s = src(14)
s = rep(s, 'if DEVICE == "cuda": load_kw["device_map"] = {"": 0}\n',
        'N_GPU = torch.cuda.device_count() if DEVICE == "cuda" else 0\n'
        'if DEVICE == "cuda": load_kw["device_map"] = {"": 0} if N_GPU == 1 else "auto"      # 2장 이상(B200 2장 등)이면 층을 나눠 올림\n')
s = rep(s, '''        print("⚠️⚠️ flash-linear-attention이 없습니다 → 선형 어텐션이 순수 PyTorch로 돌아 OOM/저속 위험. 2번 셀을 실행하고 '세션 다시 시작' 후 처음부터 실행하세요.")\n''',
        '''        print("USE_FLA = False → 선형 어텐션을 PyTorch 기본 구현으로 돌립니다 (느림)" if not USE_FLA else "⚠️⚠️ flash-linear-attention이 없습니다 → 선형 어텐션이 순수 PyTorch로 돌아 OOM/저속 위험. 2번 셀을 다시 실행한 뒤 Kernel → Restart Kernel and Run All Cells")\n''')
s = rep(s, 'print(f"모델 로드 {time.time() - t0:.0f}s | {type(base).__name__}")\n',
        'MODEL_LOAD_S = time.time() - t0\n'
        'print(f"모델 로드 {MODEL_LOAD_S:.0f}s | {type(base).__name__}")\n'
        'if DEVICE == "cuda":\n'
        '    print("가중치를 올린 뒤 GPU 메모리: " + " · ".join(f"GPU{i} {torch.cuda.memory_allocated(i) / 2**30:.0f}/{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GiB" for i in range(N_GPU)))\n')
code(s)

# ───────────────────────── 8. 채점 함수: 진행 로그 · 속도 기록 ─────────────────────────
md(src(15) + "\n진행 상황은 `RUN_DIR/progress.log`에도 적습니다(브라우저를 닫았다 다시 열었을 때 확인용).\n")
s = src(16)
s = rep(s, 'IDENTITY = (0, 1, 2, 3)\n',
        'def _plog(msg):                                  # 화면 출력 + RUN_DIR/progress.log (브라우저를 닫아도 진행 상황이 남음)\n'
        '    print(msg, flush=True)\n'
        '    try:\n'
        '        with open(os.path.join(RUN_DIR, "progress.log"), "a", encoding="utf-8") as _f: _f.write(time.strftime("%m-%d %H:%M:%S ") + str(msg) + "\\n")\n'
        '    except Exception: pass\n'
        'EVAL_TIMES = {}                                  # (해상도, TTA) → [초, 문항 수] — 첫 배치(커널 준비 시간)는 뺌. DRY_RUN 끝 예상 시간 계산용\n'
        'IDENTITY = (0, 1, 2, 3)\n')
s = rep(s, '''    rows = todo.to_dict("records"); recs, t0, n_done, nb = [], time.time(), 0, 0
''', '''    rows = todo.to_dict("records"); recs, t0, n_done, nb = [], time.time(), 0, 0
    _tb = []                                        # 배치마다 (걸린 초, 문항 수)
''')
s = rep(s, '''        batch = rows[n_done: n_done + BS]
        probs, mass, ntok = score_rows_tta(batch, tta)
''', '''        batch = rows[n_done: n_done + BS]; _t1 = time.time()
        probs, mass, ntok = score_rows_tta(batch, tta)
        _tb.append((time.time() - _t1, len(batch)))
''')
s = rep(s, '''                print(f"  {n_done:>5}/{len(rows)} | {el / n_done:.2f}s/문항 | 남은 시간 ~{(len(rows) - n_done) * el / n_done / 60:.0f}분", flush=True)
    if was_training: model.train()
''', '''                _plog(f"  {os.path.basename(out_path)} {n_done:>5}/{len(rows)} | {el / n_done:.2f}s/문항 | 남은 시간 ~{(len(rows) - n_done) * el / n_done / 60:.0f}분")
    if _tb:
        _use = _tb[1:] if len(_tb) > 1 else _tb
        _e = EVAL_TIMES.setdefault((RES_PRESET, bool(tta)), [0.0, 0]); _e[0] += sum(t for t, _ in _use); _e[1] += sum(n for _, n in _use)
    if was_training: model.train()
''')
s = rep(s, '''    print(f"▶ {tag} | {RES_PRESET}{' +TTA4' if tta else ''} | 정확도 {acc * 100:.2f}% (n={n}, ±{1.96 * se * 100:.1f}%p) | letter_mass {m.letter_mass.mean():.3f}")''',
        '''    _plog(f"▶ {tag} | {RES_PRESET}{' +TTA4' if tta else ''} | 정확도 {acc * 100:.2f}% (n={n}, ±{1.96 * se * 100:.1f}%p) | letter_mass {m.letter_mass.mean():.3f}")''')
code(s)

# ───────────────────────── 9. 학습 준비: 정답 텐서 장치 ─────────────────────────
md(src(17))
s = src(18)
s = rep(s, "    loss = F.cross_entropy(logits, target)\n", "    loss = F.cross_entropy(logits, target.to(logits.device))       # GPU 여러 장이면 로짓이 마지막 GPU에 있음\n")
code(s)

# ───────────────────────── 10. 학습 루프: 게이트 · 진행 로그 · 예상표 ─────────────────────────
md(src(19) + """
**122B 전용 안전장치**
- 본 학습(`DRY_RUN = False`)에서 zero-shot이 `ZERO_SHOT_MIN_ACC`(93%)보다 낮으면 학습 전에 멈춥니다. 무언가 고장 난 상태에서 비싼 학습을 하지 않기 위해서입니다.
- DRY_RUN 끝에는 실제로 잰 속도로 **본 실행 전체의 예상 시간·비용 표**를 보여 줍니다.
""")
s = src(20)
s = rep(s, '''    set_resolution(TRAIN_PRESET)
    if ZERO_SHOT_ONLY:
''', '''    set_resolution(TRAIN_PRESET)
    if not DRY_RUN and ZERO_SHOT_MIN_ACC and zs_acc < ZERO_SHOT_MIN_ACC:
        raise SystemExit(f"❌ zero-shot {zs_acc * 100:.2f}% < 기준 {ZERO_SHOT_MIN_ACC * 100:.0f}% → 프롬프트·해상도·커널 중 무언가가 고장 난 것으로 보고 학습 전에 멈춥니다 (비용 절약). "
                         "이 셀의 출력을 확인하세요.")
    if ZERO_SHOT_ONLY:
''')
s = rep(s, '''t_start, run_loss, run_hit, n_micro = time.time(), 0.0, 0.0, 0
''', '''t_start, run_loss, run_hit, n_micro = time.time(), 0.0, 0.0, 0
STEP_TIMES = []                                    # 옵티마이저 스텝마다 학습에 걸린 초 (평가·저장 제외)
''')
s = rep(s, '''    idx = order[k * STEP_SAMPLES: (k + 1) * STEP_SAMPLES]
    while True:''', '''    idx = order[k * STEP_SAMPLES: (k + 1) * STEP_SAMPLES]
    _ts = time.time()
    while True:''')
s = rep(s, '''    state["global_step"] = step + 1
''', '''    state["global_step"] = step + 1
    STEP_TIMES.append(time.time() - _ts)
''')
s = rep(s, '''        print(f"step {step + 1:>4}/{total_steps} | ep {epoch + 1} | loss {run_loss / n_micro:.4f} | train_acc {run_hit / n_micro * 100:5.1f}% | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | gnorm {float(gnorm):.2f} | {el / done:.1f}s/step | 남은 시간 ~{(total_steps - step - 1) * el / done / 60:.0f}분", flush=True)''',
        '''        _plog(f"step {step + 1:>4}/{total_steps} | ep {epoch + 1} | loss {run_loss / n_micro:.4f} | train_acc {run_hit / n_micro * 100:5.1f}% | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | gnorm {float(gnorm):.2f} | {el / done:.1f}s/step | 남은 시간 ~{(total_steps - step - 1) * el / done / 60:.0f}분")''')
s = rep(s, '''if DEVICE == "cuda": print(f"최대 VRAM 사용: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB / {VRAM_GB:.0f}GB")
if DRY_RUN and total_steps > step0:
    per_step = (time.time() - t_start) / (total_steps - step0)
    full_steps = (len(train_df if TRAIN_ON_FULL else tr_df) // STEP_SAMPLES) * EPOCHS
    print(f"\\n✅ DRY RUN 통과. 본 학습 예상: {full_steps} 스텝 x {per_step:.1f}s ≈ {full_steps * per_step / 3600:.1f}시간 (+평가 약 15분). DRY_RUN=False로 바꿔 1번 셀부터 다시 실행하세요.")
''', '''if DEVICE == "cuda":
    _plog("최대 GPU 메모리: " + " · ".join(f"GPU{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f}/{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GiB" for i in range(N_GPU)))
if DRY_RUN and total_steps > step0:
    _st = STEP_TIMES[1:] if len(STEP_TIMES) > 1 else STEP_TIMES          # 첫 스텝은 커널 준비 시간이 섞여 뺌
    per_step = float(np.median(_st)) if _st else 0.0
    _spe_full = len(train_df if TRAIN_ON_FULL else tr_df) // STEP_SAMPLES   # 본 실행의 epoch당 스텝 (DRY_RUN의 steps_per_epoch는 3)
    full_steps = _spe_full * EPOCHS
    if MAX_STEPS: full_steps = min(full_steps, int(MAX_STEPS))
    def _spq(preset, tta=False):                                        # 문항당 초 (DRY_RUN에서 잰 값, 없으면 가장 비싼 값)
        v = EVAL_TIMES.get((preset, tta)) or EVAL_TIMES.get((preset, False))
        if v and v[1]: return v[0] / v[1] * (4 if (tta and (preset, tta) not in EVAL_TIMES) else 1)
        return max((t / n for t, n in EVAL_TIMES.values() if n), default=0.0)
    _n_quick = sum(1 for s_ in range(1, full_steps + 1) if s_ % EVAL_EVERY == 0 and s_ % _spe_full != 0 and s_ != full_steps)
    _test_preset = TEST_PRESET if TEST_PRESET != "auto" else EVAL_PRESETS[0]
    _rows = [("가중치 GPU에 올리기", globals().get("MODEL_LOAD_S", 0.0)),
             (f"zero-shot {ZERO_SHOT_N}문항 ({ZERO_SHOT_PRESET})", (ZERO_SHOT_N or 0) * _spq(ZERO_SHOT_PRESET)),
             (f"학습 {full_steps}스텝 × {per_step:.1f}초", full_steps * per_step),
             (f"학습 중 빠른 검증 {_n_quick}번 × {QUICK_EVAL_N}문항 ({TRAIN_PRESET})", _n_quick * QUICK_EVAL_N * _spq(TRAIN_PRESET)),
             (f"epoch 끝 검증 {len(val_df)}문항 ({TRAIN_PRESET})", EPOCHS * len(val_df) * _spq(TRAIN_PRESET))]
    _rows += [(f"최종 검증 {len(val_df)}문항 ({p_})", len(val_df) * _spq(p_)) for p_ in EVAL_PRESETS if p_ != TRAIN_PRESET]
    if EVAL_TTA: _rows.append((f"최종 검증 TTA ({EVAL_PRESETS[0]})", len(val_df) * _spq(EVAL_PRESETS[0], True)))
    if MAKE_SUBMISSION: _rows.append((f"test {len(test_df)}문항 ({_test_preset}{' +TTA4' if TEST_TTA else ''})", len(test_df) * _spq(_test_preset, TEST_TTA)))
    _tot = sum(v for _, v in _rows)
    print("\\n✅ DRY RUN 통과 — 본 실행(DRY_RUN=False) 예상 (이 GPU에서 잰 속도 기준):")
    for k_, v_ in _rows: print(f"    · {k_}: {v_ / 60:.1f}분")
    print(f"    ▶ 합계 {_tot / 60:.1f}분 ≈ {_tot / 3600:.1f}시간 → 비용 약 ${_tot / 3600 * GPU_PRICE_PER_HOUR:.0f} (시간당 ${GPU_PRICE_PER_HOUR})")
    print("    (설치·데이터·가중치는 이미 끝나 있어 빠짐. 실제로는 ±30% 정도 달라질 수 있음)")
    print("→ 1번 셀에서 DRY_RUN = False로 바꾸고 Kernel → Restart Kernel and Run All Cells")
''')
code(s)

# ───────────────────────── 11~13 그대로 (요약에 다음 단계 안내) ─────────────────────────
for i in (21, 22, 23, 24, 25):
    (md if C[i]["cell_type"] == "markdown" else code)(src(i))
s = src(26).rstrip("\n") + "\n" + '''if not DRY_RUN: print("\\n다음: 14번 셀로 결과를 HF 비공개 저장소에 올리세요 (pod를 끄기 전에)")
'''
code(s)

# ───────────────────────── 14. 결과 올리기 (새) ─────────────────────────
md("""
## 14. 결과를 HF 비공개 저장소로 올리기 — pod를 끄기 전에 꼭
- 올리기 전에 ens_search가 읽는 파일을 확인합니다.
  - 검증 예측 1,000문항
  - test 예측 6,714문항
  - `train_log.json`
  - 최종 어댑터
- 체크포인트(`ckpt_last`, 이어하기용)는 올리지 않습니다. 최종 어댑터 `adapter_ep*`는 재현성 제출용으로 올립니다.
- 다 올라가면 Colab 브리지 노트북 **B 단계**로 Drive에 받습니다. Drive에 들어온 것을 확인한 뒤 RunPod에서 pod를 **Terminate**하세요.
""")
code(r'''
import re, glob, shutil, time
from huggingface_hub import HfApi
assert not DRY_RUN, "DRY_RUN 결과는 올리지 않습니다 → DRY_RUN = False로 본 실행을 끝낸 뒤 이 셀을 실행하세요"
_fs = os.listdir(RUN_DIR)
_val = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+__(.+)__" + str(len(val_df)) + r"\.csv$", f))
_tst = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+_test__(.+)__" + str(len(test_df)) + r"\.csv$", f))
assert _val, f"최종 검증 예측 파일이 없습니다 ({RUN_DIR}) → 11번 셀까지 끝났는지 확인하세요"
assert _tst, f"test 예측 파일이 없습니다 ({RUN_DIR}) → 12번 셀까지 끝났는지 확인하세요"
for f, n in [(x, len(val_df)) for x in _val] + [(x, len(test_df)) for x in _tst]:
    _d = pd.read_csv(os.path.join(RUN_DIR, f))
    assert _d.id.nunique() == n and {"p_a", "p_b", "p_c", "p_d"} <= set(_d.columns) and _d[["p_a", "p_b", "p_c", "p_d"]].notna().all().all(), f"{f}: 문항 {_d.id.nunique()}/{n} 또는 확률 열 이상"
_log = json.load(open(os.path.join(RUN_DIR, "train_log.json")))
assert _log["config"]["MODEL_ID"] == MODEL_ID, _log["config"]
_ad = sorted(glob.glob(os.path.join(RUN_DIR, "adapter_ep*", "adapter_config.json")))
assert _ad, "최종 어댑터(adapter_ep*)가 없습니다"
_lp = os.path.join(OUT_DIR, "results_log_s2.csv")
if os.path.exists(_lp): shutil.copy(_lp, os.path.join(RUN_DIR, "results_log_s2_runpod.csv"))
api = HfApi(token=HF_TOKEN)
api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(HF_REPO, repo_type="dataset").private, f"{HF_REPO}가 공개 저장소입니다 → HF 웹에서 Private로 바꾼 뒤 다시 실행하세요"
t0 = time.time(); _pre = f"vqa_runs/{RUN_NAME}"
print(f"올리는 중: {RUN_DIR} → {HF_REPO}/{_pre} (체크포인트 제외)")
api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=RUN_DIR, path_in_repo=_pre,
                  ignore_patterns=["ckpt_last/*", "ckpt_last/**", "ckpt_last_tmp/*", "ckpt_last_tmp/**"], commit_message=f"RunPod {RUN_NAME}")
_remote = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
_need = [f"{_pre}/{f}" for f in _val + _tst + ["train_log.json"]] + [f"{_pre}/" + os.path.relpath(a, RUN_DIR).replace(os.sep, "/") for a in _ad]
_miss = [f for f in _need if f not in _remote]
assert not _miss, f"올라가지 않은 파일: {_miss}"
print(f"✅ 올리기 완료 ({time.time() - t0:.0f}초) | 검증 {_val} · test {_tst} · 어댑터 {len(_ad)}개")
print(f"   검증 정확도(epoch 끝): {_log['epoch_acc']}")
print("다음: ① Colab 브리지 노트북 B 단계로 Drive에 받기 → ② Drive에 들어온 것을 확인한 뒤 RunPod에서 이 pod를 Terminate")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
