# -*- coding: utf-8 -*-
"""Qwen3.5-122B-A10B 공식 GPTQ-Int4를 vLLM으로 올려 zero-shot 채점하는 노트북 빌더.
- 분할·프롬프트·해상도·글자 확률 합산은 stage2(src_stage2.ipynb)와 같은 코드를 그대로 씀
- 검증 앞 300 → 기준 이상이면 검증 1,000 + test 6,714 → E14가 읽는 파일 이름으로 저장 (끊겨도 이어서)
사용: python3 build_zs122b.py [출력 경로]"""
import sys, json, uuid

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/zs122b/out/zs122b_vllm.ipynb"
SRC = json.load(open("/home/user/vqa/src_stage2.ipynb", encoding="utf-8"))
S = ["".join(c["source"]) for c in SRC["cells"]]

def grab(cell, start, end=None):
    """src_stage2 셀에서 start~end(미포함) 구간을 그대로 가져옴 (같은 코드를 쓰기 위함)"""
    t = S[cell]; i = t.index(start); j = t.index(end, i) if end else len(t)
    return t[i:j]

RES_BLOCK = grab(2, "RES_PRESETS = {", "# Gemma 4 계열")
DATA_CELL = S[6]
CSV_BLOCK = grab(8, "train_df = pd.read_csv")
SPLIT_CELL = S[10]
PROMPT_BLOCK = grab(12, 'PROMPT_VERSION = "baseline_v1"')
LETTER_BLOCK = grab(14, "LETTER_IDS = []")
CATS_BLOCK = grab(16, "CATS = {", "def evaluate(")
assert "df05e4bc19d27ebceeef2570672149c2" in SPLIT_CELL and "build_mc_prompt" in PROMPT_BLOCK and "up2.0" in RES_BLOCK

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# Qwen3.5-122B-A10B zero-shot (Qwen 공식 GPTQ-Int4 · vLLM) → 검증 1,000 + test 6,714 · E14 형식

**왜 zero-shot인가**
- 122B-A10B는 OCRBench 92.1로 후보 중 가장 높지만 bf16이 244GB입니다.
- 남이 올린 4bit 판을 transformers 5.17로 불러오면 전문가 층을 bf16으로 다시 풀어 버립니다(`DecompressExperts`). bitsandbytes는 합쳐진 전문가 층(3D 파라미터)을 양자화하지 못합니다.
- 그래서 우리 LoRA 파이프라인으로는 학습할 수 없고, 추론만 가능합니다.
- 이 노트북은 Qwen 공식 GPTQ-Int4를 추론 엔진 vLLM으로 올려, 학습 없이 다른 멤버와 같은 방식으로 채점합니다.
  - 전문가 층만 int4(그룹 128)이고 어텐션·공유 전문가·비전·임베딩은 bf16입니다. 파일은 78.9GB입니다.

**다른 멤버와 같은 점**
- 고정 검증 분할 (md5 `df05e4bc…`)
- 프롬프트 `baseline_v1` (시스템 지시 + 이미지 + 질문·보기, 생각 끔)
- 평가 해상도 up2.0
  - 이미지 크기를 transformers 5.17 프로세서와 똑같이 미리 맞춰서 넘깁니다 (이 대회 사진은 대부분 720×960이라 up2.0에서 1440×1920으로 커짐).
  - 셀에서 픽셀이 같은지 대조하고, 문항마다 이미지 토큰 수도 대조합니다.
- 답 자리의 a/b/c/d 변형(소문자·대문자·앞 공백) 확률 합산

**다른 점**: 확률을 vLLM이 돌려주는 답 자리 상위 64개 토큰의 로그확률에서 모읍니다. letter_mass가 1에 가까우면 전체 어휘에서 모은 값과 사실상 같습니다.

**실행**
- G4(96GB)에서 위에서부터 전체 실행합니다.
- 흐름: ① 검증 앞 300문항 → ② `GATE_ACC` 이상이면 검증 나머지 700 + test 6,714 → ③ E14가 읽는 파일 이름으로 저장.
- 끊기면 다시 전체 실행하면 이어서 합니다.
- 다른 모델 zero-shot (같은 300문항): Q35-27B 96.33 · Qwen3.6-35B-A3B 96.33 · HCX 95.67 · Flash-Next IQ3_XXS 95.33 · Gemma4-31B 97.67.
""")

md("## 1. 설정")
code('''
RUN_NAME   = "q35_122b_a10b_gptq_zs"            # 저장 폴더 = vqa_runs/<RUN_NAME>. 끝나면 E14가 자동으로 멤버로 잡음 (계열 = MODEL_ID)
MODEL_ID   = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" # Qwen 공식 4bit (전문가 층만 int4). 파일 78.9GB
STAGE      = "auto"     # "auto": 앞 300문항 정확도가 GATE_ACC 이상이면 이어서 전체 / "quick": 앞 300문항만 / "full": 무조건 전체
QUICK_N    = 300        # 다른 모델 zero-shot 기록과 같은 문항 (검증 앞 300)
GATE_ACC   = 96.0       # auto일 때 전체로 넘어가는 기준 (%). 같은 300문항: Q35-27B zero-shot 96.33 · Gemma4-31B 97.67
PRESET     = "up2.0"    # 다른 멤버의 평가 해상도와 같음
MAKE_SUBMISSION = True  # 전체가 끝나면 이 모델 단독 제출 파일도 저장

# ── vLLM ──
QUANTIZATION  = "moe_wna16"   # Qwen 모델 카드 권장값. 모델 로드에서 양자화 관련 오류가 나면 None (vLLM이 gptq_marlin을 고름)
MOE_BACKEND   = "auto"        # 전문가 층 커널. 첫 배치에서 커널 오류가 나면 "triton"
GPU_MEM_UTIL  = 0.92
MAX_MODEL_LEN = 8192          # 프롬프트 = 이미지 토큰(up2.0 최대 약 3,320) + 글자 수백
MAX_NUM_SEQS  = 16            # Qwen3.5(선형 어텐션 혼합)는 기본값(1024)이면 시작이 실패한 사례가 있어 작게
MAX_BATCHED_TOKENS = 16384
TOPK_LOGPROBS = 64            # 답 자리 상위 64개 토큰에서 a~d 변형을 모음
CHUNK         = 64            # 한 번에 넘기는 문항 수 = 저장 단위
IMG_WORKERS   = 6             # 이미지 읽기·크기 맞추기 병렬 수 (GPU가 도는 동안 다음 묶음을 미리 준비)
GDN_PREFILL_BACKEND = "triton"  # 선형 어텐션(GDN) 커널. vLLM 기본값(auto)은 Blackwell에서 FlashInfer판을 고르는데, 그 판은 CUTLASS 파이썬 DSL이 필요해 Colab에서 실패함 → Triton판 (A100 기본값과 같음)
ENGINE_KW     = {}            # 추가 vLLM 인자. 예: 어텐션 커널 오류 → {"attention_backend": "TRITON_ATTN"}

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
HF_CACHE     = "/content/hf_cache"     # 79GB 모델은 Drive가 아니라 런타임 로컬 디스크에 받음
SEED, VAL_SIZE = 42, 1000              # 검증 분할용 — 팀 공통값이므로 절대 바꾸지 마세요
HF_TOKEN   = ""                        # 필요 없음 (공개 모델). 비워 두면 Colab 보안 비밀 'HF_TOKEN'을 찾아봄

''' + RES_BLOCK.rstrip() + '''

import os, sys
# vLLM이 같이 까는 torchaudio(CUDA 12.8용)가 torch 2.13(CUDA 13)과 섞이면 import 순간 RuntimeError가 남 (2번 셀이 지우지만, 셀 순서가 꼬여도 안전하게)
# (transformers의 AutoProcessor와 vLLM이 torchaudio가 '있으면' 불러옴). 이 노트북은 오디오를 쓰지 않으므로
# 이 프로세스에서는 torchaudio를 '없는 것'으로 취급 → 둘 다 torchaudio 없이 동작. 반드시 transformers·vLLM보다 먼저 실행되는 이 셀에 둠
sys.modules["torchaudio"] = None
os.environ["HF_HOME"] = HF_CACHE                                  # 모델·프로세서 모두 로컬 디스크 캐시로 (huggingface_hub import 전에 설정)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")             # 대용량 다운로드 가속
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")      # vLLM 엔진을 노트북 프로세스 안에서 실행 (Colab의 fork/spawn 문제 회피)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")         # 초기화 때 top-k/top-p 표본 추출을 FlashInfer(JIT 컴파일 필요)로 시험하지 않게 → PyTorch판. 이 노트북은 그리디 1토큰이라 무관
''')

md("""
## 2. 설치 (vLLM 0.30.0 · transformers 5.17.0)
- vLLM 0.30.0의 기본 휠은 CUDA 13.0용입니다. 드라이버가 CUDA 13을 지원하지 않으면 CUDA 12.9용 휠을 받습니다.
- torch가 vLLM이 요구하는 버전(2.13)으로 바뀌므로 **새 런타임에서** 실행하세요. 다른 노트북과 같은 런타임을 쓰면 안 됩니다.
- vLLM 0.30은 의존성으로 torchaudio 2.11.0(CUDA 12.8용)을 같이 깝니다. 이것이 새 torch(CUDA 13)와 섞이면 transformers·vLLM을 import할 때 오류가 납니다. 이 노트북은 오디오를 쓰지 않으므로 두 겹으로 막습니다.
  - 이 셀이 torchaudio를 지웁니다. vLLM은 모델 구조를 별도 프로세스에서 검사하는데, 거기에는 아래 차단이 닿지 않기 때문입니다.
  - 1번 셀이 노트북 프로세스 안에서 torchaudio를 '없는 것'으로 취급합니다(`sys.modules["torchaudio"] = None`). 셀 순서가 꼬여도 이 프로세스는 안전합니다.
- 설치가 끝나면 vLLM의 검사 프로세스와 같은 조건(차단 없는 새 프로세스)에서 Qwen3.5 모델 모듈과 AutoProcessor를 import해 보고 버전을 출력합니다. 여기서 실패하면 멈춥니다.
- 이미 설치돼 있으면 설치는 건너뜁니다. torchaudio 지우기와 점검은 매번 합니다.
""")
code('''
import subprocess, re, shutil, sys
from importlib.metadata import version as _ver, PackageNotFoundError
VLLM_VERSION = "0.30.0"
def _installed(pkg):
    try: return _ver(pkg)
    except PackageNotFoundError: return None
_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
print(_smi[:800])
_m = re.search(r"CUDA Version:\\s*([\\d.]+)", _smi); DRIVER_CUDA = float(_m.group(1)) if _m else 0.0
if _installed("vllm") == VLLM_VERSION and _installed("transformers") == "5.17.0":
    print(f"vLLM {VLLM_VERSION} 이미 설치됨 → 건너뜀")
elif DRIVER_CUDA >= 13.0:
    !pip -q install "vllm=={VLLM_VERSION}" "transformers==5.17.0"
else:
    print(f"드라이버가 지원하는 CUDA {DRIVER_CUDA} < 13.0 → CUDA 12.9용 vLLM 휠을 받습니다")
    !pip -q install "https://github.com/vllm-project/vllm/releases/download/v{VLLM_VERSION}/vllm-{VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" "transformers==5.17.0" --extra-index-url https://download.pytorch.org/whl/cu129
print("vllm", _installed("vllm"), "| transformers", _installed("transformers"), "| torch", _installed("torch"))
# vLLM 0.30은 의존성으로 torchaudio==2.11.0(PyPI판 = CUDA 12.8용)을 같이 깔고, 이것이 torch 2.13(CUDA 13)과 섞이면 import 순간 RuntimeError가 남.
# 1번 셀의 차단은 이 노트북 프로세스에만 닿고, vLLM은 모델 구조를 '별도 프로세스'에서 검사하므로 거기서는 막을 수 없음 → 아예 지움 (오디오 미사용)
if _installed("torchaudio"):
    !pip -q uninstall -y torchaudio
    print("torchaudio 제거 (vLLM이 같이 깐 CUDA 12.8용 · 오디오 미사용)")
# 설치 점검: vLLM의 모델 검사 프로세스와 같은 조건(차단 없는 새 프로세스)에서 Qwen3.5 모델 모듈과 AutoProcessor를 import해 봄
_chk = subprocess.run([sys.executable, "-c", "import torch, torchvision, transformers, vllm; from transformers import AutoProcessor; "
                       "import vllm.model_executor.models.qwen3_5; "
                       "print('CHECK', torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, vllm.__version__)"],
                      capture_output=True, text=True)
_line = [l for l in _chk.stdout.splitlines() if l.startswith("CHECK")]
if _chk.returncode != 0 or not _line:
    print(_chk.stdout[-1500:], _chk.stderr[-3000:], sep="\\n")
    if "TorchAudio" in _chk.stderr: print("→ torchaudio가 아직 남아 있습니다. 새 셀에서 !pip uninstall -y torchaudio 를 한 번 더 실행한 뒤 이 셀을 다시 실행하세요")
    raise RuntimeError("설치 점검 실패 (위 오류) → 위 출력을 확인하세요")
print("✅ 설치 점검 OK (torch · CUDA · torchvision · transformers · vllm):", _line[-1][6:])
os.makedirs(HF_CACHE, exist_ok=True)
_free = shutil.disk_usage(HF_CACHE).free / 1e9
print(f"로컬 디스크 여유 {_free:.0f}GB (모델 78.9GB 필요)" + ("" if _free > 90 else " ⚠️ 부족할 수 있음 → 불필요한 파일을 지우세요"))
''')

md("## 3. 데이터 준비 (stage2와 같음)")
code(DATA_CELL)

md("## 4. CSV 로드 · 고정 검증 분할 (stage2와 같음)")
code('''
import re, json, time, math, hashlib, random, datetime, glob
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
random.seed(SEED); np.random.seed(SEED)

''' + CSV_BLOCK.rstrip() + '''

''' + SPLIT_CELL.rstrip() + '''

# E14 조합(검증 97.6%)의 오답 24개 — 라벨 오류 10 · 둘 다 답 8 · 진짜 실수 6 (OCR 실험 때와 같은 목록)
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}
E14_ERR = {k: [f"train_{x}.jpg" for x in v.split()] for k, v in E14_ERR.items()}
''')

md("""
## 5. 프롬프트 · 프로세서 (stage2와 같은 프롬프트)
프로세서(토크나이저·채팅 템플릿·이미지 처리)는 모델 저장소의 것을 transformers로 불러옵니다. vLLM도 내부에서 같은 프로세서를 씁니다.
""")
code(PROMPT_BLOCK.rstrip() + '''

from transformers import AutoProcessor
import transformers
processor = AutoProcessor.from_pretrained(MODEL_ID)
tok = processor.tokenizer
_tmpl = processor.chat_template if isinstance(processor.chat_template, str) else json.dumps(processor.chat_template)
TEMPLATE_KW = {"enable_thinking": False} if "enable_thinking" in (_tmpl or "") else {}
try:
    processor.apply_chat_template(build_messages({"question": "q", "a": "1", "b": "2", "c": "3", "d": "4"}), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW); USE_SYSTEM_ROLE = True
except Exception as _e:
    USE_SYSTEM_ROLE = False; print("system 역할 미지원 템플릿 → 지시문을 user 메시지 앞에 붙입니다")
IMAGE_TOKEN = getattr(processor, "image_token", "<|image_pad|>")
IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
if IMAGE_TOKEN_ID is None: IMAGE_TOKEN_ID = tok.convert_tokens_to_ids(IMAGE_TOKEN)

def render(row):
    return processor.apply_chat_template(build_messages(row), tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)

_p = render(val_df.iloc[0].to_dict())
assert _p.count(IMAGE_TOKEN) == 1, f"프롬프트에 이미지 자리({IMAGE_TOKEN})가 정확히 하나여야 합니다: {_p[:300]!r}"
print(f"transformers {transformers.__version__} | {type(processor).__name__} | 템플릿 추가 인자 {TEMPLATE_KW} | system 역할 {USE_SYSTEM_ROLE} | 이미지 토큰 {IMAGE_TOKEN}={IMAGE_TOKEN_ID}")
print("프롬프트 끝부분:", repr(_p[-120:]))

''' + LETTER_BLOCK.rstrip() + '''
''')

md("""
## 6. 이미지: transformers 5.17 프로세서와 똑같이 크기 미리 맞추기 + 대조
- 원본을 up2.0 범위(270만~340만 픽셀, 가로·세로 32의 배수)로 키우거나 줄여서 넘깁니다. 크기 계산(`smart_resize`)과 보간(bicubic, antialias)은 5.17 코드와 같습니다.
- vLLM 쪽 프로세서에는 넓은 범위를 줘서 미리 맞춘 이미지를 다시 바꾸지 않게 합니다.
- 아래 점검은 검증 3장에서 두 경로의 픽셀과 격자가 같은지 봅니다. 하나라도 다르면 미리 맞추기를 끄고, vLLM 프로세서가 원본을 up2.0 범위로 직접 맞추게 합니다(stage2와 같은 경로).
""")
code('''
from torchvision.transforms.v2 import functional as tvF
from torchvision.transforms import InterpolationMode
IP = processor.image_processor
FACTOR = int(IP.patch_size) * int(IP.merge_size)              # 16 × 2 = 32 픽셀 = 이미지 토큰 1개의 한 변
MIN_PX, MAX_PX = RES_PRESETS[PRESET]

def smart_resize(height, width, factor, min_pixels, max_pixels):     # transformers 5.17 qwen2_vl 이미지 프로세서와 같은 계산
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def prep_image(rel):
    im = load_image(rel)
    rh, rw = smart_resize(im.height, im.width, FACTOR, MIN_PX, MAX_PX)
    if (rh, rw) != (im.height, im.width):
        t = tvF.resize(tvF.pil_to_tensor(im), [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True)
        im = Image.fromarray(t.permute(1, 2, 0).contiguous().numpy())
    return im, (rh, rw)

PRE_RESIZE = True
VLLM_MM_KW = {"min_pixels": FACTOR * FACTOR, "max_pixels": int(MAX_PX * 1.1)}     # 미리 맞춘 크기를 그대로 두는 넓은 범위 (vLLM 메모리 예약도 이 상한 기준)
for _r in val_df.head(3).to_dict("records"):
    _a = IP(images=[load_image(_r["path"])], min_pixels=MIN_PX, max_pixels=MAX_PX, return_tensors="pt")
    _im, (_h, _w) = prep_image(_r["path"])
    _b = IP(images=[_im], **VLLM_MM_KW, return_tensors="pt")
    _same = _a["image_grid_thw"].tolist() == _b["image_grid_thw"].tolist() and _a["pixel_values"].shape == _b["pixel_values"].shape
    _diff = float((_a["pixel_values"].float() - _b["pixel_values"].float()).abs().max()) if _same else float("inf")
    _ntok = int(_a["image_grid_thw"][0].prod()) // int(IP.merge_size) ** 2
    print(f"  {_r['id']}: 원본 {load_image(_r['path']).size} → {(_w, _h)} | 이미지 토큰 {_ntok} (계산 {(_h // FACTOR) * (_w // FACTOR)}) | 픽셀 최대 차이 {_diff:.2e}")
    if not (_same and _diff < 1e-6 and _ntok == (_h // FACTOR) * (_w // FACTOR)): PRE_RESIZE = False
if not PRE_RESIZE:
    VLLM_MM_KW = {"min_pixels": int(MIN_PX), "max_pixels": int(MAX_PX)}
    print("⚠️ 미리 맞춘 이미지가 프로세서 결과와 달라 미리 맞추기를 끕니다 → vLLM 프로세서가 원본을 up2.0 범위로 직접 맞춤 (stage2와 같은 경로)")
else:
    print(f"✅ 미리 맞추기 = 5.17 프로세서와 같은 픽셀 | vLLM 프로세서 범위 {VLLM_MM_KW}")
''')

md("""
## 7-0. 모델 파일 받기 (79GB · 진행 막대 표시 · 끊기면 다시 실행하면 이어받기)
vLLM은 모델을 받을 때 진행 막대를 숨깁니다. 그래서 여기서 먼저 받아 둡니다. 받는 동안 멈춘 것처럼 보이지 않게 하려는 것이고, 다음 셀은 이 캐시에서 바로 불러옵니다.
""")
code('''
from huggingface_hub import snapshot_download
t0 = time.time()
MODEL_DIR = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
_sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(MODEL_DIR) for f in fs) / 1e9
_nsh = len([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
print(f"모델 파일 {_sz:.1f}GB · 가중치 조각 {_nsh}개 | {time.time() - t0:.0f}s | {MODEL_DIR}")
if _sz < 70: print("⚠️ 78.9GB보다 많이 작음 → 다운로드가 덜 된 것 같습니다. 이 셀을 다시 실행하면 이어서 받습니다")
''')

md("""
## 7. vLLM으로 모델 로드
- 7-0에서 받은 파일을 불러옵니다. 로드가 끝나면 GPU 메모리 사용량이 출력됩니다.
- Colab의 노트북 출력 스트림에는 파일 번호가 없어, vLLM이 초기화 중 `sys.stdout.fileno()`에서 멈춥니다(`UnsupportedOperation: fileno`). 이 셀은 그 부분(C 라이브러리 출력 숨기기)만 꺼서 우회합니다.
- 오류가 나면 1번 셀에서 아래처럼 바꾼 뒤 **세션 다시 시작 → 1번 셀부터** 실행하세요 (이미 받은 모델은 다시 받지 않음).

| 증상 | 바꿀 설정 |
|---|---|
| 양자화 관련 오류 (`moe_wna16`, `gptq`) | `QUANTIZATION = None` |
| 첫 배치에서 전문가 층 커널 오류 | `MOE_BACKEND = "triton"` |
| 어텐션 백엔드 오류 | `ENGINE_KW = {"attention_backend": "TRITON_ATTN"}` |
| 시작할 때 메모리 부족 | `MAX_NUM_SEQS = 8`, `MAX_BATCHED_TOKENS = 8192` |
| `No module named 'cutlass'` (선형 어텐션 FlashInfer 커널) | `GDN_PREFILL_BACKEND = "triton"` (기본값으로 이미 설정) |

로드 도중 오류가 나면 이 셀을 그대로 다시 실행하지 말고 세션을 다시 시작하세요. 엔진이 노트북 프로세스 안에 있어 반쯤 초기화된 상태가 남기 때문입니다.
""")
code('''
import vllm, contextlib, io, sys
from vllm import LLM, SamplingParams
try:
    sys.stdout.fileno(); _FILENO_OK = True
except (io.UnsupportedOperation, AttributeError, ValueError):
    _FILENO_OK = False
if not _FILENO_OK:
    # Colab 노트북의 출력 스트림에는 파일 번호가 없음. vLLM은 분산 초기화 때 C 라이브러리 출력을 숨기려고 sys.stdout.fileno()를 부르다 멈춤
    # (엔진을 노트북 프로세스 안에서 돌리기 때문) → 그 숨기기 기능만 끔. 결과에는 영향 없고, gloo 초기화 메시지가 숨겨지지 않을 뿐
    import vllm.utils.system_utils as _su, vllm.distributed.parallel_state as _ps, vllm.distributed.utils as _du
    @contextlib.contextmanager
    def _no_suppress_stdout(): yield
    for _mod in (_su, _ps, _du): _mod.suppress_stdout = _no_suppress_stdout
    print("Colab 출력 스트림 → vLLM의 stdout 숨기기(suppress_stdout)를 끔")
if "llm" not in globals():
    t0 = time.time()
    llm = LLM(model=MODEL_ID, quantization=QUANTIZATION, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEM_UTIL, max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_BATCHED_TOKENS,
              limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_kwargs=VLLM_MM_KW, mm_processor_cache_gb=0,
              enforce_eager=True, enable_prefix_caching=False, max_logprobs=TOPK_LOGPROBS, seed=SEED, moe_backend=MOE_BACKEND,
              gdn_prefill_backend=GDN_PREFILL_BACKEND, **ENGINE_KW)
    print(f"모델 로드 {time.time() - t0:.0f}s")
else:
    print("이미 로드된 모델을 그대로 씀 (설정을 바꿨다면 세션을 다시 시작하세요)")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
_free, _tot = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
print(f"vllm {vllm.__version__} | {MODEL_ID} | {GPU_NAME} | GPU 사용 {(_tot - _free) / 2**30:.1f} / {_tot / 2**30:.0f}GiB")
SP = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK_LOGPROBS, detokenize=False)
''')

md("""
## 8. 채점 함수 (끊겨도 이어서)
- 문항마다 답 자리 상위 64개 토큰에서 글자별로 변형 확률을 합칩니다. 그 뒤 네 글자끼리 정규화합니다(stage2와 같은 계산).
- 저장 열은 stage2와 같습니다: `id, pred, p_a~p_d, letter_mass, img_tokens`.
- 점검용 열을 더 저장합니다.
  - `img_tokens_exp`: 5.17 계산으로 기대한 이미지 토큰 수
  - `n_missing`: 상위 64개에 한 변형도 없어 0으로 둔 글자 수
- 첫 묶음 뒤에 letter_mass, 이미지 토큰 수 일치, 글자 누락을 확인합니다.
""")
code('''
from concurrent.futures import ThreadPoolExecutor
IMG_POOL, PREFETCH = ThreadPoolExecutor(max_workers=IMG_WORKERS), ThreadPoolExecutor(max_workers=1)

def _make_req(r):
    if PRE_RESIZE:
        im, (h, w) = prep_image(r["path"])
    else:
        im = load_image(r["path"]); h, w = smart_resize(im.height, im.width, FACTOR, MIN_PX, MAX_PX)
    return {"prompt": render(r), "multi_modal_data": {"image": im}}, (h // FACTOR) * (w // FACTOR)

def _prep_chunk(rows): return list(IMG_POOL.map(_make_req, rows))

def _parse(r, n_exp, o):
    lp = o.outputs[0].logprobs[0]                                       # {토큰 id: Logprob} (답 자리 상위 TOPK_LOGPROBS개)
    slot = np.array([np.logaddexp.reduce([lp[i].logprob for i in ids if i in lp]) if any(i in lp for i in ids) else -np.inf
                     for ids in LETTER_IDS], dtype=np.float64)
    mass = float(np.exp(slot).sum())
    p = np.exp(slot - slot.max()) if np.isfinite(slot).any() else np.ones(4)
    p = p / p.sum()
    top = max(lp.items(), key=lambda kv: kv[1].logprob)[0]
    n_img = sum(1 for t in o.prompt_token_ids if t == IMAGE_TOKEN_ID)
    return {"id": r["id"], "pred": LETTERS[int(p.argmax())], "p_a": p[0], "p_b": p[1], "p_c": p[2], "p_d": p[3], "letter_mass": mass,
            "img_tokens": int(n_img), "img_tokens_exp": int(n_exp), "n_missing": int((~np.isfinite(slot)).sum()),
            "top_token": int(top), "prompt_len": len(o.prompt_token_ids)}

def _check_first(recs, ch, reqs):
    d = pd.DataFrame(recs); mism = d[d.img_tokens != d.img_tokens_exp]
    print(f"  첫 묶음 점검: letter_mass 평균 {d.letter_mass.mean():.3f} (최소 {d.letter_mass.min():.3f}) | 이미지 토큰 불일치 {len(mism)}/{len(d)} | "
          f"글자 누락 있는 문항 {int((d.n_missing > 0).sum())} | 프롬프트 길이 최대 {d.prompt_len.max()}")
    if len(mism): print("  ⚠️ 이미지 토큰 수가 5.17 계산과 다름 (해상도가 다른 멤버와 조금 다를 수 있음). 예:", mism[["id", "img_tokens", "img_tokens_exp"]].head(3).to_dict("records"))
    if d.letter_mass.mean() < 0.5:
        for r, x in list(zip(ch, recs))[:3]:
            print(f"    {r['id']} 답 자리 최상위 토큰: {tok.decode([x['top_token']])!r}")
        raise RuntimeError("letter_mass가 너무 낮음 → 답 자리에서 a~d가 아닌 토큰을 냄 (템플릿·생각 모드 문제일 수 있음). 위 출력을 확인하세요.")

def _fix_tail(path):                                                    # 저장 도중 끊겨 마지막 줄이 줄바꿈 없이 끝났으면 보정
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\\n": f.write(b"\\n")

def read_preds(path):
    # 끊겨서 반쯤 쓴 줄은 마지막 열(prompt_len)이 비어 있음 → 버리고 다시 채점
    d = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["id", "p_a", "p_b", "p_c", "p_d", "prompt_len"]).drop_duplicates("id", keep="last")
    for c in ["img_tokens", "img_tokens_exp", "n_missing", "top_token", "prompt_len"]:
        if c in d.columns: d[c] = d[c].astype("int64")             # 반쯤 쓴 줄 때문에 실수형이 된 정수 열 되돌리기
    return d

def predict(df, out_path, label):
    _fix_tail(out_path)
    done = set(read_preds(out_path).id) if os.path.exists(out_path) else set()
    rows = df[~df.id.isin(done)].to_dict("records")
    print(f"[{label}] {os.path.basename(out_path)} | 전체 {len(df):,} | 완료 {len(df) - len(rows):,} | 남음 {len(rows):,}", flush=True)
    if not rows: return
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    fut, t0, n = PREFETCH.submit(_prep_chunk, chunks[0]), time.time(), 0
    for k, ch in enumerate(chunks):
        reqs = fut.result()
        if k + 1 < len(chunks): fut = PREFETCH.submit(_prep_chunk, chunks[k + 1])      # GPU가 도는 동안 다음 묶음 이미지 준비
        outs = llm.generate([q for q, _ in reqs], SP, use_tqdm=False)
        if len(outs) > len(ch):      # 이전 실행을 중간에 멈춰 엔진에 남아 있던 요청 → 요청 번호가 작아 앞쪽에 옴 → 버림
            print(f"  (엔진에 남아 있던 이전 요청 {len(outs) - len(ch)}개의 결과는 버림)"); outs = outs[-len(ch):]
        assert len(outs) == len(ch) and all(getattr(o, "prompt", None) in (None, q["prompt"]) for o, (q, _) in zip(outs, reqs)), "요청과 결과의 순서가 맞지 않음"
        recs = [_parse(r, e, o) for r, (_, e), o in zip(ch, reqs, outs)]
        pd.DataFrame(recs).to_csv(out_path, mode="a", header=not os.path.exists(out_path) or os.path.getsize(out_path) == 0, index=False)
        n += len(ch)
        if k == 0 and not done: _check_first(recs, ch, reqs)
        if k == 0 or (k + 1) % 10 == 0 or k + 1 == len(chunks):
            el = time.time() - t0
            print(f"  {n:>5}/{len(rows)} | {el / n:.2f}s/문항 | 남은 시간 ~{(len(rows) - n) * el / n / 60:.0f}분", flush=True)
''')

md("""
## 9. 결과 정리 · 같은 문항 비교 함수
- **zero-shot 기록**: Drive의 다른 실행 `preds_zeroshot*.csv` (E5 폴더의 Q35, Flash-Next, stage2_newfam 실행들)
- **학습한 멤버**: `preds_final_ep*__up2.0__1000.csv`. 이 모델이 고친 문항 / 새로 틀린 문항 수와 부호 검정 p값을 보입니다.
- **E14 오답 24개**: 라벨 오류 10 · 둘 다 답 8 · 진짜 실수 6 중 이 모델이 맞힌 것
""")
code(CATS_BLOCK.rstrip() + '''

RUN_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
VAL_WORK  = os.path.join(RUN_DIR, f"preds_zeroshot__{PRESET}__{len(val_df)}.csv")         # 진행 중 파일 (다른 노트북의 zero-shot 비교에도 잡힘)
TEST_WORK = os.path.join(RUN_DIR, f"preds_zeroshot_test__{PRESET}__{len(test_df)}.csv")
VAL_FINAL  = os.path.join(RUN_DIR, f"preds_final_ep1__{PRESET}__{len(val_df)}.csv")       # E14가 읽는 이름 (전체가 끝난 뒤에만 씀)
TEST_FINAL = os.path.join(RUN_DIR, f"preds_final_ep1_test__{PRESET}__{len(test_df)}.csv")
SUMMARY = {}

def report(df, label):
    m = df.merge(read_preds(VAL_WORK), on="id", how="inner"); m["correct"] = m.pred == m.answer
    acc, n = float(m.correct.mean()), len(m); se = math.sqrt(max(acc * (1 - acc), 1e-9) / n)
    print(f"▶ {label} | {PRESET} | 정확도 {acc * 100:.2f}% ({int(m.correct.sum())}/{n}, ±{1.96 * se * 100:.1f}%p) | letter_mass {m.letter_mass.mean():.3f} | "
          f"이미지 토큰 불일치 {int((m.img_tokens != m.img_tokens_exp).sum())} | 글자 누락 문항 {int((m.n_missing > 0).sum())}")
    for k, pat in CATS.items():
        sel = m.question.str.contains(pat)
        if sel.sum() >= 10: print(f"    {k:<12} {m.correct[sel].mean() * 100:5.1f}%  (n={int(sel.sum())})")
    lp = os.path.join(OUT_DIR, "results_log_s2.csv")
    log = {"time": datetime.datetime.now().strftime("%m-%d %H:%M"), "run": f"{RUN_NAME}/zeroshot_{n}", "model": MODEL_ID, "res": PRESET, "tta": 0,
           "n": n, "acc": round(acc * 100, 2), "ci95": round(1.96 * se * 100, 2), "letter_mass": round(float(m.letter_mass.mean()), 3),
           "gpu": GPU_NAME, "split_md5": split_md5[:8]}
    pd.DataFrame([log]).to_csv(lp, mode="a", header=not os.path.exists(lp), index=False)
    return m, acc

def _sign_p(k, n):                                                        # 양쪽 부호 검정 (고침 k · 새로 틀림 n-k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0

def compare(m, label, top=12):
    ids, mine = set(m.id), m.set_index("id").pred
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*", "preds_zeroshot*.csv")) + glob.glob(os.path.join(OUT_DIR, "*", f"preds_final_ep*__{PRESET}__1000.csv"))):
        run = os.path.basename(os.path.dirname(f))
        if os.path.abspath(os.path.dirname(f)) == os.path.abspath(RUN_DIR) or run.endswith("_dry"): continue
        try: o = pd.read_csv(f, usecols=["id", "pred"]).drop_duplicates("id", keep="last").set_index("id").pred
        except Exception: continue
        if not ids <= set(o.index): continue                              # 같은 문항을 다 가진 파일만
        o = o.loc[m.id].to_numpy(); ans = m.answer.to_numpy(); me = m.pred.to_numpy()
        fix, brk = int(((me == ans) & (o != ans)).sum()), int(((me != ans) & (o == ans)).sum())
        rows.append({"kind": "zero-shot" if "preds_zeroshot" in os.path.basename(f) else "학습", "run": run, "file": os.path.basename(f),
                     "acc": (o == ans).mean() * 100, "fix": fix, "break": brk, "p": _sign_p(fix, fix + brk)})
    print(f"\\n같은 {len(m)}문항 비교 ({label}) — 이 모델 {(m.pred == m.answer).mean() * 100:.2f}% | '고침'= 이 모델만 맞힘, '새로 틀림'= 상대만 맞힘")
    for kind in ["zero-shot", "학습"]:
        R = sorted([r for r in rows if r["kind"] == kind], key=lambda r: -r["acc"])
        if not R: print(f"  ({kind} 비교 파일 없음)"); continue
        print(f"  [{kind}] {len(R)}개" + (f" (정확도 상위 {top}개만)" if len(R) > top else ""))
        for r in R[:top]:
            print(f"    {r['acc']:6.2f}%  {r['run']}/{r['file']} | 고침 {r['fix']} · 새로 틀림 {r['break']} (p={r['p']:.3f})")
    print("  E14 오답 24개 중 이 모델이 맞힌 것 (이 범위에 든 문항만):")
    for cat, lst in E14_ERR.items():
        sub = m[m.id.isin(lst)]
        if len(sub): print(f"    {cat}: {int(sub.correct.sum())}/{len(sub)} | " + " ".join(f"{i.replace('train_', '').replace('.jpg', '')}{'○' if c else '×'}" for i, c in zip(sub.id, sub.correct)))
    return rows
''')

md("""
## 10. ① 검증 앞 300문항 → 계속할지 판단
`STAGE="auto"`면 정확도가 `GATE_ACC`(96.0%) 이상일 때만 ②로 넘어갑니다.
""")
code('''
quick_df = val_df.head(QUICK_N)
predict(quick_df, VAL_WORK, f"검증 앞 {QUICK_N}")
m_q, acc_q = report(quick_df, f"zero-shot 검증 앞 {len(quick_df)}")
compare(m_q, f"검증 앞 {len(quick_df)}")
SUMMARY["quick"] = {"n": len(m_q), "acc": round(acc_q * 100, 2), "letter_mass": round(float(m_q.letter_mass.mean()), 4)}
GO_FULL = STAGE == "full" or (STAGE == "auto" and acc_q * 100 >= GATE_ACC)
print(f"\\n→ STAGE={STAGE}, 앞 {len(m_q)}문항 {acc_q * 100:.2f}% (기준 {GATE_ACC}%) → " + ("② 검증 1,000 + test로 계속" if GO_FULL else "여기서 멈춤"))
''')

md("""
## 11. ② 검증 1,000 + test 6,714 → E14 형식으로 저장
- 검증 나머지 700을 채점한 뒤 같은 문항 비교를 하고, 이어서 test를 채점합니다.
- 둘 다 끝나면 E14가 읽는 이름으로 씁니다.
  - `preds_final_ep1__up2.0__1000.csv`, `preds_final_ep1_test__up2.0__6714.csv`
  - `train_log.json` (`MODEL_ID` = 새 계열, `ZERO_SHOT: true`)
- `MAKE_SUBMISSION=True`면 이 모델 단독 제출 파일도 씁니다.
""")
code('''
if GO_FULL:
    predict(val_df, VAL_WORK, "검증 1,000")
    m_v, acc_v = report(val_df, f"zero-shot 검증 {len(val_df)}")
    cmp_rows = compare(m_v, f"검증 {len(val_df)}")
    SUMMARY["val"] = {"n": len(m_v), "acc": round(acc_v * 100, 2), "letter_mass": round(float(m_v.letter_mass.mean()), 4),
                      "img_tok_mismatch": int((m_v.img_tokens != m_v.img_tokens_exp).sum()), "missing_letter_items": int((m_v.n_missing > 0).sum()),
                      "e14_err_right": {k: int(m_v[m_v.id.isin(v)].correct.sum()) for k, v in E14_ERR.items()}}
    predict(test_df, TEST_WORK, "test")
    pv = read_preds(VAL_WORK).set_index("id"); pt = read_preds(TEST_WORK).set_index("id")
    miss_v, miss_t = [i for i in val_df.id if i not in pv.index], [i for i in test_df.id if i not in pt.index]
    assert not miss_v and not miss_t, f"예측 누락: 검증 {len(miss_v)} · test {len(miss_t)} → 이 셀을 다시 실행하면 남은 것만 채점합니다"
    pv.loc[val_df.id].reset_index().to_csv(VAL_FINAL, index=False)
    pt.loc[test_df.id].reset_index().to_csv(TEST_FINAL, index=False)
    json.dump({"epoch_acc": {"1": acc_v}, "history": [{"step": 0, "zero_shot_acc": acc_v, "zero_shot_n": len(m_v)}],
               "config": {"MODEL_ID": MODEL_ID, "ZERO_SHOT": True, "TRAIN_ON_FULL": False, "SEED": SEED, "EVAL_PRESET": PRESET,
                          "PROMPT_VERSION": PROMPT_VERSION, "TEMPLATE_KW": TEMPLATE_KW, "ENGINE": f"vllm {vllm.__version__}", "QUANTIZATION": QUANTIZATION,
                          "MOE_BACKEND": MOE_BACKEND, "TOPK_LOGPROBS": TOPK_LOGPROBS, "PRE_RESIZE": PRE_RESIZE, "VLLM_MM_KW": VLLM_MM_KW,
                          "SPLIT_MD5": split_md5}},
              open(os.path.join(RUN_DIR, "train_log.json"), "w"), indent=1, default=str)
    print(f"\\n✅ E14 형식 저장: {os.path.basename(VAL_FINAL)} · {os.path.basename(TEST_FINAL)} · train_log.json → {RUN_DIR}")
    if MAKE_SUBMISSION:
        sub = sub_df[["id"]].merge(pt["pred"].rename("answer").reset_index(), on="id", how="left")
        assert len(sub) == len(sub_df) and list(sub.id) == list(sub_df.id) and sub.answer.notna().all() and set(sub.answer) <= set(LETTERS)
        sp = os.path.join(RUN_DIR, f"submission_{RUN_NAME}_final_ep1_{PRESET}.csv"); sub.to_csv(sp, index=False)
        print("단독 제출 파일:", sp, "| 예측 분포:", sub.answer.value_counts().sort_index().to_dict())
    SUMMARY["test"] = {"n": len(pt), "letter_mass": round(float(pt.letter_mass.mean()), 4), "img_tok_mismatch": int((pt.img_tokens != pt.img_tokens_exp).sum()),
                       "pred_dist": pt.pred.value_counts().sort_index().to_dict()}
else:
    print("② 건너뜀 (STAGE / GATE_ACC). 계속하려면 1번 셀에서 STAGE=\\"full\\"로 바꾸고 다시 실행하세요 — ①의 결과는 재사용됩니다.")
''')

md("## 12. 결과 요약")
code('''
print("RUN:", RUN_NAME, "| MODEL:", MODEL_ID, "| vllm", vllm.__version__, "| GPU:", GPU_NAME, "| 양자화:", QUANTIZATION, "| 크기 미리 맞추기:", PRE_RESIZE)
for k, v in SUMMARY.items(): print(f"  {k}: {v}")
print("  참고 (같은 앞 300문항 zero-shot): Q35-27B 96.33 · Qwen3.6-35B-A3B 96.33 · HCX 95.67 · Flash-Next IQ3_XXS 95.33 · InternVL 94.33 · Gemma4-31B 97.67")
print("  파일:", sorted(f for f in os.listdir(RUN_DIR) if f.endswith((".csv", ".json"))))
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "G4", "machine_shape": "hm"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
