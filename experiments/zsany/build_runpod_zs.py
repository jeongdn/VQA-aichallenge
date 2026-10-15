# -*- coding: utf-8 -*-
"""RunPod(B300 등)용 '아무 모델 zero-shot' 노트북 빌더.
zs_any_vllm.ipynb(Colab판)을 그대로 쓰고, 경로·데이터·결과 올리기만 RunPod 방식(아까 Flash-Next 때와 같음)으로 바꾼다.
- 데이터: HF 비공개 저장소의 대회 zip → /workspace/data (runpod_flashnext 노트북 3번 셀과 같음)
- 가중치 캐시: /workspace/hf 한 곳 (HF_HOME·HF_HUB_CACHE·HF_XET_CACHE를 모두 고정)
- 결과: /workspace/vqa_runs/<RUN_NAME>/ → 마지막 셀이 HF 비공개 저장소 vqa_runs/<RUN_NAME>/로 올림 → Colab 브리지 노트북 B 단계로 Drive에 받기
사용: python3 build_runpod_zs.py [출력 경로]"""
import sys, json, uuid, os, io, contextlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/zsany/out/runpod_zs_any.ipynb"
SPD = "/tmp/vqa_scratch"
_a = {}
_argv = sys.argv; sys.argv = ["build_zsany.py", f"{SPD}/_zsany_tmp_rp.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/zsany/build_zsany.py", encoding="utf-8").read(), _a)
sys.argv = _argv
A = _a["cells"]
def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:90]); return text.replace(old, new)
def find(marker):
    hits = [i for i, c in enumerate(A) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(hits) == 1, (marker, hits); return hits[0]

# ── 1. 설정: RunPod 경로 · HF 저장소 · 캐시 고정 ──
i_cfg = find('MODEL_ID  = "google/gemma-4-31B-it"')
s = A[i_cfg]["source"]
s = rep(s, '''DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
HF_CACHE     = "/content/hf_cache"
SEED, VAL_SIZE = 42, 1000
HF_TOKEN   = ""                 # Gemma처럼 라이선스 동의가 필요한 모델: 비워 두면 Colab 보안 비밀 'HF_TOKEN'을 씀
''', '''# ── RunPod 경로 (바꾸지 마세요. 모두 볼륨 디스크 /workspace 안) ──
ZIP_PATH     = "/workspace/ssafy-16-2-ai.zip"   # 대회 데이터 zip. 없으면 3번 셀이 HF 비공개 저장소에서 받음
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/workspace/data"
OUT_DIR      = "/workspace/vqa_runs"
HF_CACHE     = "/workspace/hf"                  # 가중치 캐시는 볼륨 디스크의 이 한 곳 (Flash-Next 노트북·터미널 다운로드와 같은 곳)
HF_REPO      = ""                               # 비워 두면 "<내 HF 아이디>/ssafy-vqa-private" (Colab 브리지 노트북과 같은 규칙)
HF_ZIP_NAME  = "ssafy-16-2-ai.zip"
SEED, VAL_SIZE = 42, 1000
HF_TOKEN   = ""                 # 비워 두면 pod 환경 변수 HF_TOKEN → 그것도 없으면 3번 셀에서 입력창 (Write 권한 토큰. Gemma는 라이선스 동의한 계정)
''')
s = rep(s, 'os.environ["HF_HOME"] = HF_CACHE                                  # 모델·프로세서 모두 로컬 디스크 캐시로 (huggingface_hub import 전에 설정)',
        '_HF_PRE = {k: os.environ[k] for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE") if os.environ.get(k) and not os.environ[k].startswith(HF_CACHE)}\n'
        'os.environ.update({"HF_HOME": HF_CACHE, "HF_HUB_CACHE": HF_CACHE + "/hub", "HF_XET_CACHE": HF_CACHE + "/xet"}); os.environ.pop("HUGGINGFACE_HUB_CACHE", None)   # pod에 미리 정해진 캐시 경로를 덮어써 두 번 받지 않게\n'
        'if _HF_PRE: print(f"(pod에 미리 정해진 HF 캐시 경로 {_HF_PRE} 대신 {HF_CACHE} 하나만 씁니다)")\n'
        'assert os.path.isdir("/workspace"), "/workspace가 없습니다 → pod 설정에서 Volume Disk(마운트 경로 /workspace)를 켰는지 확인하세요"')
s = rep(s, 'MAX_NUM_SEQS  = 32', 'MAX_NUM_SEQS  = 32              # B300이면 64로 올려도 됨 (시작이 실패하면 16)')
s = rep(s, 'VAL_N     = 6714                      # 검증에 쓸 train 문항 수 (앞 1,000 = 팀 공통 검증). 빨리 보려면 1000 → 나중에 늘리면 이어서 채점',
        'VAL_N     = 1000                      # 1000 = 팀 공통 검증 (기준 교체 노트북이 쓰는 문항 · Flash-Next가 학습에 안 쓴 문항). 6714로 늘리면 새 문항만 이어서 채점')
s = rep(s, 'MAKE_SUBMISSION = True\n', 'MAKE_SUBMISSION = True\nGPU_ID    = None                      # GPU 여러 장 pod에서 노트북 여러 개를 동시에: 노트북마다 0, 1, 2, … (None = 전부 보이게, GPU 1장이면 그대로)\n')
s = rep(s, 'sys.modules["torchaudio"] = None\n', 'sys.modules["torchaudio"] = None\n'
        'if GPU_ID is not None:\n'
        '    assert "torch" not in sys.modules, "GPU_ID는 커널을 새로 시작한 직후에만 적용됩니다 → Kernel → Restart Kernel and Run All Cells"\n'
        '    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID); print(f"이 노트북은 GPU {GPU_ID}번만 씁니다")\n')
s = rep(s, 'MOE_BACKEND   = "auto"\n',
        'MOE_BACKEND   = "triton"        # B300: FlashInfer MoE 커널은 즉석 컴파일이 필요한데 pod의 nvcc가 B300(sm_103)을 몰라 실패 → Triton (MoE 모델만 해당)\n'
        'ATTN_BACKEND  = "TRITON_ATTN"   # B300: 같은 이유로 FlashInfer 어텐션 대신 Triton 어텐션. None = vLLM 자동 선택(G4·H100 등)\n')
s = rep(s, 'for _k, _v in MODEL_PRESETS.get(MODEL_ID, {}).items(): globals()[_k] = _v\n',
        'for _k, _v in MODEL_PRESETS.get(MODEL_ID, {}).items(): globals()[_k] = _v\n'
        'if ATTN_BACKEND: ENGINE_KW.setdefault("attention_config", {"backend": ATTN_BACKEND})\n'
        'ENGINE_KW.setdefault("enable_flashinfer_autotune", False)   # FlashInfer 자동 튜닝(즉석 컴파일)도 건너뜀 → "nvcc fatal: Unsupported gpu architecture \'compute_103a\'" 방지\n')
A[i_cfg]["source"] = s

# ── 2. 설치: zs122b와 같은 vLLM 설치 + 7z (대회 zip 풀기) ──
i_ins = find('VLLM_VERSION = "0.30.0"')
A[i_ins]["source"] = A[i_ins]["source"].rstrip("\n") + r'''
if shutil.which("7z") is None:
    subprocess.run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq p7zip-full unzip", shell=True, capture_output=True, text=True)
print("압축 도구:", shutil.which("7z") or shutil.which("unzip") or "없음 → 3번 셀이 파이썬 zipfile로 풉니다(느림)")
''' + "\n"
A[i_ins]["source"] = rep(A[i_ins]["source"], 'print(f"로컬 디스크 여유 {_free:.0f}GB (모델 78.9GB 필요)" + ("" if _free > 90 else " ⚠️ 부족할 수 있음 → 불필요한 파일을 지우세요"))',
                         'print(f"볼륨 디스크(/workspace) 여유 {_free:.0f}GB — 모델 크기만큼 필요 (7-0 셀에서 확인)")')

A[i_ins]["source"] = rep(A[i_ins]["source"], 'if _installed("torchaudio"):\n',
    '# RunPod 이미지에는 pandas가 없고, cutlass 모듈이 빠져 vLLM이 멈추는 경우가 있음 (실제 B300 pod에서 확인) → 없을 때만 설치\n'
    '_extra = [p for p in ("pandas", "nvidia-cutlass") if not _installed(p)]\n'
    'if _extra:\n'
    '    _extra_s = " ".join(_extra)\n'
    '    !pip -q install {_extra_s}\n'
    '    print("추가 설치:", _extra)\n'
    'if _installed("torchaudio"):\n')

# ── 3. 데이터: HF 비공개 저장소의 zip (runpod_flashnext 3번 셀과 같음) ──
i_dat = find('drive.mount("/content/drive")')
A[i_dat]["source"] = r'''
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
    else:
        import zipfile
        with zipfile.ZipFile(ZIP_PATH) as _z: _z.extractall(DATA_DIR, pwd=ZIP_PASSWORD.encode())
    print(f"  완료 ({time.time() - t0:.0f}초)")
_dd = _find_data_dir(DATA_DIR)
assert _dd is not None, f"압축을 풀었는데 {DATA_DIR} 아래에 train.csv가 없습니다 → zip 내용을 확인하세요"
if _dd != DATA_DIR: print(f"train.csv가 하위 폴더에 있어 DATA_DIR을 바꿉니다: {_dd}"); DATA_DIR = _dd
for f in ["train.csv", "test.csv", "sample_submission.csv"]:
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"{f} 가 {DATA_DIR} 에 없습니다"
print("데이터 OK:", DATA_DIR, "| 결과 저장:", OUT_DIR)
'''.strip("\n") + "\n"

# ── 7-0. 받기 전에 디스크 공간 확인 (모델 크기를 HF에서 조회) ──
i_dl = find("MODEL_DIR = snapshot_download(MODEL_ID")
A[i_dl]["source"] = rep(A[i_dl]["source"], '''import shutil
from huggingface_hub import snapshot_download
t0 = time.time()''', '''import shutil
from huggingface_hub import snapshot_download, HfApi
try:
    _info = HfApi(token=HF_TOKEN).model_info(MODEL_ID, files_metadata=True)
    _need = sum((s.size or 0) for s in _info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
    _free0 = shutil.disk_usage("/workspace").free / 1e9
    print(f"모델 가중치 {_need:.0f}GB | 볼륨 디스크 여유 {_free0:.0f}GB (이미 받은 부분은 다시 안 받음)")
    if _need > _free0 + 1: print("⚠️ 공간이 모자랄 수 있습니다 → Pods → ⋮ → Edit Pod에서 Volume Disk를 늘리세요")
except Exception as _e:
    print(f"(모델 크기 조회 실패: {type(_e).__name__} — 게이트 모델이면 HF에서 라이선스 동의가 필요합니다)")
t0 = time.time()''')
A[i_dl]["source"] = rep(A[i_dl]["source"], 'MODEL_DIR = snapshot_download(MODEL_ID, ignore_patterns=', 'MODEL_DIR = snapshot_download(MODEL_ID, token=HF_TOKEN or None, max_workers=16, ignore_patterns=')

# ── 결과 올리기 셀 (요약 앞) ──
i_sum = find('print("RUN:", RUN_NAME, "| vllm", vllm.__version__, "| GPU:", GPU_NAME)')
UPLOAD_MD = {"cell_type": "markdown", "metadata": {}, "source": """## 11. 결과를 HF 비공개 저장소로 올리기 — pod를 끄기 전에 꼭
- 올리기 전에 ens_search·기준 교체 노트북이 읽는 파일을 확인합니다.
  - 검증 1,000문항 예측 (`preds_final_ep1__up2.0__1000.csv`)
  - test 6,714문항 예측
  - `train_log.json`
- 폴더 전체를 올립니다. train 6,714문항 검증 작업 파일도 함께 올라갑니다.
- 다 올라가면 Colab 브리지 노트북 **B 단계**로 Drive에 받습니다(`RUN_NAME="auto"`면 저장소의 실행 전부).
- 다음 모델: 1번 셀 `MODEL_ID`만 바꾸고 **Kernel → Restart Kernel and Run All Cells**. 설치·데이터는 건너뛰고, 받은 가중치는 그대로 남습니다.
"""}
UPLOAD = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": r'''
import re, time
from huggingface_hub import HfApi
_fs = os.listdir(RUN_DIR)
_val = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+__.+__" + str(len(val_df)) + r"\.csv$", f))
_tst = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+_test__.+__" + str(len(test_df)) + r"\.csv$", f))
assert _val and _tst, f"검증 1,000 · test 예측 파일이 없습니다 ({RUN_DIR}) → 9·10번 셀이 끝났는지(VAL_N ≥ 1000, DO_TEST=True) 확인하세요"
for f, n in [(x, len(val_df)) for x in _val] + [(x, len(test_df)) for x in _tst]:
    _d = pd.read_csv(os.path.join(RUN_DIR, f))
    assert _d.id.nunique() == n and _d[["p_a", "p_b", "p_c", "p_d"]].notna().all().all(), f"{f}: 문항 {_d.id.nunique()}/{n} 또는 확률 열 이상"
api = HfApi(token=HF_TOKEN)
api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
assert api.repo_info(HF_REPO, repo_type="dataset").private, f"{HF_REPO}가 공개 저장소입니다 → HF 웹에서 Private로 바꾼 뒤 다시 실행하세요"
t0 = time.time(); _pre = f"vqa_runs/{RUN_NAME}"
print(f"올리는 중: {RUN_DIR} → {HF_REPO}/{_pre}")
api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=RUN_DIR, path_in_repo=_pre, commit_message=f"RunPod zero-shot {RUN_NAME}")
_remote = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
_miss = [f for f in [f"{_pre}/{x}" for x in _val + _tst + ["train_log.json"]] if f not in _remote]
assert not _miss, f"올라가지 않은 파일: {_miss}"
print(f"✅ 올리기 완료 ({time.time() - t0:.0f}초) | {_val} · {_tst} · train_log.json")
print("다음: ① 다른 모델이면 MODEL_ID를 바꾸고 Kernel → Restart Kernel and Run All Cells ② 다 끝나면 Colab 브리지 노트북 B 단계 → Drive 확인 → pod Terminate")
'''.strip("\n") + "\n"}
A.insert(i_sum - 1, UPLOAD_MD); A.insert(i_sum, UPLOAD)
_ms = [i for i, c in enumerate(A) if c["cell_type"] == "markdown" and c["source"].startswith("## 11. 결과 요약")]
assert len(_ms) == 1; A[_ms[0]]["source"] = A[_ms[0]]["source"].replace("## 11. 결과 요약", "## 12. 결과 요약", 1)
_md = [i for i, c in enumerate(A) if c["cell_type"] == "markdown" and c["source"].startswith("## 3. 데이터 준비")]
assert len(_md) == 1; A[_md[0]]["source"] = "## 3. 데이터 준비 (HF 비공개 저장소의 대회 zip → `/workspace/data`, Flash-Next 노트북과 같음)\n"

# ── 소개 (RunPod 사용법) ──
A[0] = {"cell_type": "markdown", "metadata": {}, "source": """# RunPod · 아무 모델 zero-shot (vLLM) — `MODEL_ID`만 바꿔서 실행 · 학습 없음

학습하지 않은 대형 모델(zero-shot)의 검증·test 예측을 만들어, 교체 규칙·앙상블의 보조 모델로 쓰는 노트북입니다.
- 입력: 이미지 + 질문 + 보기
- 출력: a~d 한 글자(글자 확률). 생각 모드는 끕니다.

**하는 일**
- 1번 셀 `MODEL_ID` 모델로 train 검증(`VAL_N`, 앞 1,000 = 팀 공통 검증)과 test 6,714문항을 채점합니다.
- 결과 폴더 `/workspace/vqa_runs/<RUN_NAME>/`에 남는 것:
  - E14 형식 파일(ens_search·기준 교체 노트북이 읽는 이름)
  - 단독 제출 파일
- 마지막 셀이 HF 비공개 저장소로 올립니다. Colab 브리지 노트북 **B 단계**로 Drive에 받으면 기준 교체 노트북(`anchor_rescue.ipynb`)이 보조 모델로 자동으로 잡습니다.

**실행 순서**
1. B300 pod의 Jupyter에서 이 노트북을 열고 **Run → Run All Cells**
   - 처음 한 번만 걸리는 것: vLLM 설치 약 5분, 대회 데이터 받기, 모델 가중치 받기
2. 다음 모델: 1번 셀 `MODEL_ID`만 바꾸고 **Kernel → Restart Kernel and Run All Cells**
   - vLLM은 한 커널에 모델 하나만 올릴 수 있습니다.
   - 설치·데이터는 건너뛰고, 이미 받은 가중치는 다시 받지 않습니다.
3. 다 끝나면 Colab 브리지 노트북 B 단계 → Drive 확인 → pod **Terminate**

**기본 `VAL_N = 1000`**: 팀 공통 검증 1,000 + test 6,714 = 7,714문항입니다. 기준 교체 노트북은 이 1,000문항만 씁니다(Flash-Next가 나머지 train으로 학습했기 때문). 나중에 6714로 바꿔 다시 실행하면 새 문항만 이어서 채점합니다.
"""}
cells = A
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
