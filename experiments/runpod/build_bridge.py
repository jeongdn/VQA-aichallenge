# -*- coding: utf-8 -*-
"""Colab ↔ RunPod 브리지 노트북 (Colab CPU 런타임이면 충분)
 A. RunPod 시작 전: Drive의 대회 zip → HF 비공개 데이터셋 저장소
 B. RunPod 학습 뒤: 저장소의 vqa_runs/<실행>/ → Drive vqa_runs/<실행>/ (RUN_NAME="auto"면 저장소에 있는 실행 전부, ens_search가 읽는 파일 점검)
 C. (대회 뒤) 저장소 지우기
사용: python3 build_bridge.py [출력 경로]"""
import json, sys, os, uuid
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/runpod/out/runpod_bridge_colab.ipynb"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# Colab ↔ RunPod 브리지 (GPU 필요 없음 — 런타임 유형: CPU)

RunPod에는 Google Drive가 붙지 않아서, 대회 데이터와 학습 결과를 **Hugging Face 비공개 데이터셋 저장소**를 거쳐 옮깁니다.

| 단계 | 언제 | 하는 일 |
|---|---|---|
| **A** | RunPod을 켜기 전 | Drive의 대회 zip → 비공개 저장소 |
| **B** | RunPod 노트북 14번 셀(결과 올리기)이 끝난 뒤 | 저장소의 결과 → Drive `vqa_runs/<실행 이름>/` |
| C | 대회가 끝난 뒤 (선택) | 저장소 지우기 |

- 0번 셀은 매번 실행합니다(Drive 연결·HF 로그인). 그다음 A나 B 하나만 실행하면 됩니다.
- HF 토큰은 **Write 권한**이어야 합니다. Colab 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`으로 넣어 두면 입력창이 뜨지 않습니다.
- 저장소는 **비공개(private)** 로 만들어집니다. 공개 저장소면 멈춥니다(대회 데이터 유출 방지).
""")

md("## 0. 설정 · Drive 연결 · HF 로그인")
code(r'''
DRIVE_ZIP   = "/content/drive/MyDrive/ssafy-16-2-ai.zip"   # 대회 데이터 zip (다른 노트북과 같은 경로)
OUT_DIR     = "/content/drive/MyDrive/vqa_runs"            # 결과를 넣을 Drive 폴더 (ens_search의 OUT_DIR)
RUN_NAME    = "auto"    # "auto": 저장소에 올라온 실행 전부 받기 · 하나만 받으려면 이름 (Flash-Next "q38flash_r16_native_bf16")
HF_REPO     = ""        # 비워 두면 "<내 HF 아이디>/ssafy-vqa-private" (RunPod 노트북과 같은 규칙)
HF_ZIP_NAME = "ssafy-16-2-ai.zip"
HF_TOKEN    = ""        # 비워 두면 Colab 보안 비밀 'HF_TOKEN' → 없으면 입력창

import os, time, shutil, getpass
try:
    from google.colab import drive
    drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
if not HF_TOKEN and IN_COLAB:
    try:
        from google.colab import userdata; HF_TOKEN = userdata.get("HF_TOKEN") or ""
    except Exception: pass
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN", "") or getpass.getpass("Hugging Face 토큰(Write 권한, hf_로 시작)을 붙여 넣고 Enter: ").strip()
from huggingface_hub import HfApi, login
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi(token=HF_TOKEN)
HF_USER = api.whoami()["name"]
HF_REPO = HF_REPO or f"{HF_USER}/ssafy-vqa-private"
api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
_info = api.repo_info(HF_REPO, repo_type="dataset")
assert _info.private, f"❌ {HF_REPO}가 공개 저장소입니다 → 대회 데이터를 올리면 안 됩니다. HF 웹에서 Private로 바꾸거나 HF_REPO를 다른 이름으로 정하세요"
print(f"✅ 로그인 {HF_USER} | 비공개 저장소 {HF_REPO} (private={_info.private})")
''')

md("""
## A. RunPod을 켜기 전 — 대회 데이터 zip 올리기
- 수 GB라 몇 분 걸립니다. 같은 크기의 파일이 이미 올라가 있으면 건너뜁니다.
""")
code(r'''
assert os.path.exists(DRIVE_ZIP), f"{DRIVE_ZIP}가 없습니다 → 0번 셀의 DRIVE_ZIP 경로를 확인하세요"
_size = os.path.getsize(DRIVE_ZIP)
try:
    _r = api.get_paths_info(HF_REPO, [HF_ZIP_NAME], repo_type="dataset"); _rsize = getattr(_r[0], "size", None) if _r else None
except Exception:
    _rsize = None
if _rsize == _size:
    print(f"이미 올라가 있음 ({_size / 1e9:.2f}GB, 크기 같음) → 건너뜀")
else:
    t0 = time.time(); print(f"올리는 중: {DRIVE_ZIP} ({_size / 1e9:.2f}GB) → {HF_REPO}/{HF_ZIP_NAME}")
    api.upload_file(path_or_fileobj=DRIVE_ZIP, path_in_repo=HF_ZIP_NAME, repo_id=HF_REPO, repo_type="dataset", commit_message="competition data (private)")
    print(f"✅ 완료 ({time.time() - t0:.0f}초)")
print("다음: RunPod pod를 만들고 runpod_flashnext_bf16.ipynb를 실행하세요")
''')

md("""
## B. RunPod 학습이 끝난 뒤 — 결과를 Drive로 받기
- RunPod 노트북 **14번 셀**(결과 올리기)이 `✅ 올리기 완료`를 출력한 뒤에 실행합니다.
- ens_search가 읽는 두 파일을 점검합니다. 검증 1,000문항은 팀 공통 분할과 같아야 하고, test는 6,714문항이어야 합니다.
- 점검을 통과하면 `vqa_runs/<실행 이름>/`에 복사합니다.
- 여기까지 끝나면 RunPod pod를 **Terminate** 해도 됩니다.
""")
code(r'''
import re, json
import pandas as pd
from huggingface_hub import snapshot_download
_remote = api.list_repo_files(HF_REPO, repo_type="dataset")
_runs = sorted({f.split("/")[1] for f in _remote if f.startswith("vqa_runs/") and f.count("/") >= 2}) if RUN_NAME == "auto" else [RUN_NAME]
assert _runs, f"❌ 저장소({HF_REPO})에 vqa_runs/ 결과가 없습니다 → RunPod 노트북 14번 셀(결과 올리기)을 먼저 실행하세요"
print(f"받을 실행 {len(_runs)}개: {_runs}")
_tmp = "/content/_runpod_dl" if IN_COLAB else os.path.join(os.getcwd(), "_runpod_dl")

def _fetch(run):
    _pre = f"vqa_runs/{run}"
    shutil.rmtree(_tmp, ignore_errors=True)
    t0 = time.time(); print(f"\n받는 중: {HF_REPO}/{_pre}")
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{_pre}/*", f"{_pre}/**"], local_dir=_tmp, token=HF_TOKEN)
    _src = os.path.join(_tmp, _pre)
    assert os.path.isdir(_src), f"❌ 저장소에 {_pre}가 없습니다 → RunPod 노트북 14번 셀(결과 올리기)을 먼저 실행하세요 (RUN_NAME도 같은지 확인)"
    _fs = os.listdir(_src)
    _val = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+__up2\.0__1000\.csv$", f))
    _tst = sorted(f for f in _fs if re.match(r"^preds_final_ep\d+_test__up2\.0__6714\.csv$", f))
    assert _val and _tst, f"❌ {run}: ens_search가 읽는 파일이 없습니다 (검증 {_val}, test {_tst}) → 받은 파일: {sorted(_fs)[:20]}"
    V = pd.read_csv(os.path.join(_src, _val[-1])); T = pd.read_csv(os.path.join(_src, _tst[-1]))
    _cols = {"id", "pred", "p_a", "p_b", "p_c", "p_d"}
    assert _cols <= set(V.columns) and _cols <= set(T.columns), f"{run}: 열이 부족합니다: {list(V.columns)}"
    assert V.id.nunique() == 1000 and T.id.nunique() == 6714, f"{run}: 문항 수: 검증 {V.id.nunique()} / test {T.id.nunique()}"
    _vid = os.path.join(OUT_DIR, "valid_ids.csv")
    if os.path.exists(_vid):
        assert set(V.id) == set(pd.read_csv(_vid).id), f"❌ {run}: 검증 문항이 팀 공통 분할(valid_ids.csv)과 다릅니다 → 위 출력을 확인하세요"
        print("검증 문항 = 팀 공통 분할 ✅")
    _log = json.load(open(os.path.join(_src, "train_log.json"), encoding="utf-8"))
    _dst = os.path.join(OUT_DIR, run)
    if os.path.exists(_dst): print(f"⚠️ {_dst}가 이미 있습니다 → 같은 이름의 파일만 덮어씁니다")
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copytree(_src, _dst, dirs_exist_ok=True)
    print(f"✅ Drive에 복사: {_dst} ({time.time() - t0:.0f}초)")
    print(f"   모델 {_log['config'].get('MODEL_ID')} | epoch 끝 검증 정확도 {_log.get('epoch_acc')} | 파일 {_val[-1]} · {_tst[-1]}")

for _r in _runs: _fetch(_r)
print("\n다음: ① RunPod에서 pod를 Terminate ② ens_search.ipynb(또는 manual_ensemble)를 다시 실행하면 새 멤버로 잡힙니다")
''')

md("""
## C. (대회가 끝난 뒤, 선택) 비공개 저장소 지우기
- 저장소 안의 대회 데이터와 결과가 모두 지워집니다. 되돌릴 수 없습니다.
- Drive에 받은 결과는 그대로 남습니다.
""")
code(r'''
DELETE_REPO = False     # True로 바꾸고 실행하면 지웁니다
if DELETE_REPO:
    api.delete_repo(HF_REPO, repo_type="dataset"); print("삭제:", HF_REPO)
else:
    print("DELETE_REPO = False → 아무것도 지우지 않았습니다")
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
