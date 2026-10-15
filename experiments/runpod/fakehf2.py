# -*- coding: utf-8 -*-
"""실제 Jupyter 커널 시험용: huggingface_hub를 로컬 폴더 가짜 허브로 바꾸고, 122B 이름을 작은 모델 폴더로 돌림 (run_runpod.py와 같은 규칙)"""
import os, json, shutil, fnmatch, types
SP = "/tmp/vqa_scratch/runpod"
import os as _os
MID = _os.environ.get("FAKE_MID", "Qwen/Qwen3.8-Flash-Next")
TINY = {"Qwen/Qwen3.8-Flash-Next": f"{SP}/tinyflash", "Qwen/Qwen3.5-122B-A10B": f"{SP}/tiny122b", "zai-org/GLM-5.3-Flash": f"{SP}/tinyglm/fp8", "Qwen/Qwen3.5-397B-A17B-FP8": f"{SP}/tiny397b_fp8"}[MID]; FAKEHUB = f"{SP}/fakehub"
def _rdir(repo_id, repo_type): return os.path.join(FAKEHUB, (repo_type or "model") + "s", repo_id)
class _Info(types.SimpleNamespace): pass
class FakeApi:
    def __init__(self, token=None): self.token = token
    def whoami(self, token=None): return {"name": "tester"}
    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False, **kw):
        d = _rdir(repo_id, repo_type); meta = os.path.join(d, ".meta.json")
        if not os.path.exists(meta): os.makedirs(d, exist_ok=True); json.dump({"private": bool(private)}, open(meta, "w"))
    def repo_info(self, repo_id, repo_type=None, **kw):
        return _Info(private=json.load(open(os.path.join(_rdir(repo_id, repo_type), ".meta.json")))["private"])
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, **kw):
        dst = os.path.join(_rdir(repo_id, repo_type), path_in_repo); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(path_or_fileobj, dst)
    def upload_folder(self, repo_id, folder_path, path_in_repo=None, repo_type=None, ignore_patterns=None, **kw):
        for d, _, fs in os.walk(folder_path):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), folder_path).replace(os.sep, "/")
                if ignore_patterns and any(fnmatch.fnmatch(rel, p) for p in ignore_patterns): continue
                dst = os.path.join(_rdir(repo_id, repo_type), path_in_repo or "", rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(os.path.join(d, f), dst)
    def list_repo_files(self, repo_id, repo_type=None, **kw):
        root = _rdir(repo_id, repo_type)
        return [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, fs in os.walk(root) for f in fs if f != ".meta.json"]
    def get_paths_info(self, repo_id, paths, repo_type=None, **kw):
        root = _rdir(repo_id, repo_type)
        return [_Info(path=p, size=os.path.getsize(os.path.join(root, p))) for p in paths if os.path.isfile(os.path.join(root, p))]
    def delete_repo(self, repo_id, repo_type=None, **kw): shutil.rmtree(_rdir(repo_id, repo_type))
def install():
    import huggingface_hub as H, transformers
    def login(token=None, add_to_git_credential=False, **kw): pass
    def whoami(token=None, **kw): return {"name": "tester"}
    def hf_hub_download(repo_id, filename, repo_type=None, token=None, local_dir=None, **kw):
        src = os.path.join(_rdir(repo_id, repo_type), filename)
        if not os.path.isfile(src): raise FileNotFoundError(f"EntryNotFound {repo_id}/{filename}")
        os.makedirs(local_dir, exist_ok=True); dst = os.path.join(local_dir, filename); shutil.copy(src, dst); return dst
    def snapshot_download(repo_id, repo_type=None, allow_patterns=None, local_dir=None, token=None, max_workers=None, **kw):
        if repo_type in (None, "model") and repo_id == MID: return TINY
        root = _rdir(repo_id, repo_type)
        for rel in FakeApi().list_repo_files(repo_id, repo_type):
            if allow_patterns and not any(fnmatch.fnmatch(rel, p) for p in allow_patterns): continue
            dst = os.path.join(local_dir, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(os.path.join(root, rel), dst)
        return local_dir
    H.HfApi, H.login, H.whoami, H.hf_hub_download, H.snapshot_download = FakeApi, login, whoami, hf_hub_download, snapshot_download
    _op, _om = transformers.AutoProcessor.from_pretrained, transformers.AutoModelForImageTextToText.from_pretrained
    transformers.AutoProcessor.from_pretrained = lambda mid, *a, **k: _op(TINY if mid == MID else mid, *a, **k)
    import torch
    def _fake_gpu_load(path, *a, **k):                  # GLM: CPU에서도 FP8 그대로 불러오게 (불러오는 동안만 GPU가 있는 척)
        _ia, _gc = torch.cuda.is_available, torch.cuda.get_device_capability
        torch.cuda.is_available = lambda: True; torch.cuda.get_device_capability = lambda *x, **y: (9, 0)
        try: return _om(path, *a, **k)
        finally: torch.cuda.is_available, torch.cuda.get_device_capability = _ia, _gc
    if MID.startswith("zai-org/GLM") or "FP8" in MID:
        transformers.AutoModelForImageTextToText.from_pretrained = lambda mid, *a, **k: _fake_gpu_load(TINY, *a, **k) if mid == MID else _om(mid, *a, **k)
    else:
        transformers.AutoModelForImageTextToText.from_pretrained = lambda mid, *a, **k: _om(TINY if mid == MID else mid, *a, **k)
