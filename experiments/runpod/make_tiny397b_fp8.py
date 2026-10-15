# -*- coding: utf-8 -*-
"""RunPod 397B-FP8 노트북 시험용 '작은 Qwen3.5-397B-A17B-FP8' (qwen3_5_moe, 공식 FP8 체크포인트와 같은 규칙).
작은 122B(tiny122b, float32)를 복사해서 공식 FP8판처럼 양자화한다.
- FP8(블록 양자화 + weight_scale_inv): 전체 어텐션 q/k/v/o, 전문가 gate/up/down, 공유 전문가 gate/up/down
- 그대로(modules_to_not_convert): 선형 어텐션(GDN) 투영, 라우터(mlp.gate)·shared_expert_gate, 비전 타워, 임베딩, lm_head
블록은 128 대신 16 (작은 모델 크기에 맞춤). 저장: {SP}/tiny397b_fp8"""
import os, sys, json, shutil, re
sys.path.insert(0, "/home/user/vqa/runpod")
from safetensors.torch import load_file, save_file
from make_tinyglm import fp8_quant, BLOCK

SP = "/tmp/vqa_scratch/runpod"
SRC, D = f"{SP}/tiny122b", f"{SP}/tiny397b_fp8"
FP8_RE = re.compile(r"(self_attn\.(q|k|v|o)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj|mlp\.shared_expert\.(gate|up|down)_proj)\.weight$")


def main():
    shutil.rmtree(D, ignore_errors=True); shutil.copytree(SRC, D)
    files = sorted(f for f in os.listdir(D) if f.endswith(".safetensors"))
    idx_p = f"{D}/model.safetensors.index.json"
    wm = json.load(open(idx_p))["weight_map"] if os.path.exists(idx_p) else None
    all_lin, n_q = set(), 0
    for fn in files:
        sd = load_file(f"{D}/{fn}"); new = {}
        for k, v in sd.items():
            if k.endswith(".weight") and v.dim() == 2: all_lin.add(k[:-len(".weight")])
            if FP8_RE.search(k) and "visual" not in k:
                q, s = fp8_quant(v); new[k] = q; new[k[:-len(".weight")] + ".weight_scale_inv"] = s; n_q += 1
                if wm is not None: wm[k[:-len(".weight")] + ".weight_scale_inv"] = fn
            else:
                new[k] = v
        save_file(new, f"{D}/{fn}", metadata={"format": "pt"})
    if wm is not None:
        json.dump({"metadata": {}, "weight_map": wm}, open(idx_p, "w"), indent=1)
    c = json.load(open(f"{D}/config.json"))
    c["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [BLOCK, BLOCK],
                                "modules_to_not_convert": sorted(m for m in all_lin if not FP8_RE.search(m + ".weight"))}
    json.dump(c, open(f"{D}/config.json", "w"), indent=1)
    print("저장:", D, "| FP8로 바꾼 층", n_q, "| 그대로 둔 선형 층", len(c["quantization_config"]["modules_to_not_convert"]))


if __name__ == "__main__":
    main()
