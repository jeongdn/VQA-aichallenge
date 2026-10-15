# -*- coding: utf-8 -*-
"""RunPod GLM-5.3-Flash 노트북 시험용 '작은 GLM-5.3-Flash' (transformers 5.17 glm5_next를 아주 작게) + FP8 체크포인트.
- 구조: KDA 선형 어텐션 3층 + DSA(MLA + 희소 인덱서) 1층, 첫 층 dense MLP, 나머지 MoE(전문가 8 + 공유 1), mHC, 비전 타워
- 토크나이저: GLM식 특수 토큰, add_special_tokens=True면 [gMASK]<sop>를 앞에 붙임(실제 GLM 토크나이저처럼)
- 템플릿: 웹에서 확인한 GLM-5.3-Flash 규칙 — [gMASK]<sop> 시작, <|system|>Reasoning Effort: X, <|user|>, 생성 프롬프트 <|assistant|><think>
- 프로세서: 실제 Glm5NextProcessor · Glm5NextImageProcessor(patch 14 · merge 2 · temporal 2)
- 저장 두 가지: {D}/bf16 (float32 원본), {D}/fp8 (실제 공식 체크포인트처럼 전문가·공유 전문가·dense MLP·DSA 투영을 FP8 블록 양자화 + weight_scale_inv)"""
import os, sys, json, shutil, glob, torch
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration, Glm5NextProcessor
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.glm5_next.video_processing_glm5_next import Glm5NextVideoProcessor
from safetensors.torch import load_file, save_file

SP = "/tmp/vqa_scratch/runpod"
U = "/tmp/vqa_uploads"
D = sys.argv[1] if len(sys.argv) > 1 else f"{SP}/tinyglm"
BLOCK = 16
SPECIAL = ["<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>",
           "<|begin_of_image|>", "<|end_of_image|>", "<|image|>", "<|begin_of_video|>", "<|end_of_video|>", "<|video|>"]
TMPL = r"""{%- set effort = reasoning_effort|default('max') -%}{%- if effort not in ['low', 'high', 'max'] -%}{%- set effort = 'max' -%}{%- endif -%}
[gMASK]<sop><|system|>Reasoning Effort: {{ effort|capitalize }}
{%- for m in messages -%}
{%- if m.role == 'system' -%}<|system|>{%- if m.content is string -%}{{ m.content }}{%- else -%}{%- for c in m.content -%}{{ c.text }}{%- endfor -%}{%- endif -%}
{%- elif m.role == 'user' -%}<|user|>{%- if m.content is string -%}{{ '\n' + m.content }}{%- else -%}{%- for c in m.content -%}{%- if c.type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>{%- elif c.type == 'text' -%}{{ '\n' + c.text }}{%- endif -%}{%- endfor -%}{%- endif -%}
{%- elif m.role == 'assistant' -%}<|assistant|>{{ '\n<think></think>\n' }}{%- if m.content is string -%}{{ m.content }}{%- else -%}{%- for c in m.content -%}{{ c.text }}{%- endfor -%}{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"""
# 공식 체크포인트에서 FP8로 저장된 층 (config의 modules_to_not_convert에 없는 선형 층): 전문가·공유 전문가·dense MLP·DSA의 q_a/q_b/kv_a/o
FP8_KEYS = ("mlp.experts.", "mlp.shared_experts.", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa")


def make_processor():
    tr = pd.read_csv(f"{U}/6247e8ae-train.csv")
    corpus = [f"{r.question}\n(a) {r.a}\n(b) {r.b}\n(c) {r.c}\n(d) {r.d}\n\n정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요." for r in tr.head(2000).itertuples()]
    corpus += ["You are a helpful visual question answering assistant. Answer using exactly one letter among a, b, c, or d. No explanation.",
               "Reasoning Effort: Low High Max", " a b c d A B C D a b c d", "\n"] * 50
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    gm, sop = tk.token_to_id("[gMASK]"), tk.token_to_id("<sop>")
    tk.post_processor = processors.TemplateProcessing(single="[gMASK] <sop> $A", pair="[gMASK] <sop> $A $B", special_tokens=[("[gMASK]", gm), ("<sop>", sop)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    for L in "abcdABCD":
        assert len(tok.encode(L, add_special_tokens=False)) == 1 and len(tok.encode(" " + L, add_special_tokens=False)) == 1, L
    assert tok("x").input_ids[:2] == [gm, sop], "add_special_tokens=True면 [gMASK]<sop>를 붙여야 함"
    ip = Glm5NextImageProcessor()
    vp = Glm5NextVideoProcessor()
    proc = Glm5NextProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=TMPL)
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|image|>", "<|video|>", "<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>")}
    return proc, tok, ids


def tiny_config(tok, ids):
    text = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
                routed_scaling_factor=2.5, kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
                n_group=1, topk_group=1, mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
                layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
                indexer_types=["full"] * 4, index_topk=8, index_head_dim=16, index_n_heads=2, index_kpool=4, index_kpool_always_select_tail=True,
                linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
                hc_mult=2, hc_sinkhorn_iters=3, swiglu_limit=10.0, pad_token_id=tok.pad_token_id, eos_token_id=[tok.pad_token_id],
                rms_norm_eps=1e-5, max_position_embeddings=8192)
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14, spatial_merge_size=2,
                  temporal_patch_size=2, projection_intermediate_size=64, image_size=448)
    return Glm5NextConfig(text_config=text, vision_config=vision, image_token_id=ids["<|image|>"], video_token_id=ids["<|video|>"],
                          image_start_token_id=ids["<|begin_of_image|>"], image_end_token_id=ids["<|end_of_image|>"],
                          video_start_token_id=ids["<|begin_of_video|>"], video_end_token_id=ids["<|end_of_video|>"], tie_word_embeddings=False)


def fp8_quant(w, block=BLOCK):
    """블록(128x128 대신 16x16) 양자화: W ≈ fp8 * scale_inv (DeepSeek·GLM 공식 FP8과 같은 규칙: amax/448)"""
    out, inn = w.shape
    nb0, nb1 = (out + block - 1) // block, (inn + block - 1) // block
    wp = torch.zeros(nb0 * block, nb1 * block, dtype=torch.float32); wp[:out, :inn] = w.float()
    blocks = wp.view(nb0, block, nb1, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(nb0 * block, nb1 * block)[:out, :inn].contiguous()
    return q, scale.float().contiguous()


def main():
    proc, tok, ids = make_processor()
    cfg = tiny_config(tok, ids)
    torch.manual_seed(0)
    model = Glm5NextForConditionalGeneration(cfg)
    with torch.no_grad():                                    # 초기값이 0인 파라미터(게이트·노름 등)도 무작위로 → 시험이 의미 있게
        for n, p in model.named_parameters():
            if p.dtype.is_floating_point and p.abs().sum() == 0 and p.numel() > 1: p.normal_(0, 0.05)
    shutil.rmtree(D, ignore_errors=True)
    bf = f"{D}/bf16"; f8 = f"{D}/fp8"
    model.save_pretrained(bf, max_shard_size="200KB"); proc.save_pretrained(bf)
    # ── FP8 체크포인트: 같은 가중치를 블록 양자화 ──
    shutil.copytree(bf, f8)
    idx = json.load(open(f"{f8}/model.safetensors.index.json"))
    wm, n_q = dict(idx["weight_map"]), 0
    import re as _re
    dsa = {m.group(1) for k in wm if (m := _re.search(r"layers\.(\d+)\.self_attn\.q_a_proj\.weight$", k))}   # DSA 층의 o_proj도 FP8 (KDA 층 o_proj는 bf16)
    def is_fp8(k):
        m = _re.search(r"layers\.(\d+)\.self_attn\.o_proj\.weight$", k)
        return (any(s in k for s in FP8_KEYS) or (m is not None and m.group(1) in dsa)) and "visual" not in k and "indexer" not in k
    for fn in sorted(set(wm.values())):
        sd = load_file(f"{f8}/{fn}"); new = {}
        for k, v in sd.items():
            if k.endswith(".weight") and v.dim() == 2 and is_fp8(k):
                q, s = fp8_quant(v); new[k] = q; new[k[:-len(".weight")] + ".weight_scale_inv"] = s
                wm[k[:-len(".weight")] + ".weight_scale_inv"] = fn; n_q += 1
            else:
                new[k] = v
        save_file(new, f"{f8}/{fn}", metadata={"format": "pt"})
    idx["weight_map"] = wm
    json.dump(idx, open(f"{f8}/model.safetensors.index.json", "w"), indent=1)
    c = json.load(open(f"{f8}/config.json"))
    all_lin = sorted({k[:-len(".weight")] for k in wm if k.endswith(".weight")})
    c["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [BLOCK, BLOCK],
                                "modules_to_not_convert": [m for m in all_lin if not is_fp8(m + ".weight")]}
    json.dump(c, open(f"{f8}/config.json", "w"), indent=1)
    keys = sorted(wm)
    print("저장:", D, "| 어휘", len(tok), "| 토큰 id", ids, "| FP8로 바꾼 층", n_q)
    print("키 예:", [k for k in keys if "experts.0." in k][:3], [k for k in keys if "q_a_proj" in k][:2], [k for k in keys if "visual" in k][:1])


if __name__ == "__main__":
    main()
