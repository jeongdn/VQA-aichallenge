# -*- coding: utf-8 -*-
"""RunPod Flash-Next 노트북 시험용 '작은 Qwen3.8-Flash-Next' (transformers 5.17 qwen4_exp를 아주 작게).
실제 구조를 그대로 탄다: Gated DeltaNet + QSA(희소 어텐션 인덱서) + MoE(전문가 + 공유 전문가) + hyper-connection + PLE(n-gram 임베딩).
- ple_layer_ids=[2] (실제와 같이 2번째 층 하나), split_ngram_parts=4 → save_pretrained가 n-gram 표를 shard_0..3으로 나눠 저장(실제 체크포인트와 같은 키 이름)
- QSA budget 16 · 압축 4 → 시험 입력(약 80토큰)에서도 희소 선택이 실제로 일어남
- max_shard_size를 작게 해서 safetensors 여러 파일 + index.json (실제처럼 n-gram shard가 여러 파일에 흩어짐)
프로세서는 실제 Qwen3VLProcessor(patch 16 · merge 2) + 작은 BPE + Flash-Next식 템플릿(enable_thinking / reasoning_effort)."""
import os, sys, json, shutil, torch
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

SP = "/tmp/vqa_scratch/runpod"
U = "/tmp/vqa_uploads"
D = sys.argv[1] if len(sys.argv) > 1 else f"{SP}/tinyflash"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]
# Flash-Next 템플릿의 핵심 분기(웹에서 확인한 줄): 생각 모드가 켜져 있으면 system에 추론 강도 지시문, 생성 프롬프트는 enable_thinking=False일 때 '<think>\n\n</think>\n\n'
TMPL = r"""{%- if enable_thinking is undefined or enable_thinking is true -%}{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') -%}{%- set reasoning_instructions = 'Reasoning effort is set to ' + resolved_reasoning_effort + '.' -%}{%- else -%}{%- set reasoning_instructions = '' -%}{%- endif -%}
{%- for m in messages -%}
{%- if m.role == 'system' -%}{{- '<|im_start|>system\n' -}}{%- if reasoning_instructions -%}{{- reasoning_instructions + '\n\n' -}}{%- endif -%}{%- if m.content is string -%}{{- m.content -}}{%- else -%}{%- for c in m.content -%}{{- c.text -}}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\n' -}}
{%- else -%}{{- '<|im_start|>' + m.role + '\n' -}}{%- if m.content is string -%}{{- m.content -}}{%- else -%}{%- for c in m.content -%}{%- if c.type == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}{%- elif c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\n\n</think>\n\n' -}}{%- else -%}{{- '<think>\n' -}}{%- endif -%}{%- endif -%}"""


def tiny_config(tok, ids, **over):
    tc = dict(vocab_size=len(tok), hidden_size=64, num_hidden_layers=4,
              layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
              num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
              linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, moe_intermediate_size=32,
              shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2, hc_count=2, hc_lowrank=8,
              ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
              ngram_vocab_size_base=5000, make_ngram_vocab_size_divisible_by=4, split_ngram_parts=4,
              indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
              output_gate_type="sigmoid", eos_token_id=tok.pad_token_id, bos_token_id=tok.pad_token_id, pad_token_id=None,
              rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                               "rope_theta": 10000.0, "rope_type": "default"}, partial_rotary_factor=0.25)
    tc.update(over)
    return Qwen4ExpConfig(
        text_config=tc,
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64, deepstack_visual_indexes=[]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"], tie_word_embeddings=False)


def make_processor():
    tr = pd.read_csv(f"{U}/6247e8ae-train.csv")
    corpus = [f"{r.question}\n(a) {r.a}\n(b) {r.b}\n(c) {r.c}\n(d) {r.d}\n\n정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요." for r in tr.head(2000).itertuples()]
    corpus += ["You are a helpful visual question answering assistant. Answer using exactly one letter among a, b, c, or d. No explanation.",
               "system user assistant", " a b c d A B C D a b c d"] * 50
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    for L in "abcdABCD":
        assert len(tok.encode(L, add_special_tokens=False)) == 1 and len(tok.encode(" " + L, add_special_tokens=False)) == 1, L
    ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                               size={"shortest_edge": 65536, "longest_edge": 16777216})
    vp = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    proc = Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=TMPL)
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
    return proc, tok, ids


def main():
    proc, tok, ids = make_processor()
    cfg = tiny_config(tok, ids)
    torch.manual_seed(0)
    model = Qwen4ExpForConditionalGeneration(cfg)
    with torch.no_grad():                                   # 초기값 그대로면 conv1d·RMSNorm이 0이라 PLE·인덱서가 사실상 꺼짐 → 시험이 의미 있게 무작위 값으로
        for n, p in model.named_parameters():
            if p.dtype.is_floating_point and ("ple." in n or "indexer" in n or "hyper_connection" in n or n.endswith("norm.weight")):
                p.normal_(0, 0.3 if p.dim() > 1 else 0.1)
    shutil.rmtree(D, ignore_errors=True)
    model.save_pretrained(D, max_shard_size="120KB"); proc.save_pretrained(D)
    idx = json.load(open(f"{D}/model.safetensors.index.json"))
    ng = sorted(k for k in idx["weight_map"] if "ngram_embedding" in k)
    print("저장:", D, "| 어휘", len(tok), "| 토큰 id", ids, "| safetensors", len(set(idx["weight_map"].values())), "개")
    print("n-gram 키:", ng[:2], "...", ng[-1], f"({len(ng)}개) · 파일", sorted({idx['weight_map'][k] for k in ng}))
    assert len(ng) == 4 and all(".shard_" in k for k in ng)


if __name__ == "__main__":
    main()
