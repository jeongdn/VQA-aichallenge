# -*- coding: utf-8 -*-
"""RunPod 122B 노트북 시험용 '작은 Qwen3.5-122B-A10B':
실제 transformers 5.17 Qwen3_5MoeForConditionalGeneration(qwen3_5_moe: 선형 어텐션 + 전체 어텐션, MoE 전문가 + 공유 전문가)을 아주 작게 만들고,
실제 Qwen3VLProcessor(이미지 처리 = 실제 Qwen2VL 코드, patch 16 · merge 2)에 작은 BPE 토크나이저와 Qwen3.5식 채팅 템플릿을 붙여 한 폴더에 저장한다.
노트북은 이 폴더를 'Qwen/Qwen3.5-122B-A10B'로 알고 AutoProcessor / AutoModelForImageTextToText.from_pretrained 실제 경로를 그대로 탄다."""
import os, shutil, torch
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

SP = "/tmp/vqa_scratch/runpod"
U = "/tmp/vqa_uploads"
D = f"{SP}/tiny122b"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]
TMPL = r"""{%- for m in messages -%}
{%- if m.role == 'system' -%}{{- '<|im_start|>system\n' -}}{%- if m.content is string -%}{{- m.content -}}{%- else -%}{%- for c in m.content -%}{{- c.text -}}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\n' -}}
{%- else -%}{{- '<|im_start|>' + m.role + '\n' -}}{%- if m.content is string -%}{{- m.content -}}{%- else -%}{%- for c in m.content -%}{%- if c.type == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}{%- elif c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\n\n</think>\n\n' -}}{%- else -%}{{- '<think>\n' -}}{%- endif -%}{%- endif -%}"""

def main():
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
    cfg = Qwen3_5MoeConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=64, num_hidden_layers=4,
                         layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, moe_intermediate_size=32,
                         shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2, pad_token_id=tok.pad_token_id),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"], tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(cfg)
    shutil.rmtree(D, ignore_errors=True)
    model.save_pretrained(D); proc.save_pretrained(D)
    print("저장:", D, "| 어휘", len(tok), "| 토큰 id", ids, "|", sorted(os.listdir(D)))

if __name__ == "__main__":
    main()
