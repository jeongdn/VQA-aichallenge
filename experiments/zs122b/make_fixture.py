# -*- coding: utf-8 -*-
"""zs122b 노트북 시험용: 실제 transformers 5.17 Qwen3VLProcessor(이미지 처리 = 실제 Qwen2VL 코드, patch 16·merge 2)에
작은 BPE 토크나이저와 Qwen3.5식 채팅 템플릿(시스템/사용자/이미지 자리/생각 끔)을 붙여 로컬 폴더에 저장한다."""
import os, shutil
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

SP = "/tmp/vqa_scratch/zs122b"
U = "/tmp/vqa_uploads"
D = f"{SP}/tiny_proc"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]

# 실제 Qwen3.5 템플릿의 핵심 구조: 시스템 → 사용자(이미지 자리 = vision_start + image_pad + vision_end) → assistant + (생각 끔이면 빈 think 블록)
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
                               size={"shortest_edge": 65536, "longest_edge": 16777216})     # 실제 122B 저장소의 preprocessor_config와 같은 값
    vp = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    proc = Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=TMPL)
    shutil.rmtree(D, ignore_errors=True); proc.save_pretrained(D)
    print("저장:", D, "| 어휘", len(tok), "| image_pad id", tok.convert_tokens_to_ids("<|image_pad|>"))

if __name__ == "__main__":
    main()
