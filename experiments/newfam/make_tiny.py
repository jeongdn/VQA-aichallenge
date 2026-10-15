# -*- coding: utf-8 -*-
"""새 계열 후보(Muse Glimmer · EXAONE 4.5 · Qwen3.6-35B-A3B(qwen3_5_moe) · Gemma4 26B-A4B(MoE))의
실제 transformers 구조로 '작은 무작위 모델 + 실제 프로세서'를 만들어 로컬 폴더에 저장한다.
노트북이 AutoProcessor / AutoModelForImageTextToText.from_pretrained(로컬 폴더)로 실제 코드 경로를 그대로 타게 하기 위함.
(토크나이저·채팅 템플릿만 가짜: 실제 템플릿의 핵심 구조를 흉내 냄)"""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

SP = "/tmp/vqa_scratch/newfam"
U = "/tmp/vqa_uploads"
os.makedirs(SP, exist_ok=True)

SPECIAL = ["<|pad|>", "<|begin_of_text|>", "<|end_of_text|>",
           # Muse Glimmer
           "<|start|>", "<|message|>", "<|eot|>", "<|eom|>", "<|patch|>", "<|image_start|>", "<|image_end|>",
           "<|video|>", "<|vid_frame_separator|>", "<|vid_start|>", "<|vid_end|>",
           # EXAONE 4.5
           "[|system|]", "[|user|]", "[|assistant|]", "[|endofturn|]", "<|image_pad|>", "<|video_pad|>", "<vision>", "</vision>",
           "<think>", "</think>"]

def build_tokenizer():
    tr = pd.read_csv(f"{U}/6247e8ae-train.csv")
    corpus = []
    for _, r in tr.head(3000).iterrows():
        corpus.append(f"{r.question}\n(a) {r.a}\n(b) {r.b}\n(c) {r.c}\n(d) {r.d}\n\n정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요.")
    corpus += ["You are a helpful visual question answering assistant. Answer using exactly one letter among a, b, c, or d. No explanation.",
               "Reasoning strength: low. Knowledge cutoff: 2026-01-04 system user assistant to=self", " a b c d A B C D a b c d"] * 50
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1400, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    bos_id = tk.token_to_id("<|begin_of_text|>")
    tk.post_processor = processors.TemplateProcessing(single="<|begin_of_text|> $A", pair="<|begin_of_text|> $A $B", special_tokens=[("<|begin_of_text|>", bos_id)])   # 실제 Muse(Llama 계열) 토크나이저처럼 BOS 자동 추가
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>", pad_token="<|pad|>")
    for L in "abcdABCD":
        assert len(tok.encode(L, add_special_tokens=False)) == 1, L
    return tok

# 실제 Muse Glimmer 템플릿의 구조 (시스템 메시지 + 'Reasoning strength', 이미지 = <|patch|>, 답 = <|start|>assistant<|message|>…<|eot|>)
MUSE_TMPL = r"""{{- '<|begin_of_text|>' -}}{%- macro rc(content) -%}{%- if content is string -%}{{- content -}}{%- else -%}{%- for c in content -%}{%- if c.type == 'image' -%}{{- '<|patch|>' -}}{%- elif c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}
{%- macro dt() -%}{%- if current_date is defined and current_date -%}{{- '\nCurrent date: ' + current_date + '.' -}}{%- elif strftime_now is defined -%}{{- '\nCurrent date: ' + strftime_now('%Y-%m-%d') + '.' -}}{%- endif -%}{%- endmacro -%}
{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
{%- if messages and messages[0].role == 'system' -%}{{- '<|start|>system<|message|>' + rc(messages[0].content) + '\nKnowledge cutoff: 2026-01-04' + dt() + '\nReasoning strength: ' + rs + '.<|eot|>' -}}{%- set lm = messages[1:] -%}
{%- else -%}{{- '<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04' + dt() + '\nReasoning strength: ' + rs + '.<|eot|>' -}}{%- set lm = messages -%}{%- endif -%}
{%- for m in lm -%}{%- if m.role == 'assistant' -%}{%- if m.get('reasoning_content') -%}{{- '<|start|>assistant to=self<|message|>' + m['reasoning_content'] + '<|eom|>' -}}{%- endif -%}{{- '<|start|>assistant<|message|>' + rc(m.content) + '<|eot|>' -}}
{%- else -%}{{- '<|start|>' + m.role + '<|message|>' + rc(m.content) + '<|eot|>' -}}{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}"""

# EXAONE 4.x식 (역할 표지 + enable_thinking)
EXA_TMPL = r"""{%- for m in messages -%}{{- '[|' + m.role + '|]' -}}{%- if m.content is string -%}{{- m.content -}}{%- else -%}{%- for c in m.content -%}{%- if c.type == 'image' -%}{{- '<vision><|image_pad|></vision>' -}}{%- elif c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}{%- endif -%}{{- '[|endofturn|]\n' -}}{%- endfor -%}
{%- if add_generation_prompt -%}{{- '[|assistant|]' -}}{%- if enable_thinking is defined and enable_thinking -%}{{- '<think>\n' -}}{%- else -%}{{- '<think>\n\n</think>\n\n' -}}{%- endif -%}{%- endif -%}"""


def make_muse(tok):
    from transformers.models.muse_glimmer import (MuseGlimmerConfig, MuseGlimmerForConditionalGeneration, MuseGlimmerProcessor,
                                                  MuseGlimmerImageProcessor)
    from transformers.models.muse_glimmer.video_processing_muse_glimmer import MuseGlimmerVideoProcessor
    V = len(tok)
    cfg = MuseGlimmerConfig(
        text_config=dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, sliding_window=64, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id),
        vision_config=dict(patch_size=14, pos_emb_height=8, pos_emb_width=8, num_attention_heads=2, num_hidden_layers=2, hidden_size=32,
                           intermediate_size=64, max_position_embeddings=64),
        image_token_id=tok.convert_tokens_to_ids("<|patch|>"), video_token_id=tok.convert_tokens_to_ids("<|video|>"),
        out_hidden_size=128, projector_hidden_size=64)
    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(cfg)
    d = f"{SP}/tiny_muse"; shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d)
    proc = MuseGlimmerProcessor(image_processor=MuseGlimmerImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3, max_image_tokens=4096),
                                video_processor=MuseGlimmerVideoProcessor(), tokenizer=tok, chat_template=MUSE_TMPL)
    proc.save_pretrained(d)
    return d


def make_exaone(tok):
    from transformers.models.exaone4_5 import Exaone4_5_Config, Exaone4_5_ForConditionalGeneration, Exaone4_5_Processor
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    V = len(tok)
    cfg = Exaone4_5_Config(
        text_config=dict(model_type="exaone4", vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, sliding_window=64, sliding_window_pattern=4, pad_token_id=tok.pad_token_id,
                         bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, max_position_embeddings=8192),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, num_key_value_heads=1, out_hidden_size=64, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1]),
        image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"), video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"))
    torch.manual_seed(0)
    model = Exaone4_5_ForConditionalGeneration(cfg)
    d = f"{SP}/tiny_exaone45"; shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d)
    proc = Exaone4_5_Processor(image_processor=Qwen2VLImageProcessor(patch_size=14, merge_size=2, temporal_patch_size=2),
                               tokenizer=tok, video_processor=Qwen2VLVideoProcessor(), chat_template=EXA_TMPL)
    proc.save_pretrained(d)
    return d


def make_qwen35moe():
    from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    sys.path.insert(0, "/home/user/vqa/e1")
    from tiny_qwen import IMG, VID, VS, VE, PAD
    cfg = Qwen3_5MoeConfig(
        text_config=dict(vocab_size=300, hidden_size=64, num_hidden_layers=4,
                         layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, moe_intermediate_size=32,
                         shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2, pad_token_id=PAD),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS, vision_end_token_id=VE, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(cfg)
    d = f"{SP}/tiny_qwen35moe"; shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d)
    return d


def make_gemma4moe():
    from transformers.models.gemma4 import Gemma4Config, Gemma4ForConditionalGeneration
    cfg = Gemma4Config(
        text_config=dict(vocab_size=400, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, layer_types=["sliding_attention", "full_attention"], sliding_window=32, enable_moe_block=True,
                         num_experts=4, top_k_experts=2, moe_intermediate_size=32, vocab_size_per_layer_input=400, hidden_size_per_layer_input=16),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                           position_embedding_size=64),
        boi_token_id=390, eoi_token_id=391, image_token_id=392, video_token_id=393, boa_token_id=394, eoa_token_index=395, audio_token_id=396)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(cfg)
    d = f"{SP}/tiny_gemma4moe"; shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d)
    return d


if __name__ == "__main__":
    tok = build_tokenizer()
    print("tokenizer", len(tok))
    for f in sys.argv[1:] or ["muse", "exaone", "qwen35moe", "gemma4moe"]:
        d = {"muse": lambda: make_muse(tok), "exaone": lambda: make_exaone(tok), "qwen35moe": make_qwen35moe, "gemma4moe": make_gemma4moe}[f]()
        print("saved", f, "→", d, sorted(os.listdir(d)))
