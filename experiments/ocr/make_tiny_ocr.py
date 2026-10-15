# -*- coding: utf-8 -*-
"""작은 무작위 PaddleOCR-VL(실제 transformers 구조) + 실제 PaddleOCRVLProcessor(실제 이미지 프로세서, 1.6판 채팅 템플릿) + 가짜 토크나이저(위치 토큰 포함)."""
import os, shutil, torch, pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from transformers.models.paddleocr_vl import PaddleOCRVLConfig, PaddleOCRVLForConditionalGeneration, PaddleOCRVLProcessor, PaddleOCRVLImageProcessor

SP = "/tmp/vqa_scratch/ocrtest"
U = "/tmp/vqa_uploads"
D = f"{SP}/tiny_pvl"
TMPL = open("/home/user/vqa/ocr/pvl16_chat_template.jinja", encoding="utf-8").read()

SPECIAL = ["<unk>", "<|begin_of_sentence|>", "</s>", "<|end_of_sentence|>", "<|IMAGE_PLACEHOLDER|>", "<|IMAGE_START|>", "<|IMAGE_END|>",
           "<|video_pad|>", "<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"] + [f"<|LOC_{i}|>" for i in range(1001)]

if __name__ == "__main__":
    shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
    tr = pd.read_csv(f"{U}/6247e8ae-train.csv")
    corpus = [" ".join(map(str, [r.question, r.a, r.b, r.c, r.d])) for r in tr.head(3000).itertuples()] + ["User: Assistant: Spotting: OCR:"] * 50
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_sentence|>", eos_token="</s>", pad_token="<unk>", unk_token="<unk>",
                                  extra_special_tokens={"image_token": "<|IMAGE_PLACEHOLDER|>"})
    assert tok.image_token == "<|IMAGE_PLACEHOLDER|>" and tok.image_token_id == tok.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")
    V = len(tok); cid = tok.convert_tokens_to_ids
    cfg = PaddleOCRVLConfig(
        text_config=dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                         rope_parameters={"rope_type": "default", "rope_theta": 500000.0, "mrope_section": [2, 3, 3]}),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14,
                           spatial_merge_size=2),
        image_token_id=cid("<|IMAGE_PLACEHOLDER|>"), video_token_id=cid("<|video_pad|>"),
        vision_start_token_id=cid("<|IMAGE_START|>"), vision_end_token_id=cid("<|IMAGE_END|>"))
    torch.manual_seed(0)
    m = PaddleOCRVLForConditionalGeneration(cfg)
    m.generation_config.eos_token_id = tok.eos_token_id; m.generation_config.pad_token_id = tok.pad_token_id
    m.generation_config.use_cache = False; m.config.use_cache = False; m.config.text_config.use_cache = False     # 실제 1.6판과 같게
    m.save_pretrained(D)
    proc = PaddleOCRVLProcessor(image_processor=PaddleOCRVLImageProcessor(), tokenizer=tok, chat_template=TMPL)
    proc.save_pretrained(D)
    print("saved", D, sorted(os.listdir(D)), "vocab", V)
