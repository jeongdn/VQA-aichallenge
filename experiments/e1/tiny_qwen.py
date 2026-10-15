# 작은 Qwen3.5(model_type qwen3_5) 모델 + 가짜 프로세서: GPU 없이 노트북의 모델 셀·채점 코드를 실제로 돌려 보기 위한 도구
import torch, numpy as np
from transformers import BatchFeature
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

IMG, VID, VS, VE, PAD = 290, 291, 292, 293, 0
def tiny_config():
    return Qwen3_5Config(
        text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                         linear_conv_kernel_dim=4, pad_token_id=PAD),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64),
        image_token_id=IMG, video_token_id=VID, vision_start_token_id=VS, vision_end_token_id=VE, tie_word_embeddings=False)

class FakeTok:
    padding_side = "right"
    _T = {"a": [1], "b": [2], "c": [3], "d": [4], "A": [5], "B": [6], "C": [7], "D": [8],
          " a": [9], " b": [10], " c": [11], " d": [12], " A": [13], " B": [14], " C": [15], " D": [16]}
    def encode(self, s, add_special_tokens=False):
        return list(self._T.get(s, [20 + (ord(ch) % 250) for ch in s]))
    def convert_tokens_to_ids(self, t): return IMG
    def decode(self, ids): return "".join(f"<{i}>" for i in ids)

class FakeQwenProcessor:
    """apply_chat_template → 문자열, __call__ → Qwen3.5 입력 (이미지 1장 = 패치 16개 = 병합 후 토큰 4개)."""
    image_token_id = IMG
    chat_template = "{%- if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{%- endif %}"
    def __init__(self): self.tokenizer = FakeTok(); self.calls = []
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        parts = []
        for m in messages:
            for c in m["content"]: parts.append("<IMG>" if c["type"] == "image" else c["text"])
        return "|".join(parts) + ("<GEN>" if add_generation_prompt else "")
    def __call__(self, text, images, padding=True, return_tensors="pt", **kw):
        self.calls.append(kw)
        seqs, pvs = [], []
        for t, im in zip(text, images):
            pre, post = t.split("<IMG>")
            seqs.append([20 + (ord(ch) % 250) for ch in pre[-6:]] + [VS] + [IMG] * 4 + [VE] + [20 + (ord(ch) % 250) for ch in post[-40:]])
            a = np.asarray(im.resize((8, 8))).astype(np.float32) / 255.0
            rng = np.random.RandomState(int(a.sum() * 1000) % (2**31))
            pvs.append(torch.tensor(rng.randn(16, 3 * 2 * 16 * 16).astype(np.float32)))
        L = max(map(len, seqs))
        ids = torch.tensor([[PAD] * (L - len(s)) + s for s in seqs])
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
        mm = (ids == IMG).long()                                  # 실제 Qwen3.5 프로세서처럼 텍스트 0 / 이미지 1
        return BatchFeature({"input_ids": ids, "attention_mask": att, "mm_token_type_ids": mm, "pixel_values": torch.cat(pvs),
                             "image_grid_thw": torch.tensor([[1, 4, 4]] * len(text))})
