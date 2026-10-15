# 작은 Gemma 4 모델로: stage2 LoRA 정규식 부착 → 저장 → bf16 베이스에 재부착 → 노트북의 어댑터 대조 함수 → 켬/끔 출력 차이
import json, re, torch, shutil
from transformers.models.gemma4 import Gemma4Config, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
ROOT = "/tmp/vqa_scratch/e1test/gemma"
shutil.rmtree(ROOT, ignore_errors=True)
cfg = Gemma4Config(
    text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=32, layer_types=["sliding_attention", "full_attention"], sliding_window=16, vocab_size_per_layer_input=300,
                     hidden_size_per_layer_input=16, per_layer_config={}, pad_token_id=0, eos_token_id=1, bos_token_id=2),
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                       use_clipped_linears=True, position_embedding_size=64),
    audio_config=None, image_token_id=290, boi_token_id=291, eoi_token_id=292)
torch.manual_seed(0)
Gemma4ForConditionalGeneration(cfg).save_pretrained(f"{ROOT}/base")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"
b = Gemma4ForConditionalGeneration.from_pretrained(f"{ROOT}/base", dtype=torch.float32)
print("비전 선형층 종류:", sorted({type(m).__name__ for n, m in b.named_modules() if "vision" in n and "Linear" in type(m).__name__}))
pm = get_peft_model(b, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, bias="none", target_modules=LORA_TARGETS))
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lora_B" in n: p.normal_(0, 0.5)
n_l = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A")); n_v = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A") and "vision" in n)
print(f"학습 쪽 LoRA 층 {n_l}개 (비전 {n_v})", [n for n, _ in pm.named_modules() if n.endswith("lora_A")][:3])
pm.save_pretrained(f"{ROOT}/adapter_ep1")

# ── 노트북 7번 셀과 같은 방식으로 bf16 베이스에 재부착 + 대조 함수 실행 ──
nb = json.load(open("/home/user/vqa/e1/e1_bf16_reeval.ipynb", encoding="utf-8"))
model_cell = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][6]["source"])
fn_src = model_cell[model_cell.index("def _norm_key"):model_cell.index("_c = check_adapter_weights")]
g = {}; exec("from peft import load_peft_weights, get_peft_model_state_dict\n" + fn_src, g)
base = Gemma4ForConditionalGeneration.from_pretrained(f"{ROOT}/base", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, f"{ROOT}/adapter_ep1", is_trainable=False).eval()
print("재부착 LoRA 층:", sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")))
print("대조 결과:", g["check_adapter_weights"](model, f"{ROOT}/adapter_ep1"))
ids = torch.tensor([[2, 50, 60, 70, 80, 90]])
with torch.inference_mode():
    on = model(input_ids=ids, logits_to_keep=1).logits.float()
    with model.disable_adapter(): off = model(input_ids=ids, logits_to_keep=1).logits.float()
print(f"켬/끔 로짓 최대 차이: {(on - off).abs().max().item():.4f}")
