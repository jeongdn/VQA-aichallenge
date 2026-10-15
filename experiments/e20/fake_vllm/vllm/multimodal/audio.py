# 실제 vLLM 0.30.0 vllm/multimodal/audio.py와 같은 torchaudio 가져오기 (ImportError만 건너뜀 — RuntimeError는 그대로 터짐)
try:
    import torchaudio
except ImportError:
    torchaudio = None
