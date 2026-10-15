# -*- coding: utf-8 -*-
"""노트북 시험용 가짜 vLLM (CPU). 실제 vLLM 0.30.0처럼:
- LLM(model, **kw): 인자 이름을 실제 0.30.0 소스에서 뽑은 목록(vllm_api.json)과 대조
- generate(prompts, sp, use_tqdm=...): 실제처럼 프롬프트 + 이미지를 HF 프로세서(엔진 수준 mm_processor_kwargs)로 처리해
  prompt_token_ids(이미지 토큰 펼침 포함)를 만들고, 답 자리 상위 k개 {토큰 id: Logprob}를 돌려줌
정답을 아는 문항(학습 CSV)은 STATE['acc'] 확률로 정답 글자에 확률을 몰아 줌."""
import os, json, hashlib
import numpy as np

__version__ = "0.30.0+fake"
_API = json.load(open(os.environ["FAKE_VLLM_API"]))
STATE = {"init_kw": None, "n_init": 0, "calls": 0, "crash_at": None, "mode": "normal", "answers": {}, "acc": 0.97, "n_prompts": 0}


class SamplingParams:
    def __init__(self, **kw):
        for k in kw: assert k in _API["sp_fields"], f"SamplingParams에 없는 인자: {k}"
        self.max_tokens, self.temperature, self.logprobs = kw.get("max_tokens", 16), kw.get("temperature", 1.0), kw.get("logprobs")
        self.kw = kw


class Logprob:
    def __init__(self, logprob, rank=None, decoded_token=None): self.logprob, self.rank, self.decoded_token = logprob, rank, decoded_token


class CompletionOutput:
    def __init__(self, token_ids, logprobs): self.index, self.text, self.token_ids, self.cumulative_logprob, self.logprobs = 0, "", token_ids, None, logprobs


class RequestOutput:
    def __init__(self, rid, prompt, ids, outputs): self.request_id, self.prompt, self.prompt_token_ids, self.outputs = rid, prompt, ids, outputs


class LLM:
    def __init__(self, model, **kw):
        ok = set(_API["llm_params"]) | set(_API["engine_fields"])
        for k in kw: assert k in ok, f"vLLM 0.30.0 LLM/EngineArgs에 없는 인자: {k}"
        STATE["init_kw"] = dict(kw, model=model); STATE["n_init"] += 1
        import subprocess, sys as _sys
        _r = subprocess.run([_sys.executable, "-c", "import vllm.model_executor.models.qwen3_5"], capture_output=True, text=True)
        if _r.returncode != 0:                                               # 실제: registry._run_in_subprocess 실패 → ModelConfig 검증 오류
            print(_r.stderr[-800:])
            raise ValueError("1 validation error for ModelConfig\n  Value error, Model architectures ['Qwen3_5MoeForConditionalGeneration'] failed to be inspected. Please check the logs for more details.")
        import vllm.multimodal.parse  # noqa: F401  (실제 vLLM처럼 멀티모달 처리 모듈 → torchaudio 가져오기 시도)
        from vllm.distributed import parallel_state as _ps, utils as _du    # 실제 vLLM처럼 분산 초기화 중 gloo 그룹 생성 시 stdout 숨기기
        with _ps.suppress_stdout(): pass
        with _du.suppress_stdout(): pass
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(os.environ["FAKE_PROC_DIR"])
        self.mm_kw = dict(kw.get("mm_processor_kwargs") or {})
        self.topk = kw.get("max_logprobs", 20)
        tok = self.proc.tokenizer
        self.var = [[tok.encode(s, add_special_tokens=False)[0] for s in (L, L.upper(), " " + L, " " + L.upper())] for L in "abcd"]
        allv = {i for v in self.var for i in v}
        self.filler = [i for i in range(10, len(tok)) if i not in allv][:400]

    def generate(self, prompts, sampling_params=None, *, use_tqdm=True, **kw):
        for k in kw: assert k in _API["generate_params"], k
        STATE["calls"] += 1
        if STATE["crash_at"] and STATE["calls"] == STATE["crash_at"]:
            raise RuntimeError("가짜 중단 (시험)")
        sp = sampling_params
        assert sp.max_tokens == 1 and sp.temperature == 0.0 and sp.logprobs is not None and sp.logprobs <= self.topk, vars(sp)
        outs = []
        todo = list(STATE.pop("leftover", None) or []) + list(prompts)       # 중간에 멈춘 이전 호출의 요청이 엔진에 남아 있는 상황 흉내 (번호가 더 작음)
        for n, q in enumerate(todo):
            assert set(q) <= {"prompt", "multi_modal_data", "mm_processor_kwargs"}, set(q)
            text, im = q["prompt"], q["multi_modal_data"]["image"]
            mk = dict(self.mm_kw)
            if STATE["mode"] == "mismatch": mk["max_pixels"] = int(mk["max_pixels"] * 0.5)   # 엔진 쪽 프로세서가 이미지를 다시 줄이는 상황 흉내
            enc = self.proc(text=[text], images=[im], return_tensors="pt", **mk)
            ids = enc["input_ids"][0].tolist()
            outs.append(RequestOutput(str(STATE["n_prompts"]), text, ids, [CompletionOutput(*self._answer(text, sp.logprobs))]))
            STATE["n_prompts"] += 1
        return outs

    def _answer(self, text, k):
        user = text.split("<|vision_end|>", 1)[1].split("<|im_end|>", 1)[0]
        h = int(hashlib.md5(user.encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % (2 ** 31))
        gold = STATE["answers"].get(user)
        if gold is None: fav = rng.randint(4)
        else: fav = gold if rng.rand() < STATE["acc"] else (gold + 1 + rng.randint(3)) % 4
        letters_mass = 0.1 if STATE["mode"] == "lowmass" else 0.995
        probs = {}
        w = np.array([0.93 if L == fav else 0.02 for L in range(4)]); w = w / w.sum() * letters_mass
        for L in range(4):
            for j, tid in enumerate(self.var[L]): probs[tid] = w[L] * [0.9, 0.05, 0.03, 0.02][j]
        drop = (h % 29 == 0) and fav != 3                                        # 가끔 'd' 변형이 상위 k개에 하나도 없음
        if drop:
            for tid in self.var[3]: probs.pop(tid, None)
        rest = 1 - sum(probs.values())
        fl = rng.dirichlet(np.ones(len(self.filler)) * 0.3) * rest
        if STATE["mode"] == "lowmass":                                           # 답 자리에서 글자가 아닌 토큰(예: '이')을 내려는 상황
            fl = np.full(len(self.filler), rest * 0.1 / (len(self.filler) - 1)); fl[0] = rest * 0.9
        for tid, p in zip(self.filler, fl): probs[tid] = p
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:k]
        lp = {tid: Logprob(float(np.log(max(p, 1e-30))), r + 1) for r, (tid, p) in enumerate(top)}
        return [top[0][0]], [lp]
