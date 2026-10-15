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
STATE = {"init_kw": None, "n_init": 0, "calls": 0, "crash_at": None, "mode": "normal", "answers": {}, "acc": 0.97, "n_prompts": 0,
         "xc_gold": {}, "xc_override": {}, "xc_invalid": {}, "xc_mode": "good", "gen_log": []}


_HCX_IMG = "<|image_start|><|IMAGE_PAD|><|image_end|>"
def _to_qwen(text):
    # 시험용: HyperCLOVA X 형식 이미지 자리(vLLM 전용 구현이 <|IMAGE_PAD|>를 펼침)를 작은 Qwen 프로세서가 아는 자리로 바꿔 같은 처리를 흉내
    if "<|IMAGE_PAD|>" in text:
        assert text.count(_HCX_IMG) == 1 and text.count("<|IMAGE_PAD|>") == 1, "HyperCLOVA X 프롬프트의 이미지 자리가 한 개가 아님"
        text = text.replace(_HCX_IMG, "<|vision_start|><|image_pad|><|vision_end|>")
    return text


class SamplingParams:
    def __init__(self, **kw):
        for k in kw: assert k in _API["sp_fields"], f"SamplingParams에 없는 인자: {k}"
        self.max_tokens, self.temperature, self.logprobs = kw.get("max_tokens", 16), kw.get("temperature", 1.0), kw.get("logprobs")
        self.n, self.seed, self.top_p, self.top_k = kw.get("n", 1), kw.get("seed"), kw.get("top_p", 1.0), kw.get("top_k", 0)
        self.kw = kw


class Logprob:
    def __init__(self, logprob, rank=None, decoded_token=None): self.logprob, self.rank, self.decoded_token = logprob, rank, decoded_token


class CompletionOutput:
    def __init__(self, token_ids, logprobs, text="", finish_reason="stop", index=0):
        self.index, self.text, self.token_ids, self.cumulative_logprob, self.logprobs, self.finish_reason = index, text, token_ids, None, logprobs, finish_reason


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
        if isinstance(sp, (list, tuple)) or sp.max_tokens > 1:               # 생성 모드 (교차 검증): 표본 n개 글자 출력
            return self._generate_text(prompts, sp)
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

    def _generate_text(self, prompts, sps):
        import re as _re
        sps = list(sps) if isinstance(sps, (list, tuple)) else [sps] * len(prompts)
        assert len(sps) == len(prompts), "SamplingParams 목록과 프롬프트 수가 다름"
        outs = []
        todo = list(STATE.pop("leftover", None) or []) + list(prompts)
        sps = [sps[0]] * (len(todo) - len(prompts)) + sps
        for q, sp in zip(todo, sps):
            assert set(q) <= {"prompt", "multi_modal_data", "mm_processor_kwargs"}, set(q)
            assert sp.n >= 1 and sp.temperature > 0 and sp.max_tokens > 1 and sp.seed is not None, vars(sp)
            text, im = q["prompt"], q["multi_modal_data"]["image"]
            STATE.setdefault("prompts", []).append(text)
            orig, text = text, _to_qwen(text)
            enc = self.proc(text=[text], images=[im], return_tensors="pt", **dict(self.mm_kw))
            ids = enc["input_ids"][0].tolist()
            user = text.split("<|vision_end|>", 1)[1].split("<|im_end|>", 1)[0]
            key = user.split("\n\n[작전 지시")[0].lstrip("\n")     # HyperCLOVA X 템플릿은 이미지와 글 사이에 줄바꿈 하나
            mA, mB = _re.search(r"모델 A의 답: \(([abcd])\)", user), _re.search(r"모델 B의 답: \(([abcd])\)", user)
            assert mA and mB and mA.group(1) != mB.group(1), user[-300:]
            A, B = mA.group(1), mB.group(1)
            STATE["gen_log"].append({"key": key, "A": A, "B": B, "seed": sp.seed, "n": sp.n, "max_tokens": sp.max_tokens, "thinking": "<think>\n\n</think>" not in text})
            gold = STATE["xc_gold"].get(key); ov = STATE["xc_override"].get(key); ninv = STATE["xc_invalid"].get(key, 0)
            res = []
            for j in range(sp.n):
                if STATE["xc_mode"] == "noformat" or j < ninv:
                    res.append(("이미지를 보니 판단하기 어렵다. 두 답 모두 가능해 보인다.", "length" if j % 2 else "stop")); continue
                if ov is not None: L = ov
                elif gold is not None: L = gold if j < sp.n - 1 else (B if gold == A else A)
                else: L = A
                body = f"1. 이미지에서 관련 글자를 읽었다.\n2. 모델 A({A})와 모델 B({B})의 근거를 비교했다.\n3. 착각 패턴을 배제했다.\n"
                if STATE["xc_mode"] == "think":
                    wrong = B if L == A else A
                    res.append((f"<think>처음에는 정답: ({wrong})라고 생각했다. 다시 보자.</think>\n{body}최종 정답: ({L})", "stop"))
                else:
                    res.append((body + f"정답: ({L})", "stop"))
            outs.append(RequestOutput(str(STATE["n_prompts"]), orig, ids,
                                      [CompletionOutput(list(range(len(t) // 2)), None, text=t, finish_reason=fr, index=j) for j, (t, fr) in enumerate(res)]))
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
