# === Qwen3.8-Flash-Next(qwen4_exp) 보정: transformers 5.17 그대로는 학습이 안 되는 두 곳 ===
# (1) QSA 인덱서: 원래 코드는 질의 토큰 하나마다 파이썬 루프(nonzero·topk…)를 돌아 3천 토큰 입력이면 문항당 수십 초.
#     → 같은 계산을 질의 전체에 한 번에(벡터화). 선택되는 토큰 마스크는 원래 구현과 같음(작은 모델로 대조 시험).
#     캐시를 쓰는 생성 경로와 예상 밖 마스크 모양은 원래 구현으로 넘김.
# (2) n-gram 임베딩 표(51.2B = 95.4GiB): 체크포인트에 128조각으로 나뉘어 있고, transformers는 CPU에서 한꺼번에 이어 붙여
#     CPU RAM을 약 190GiB까지 씀. → 조각은 불러오기에서 빼고(key_mapping), 빈 표를 만든 뒤 조각을 하나씩 복사.
#     GPU 1장이면 표를 CPU RAM에 둠(순전파마다 조회 한 번 → 속도 영향 작음). GPU 여러 장이면 device_map="auto"로 GPU에.
import math, re, json, os
import torch
from torch import nn


def install_qsa_patch():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _q4
    cls = _q4.Qwen4ExpTextQSAIndexer
    if getattr(cls, "_fast_qsa", False): return cls
    orig = cls.forward

    def fast_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        if past_key_values is not None or attention_mask is None or attention_mask.dim() != 4:
            return orig(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        B, S, _ = hidden_states.shape
        D, R, K = self.index_head_dim, self.compress_ratio, self.block_topk
        vis = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        vis = vis[:, 0]                                                    # (B, S, KV) — 보이는 키
        KV = vis.shape[-1]
        if KV != S or vis.shape[1] != S:
            return orig(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        full_cos, full_sin = position_embeddings
        with torch.no_grad():                                              # 선택 결과(불리언 마스크)만 쓰므로 기울기 불필요
            qk = self.index_qk_proj(hidden_states)
            q, token_k = torch.split(qk, [self.index_n_heads * D, self.index_kv_heads * D], dim=-1)
            q = q.reshape(B, S, -1, D)
            raw_keys = token_k.reshape(B, S, -1, D).squeeze(2)
            q = self.q_layernorm(q)
            q = _q4.apply_rotary_pos_emb(q, cos=full_cos[:, -S:, :], sin=full_sin[:, -S:, :], unsqueeze_dim=2)
            ar = torch.arange(S, device=vis.device)
            causal = ar[None, :] <= ar[:, None]                            # (S, KV)
            sel = torch.zeros((B, S, KV), dtype=torch.bool, device=vis.device)
            for b in range(B):
                vb = vis[b]
                keyvalid = vb.any(dim=0)
                if not bool(keyvalid.any()): continue                      # 전부 가려진 행뿐 → 원래 구현도 선택 없음
                p = int(torch.argmax(keyvalid.to(torch.uint8)))            # 첫 유효 토큰 (왼쪽 패딩)
                if not torch.equal(vb, causal & keyvalid[None, :] & (ar >= p)[:, None]):
                    sb = orig(self, hidden_states[b:b + 1], (full_cos[b:b + 1], full_sin[b:b + 1]), attention_mask[b:b + 1], None)
                    sel[b] = (sb if sb.dtype == torch.bool else sb == 0)[0, 0]
                    continue
                n = S - p                                                  # 유효 토큰 수 (질의 = 키 = p..S-1)
                r = torch.arange(n, device=vis.device)                     # 질의의 순위 (보이는 토큰 수 - 1)
                nb_row = (r + 1) // R                                      # 질의마다 완성된 블록 수
                nblk = n // R
                tok = torch.zeros((n, n), dtype=torch.bool, device=vis.device)
                if nblk > 0:
                    kg = raw_keys[b, p:p + nblk * R].view(nblk, R, D)
                    pooled = self.k_layernorm(kg.float().mean(dim=1).to(raw_keys.dtype))
                    starts = p + torch.arange(nblk, device=vis.device) * R
                    bk = _q4.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=full_cos[b].index_select(0, starts),
                                                  sin=full_sin[b].index_select(0, starts)).squeeze(1)       # (nblk, D)
                    sc = torch.einsum("nhd,jd->njh", q[b, p:].float(), bk.float())
                    sc = torch.relu(sc).sum(dim=-1) / math.sqrt(D)                                          # (n, nblk)
                    jj = torch.arange(nblk, device=vis.device)
                    sc = sc.masked_fill(jj[None, :] >= nb_row[:, None], float("-inf"))
                    top = sc.topk(min(K, nblk), dim=-1).indices
                    keep = top < nb_row[:, None]                                  # 보이는 블록이 K개보다 적은 질의: -inf로 뽑힌 자리는 버림
                    blk = torch.zeros((n, nblk), dtype=torch.bool, device=vis.device).scatter_(1, top, keep)
                    tok[:, :nblk * R] = blk.repeat_interleave(R, dim=1)
                tt = torch.arange(n, device=vis.device)
                tok |= (tt[None, :] >= (nb_row * R)[:, None]) & (tt[None, :] <= r[:, None])                  # 꼬리(미완성 블록)
                sel[b, p:, p:] = tok
        sel = sel.unsqueeze(1)
        if attention_mask.is_floating_point():
            sel = torch.where(sel, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return sel

    cls.forward = fast_forward
    cls._fast_qsa = True
    cls._orig_forward = orig
    return cls


NGRAM_SHARD_RE = re.compile(r"^(.*\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$")


def ngram_shards(model_dir):
    """체크포인트 색인에서 n-gram 표 조각 [(키, 파일, 번호)]과 모듈 경로를 찾음"""
    ip = os.path.join(model_dir, "model.safetensors.index.json")
    wm = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None
    if wm is None:
        from safetensors import safe_open
        f1 = os.path.join(model_dir, "model.safetensors")
        with safe_open(f1, "pt") as f: wm = {k: "model.safetensors" for k in f.keys()}
    out = []
    for k, fn in wm.items():
        m = NGRAM_SHARD_RE.match(k)
        if m: out.append((k, fn, int(m.group(2)), m.group(1)))
    out.sort(key=lambda x: (x[3], x[2]))
    return out


def flash_device_map_1gpu(config, gpu=0):
    """GPU 1장: n-gram 표만 device_map에서 빼서 CPU에 두고 나머지는 전부 gpu에. 값이 한 종류라 accelerate 훅이 붙지 않음"""
    from transformers import AutoModelForImageTextToText
    with torch.device("meta"):
        meta = AutoModelForImageTextToText.from_config(config)
    skip = {n for n, _ in meta.named_parameters() if n.endswith("ple_embedding.ngram_embedding.weight")}
    dm = {}

    def rec(mod, name):
        inside = [s for s in skip if name == "" or s.startswith(name + ".")]
        if not inside:
            dm[name] = gpu; return
        for pn, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            full = f"{name}.{pn}" if name else pn
            if full not in skip: dm[full] = gpu
        for cn, ch in mod.named_children():
            rec(ch, f"{name}.{cn}" if name else cn)
    rec(meta, "")
    return dm, sorted(skip)


def load_flash_model(model_dir, load_kw, ngram_mode, log=print, gpu=0):
    """ngram_mode: "cpu"(GPU 1장, 표를 CPU RAM에) 또는 "gpu"(device_map 그대로, 보통 "auto")"""
    import time
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _q4
    from safetensors import safe_open
    shards = ngram_shards(model_dir)
    assert shards, f"n-gram 조각을 체크포인트에서 찾지 못했습니다: {model_dir}"
    kw = dict(load_kw)
    if ngram_mode == "cpu":
        kw["device_map"], skip = flash_device_map_1gpu(AutoConfig.from_pretrained(model_dir), gpu=gpu)
    kw["key_mapping"] = {r"ple_embedding\.ngram_embedding\.shard_(\d+)\.weight": r"ple_embedding.ngram_skipped_shard_\1"}
    big = nn.Embedding
    _orig_init = _q4.Qwen4ExpPreTrainedModel._init_weights

    def _init_skip_big(self, module):                                      # 95GiB 표를 무작위로 초기화하지 않음 (곧 체크포인트 값으로 덮어씀)
        if isinstance(module, big) and module.weight.numel() > 1e9: return
        return _orig_init(self, module)
    _q4.Qwen4ExpPreTrainedModel._init_weights = _init_skip_big
    try:
        t0 = time.time()
        model, info = AutoModelForImageTextToText.from_pretrained(model_dir, output_loading_info=True, **kw)
    finally:
        _q4.Qwen4ExpPreTrainedModel._init_weights = _orig_init
    miss = sorted(info.get("missing_keys", []))
    prefixes = sorted({s[3] for s in shards})
    assert miss == [p + ".weight" for p in prefixes], f"불러오지 못한 가중치가 n-gram 표 말고도 있습니다: {miss[:8]}"
    log(f"  n-gram 표를 뺀 가중치 로드 {time.time() - t0:.0f}s → 표 채우기 ({len(shards)}조각)")
    t0 = time.time()
    for pref in prefixes:
        W = model.get_submodule(pref).weight
        off, files = 0, {}
        for k, fn, i, p in [s for s in shards if s[3] == pref]:
            with safe_open(os.path.join(model_dir, fn), "pt") as f:
                t = f.get_tensor(k)
            W.data[off:off + t.shape[0]].copy_(t.to(W.dtype))
            off += t.shape[0]; del t
        assert off == W.shape[0], f"n-gram 표 행 수 불일치: 조각 합 {off} ≠ 표 {W.shape[0]}"
        log(f"  n-gram 표 {tuple(W.shape)} {W.dtype} → {W.device} ({W.numel() * W.element_size() / 2**30:.1f}GiB, {time.time() - t0:.0f}s)")
    return model
