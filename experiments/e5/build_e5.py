# -*- coding: utf-8 -*-
"""E5 노트북 빌더: 기준 앙상블이 애매한 문항만 Qwen 멤버(Q35·Q38)를 더 큰 해상도(up3.0, up4.0)로 다시 채점 → 여러 해상도 평균 → 앙상블 비교 → 제출 후보.
E9 빌더의 분할·게이트·채점 함수·멤버 로드 셀을 그대로 가져오고, 채점 루프와 판정만 해상도용으로 바꿉니다.
"""
import sys, uuid
import nbformat

OUT_NB = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e5/out/e5_hires_gated.ipynb"
_ns = {}
_argv = sys.argv; sys.argv = ["build_e9.py", "/tmp/vqa_scratch/_e9_tmp.ipynb"]
exec(open("/home/user/vqa/e9/build_e9.py", encoding="utf-8").read(), _ns)
sys.argv = _argv
S, rep = _ns["S"], _ns["rep"]

CFG = r'''
MEMBERS_HR = ["q35_27b_r16_native", "q38_27b_r16_native"]
#   고해상도로 다시 채점할 멤버. Qwen(픽셀 기준 해상도)만 가능 — Gemma는 이미지 토큰 1,120개가 최대라 더 못 올림, HCX는 설치가 달라 이 노트북으로 안 됨
HR_PRESETS = ["up3.0", "up4.0"]   # 더 키울 해상도 (원본 720×960 기준 가로세로 약 3배·4배). 메모리가 모자라면 그 해상도만 자동으로 건너뜀
HR_BS      = {"up2.0": 8, "up3.0": 2, "up4.0": 1}   # 해상도별 채점 배치 (OOM이면 자동으로 절반)
PRESET     = "up2.0"        # 기존 멤버 예측 파일의 해상도 (기준)
TAU_GATE   = 0.8            # 기준 앙상블 3개 중 하나라도 1·2위 확률 차 < 0.8인 문항만 채점 (E9와 같은 문항)
RUN_TEST   = True           # test 애매한 문항도 채점 (제출 후보를 만들려면 필요)
RUN_NAME   = "e5_hires"     # 결과 폴더: vqa_runs/e5_hires/<멤버>/  (기존 멤버 폴더는 읽기만 함)
MODEL_ID   = "Qwen/Qwen3.5-27B"   # 자리표시(2번 셀 설치 조건용). 멤버별 베이스 모델은 어댑터 기록에서 자동으로 읽음
DELETE_HF_CACHE = True      # 다른 베이스로 넘어갈 때 이전 베이스 파일을 지워 디스크 확보
RETRY_SKIPPED = False       # True: 예전 실행에서 메모리 부족으로 건너뛴 해상도를 다시 시도 (더 큰 GPU로 바꿨을 때)

# ── 기준 앙상블 레시피 (E8·E9와 같음) ──
Q35, Q38 = ("q35_27b_r16_native", PRESET), ("q38_27b_r16_native", PRESET)
GDEV, G, HCX = ("gemma4_31b_r16_up20_dev", PRESET), ("gemma4_31b_r16_up20", PRESET), ("hcx32b_r16_native", PRESET)
RECIPES = {
    "clean4_qg_hcx":    dict(file="submission_clean4_qg_hcx.csv",    members=[Q35, Q38, GDEV, G, HCX], weights=[1, 1, 1, 1, 2], cal=False),
    "ens_q38_gemma4":   dict(file="submission_ens_q38_gemma4.csv",   members=[Q38, G],                 weights=[1, 1],          cal=False),
    "ens_q38_gdev_cal": dict(file="submission_ens_q38_gdev_cal.csv", members=[Q38, GDEV],              weights=[1, 1],          cal=True),
}
VOTE = dict(name="majority_vote", file="submission_majority_vote.csv", voters=["ens_q38_gemma4", "clean4_qg_hcx", "ens_q38_gdev_cal"])
PUBLIC = {"clean4_qg_hcx": 0.97646, "majority_vote": 0.97676}; N_PUBLIC = 3357
ENSEMBLE_SUBDIR = "ensembles"

# ── 사전 기준 (결과를 보기 전에 고정) ──
# H1: 애매한 문항에서 Q35·Q38 확률을 {up2.0(기존), up3.0, up4.0} 동일 가중 평균으로 바꿈 (메모리 부족으로 빠진 해상도는 제외)
NET_MIN = 3        # 검증 순이득 + 공개 순이득(clean4 후보) ≥ +3 이고 공개 순이득 ≥ 0 → 채택 / 검증 순이득 ≤ −3이면 공개 확인 없이 불채택

DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
BATCH_SIZE   = "auto"
SEED, VAL_SIZE = 42, 1000
SKIP_MISSING_IMAGES = False
HF_TOKEN     = ""

RES_PRESETS = {
    "baseline384": (384 * 384, 384 * 384),
    "mid":         (256 * 28 * 28, 512 * 28 * 28),
    "native":      (256 * 28 * 28, 1280 * 28 * 28),
    "up1.5":       (1_550_000, 2_100_000),
    "up2.0":       (2_700_000, 3_400_000),
    "up3.0":       (6_000_000, 7_600_000),      # 720×960 사진 → 약 2,120×2,830 (가로세로 약 2.95배)
    "up4.0":       (10_500_000, 12_500_000),    # 720×960 사진 → 약 2,810×3,740 (가로세로 약 3.9배)
}
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}
RES_PRESET = PRESET
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
'''

DATA = _ns["DATA"]
for o, n in [("for run in MEMBERS_TTA:", "for run in MEMBERS_HR:"),
             ("1번 셀 MEMBERS_TTA를 확인하세요", "1번 셀 MEMBERS_HR를 확인하세요"),
             ('    assert not ("hyperclova" in mid.lower() or "hcx" in run.lower()), f"{run}: HCX는 이 노트북으로 채점할 수 없습니다 → MEMBERS_TTA에서 빼세요"',
              '    assert not ("hyperclova" in mid.lower() or "hcx" in run.lower()), f"{run}: HCX는 이 노트북으로 채점할 수 없습니다 → MEMBERS_HR에서 빼세요"\n'
              '    assert "gemma" not in mid.lower() and "qwen" in mid.lower(), f"{run}({mid}): 픽셀 기준 해상도를 쓰는 Qwen 멤버만 됩니다 (Gemma는 이미지 토큰 1,120개가 최대) → MEMBERS_HR에서 빼세요"'),
             ('E9_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(E9_DIR, exist_ok=True)\nINFO_PATH = os.path.join(E9_DIR, "e9_info.json")\nE9_INFO = json.load(open(INFO_PATH, encoding="utf-8")) if os.path.exists(INFO_PATH) else {}',
              'E5_DIR = os.path.join(OUT_DIR, RUN_NAME); os.makedirs(E5_DIR, exist_ok=True)\nINFO_PATH = os.path.join(E5_DIR, "e5_info.json")\nE5_INFO = json.load(open(INFO_PATH, encoding="utf-8")) if os.path.exists(INFO_PATH) else {}'),
             ('    old = (E9_INFO.get("members") or {}).get(run)', '    old = (E5_INFO.get("members") or {}).get(run)'),
             ('        raise RuntimeError(f"{E9_DIR}/{run}에는 다른 어댑터로 만든 결과가 있습니다 → 1번 셀 RUN_NAME을 바꾸세요")',
              '        raise RuntimeError(f"{E5_DIR}/{run}에는 다른 어댑터로 만든 결과가 있습니다 → 1번 셀 RUN_NAME을 바꾸세요")'),
             ('print("데이터 OK:", DATA_DIR, "| 결과 폴더(새로 씀):", E9_DIR)', 'print("데이터 OK:", DATA_DIR, "| 결과 폴더(새로 씀):", E5_DIR)')]:
    DATA = rep(DATA, o, n)

GATE = _ns["GATE"]
GATE = rep(GATE, '''print(f"→ 멤버 {len(MEMBERS_TTA)}개 × (검증 {len(VAL_G)} + test {len(TEST_G) if RUN_TEST else 0}) × 순서 4가지" + (" × LoRA 켬/끔" if SCORE_BASE else ""))''',
           '''print(f"→ 멤버 {len(MEMBERS_HR)}개 × (검증 {len(VAL_G)} × 해상도 {1 + len(HR_PRESETS)} + test {len(TEST_G) if RUN_TEST else 0} × 해상도 {len(HR_PRESETS)})  (up2.0 검증은 기존 예측 재현 확인용)")''')
_old_tta = GATE[GATE.index("# 예전에 해 본 TTA 기록"):]
GATE = GATE.replace(_old_tta, "").rstrip() + "\n"          # TTA 기록 출력은 E9에만

SCORING, LOADER, SPLIT = _ns["SCORING"], _ns["LOADER"], _ns["SPLIT"]

RUNCELL = r'''
# ── 멤버마다: 모델 로드 → 사전 점검 → 애매한 문항을 해상도별로 채점 (검증 up2.0 재현 확인 → 검증 고해상도 → test 고해상도). 끊겨도 다시 실행하면 남은 것만 ──
RES_COLS = ["id", "preset", "p_a", "p_b", "p_c", "p_d", "letter_mass", "img_tokens"]
SPLIT_DF = {"val": VAL_G, "test": TEST_G}
def _res_path(run, split): return os.path.join(E5_DIR, run, f"res_{split}.csv")
def _done(run, split):
    p = _res_path(run, split)
    if not os.path.exists(p): return set()
    d = pd.read_csv(p); return set(zip(d.id, d.preset))
def _plan():
    return [("val", "up2.0")] + [("val", pr) for pr in HR_PRESETS] + ([("test", pr) for pr in HR_PRESETS] if RUN_TEST else [])
def _skipped(run): return set() if RETRY_SKIPPED else set((E5_INFO.get("infeasible") or {}).get(run, []))
def _remaining(run):
    dn = {s: _done(run, s) for s in ("val", "test")}
    return {(s, pr): 0 if pr in _skipped(run) else sum((i, pr) not in dn[s] for i in SPLIT_DF[s].id) for s, pr in _plan()}
def score_res(run, split, preset, save_every=10):
    global BS
    path = _res_path(run, split); os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _done(run, split); rows = [r for r in SPLIT_DF[split].to_dict("records") if (r["id"], preset) not in done]
    set_resolution(preset); BS = min(HR_BS.get(preset, 8), (8 if VRAM_GB >= 38 else 4 if VRAM_GB >= 20 else 2))
    print(f"  [{split} | {preset}] {len(rows):,}문항 남음 (배치 {BS})", flush=True)
    recs, n, nb, t0 = [], 0, 0, time.time()
    try:
        while n < len(rows):
            batch = rows[n: n + BS]
            p, m, nt = score_rows(batch)
            if n == 0 and preset != PRESET:
                # 해상도가 실제로 바뀌었는지: 모든 사진이 up2.0에선 최대 픽셀 이하, 고해상도에선 최소 픽셀 이상이어야 하므로 토큰 수로 바로 확인됨
                _lo = 0.85 * RES_PRESETS[preset][0] / RES_PRESETS[PRESET][1] * UP2_TOK_MAX
                if min(nt) < _lo:
                    raise RuntimeError(f"{preset} 해상도가 적용되지 않은 것 같습니다: 이미지 토큰 {list(nt)} < 기대 최소 {_lo:.0f} "
                                       f"(up2.0 사전 점검 최대 {UP2_TOK_MAX}) → 위 출력을 확인하세요")
            for r, pp, mm, tt in zip(batch, p, m, nt):
                recs.append({"id": r["id"], "preset": preset, "p_a": pp[0], "p_b": pp[1], "p_c": pp[2], "p_d": pp[3], "letter_mass": float(mm), "img_tokens": int(tt)})
            n += len(batch); nb += 1
            if nb % save_every == 0 or n >= len(rows):
                pd.DataFrame(recs, columns=RES_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False); recs = []
                el = time.time() - t0; print(f"    {n:>5}/{len(rows)} | {el / n:.2f}s/문항 | 이미지 토큰 약 {int(np.mean(nt)):,}개 | 남은 시간 ~{(len(rows) - n) * el / n / 60:.0f}분", flush=True)
    except RuntimeError as e:
        if "메모리 부족" not in str(e): raise
        if recs: pd.DataFrame(recs, columns=RES_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        E5_INFO.setdefault("infeasible", {}).setdefault(run, [])
        if preset not in E5_INFO["infeasible"][run]: E5_INFO["infeasible"][run].append(preset)
        json.dump(E5_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        torch.cuda.empty_cache(); print(f"  ⚠️ {preset}: 배치 1에서도 GPU 메모리 부족 → 이 멤버는 {preset}를 건너뜁니다")
    set_resolution(PRESET)

E5_INFO.setdefault("members", {})
for i, run in enumerate(MEMBERS_HR):
    rem = _remaining(run)
    print(f"\n▶ {run} ({i + 1}/{len(MEMBERS_HR)}) | 남은 채점: " + ", ".join(f"{s}·{pr} {n:,}" for (s, pr), n in rem.items()))
    if sum(rem.values()) == 0: print("  이미 끝났습니다 → 건너뜀"); continue
    _free(); load_member(run)
    # 사전 점검: 어댑터 켬/끔 차이, letter_mass, 이미지 토큰 수가 기존 예측과 같은지 (up2.0)
    set_resolution(PRESET)
    _chk = VAL_G.head(4).to_dict("records") or val_df.head(4).to_dict("records")
    p_on, m_on, nt_on = score_rows(_chk)
    with model.disable_adapter(): p_off, _, _ = score_rows(_chk)
    _d = float(np.abs(p_on - p_off).max())
    _ref = pd.read_csv(MEMBER_INFO[run]["ref_val"]).drop_duplicates("id", keep="last").set_index("id").reindex([r["id"] for r in _chk])
    print(f"  사전 점검(up2.0): 어댑터 켬/끔 확률 최대 차이 {_d:.3f} | letter_mass {m_on.mean():.3f} | 이미지 토큰 {list(nt_on)}"
          + (f" (기존 {[int(t) for t in _ref.img_tokens]})" if "img_tokens" in _ref.columns else ""))
    assert _d > 1e-3, "어댑터를 켜고 꺼도 출력이 같습니다 → 어댑터가 적용되지 않았습니다. 위 출력을 확인하세요"
    assert m_on.mean() > 0.9, "letter_mass가 낮습니다 → 프롬프트·토큰 처리 문제. 위 출력을 확인하세요"
    UP2_TOK_MAX = int(max(nt_on))
    if "img_tokens" in _ref.columns and not _ref.img_tokens.isna().any():
        assert [int(t) for t in _ref.img_tokens] == [int(t) for t in nt_on], "이미지 토큰 수가 기존 예측과 다릅니다 → 해상도·프로세서가 달라 비교할 수 없습니다"
    E5_INFO["members"][run] = {"md5": MEMBER_INFO[run]["md5"], "model_id": MODEL_ID, "adapter": MEMBER_INFO[run]["adapter_dir"], "precision": "bf16", "gpu": GPU_NAME,
                               "presets": {pr: list(RES_PRESETS[pr]) for pr in ["up2.0"] + HR_PRESETS}, "tau_gate": TAU_GATE}
    json.dump(E5_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for s, pr in _plan():
        if pr in _skipped(run): print(f"  [{s} | {pr}] 메모리 부족으로 건너뜀(이전 실행 기록)"); continue
        score_res(run, s, pr)
    nxt = MEMBER_INFO[MEMBERS_HR[i + 1]]["model_id"] if i + 1 < len(MEMBERS_HR) else None
    if DELETE_HF_CACHE and nxt != MODEL_ID:
        _free(); _drop_hf_cache(MODEL_ID)
print("\n채점 끝 → 11번 셀에서 판정")
'''

EVAL = r'''
# ── 판정 (GPU 불필요): 애매한 문항에서만 Q35·Q38 확률을 여러 해상도 평균으로 바꿔 같은 레시피로 앙상블 → 같은 문항끼리 비교 ──
V = {"lines": [], "cands": []}
def say(s): print(s); V["lines"].append(s)
IDX = {"val": {i: n for n, i in enumerate(VAL_IDS)}, "test": {i: n for n, i in enumerate(TEST_IDS)}}
GIDS = {"val": VAL_G.id.tolist(), "test": TEST_G.id.tolist()}
gv = np.array([IDX["val"][i] for i in GIDS["val"]], int); gt = np.array([IDX["test"][i] for i in GIDS["test"]], int)
def res_mat(run, split, preset):
    # (애매한 문항 수, 4) — 그 해상도로 모든 문항이 채점됐을 때만, 아니면 (None, None, None)
    p = _res_path(run, split)
    if not os.path.exists(p): return None, None, None
    d = pd.read_csv(p); d = d[d.preset == preset].drop_duplicates("id", keep="last").set_index("id").reindex(GIDS[split])
    if d.p_a.isna().any(): return None, None, None
    return d[["p_a", "p_b", "p_c", "p_d"]].to_numpy(dtype=float), float(d.letter_mass.mean()), float(d.img_tokens.mean())
MEM_ROWS, OVER1, OVER_SINGLE, health, USED = [], {}, {p: {} for p in HR_PRESETS}, [], {}
for run in MEMBERS_HR:
    Pv0, Pt0 = load(run, PRESET, "val"), load(run, PRESET, "test")
    R2, m2, t2 = res_mat(run, "val", "up2.0")
    avail = []
    for pr in HR_PRESETS:
        if pr in _skipped(run): continue
        Rv, mv, tv = res_mat(run, "val", pr); Rt = res_mat(run, "test", pr)[0] if RUN_TEST else None
        if Rv is None or (RUN_TEST and Rt is None): continue
        avail.append((pr, Rv, Rt, mv, tv))
    row = {"멤버": run}
    if R2 is None or not avail:
        row["상태"] = "채점 미완료 → 원래 확률 사용"; MEM_ROWS.append(row); continue
    yg = Y_VAL[gv]; base_g = Pv0[gv]
    rep_ = float((R2.argmax(1) == base_g.argmax(1)).mean() * 100)
    row.update({"상태": "✅", "up2.0 재현(%)": round(rep_, 1), "애매한 검증 정답: up2.0(기존)": int((base_g.argmax(1) == yg).sum())})
    for pr, Rv, Rt, mv, tv in avail:
        row[f"{pr} 정답"] = int((Rv.argmax(1) == yg).sum()); row[f"{pr} letter_mass"] = round(mv, 3); row[f"{pr} 이미지 토큰"] = int(tv)
        if mv < 0.9: health.append(f"{run} {pr}: letter_mass {mv:.3f} < 0.9")
    row["해상도 평균 정답"] = int((np.mean([base_g] + [a[1] for a in avail], 0).argmax(1) == yg).sum())
    row["문항 수"] = len(gv)
    if rep_ < 85: health.append(f"{run}: up2.0 재현 {rep_:.1f}% < 85%")
    if m2 < 0.9: health.append(f"{run} up2.0: letter_mass {m2:.3f}")
    Pv1, Pt1 = Pv0.copy(), Pt0.copy()
    Pv1[gv] = np.mean([base_g] + [a[1] for a in avail], 0)
    if RUN_TEST: Pt1[gt] = np.mean([Pt0[gt]] + [a[2] for a in avail], 0)
    OVER1[run] = (Pv1, Pt1); USED[run] = [a[0] for a in avail]
    for pr, Rv, Rt, _, _ in avail:                                    # 참고용: 기존 대신 그 해상도만 쓴 경우
        Pv_, Pt_ = Pv0.copy(), Pt0.copy(); Pv_[gv] = Rv
        if RUN_TEST: Pt_[gt] = Rt
        OVER_SINGLE[pr][run] = (Pv_, Pt_)
    MEM_ROWS.append(row)
print(pd.DataFrame(MEM_ROWS).to_string(index=False))
COMPLETE = len(OVER1) == len(MEMBERS_HR) and all(set(USED[r]) == set(HR_PRESETS) - _skipped(r) for r in OVER1)
if not OVER1:
    say("채점이 끝난 멤버가 없습니다 → 10번 셀을 먼저 끝까지 실행하세요")
else:
    def build(over):
        P = {n: {s: ens_probs(r, s, over) for s in ("val", "test")} for n, r in RECIPES.items()}
        A = {n: {s: P[n][s].argmax(1) for s in ("val", "test")} for n in RECIPES}
        A[VOTE["name"]] = {s: vote_of([A[v][s] for v in VOTE["voters"]]) for s in ("val", "test")}
        return A
    A1 = build(OVER1)
    say(f"[H1: 해상도 평균] 반영 멤버: " + ", ".join(f"{r}(up2.0+{'+'.join(USED[r])})" for r in OVER1) + ("" if COMPLETE else " (미완료 있음)"))
    RES = {}
    for nm in ("clean4_qg_hcx", VOTE["name"]):
        f, b, n, D = paired(A0[nm]["val"], A1[nm]["val"], Y_VAL); RES[nm] = (n, D)
        say(f"  {nm}: 검증 {(A0[nm]['val'] == Y_VAL).mean() * 100:.2f}% → {(A1[nm]['val'] == Y_VAL).mean() * 100:.2f}% | 고침 {f} | 새로 틀림 {b} | 순이득 {n:+d} | 바뀐 답 {D}"
            + ("  (바뀐 답 10개 미만 → 검증만으로는 판단 안 함)" if D < 10 else ""))
    for pr in HR_PRESETS:                                              # 참고용 (판정에 쓰지 않음)
        if len(OVER_SINGLE[pr]) == len(OVER1):
            As = build(OVER_SINGLE[pr]); f, b, n, D = paired(A0["clean4_qg_hcx"]["val"], As["clean4_qg_hcx"]["val"], Y_VAL)
            say(f"  (참고) {pr}만 쓴 경우 clean4: 고침 {f} / 새로 틀림 {b} / 순이득 {n:+d} (바뀐 답 {D})")
    n1, D1 = RES["clean4_qg_hcx"]
    # 해상도를 올려 답이 바뀐 검증 문항을 이미지와 함께 저장 (작은 글씨를 실제로 더 잘 읽었는지 눈으로 확인)
    try:
        import base64, io, html as _h
        ch = np.where(A0["clean4_qg_hcx"]["val"] != A1["clean4_qg_hcx"]["val"])[0]
        parts = [f"<h2>E5 — 해상도 평균으로 clean4 답이 바뀐 검증 문항 {len(ch)}개</h2>"]
        for i in ch:
            r = val_df.iloc[i]; im = load_image(r.path); im.thumbnail((1000, 1000)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
            res = "고침" if A1["clean4_qg_hcx"]["val"][i] == Y_VAL[i] else ("새로 틀림" if A0["clean4_qg_hcx"]["val"][i] == Y_VAL[i] else "둘 다 틀림")
            opts = "".join(f"<li>({k}) {_h.escape(str(r[k]))}{' <b>← 정답</b>' if k == r.answer else ''}{' ← 기존' if LETTERS[A0['clean4_qg_hcx']['val'][i]] == k else ''}{' ← 해상도 평균' if LETTERS[A1['clean4_qg_hcx']['val'][i]] == k else ''}</li>" for k in LETTERS)
            parts.append(f"<div style='margin:18px 0;border-top:1px solid #ccc;padding-top:12px'><b>{r.id} — {res}</b><p>{_h.escape(str(r.question))}</p><ul>{opts}</ul>"
                         f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'></div>")
        _hp = os.path.join(E5_DIR, "e5_changed_val.html")
        open(_hp, "w", encoding="utf-8").write("<meta charset='utf-8'><body style='font-family:sans-serif;max-width:1100px;margin:auto'>" + "".join(parts))
        print("답이 바뀐 검증 문항 리포트:", _hp)
    except Exception as e:
        print("(리포트 생략:", e, ")")
    if health:
        say("판정: 확인 필요 — " + "; ".join(health) + " → 위 출력을 확인하세요")
    elif not COMPLETE:
        say("판정: 대기 — 모든 멤버의 채점이 끝나면 다시 실행")
    elif n1 <= -3:
        say(f"판정: 불채택 — H1 검증 순이득 {n1:+d} (≤ −3). 공개로 확인할 필요 없음")
    elif RUN_TEST:
        say(f"판정: 공개 확인 — 검증 순이득 {n1:+d}(바뀐 답 {D1}). 아래 clean4 후보의 공개 점수로 결정합니다")
        def _write(name, base_name, A_new, A_ref):
            fa = file_answers(RECIPES[base_name]["file"] if base_name in RECIPES else VOTE["file"])
            base_ans = fa if fa is not None else A_ref
            chg = A_new != A_ref; cand = np.where(chg, A_new, base_ans)
            path = os.path.join(E5_DIR, f"submission_{name}.csv"); sub_df[["id"]].assign(answer=[LETTERS[i] for i in cand]).to_csv(path, index=False)
            return os.path.basename(path), int((cand != base_ans).sum())
        need = max(0, NET_MIN - n1)
        for nm in ("clean4_qg_hcx", VOTE["name"]):
            fn, Dn = _write(f"{nm}__e5_hires", nm, A1[nm]["test"], A0[nm]["test"]); b0 = count_of(PUBLIC[nm]); V["cands"].append((fn, Dn))
            line = f"  제출 후보: {fn} | 기존 {nm}({PUBLIC[nm]:.5f} = {b0:,}문항)과 다른 답 {Dn}개 (공개에 약 {Dn // 2}개)"
            if nm == "clean4_qg_hcx":
                line += f" | ◀ 판정용: 공개 {score_of(b0 + need):.5f}({b0 + need:,}문항) 이상이면 채택 (검증 {n1:+d} + 공개 ≥ +{NET_MIN}, 공개 ≥ 0)"
            say(line)
        say("  제출 순서: clean4 후보(판정용) → 다수결 후보(실제로 쓸 것). 판정은 clean4 후보 한 줄로만 합니다")
    else:
        say(f"판정: 검증만 확인 — 검증 순이득 {n1:+d}(바뀐 답 {D1}). 제출 후보를 만들려면 1번 셀 RUN_TEST=True로 10번 셀부터 다시 실행")
    E5_INFO.update({"verdict_lines": V["lines"], "cands": V["cands"]})
    json.dump(E5_INFO, open(INFO_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
'''

SUMMARY = r'''
print(f"E5 | 애매한 문항 고해상도 | 기준 {PRESET} + {HR_PRESETS} | 채점 대상 검증 {len(VAL_G)} / test {len(TEST_G)} | GPU {globals().get('GPU_NAME', '-')}")
if E5_INFO.get("infeasible"): print("  메모리 부족으로 건너뛴 해상도:", E5_INFO["infeasible"])
for r in MEM_ROWS: print("  ", r)
for l in V["lines"]: print(l)
'''

INTRO = r'''# E5 · 애매한 문항만 더 큰 해상도로 다시 채점 (학습 없음)

**왜**: 대회 이미지는 대부분 720×960(약 0.7MP)입니다. 지금 쓰는 up2.0(2.7~3.4MP)도 이미 가로세로 약 2배로 키운 것이고, native보다 up2.0 점수가 높았습니다. 원본에 없는 글씨가 새로 생기지는 않지만, 크게 키우면 작은 글자 하나가 더 많은 이미지 토큰에 걸쳐 들어가 모델이 구분하기 쉬워질 수 있습니다. 그래서 **up3.0(약 3배), up4.0(약 4배)**까지 올려 봅니다.

**하는 일**
- E9와 같은 애매한 문항(세 앙상블 중 하나라도 1·2위 확률 차 < 0.8)만 **Q35·Q38**을 기존 어댑터 그대로 up3.0·up4.0으로 다시 채점합니다. Gemma는 이미지 토큰 1,120개가 최대라 더 못 올리고, HCX는 설치가 달라 이 노트북에서 빠집니다(원래 확률 그대로).
- 먼저 검증 문항을 up2.0으로 다시 채점해 기존 예측이 재현되는지 확인합니다(85% 이상이어야 정상). 해상도마다 letter_mass와 이미지 토큰 수도 기록합니다.
- 해상도를 올려 앙상블 답이 바뀐 검증 문항은 `e5_changed_val.html`에 이미지와 함께 저장합니다. 작은 글씨를 정말 더 잘 읽었는지 눈으로 볼 수 있습니다.

**사전 기준 (결과를 보기 전에 고정)**
- **H1**: 애매한 문항에서 Q35·Q38 확률을 {up2.0(기존), up3.0, up4.0} 동일 가중 평균으로 바꿉니다. 메모리 부족으로 빠진 해상도는 평균에서 제외합니다. 해상도 하나만 쓴 결과는 참고용입니다.
- 검증 순이득 + clean4 후보 공개 순이득 ≥ +3이고 공개 순이득 ≥ 0 → 채택. 검증 순이득 ≤ −3이면 공개 확인 없이 불채택. 검증에서 바뀐 답이 10개 미만이면 검증만으로 판단하지 않습니다.

**실행**: 위에서부터 끝까지. 끊기면 다시 실행하면 남은 채점만 합니다. up4.0은 문항당 이미지 토큰이 약 1만 개라 배치 1로 느리게 돕니다. 메모리가 모자라면 그 해상도만 건너뛰고 기록합니다.
**런타임**: G4 권장 (A100 80GB도 가능). 결과는 `vqa_runs/e5_hires/`에만 씁니다.'''

cells = [
    ("markdown", INTRO),
    ("markdown", "## 1. 설정"), ("code", CFG),
    ("markdown", "## 2. 라이브러리 설치 (학습 때와 같은 버전)"), ("code", S(4)),
    ("markdown", "## 3. Drive 연결 → 멤버 확인 → 데이터 준비"), ("code", DATA),
    ("markdown", "## 4. 공통 import · CSV 로드"), ("code", S(8)),
    ("markdown", "## 5. 고정 검증셋 분할"), ("code", SPLIT),
    ("markdown", "## 6. 프롬프트 (학습 때와 동일)"), ("code", S(12)),
    ("markdown", "## 7. 기준 앙상블 → 애매한 문항 고르기 (GPU 불필요, E9와 같은 문항)"), ("code", GATE),
    ("markdown", "## 8. 채점 함수 (stage2와 동일, 증강만 제거)"), ("code", SCORING),
    ("markdown", "## 9. 멤버 로드 함수 (bf16 베이스 + 기존 어댑터, merge 없음)"), ("code", LOADER),
    ("markdown", "## 10. 멤버별 채점: 검증 up2.0 재현 확인 → 검증 up3.0·up4.0 → test up3.0·up4.0"), ("code", RUNCELL),
    ("markdown", "## 11. 판정 → 제출 후보\n멤버별로 `up2.0 재현(%)`이 85% 이상, 해상도별 letter_mass가 0.9 이상이어야 정상입니다. 해상도별 정답 수(애매한 검증 문항 기준)도 함께 봅니다."), ("code", EVAL),
    ("markdown", "## 12. 결과 요약"), ("code", SUMMARY),
]
nb = nbformat.v4.new_notebook()
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
for kind, text in cells:
    text = text.strip("\n")
    c = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]; nb.cells.append(c)
nbformat.validate(nb); nbformat.write(nb, OUT_NB)
print("저장:", OUT_NB, "| 셀", len(nb.cells))
