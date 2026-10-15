# -*- coding: utf-8 -*-
"""팀의 stage2 bf16 노트북(Qwen3.5-27B / Gemma4-31B) → OCR 글자를 프롬프트에 넣는 판.
바꾼 것: 설정(RUN_NAME·OCR·비교 대상·MAKE_SUBMISSION) · 프롬프트(OCR 블록) · 학습 기록(OCR 설정) · 오답 리포트(OCR 표시) · 11-2 OCR 효과 비교 셀 · 요약.
학습·평가·제출·이어하기 코드는 그대로."""
import json, copy

U = "/tmp/vqa_uploads"
JOBS = [
    dict(src="/home/user/vqa/src_stage2.ipynb", dst="/home/user/vqa/ocr/out/stage2_ocr_qwen35.ipynb",
         run_old='"q35_27b_r16_native_bf16"', run_new='"q35_27b_r16_native_ocr_bf16"',
         compare='["q35_27b_r16_native_bf16", "q35_27b_r16_native"]', label="Qwen3.5-27B"),
    dict(src=f"{U}/98af8ea5-stage2_gemma4_31b_bf16.ipynb", dst="/home/user/vqa/ocr/out/stage2_ocr_gemma4.ipynb",
         run_old='"gemma4_31b_r16_up20_bf16"', run_new='"gemma4_31b_r16_up20_ocr_bf16"',
         compare='["gemma4_31b_r16_up20_bf16", "gemma4_31b_r16_up20"]', label="Gemma4-31B"),
]

def build(job):
    nb = json.load(open(job["src"], encoding="utf-8")); C = nb["cells"]
    def get(i): return "".join(C[i]["source"])
    def put(i, s):
        lines = s.split("\n"); C[i]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        if C[i]["cell_type"] == "code": C[i]["outputs"] = []; C[i]["execution_count"] = None
    def rep(i, old, new, count=1):
        s = get(i); n = s.count(old); assert n == count, f"{job['label']} cell {i}: '{old[:70]}' {n}번 (기대 {count})"; put(i, s.replace(old, new))

    # 0. 제목
    rep(0, "# Stage 2 · LoRA 파인튜닝 → 검증 → 제출\n",
        f"# Stage 2 + OCR ({job['label']}) · LoRA 파인튜닝 → 검증 → 제출\n\n"
        "기존 bf16 stage2와 학습·평가·제출 방식은 같고, **프롬프트에 PaddleOCR-VL이 읽은 이미지 속 글자를 추가**한 것만 다릅니다.\n"
        "- 먼저 `ocr_paddle.ipynb`로 `OUT_DIR/ocr_paddle/<OCR_TAG>/ocr_text.csv`를 만들어 두어야 합니다 (train·test 전부).\n"
        "- 프롬프트: `[이미지 속 글자 — OCR 자동 인식 결과 …]` 아래에 글자 줄(위→아래, 줄마다 대략의 위치) → 질문 → 보기 → 답 지시. 학습·검증·test 모두 같은 형식이라 LoRA가 OCR을 믿을지 말지를 함께 배웁니다.\n"
        f"- OCR 글자가 `OCR_MAX_CHARS`를 넘으면 질문·보기와 글자가 많이 겹치는 줄부터 남깁니다(원래 순서 유지).\n"
        "- **11-2번 셀**: OCR만 없는 같은 레시피 실행(`COMPARE_WITH`)과 같은 문항 비교 (고침 / 새로 틀림 / 부호 검정), E14 오답 24개(라벨 오류 10 · 둘 다 답 8 · 진짜 실수 6)에서의 결과.\n"
        "- 결과 파일 이름은 기존과 같아 E14가 자동으로 멤버로 잡습니다. E14의 계열은 MODEL_ID 기준이라 이 실행은 OCR 없는 같은 모델 실행들과 같은 계열로 묶입니다.\n\n---\n")

    # 2. 설정
    rep(2, f"RUN_NAME   = {job['run_old']}", f"RUN_NAME   = {job['run_new']}")
    rep(2, 'MAKE_SUBMISSION = False     # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성\n',
        'MAKE_SUBMISSION = True      # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성 (E14 멤버가 되려면 test 예측이 필요)\n')
    rep(2, 'TEST_TTA     = False       # True면 test 추론 시간이 4배. 검증에서 TTA 이득이 확인된 뒤 켜서 12번 셀만 다시 실행\n',
        'TEST_TTA     = False       # True면 test 추론 시간이 4배. 검증에서 TTA 이득이 확인된 뒤 켜서 12번 셀만 다시 실행\n\n'
        '# ── OCR (ocr_paddle.ipynb 결과를 프롬프트에 넣음) ──\n'
        'OCR_TAG       = "pvl16_spotting"   # ocr_paddle.ipynb의 OCR_TAG와 같게 → OUT_DIR/ocr_paddle/<OCR_TAG>/ocr_text.csv\n'
        'OCR_MAX_CHARS = 1000               # 프롬프트에 넣을 OCR 글자 수 상한. 넘으면 질문·보기와 글자가 많이 겹치는 줄부터 남김\n'
        'OCR_POS_TAGS  = True               # 줄마다 대략의 위치 "(위 왼쪽)" (spotting 결과일 때만 있음)\n'
        f'COMPARE_WITH  = {job["compare"]}   # OCR만 없는 같은 레시피 실행 → 11-2번 셀에서 같은 문항 비교 (있는 것만)\n')

    # 6. 프롬프트
    rep(11, "## 6. 프롬프트 (Stage 1과 동일)", "## 6. 프롬프트 (+ OCR 글자)\n"
        "user 메시지 = 이미지 → `[이미지 속 글자 — OCR 자동 인식 결과 …]` + 글자 줄 → 질문 → 보기 → 답 지시. system 지시문·답 자리는 기존과 같습니다.")
    rep(12, 'PROMPT_VERSION = "baseline_v1"\n', '''PROMPT_VERSION = f"ocr_v2_{OCR_TAG}"      # v2: 생성 루프로 생긴 반복 줄 정리
''')
    rep(12, 'LETTERS = "abcd"\n', '''LETTERS = "abcd"

# ── OCR 결과 불러오기 (ocr_paddle.ipynb가 만든 파일) ──
OCR_FILE = os.path.join(OUT_DIR, "ocr_paddle", OCR_TAG, "ocr_text.csv")
assert os.path.exists(OCR_FILE), (f"OCR 결과가 없습니다: {OCR_FILE}\\n→ ocr_paddle.ipynb를 먼저 끝까지 실행하세요 (여러 기기로 나눠 돌렸다면 7번 '합치기' 셀까지). "
                                  f"폴더 내용: {sorted(os.listdir(os.path.dirname(OCR_FILE))) if os.path.isdir(os.path.dirname(OCR_FILE)) else '폴더 없음'}")
_ocr = pd.read_csv(OCR_FILE, keep_default_na=False)
# OCR이 생성 루프에 빠진 흔적 정리 (예: "1000000…"이 상한까지 이어짐. 이런 이미지는 hit_max로 기록됨)
def _clean_line(x):
    y = re.sub(r"(.)\\1{5,}", r"\\1" * 6, x)                  # 같은 글자가 6번 넘게 이어지면 6번으로
    y = re.sub(r"(.{2,12}?)\\1{3,}", r"\\1\\1\\1", y)             # 짧은 조각이 4번 넘게 반복되면 3번으로
    return y.strip()
def _clean_lines(lines):
    out, seen = [], {}
    for t, x in lines:
        y = _clean_line(x)
        if not y or len(y) < 0.4 * len(x): continue            # 대부분이 반복이던 줄(루프)은 버림
        seen[y] = seen.get(y, 0) + 1
        if seen[y] > 3: continue                                # 같은 글자 줄은 3번까지
        out.append((t, y))
    return out
_raw_lines = {p: [(t or None, str(x)) for t, x in json.loads(j)] for p, j in zip(_ocr.path, _ocr.lines_json)}
OCR_LINES = {p: _clean_lines(v) for p, v in _raw_lines.items()}
_nchg = sum(1 for p in OCR_LINES if OCR_LINES[p] != _raw_lines[p]); _drop = sum(len(_raw_lines[p]) - len(OCR_LINES[p]) for p in OCR_LINES)
print(f"OCR 반복 정리: {_nchg:,}장에서 줄 {_drop:,}개를 버리거나 줄임 | 상한 도달(hit_max) {(_ocr.hit_max.astype(str).str.lower() == 'true').mean() * 100:.1f}%")
_need = {"train": list(train_df.path)}
if MAKE_SUBMISSION: _need["test"] = list(test_df.path)
if EXTRA_TRAIN_CSV: _need["추가 학습 데이터"] = list(pd.read_csv(EXTRA_TRAIN_CSV).path)
for _k, _ps in _need.items():
    _miss = [p for p in _ps if p not in OCR_LINES]
    assert not _miss, f"{_k} 이미지 {len(_miss)}장의 OCR 결과가 없습니다 (예: {_miss[:3]}) → ocr_paddle.ipynb의 SPLITS·샤드 완료 여부를 확인하세요"
_info_fp = os.path.join(os.path.dirname(OCR_FILE), "ocr_info.json")
OCR_INFO = json.load(open(_info_fp)) if os.path.exists(_info_fp) else {}
_nc = pd.to_numeric(_ocr.n_chars, errors="coerce").fillna(0)
print(f"OCR {OCR_TAG}: {len(OCR_LINES):,}장 | {OCR_INFO.get('OCR_MODEL_ID', '?')} · {OCR_INFO.get('OCR_TASK', '?')}")
print(f"  이미지당 글자 수 중앙값 {_nc.median():.0f} · 90% {_nc.quantile(0.9):.0f} | {OCR_MAX_CHARS}자 초과 {(_nc > OCR_MAX_CHARS).mean() * 100:.1f}% (질문·보기와 겹치는 줄 우선으로 줄임) | 글자 없음 {(pd.to_numeric(_ocr.n_lines, errors='coerce').fillna(0) == 0).mean() * 100:.1f}%")

def _norm(s): return re.sub(r"[\\s\\W_]+", "", str(s).lower())
def _bg(s): return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())
OCR_HEADER = "[이미지 속 글자 — OCR 자동 인식 결과. 틀리거나 빠진 글자가 있을 수 있음]"

def ocr_block(row):
    # 선지 순서와 무관 (학습 중 선지를 섞어도 같은 블록)
    lines = OCR_LINES.get(row.get("path") if hasattr(row, "get") else None, [])
    fmt = [((f"({t}) " if (OCR_POS_TAGS and t) else "") + x)[:200] for t, x in lines]
    if sum(len(f) + 1 for f in fmt) > OCR_MAX_CHARS:            # 길면 질문·보기와 겹치는 두 글자 조각이 많은 줄부터 남기고 원래 순서로
        q = _bg(_norm(" ".join([str(row["question"])] + [str(row[k]) for k in LETTERS])))
        score = [len(_bg(_norm(x)) & q) for _, x in lines]
        keep, used = set(), 0
        for i in sorted(range(len(fmt)), key=lambda i: (-score[i], i)):
            if used + len(fmt[i]) + 1 <= OCR_MAX_CHARS: keep.add(i); used += len(fmt[i]) + 1
        fmt = [fmt[i] for i in sorted(keep)] + ["…(일부 줄 생략)"]
    return OCR_HEADER + "\\n" + ("\\n".join(fmt) if fmt else "(인식된 글자 없음)")
''')
    rep(12, '    user_text = build_mc_prompt(str(row["question"]), opts)\n',
        '    user_text = ocr_block(row) + "\\n\\n" + build_mc_prompt(str(row["question"]), opts)      # OCR 블록 → 질문·보기·답 지시\n')

    # 9. 학습 기록에 OCR 설정
    rep(18, '"EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS"]}}', '"EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "PROMPT_VERSION", "OCR_TAG", "OCR_MAX_CHARS", "OCR_POS_TAGS"]}}')

    # 11. 오답 리포트에 OCR 표시
    rep(22, '''f"<div><b>{r.id}</b><p>{_html.escape(str(r.question))}</p><ul>{opts}</ul></div></div>")''',
        '''f"<div><b>{r.id}</b><p>{_html.escape(str(r.question))}</p><ul>{opts}</ul>"
                     f"<pre style='white-space:pre-wrap;font-size:12px;background:#f5f5f5;padding:6px'>{_html.escape(ocr_block(r))}</pre></div></div>")''')

    # 11-2. OCR 효과 비교 (새 셀)
    cmp_md = {"cell_type": "markdown", "metadata": {}, "source": [
        "## 11-2. OCR 효과: OCR만 없는 같은 레시피 실행과 같은 문항 비교\n",
        "- `COMPARE_WITH`의 실행 중 검증 예측이 있는 것마다: 정확도, 고침 / 새로 틀림 / 순이득, 부호 검정 p값, 바뀐 문항 목록.\n",
        "- E14 조합의 검증 오답 24개(라벨 오류 10 · 둘 다 답 8 · 진짜 모델 실수 6)에서 이 실행이 맞힌 수.\n",
        "- 앙상블 판단은 E14에서 (이 실행의 test 예측까지 끝난 뒤 다시 실행하면 자동 포함)."]}
    cmp_code = '''import glob
E14_ERR = {"라벨 오류": "0175 1878 3551 4755 6545 2114 4801 3823 6130 6211", "둘 다 답": "5008 1900 1911 2910 5811 2462 2186 1785",
           "진짜 실수": "2112 0936 5797 5209 1045 1000"}
E14_ERR = {k: [f"train_{x}.jpg" for x in v.split()] for k, v in E14_ERR.items()}
CAT_OF = {i: k for k, ids in E14_ERR.items() for i in ids}
OCR_CMP = []
if DRY_RUN or len(final_eval_df) != len(val_df):
    print("DRY_RUN → 비교 생략")
else:
    _pre = "up2.0" if "up2.0" in EVAL_PRESETS else BEST_PRESET
    _mine = pd.read_csv(os.path.join(RUN_DIR, f"preds_{FINAL_TAG}__{_pre}__{len(val_df)}.csv")).drop_duplicates("id", keep="last")
    CMP = val_df[["id", "question", "answer"]].merge(_mine[["id", "pred"]], on="id"); CMP["ok"] = CMP.pred == CMP.answer
    _lab = set(E14_ERR["라벨 오류"])
    print(f"OCR 실행 {RUN_NAME} @ {_pre}: {CMP.ok.mean() * 100:.2f}% ({int(CMP.ok.sum())}/{len(CMP)}) | 라벨 오류 10개 제외 {CMP[~CMP.id.isin(_lab)].ok.mean() * 100:.2f}%")
    for base in COMPARE_WITH:
        fs = sorted(glob.glob(os.path.join(OUT_DIR, base, f"preds_final_ep*__{_pre}__{len(val_df)}.csv")))
        if not fs: print(f"\\n  {base}: {_pre} 검증 예측 없음 → 건너뜀"); continue
        b = pd.read_csv(fs[-1]).drop_duplicates("id", keep="last")[["id", "pred"]].rename(columns={"pred": "pred_base"})
        M = CMP.merge(b, on="id"); M["ok_base"] = M.pred_base == M.answer
        fix, brk = M[M.ok & ~M.ok_base], M[~M.ok & M.ok_base]
        n, k = len(fix) + len(brk), len(fix)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0     # 양쪽 부호 검정
        OCR_CMP.append({"base": base, "n": len(M), "base_acc": M.ok_base.mean(), "ocr_acc": M.ok.mean(), "fix": len(fix), "break": len(brk), "p": p})
        print(f"\\n▶ {base} ({os.path.basename(fs[-1])}, {len(M)}문항): {M.ok_base.mean() * 100:.2f}% → OCR {M.ok.mean() * 100:.2f}% | "
              f"고침 {len(fix)} · 새로 틀림 {len(brk)} · 순 {len(fix) - len(brk):+d} (부호 검정 p={p:.3f}) | 답이 다른 문항 {int((M.pred != M.pred_base).sum())}")
        for name, sub in [("고침", fix), ("새로 틀림", brk)]:
            for r in sub.head(40).to_dict("records"):
                print(f"    {name}: {r['id']} [{CAT_OF.get(r['id'], '-')}] 정답 {r['answer']} | OCR {r['pred']} · 기준 {r['pred_base']} | {str(r['question'])[:60]}")
            if len(sub) > 40: print(f"    … {name} {len(sub) - 40}개 더")
    print("\\nE14 조합의 검증 오답 24개에서 이 실행의 결과 (○ 맞힘 · × 틀림):")
    for cat, ids in E14_ERR.items():
        sub = CMP[CMP.id.isin(ids)]
        print(f"  {cat}: {int(sub.ok.sum())}/{len(sub)} | " + " ".join(f"{i.replace('train_', '').replace('.jpg', '')}{'○' if o else '×'}" for i, o in zip(sub.id, sub.ok)))
'''
    C.insert(23, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": []})
    C.insert(23, cmp_md)
    put(24, cmp_code)

    # 13. 요약 (셀 번호가 2칸 밀림: 26 → 28)
    rep(28, '''print("RUN:", RUN_NAME, "| epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})
''', '''print("RUN:", RUN_NAME, "| epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})
print(f"OCR: {OCR_TAG} ({OCR_INFO.get('OCR_MODEL_ID', '?')} · {OCR_INFO.get('OCR_TASK', '?')}) | 글자 상한 {OCR_MAX_CHARS} | 위치 태그 {OCR_POS_TAGS} | 프롬프트 {PROMPT_VERSION}")
for _c in OCR_CMP: print(f"OCR 효과 vs {_c['base']}: {_c['base_acc'] * 100:.2f}% → {_c['ocr_acc'] * 100:.2f}% | 고침 {_c['fix']} · 새로 틀림 {_c['break']} (p={_c['p']:.3f})")
''')
    nb["metadata"].setdefault("colab", {})
    json.dump(nb, open(job["dst"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", job["dst"], "| 셀", len(C))

for j in JOBS: build(j)
