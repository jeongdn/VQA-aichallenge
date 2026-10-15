# -*- coding: utf-8 -*-
"""E8 학습 노트북 빌더: 팀의 stage2 bf16 노트북 + '레시피 복사' 셀 (기존 멤버 설정 그대로 → 전체 데이터 학습)."""
import json, sys, copy
import nbformat

SRC = "/mnt/user-data/uploads/stage2_qwen35_27b_bf16.ipynb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/e8/out/e8_full_retrain.ipynb"
nb = nbformat.read(SRC, as_version=4)
def src(i): return nb.cells[i].source
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80]); return text.replace(old, new)

INTRO = r'''# E8 · 전체 데이터 재학습 (기존 멤버와 같은 레시피 + 검증 1,000문항 → 6,714문항)

**하는 일**
- 1번 셀 `RECIPE_FROM`에 기존 멤버 폴더 이름을 넣으면, 3-1번 셀이 그 멤버의 `train_log.json`에 기록된 학습 설정(모델·학습 해상도·epoch·LR·LoRA r/α·유효 배치·선지 셔플·시드·dev 추가 데이터·이미지 증강·다해상도)을 그대로 가져와 **검증셋까지 합친 전체 데이터**로 다시 학습합니다.
- 결과 폴더는 `vqa_runs/<RECIPE_FROM>_full`입니다. 학습이 끝나면 test 추론(up2.0)까지 이어서 합니다. 기존 멤버 폴더는 읽기만 합니다.
- 검증 점수는 학습에 쓴 문항이라 **의미가 없습니다**(학습이 제대로 됐는지 보는 용도). 채택 판단은 `e8_ensemble.ipynb`에서 원래 멤버와의 test 답 일치율과 공개 점수로 합니다.

**코드**: 올려 주신 `stage2_qwen35_27b_bf16.ipynb`와 셀 코드가 같습니다(1번 셀 일부와 3-1번 셀만 다름). bf16으로 학습하고, 모델은 RECIPE_FROM의 설정에서 가져오므로 **Qwen·Gemma 멤버 모두 이 노트북 하나**로 됩니다.
**HCX**: 설치·모델 로드가 달라서 이 노트북으로는 안 됩니다. HCX 학습 노트북의 Drive 연결 셀 바로 다음(모델 로드 전)에 3-1번 셀을 그대로 복사해 넣으세요. RECIPE_FROM이 없으면 `hcx32b_r16_native`를 씁니다.

**실행**: 1번 셀 `RECIPE_FROM`만 바꾸고 위에서부터 끝까지 실행하세요. 끊기면 1번 셀부터 다시 실행하면 체크포인트에서 이어서 학습합니다.
**GPU**: 원래 그 멤버를 학습했던 GPU와 같은 종류(bf16 전용: G4 96GB 또는 A100 80GB).'''

CFG = src(2)
CFG = rep(CFG, 'RUN_NAME   = "q35_27b_r16_native_bf16"        # 실험 이름 = 저장 폴더 이름. 설정을 바꾸면 이름도 바꾸세요\nMODEL_ID   = "Qwen/Qwen3.5-27B"',
          'RECIPE_FROM = "q38_27b_r16_native"     # ★ 따라 할 기존 멤버 폴더 이름. 예) "gemma4_31b_r16_up20", "gemma4_31b_r16_up20_dev", "q35_27b_r16_native"\n'
          'RUN_NAME   = RECIPE_FROM + "_full"       # 결과 폴더 = 원래 이름 + _full (e8_ensemble.ipynb가 이 이름으로 찾습니다)\n'
          'MODEL_ID   = "Qwen/Qwen3.5-27B"          # 자리표시: 3-1번 셀이 RECIPE_FROM의 값으로 덮어씀\n'
          'E8_STAGE2_NB = True                      # (이 노트북 표시 — 지우지 마세요)')
CFG = rep(CFG, "# ── 학습 ──", "# ── 학습 ── (아래 값들은 3-1번 셀이 RECIPE_FROM의 train_log.json 값으로 덮어씁니다)")
CFG = rep(CFG, "TRAIN_ON_FULL = False      # True: 검증셋까지 합쳐 전체 6,714개로 학습 (레시피가 확정된 뒤 최종 제출용)",
          "TRAIN_ON_FULL = True       # E8: 검증셋까지 합쳐 전체 6,714개로 학습")
CFG = rep(CFG, "MAKE_SUBMISSION = False     # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성",
          "MAKE_SUBMISSION = True      # 학습·평가가 끝나면 이어서 test 추론 → 제출 파일까지 생성")

RECIPE = r'''# ── E8: RECIPE_FROM 멤버의 학습 설정을 그대로 가져와 '전체 데이터(검증 포함)'로 학습 ──
# HCX 노트북에 붙여 넣을 때도 이 셀 그대로: Drive 연결 셀 다음, 모델 로드 셀 전에 실행
import os, json
RECIPE_FROM = globals().get("RECIPE_FROM") or "hcx32b_r16_native"
_lp = os.path.join(OUT_DIR, RECIPE_FROM, "train_log.json")
assert os.path.exists(_lp), f"{_lp} 가 없습니다 → RECIPE_FROM(기존 멤버 폴더 이름)을 확인하세요"
_log = json.load(open(_lp, encoding="utf-8"))
_cfg = dict(_log.get("config") or {})
assert _cfg, f"{_lp}에 학습 설정(config)이 없습니다 → 위 출력을 확인하세요"
assert not _cfg.get("TRAIN_ON_FULL"), f"{RECIPE_FROM}은 이미 전체 데이터로 학습한 실행입니다 → 검증셋을 뺀 원래 멤버 이름을 넣으세요"
_DEF = {"TRAIN_SEED": 42, "MAX_STEPS": None, "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None}   # 예전 기록에 없는 항목의 기본값 (stage2 이어하기 규칙과 같음)
_COPY = ["MODEL_ID", "TRAIN_PRESET", "EPOCHS", "MAX_STEPS", "LR", "LORA_R", "LORA_ALPHA", "SHUFFLE_OPTIONS", "SEED", "TRAIN_SEED",
         "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS"]
_missing = [k for k in _COPY if k not in _cfg and k not in _DEF]
assert not _missing, f"train_log.json의 config에 {_missing}이 없습니다 → 위 출력을 확인하세요"
for _k in _COPY: globals()[_k] = _cfg.get(_k, _DEF.get(_k))
assert SEED == 42, f"검증 분할 시드가 팀 공통값(42)이 아닙니다: {SEED}"
if globals().get("E8_STAGE2_NB") and ("hyperclova" in str(MODEL_ID).lower() or "hcx" in RECIPE_FROM.lower()):
    raise RuntimeError("HCX는 이 노트북이 아니라 HCX 학습 노트북에 이 셀을 붙여 넣어 쓰세요 (설치·모델 로드가 다름)")
_ea = {int(k): v for k, v in (_log.get("epoch_acc") or {}).items()}
_best = max(_ea, key=lambda e: (_ea[e], e)) if _ea else None
if _best is not None and _best != int(EPOCHS):
    raise RuntimeError(f"{RECIPE_FROM}은 EPOCHS={EPOCHS}로 학습하고 검증으로 epoch {_best}를 골랐습니다. 전체 데이터에서는 검증으로 epoch를 "
                       "고를 수 없어 멈춥니다 → 위 출력을 확인하세요")
if MAX_STEPS:
    _old = int(MAX_STEPS); MAX_STEPS = int(round(_old * 6714 / 5714))
    print(f"⚠️ 원래 실행은 MAX_STEPS={_old}에서 끊었습니다 → 데이터가 늘어난 비율(6,714/5,714)만큼 {MAX_STEPS}로 늘립니다")
if EXTRA_TRAIN_CSV:
    assert os.path.exists(EXTRA_TRAIN_CSV), f"추가 학습 데이터(EXTRA_TRAIN_CSV)가 없습니다: {EXTRA_TRAIN_CSV}"
_ss = int(_cfg.get("STEP_SAMPLES", MICRO_BS * GRAD_ACCUM))
if globals().get("E8_STAGE2_NB"):                  # 원래 노트북 기본값: Qwen native 2×8, Gemma up2.0 1×16 (유효 배치는 기록값 그대로)
    MICRO_BS = 2 if (str(TRAIN_PRESET) == "native" and not TRAIN_PRESETS) else 1
if MICRO_BS * GRAD_ACCUM != _ss:
    assert _ss % MICRO_BS == 0, f"유효 배치 {_ss}를 MICRO_BS={MICRO_BS}로 나눌 수 없습니다"
    GRAD_ACCUM = _ss // MICRO_BS
TRAIN_ON_FULL = True
RUN_NAME = RECIPE_FROM + "_full"
if "RUN_DIR" in globals() and os.path.basename(str(RUN_DIR)).replace("_dry", "") != RUN_NAME:
    raise RuntimeError(f"결과 폴더(RUN_DIR={RUN_DIR})가 이미 정해졌습니다 → 이 셀을 모델 로드 셀보다 앞에 두고 1번 셀부터 다시 실행하세요 (원래 멤버 폴더를 덮어쓰지 않도록 멈춤)")
RES_PRESET = TRAIN_PRESET
MAKE_SUBMISSION, TEST_PRESET, TEST_TTA = True, "up2.0", False
EVAL_PRESETS, EVAL_TTA = ["up2.0"], False        # 검증 점수는 의미 없음(학습에 포함) — 학습 확인·파일 위치용
print(f"따라 할 멤버: {RECIPE_FROM} (검증으로 고른 epoch {_best}) → 새 실행 폴더: {RUN_NAME}{'_dry' if DRY_RUN else ''}")
for _k in _COPY: print(f"  {_k:<16} = {globals()[_k]!r}" + ("" if _k in _cfg else "  (기록 없음 → 기본값)"))
print(f"  유효 배치         = {MICRO_BS} × {GRAD_ACCUM} = {MICRO_BS * GRAD_ACCUM} (기록 {_ss}) | TRAIN_ON_FULL=True | 끝나면 test 추론 {TEST_PRESET}")
print("  정밀도: bf16 (이 노트북 고정)" if globals().get("E8_STAGE2_NB") else "  ⚠️ 정밀도(USE_4BIT)는 기록에 없습니다 → 설정 셀의 USE_4BIT이 원래 실행과 같은지 확인하세요")
_prev = os.path.join(OUT_DIR, RUN_NAME)
if os.path.isdir(_prev) and any(f.startswith("preds_final_") and "_test__" in f for f in os.listdir(_prev)):
    print(f"ℹ️ {RUN_NAME}에는 이미 test 예측이 있습니다 (끝난 실행). 다시 돌릴 필요가 없습니다.")'''

cells = list(nb.cells)
cells[0] = nbformat.v4.new_markdown_cell(INTRO)
cells[2] = nbformat.v4.new_code_cell(CFG)
assert "drive.mount" in cells[6].source and cells[6].cell_type == "code"
new = [nbformat.v4.new_markdown_cell("## 3-1. E8 레시피 복사 (RECIPE_FROM과 같은 설정 → 전체 데이터)\n기존 멤버의 `train_log.json` 설정을 가져와 덮어쓰고, 무엇을 가져왔는지 표로 보여 줍니다. **표의 값이 원래 멤버와 같은지** 한 번 보고 넘어가세요."),
       nbformat.v4.new_code_cell(RECIPE)]
cells = cells[:7] + new + cells[7:]
import uuid
for c in cells:
    if c.cell_type == "code": c.outputs, c.execution_count = [], None
    c["id"] = uuid.uuid4().hex[:8]
out = nbformat.v4.new_notebook(); out.metadata = nb.metadata; out.cells = cells
nbformat.validate(out); nbformat.write(out, OUT)
print("저장:", OUT, "| 셀", len(cells))
