# -*- coding: utf-8 -*-
"""bf16 복제 학습 노트북 빌더: stage2 bf16 노트북(Qwen·Gemma 공용 코드) + '레시피 복사' 셀.
기존 4bit 멤버의 train_log.json 설정을 그대로 가져와 bf16으로 같은 분할(5,714/1,000)에서 다시 학습 → val·test 예측까지.
옵션: NEW_SEED(시드만 바꾼 두 번째 실행), EXTRA_FROM(다른 실행의 dev 의사 라벨을 빌려 추가 학습)."""
import sys, uuid
import nbformat
SRC = "/home/user/vqa/bf16/out/stage2_qwen38_27b_bf16.ipynb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/clone/out/stage2_bf16_clone.ipynb"
nb = nbformat.read(SRC, as_version=4)
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80]); return text.replace(old, new)

INTRO = r'''# Stage 2 · bf16 복제 학습 (기존 4bit 멤버와 같은 레시피 → bf16, 같은 5,714/1,000 분할)

**하는 일**: 1번 셀 `RECIPE_FROM`에 기존 멤버 폴더 이름을 넣으면, 3-1번 셀이 그 멤버의 `train_log.json`에 기록된 학습 설정(모델·학습 해상도·epoch·LR·LoRA r/α·유효 배치·선지 셔플·시드·dev 추가 데이터·이미지 증강·다해상도)을 그대로 가져와 **bf16으로** 다시 학습하고, 검증(up2.0) → test 추론(up2.0)까지 이어서 합니다. 기존 멤버 폴더는 읽기만 합니다.

| 하고 싶은 것 | 1번 셀 | 새 폴더 이름 (E13이 자동으로 찾음) |
|---|---|---|
| GDEV를 bf16으로 | `RECIPE_FROM="gemma4_31b_r16_up20_dev"` | `gemma4_31b_r16_up20_dev_bf16` |
| Q38 + dev 의사 라벨 (GDEV와 같은 dev 데이터) | `RECIPE_FROM="q38_27b_r16_native"`, `EXTRA_FROM="gemma4_31b_r16_up20_dev"` | `q38_27b_r16_dev_bf16` |
| Q38 bf16 두 번째 시드 | `RECIPE_FROM="q38_27b_r16_native"`, `NEW_SEED=43` | `q38_27b_r16_native_bf16_s43` |

**코드**: 지금 돌고 있는 `stage2_qwen38_27b_bf16.ipynb`와 셀 코드가 같습니다(1번 셀 일부와 3-1번 셀만 다름). 모델은 RECIPE_FROM의 설정에서 가져오므로 **Qwen·Gemma 멤버 모두 이 노트북 하나**로 됩니다. HCX는 설치·모델 로드가 달라 `stage2_hcx_32b_bf16.ipynb`를 쓰세요.

**실행**: 1번 셀의 세 줄만 바꾸고 위에서부터 끝까지. 끊기면 1번 셀부터 다시 실행하면 체크포인트에서 이어서 학습합니다.
**GPU**: bf16 전용 — G4 96GB 또는 A100/H100 80GB.'''

CFG = nb.cells[2].source
CFG = rep(CFG, 'RUN_NAME   = "q38_27b_r16_native_bf16"        # 실험 이름 = 저장 폴더 이름. 설정을 바꾸면 이름도 바꾸세요\nMODEL_ID   = "Qwen/Qwen3.8-27B"',
          'RECIPE_FROM = "gemma4_31b_r16_up20_dev"   # ★ 따라 할 기존 멤버 폴더 이름. 예) "q38_27b_r16_native", "gemma4_31b_r16_up20", "q35_27b_r16_native"\n'
          'NEW_SEED    = None        # ★ 숫자(예: 43)면 학습 시드만 바꾼 두 번째 실행 (시드 앙상블). None이면 원래 시드 그대로\n'
          'EXTRA_FROM  = None        # ★ 다른 실행의 추가 학습 데이터(dev 의사 라벨)를 빌려 씀. 예) "gemma4_31b_r16_up20_dev". None이면 RECIPE_FROM 기록 그대로\n'
          'RUN_NAME   = "(3-1번 셀이 정함)"      # 결과 폴더 이름: 3-1번 셀이 위 세 값으로 정합니다 (E13이 찾는 규칙)\n'
          'MODEL_ID   = "Qwen/Qwen3.8-27B"      # 자리표시: 3-1번 셀이 RECIPE_FROM의 값으로 덮어씀')
CFG = rep(CFG, "# ── 학습 ──", "# ── 학습 ── (아래 값들은 3-1번 셀이 RECIPE_FROM의 train_log.json 값으로 덮어씁니다)")

RECIPE = r'''# ── 레시피 복사: RECIPE_FROM 멤버의 학습 설정을 그대로 가져와 bf16으로 같은 분할에서 다시 학습 ──
import os, json
import pandas as pd
_lp = os.path.join(OUT_DIR, RECIPE_FROM, "train_log.json")
assert os.path.exists(_lp), f"{_lp} 가 없습니다 → RECIPE_FROM(기존 멤버 폴더 이름)을 확인하세요"
_log = json.load(open(_lp, encoding="utf-8"))
_cfg = dict(_log.get("config") or {})
assert _cfg, f"{_lp}에 학습 설정(config)이 없습니다 → 위 출력을 확인하세요"
assert not _cfg.get("TRAIN_ON_FULL"), f"{RECIPE_FROM}은 전체 데이터(검증 포함)로 학습한 실행입니다 → 검증셋을 뺀 원래 멤버 이름을 넣으세요"
_DEF = {"TRAIN_SEED": 42, "MAX_STEPS": None, "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None}   # 예전 기록에 없는 항목의 기본값 (stage2 이어하기 규칙과 같음)
_COPY = ["MODEL_ID", "TRAIN_PRESET", "EPOCHS", "MAX_STEPS", "LR", "LORA_R", "LORA_ALPHA", "SHUFFLE_OPTIONS", "SEED", "TRAIN_SEED",
         "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS"]
_missing = [k for k in _COPY if k not in _cfg and k not in _DEF]
assert not _missing, f"train_log.json의 config에 {_missing}이 없습니다 → 위 출력을 확인하세요"
for _k in _COPY: globals()[_k] = _cfg.get(_k, _DEF.get(_k))
assert SEED == 42, f"검증 분할 시드가 팀 공통값(42)이 아닙니다: {SEED}"
if "hyperclova" in str(MODEL_ID).lower() or "hcx" in RECIPE_FROM.lower():
    raise RuntimeError("HCX는 이 노트북이 아니라 stage2_hcx_32b_bf16.ipynb를 쓰세요 (설치·모델 로드가 다름)")
_changed = []
if EXTRA_FROM:
    _ep = os.path.join(OUT_DIR, EXTRA_FROM, "train_log.json"); assert os.path.exists(_ep), f"{_ep} 가 없습니다 → EXTRA_FROM을 확인하세요"
    _x = (json.load(open(_ep, encoding="utf-8")).get("config") or {}).get("EXTRA_TRAIN_CSV")
    assert _x, f"{EXTRA_FROM}은 추가 학습 데이터(EXTRA_TRAIN_CSV) 없이 학습한 실행입니다 → dev 의사 라벨로 학습한 실행 이름을 넣으세요"
    assert not EXTRA_TRAIN_CSV or EXTRA_TRAIN_CSV == _x, f"{RECIPE_FROM}은 이미 다른 추가 데이터로 학습했습니다: {EXTRA_TRAIN_CSV}"
    if EXTRA_TRAIN_CSV != _x: _changed.append(f"EXTRA_TRAIN_CSV ← {EXTRA_FROM}의 값")
    EXTRA_TRAIN_CSV = _x
if EXTRA_TRAIN_CSV:
    assert os.path.exists(EXTRA_TRAIN_CSV), f"추가 학습 데이터(EXTRA_TRAIN_CSV)가 없습니다: {EXTRA_TRAIN_CSV}"
    if MAX_STEPS and any("EXTRA_TRAIN_CSV" in c for c in _changed):      # 원래 실행이 스텝 수로 끊었다면 늘어난 데이터 비율만큼 늘림
        _n = len(pd.read_csv(EXTRA_TRAIN_CSV)); _old = int(MAX_STEPS); MAX_STEPS = int(round(_old * (5714 + _n) / 5714))
        _changed.append(f"MAX_STEPS {_old} → {MAX_STEPS} (추가 데이터 {_n:,}문항 비율)")
if NEW_SEED is not None:
    assert int(NEW_SEED) != int(TRAIN_SEED), f"NEW_SEED={NEW_SEED}가 원래 시드와 같습니다 → 다른 숫자를 넣으세요"
    _changed.append(f"TRAIN_SEED {TRAIN_SEED} → {int(NEW_SEED)}"); TRAIN_SEED = int(NEW_SEED)
_ss = int(_cfg.get("STEP_SAMPLES", MICRO_BS * GRAD_ACCUM))
MICRO_BS = 2 if (str(TRAIN_PRESET) == "native" and not TRAIN_PRESETS) else 1     # 원래 노트북 기본값: Qwen native 2×8, Gemma up2.0 1×16 (유효 배치는 기록값 그대로)
if MICRO_BS * GRAD_ACCUM != _ss:
    assert _ss % MICRO_BS == 0, f"유효 배치 {_ss}를 MICRO_BS={MICRO_BS}로 나눌 수 없습니다"
    GRAD_ACCUM = _ss // MICRO_BS
TRAIN_ON_FULL, USE_4BIT = False, False
_stem = (RECIPE_FROM.replace("_native", "_dev") if "_native" in RECIPE_FROM else RECIPE_FROM + "_dev") if (EXTRA_FROM and any("EXTRA_TRAIN_CSV" in c for c in _changed)) else RECIPE_FROM
RUN_NAME = _stem + "_bf16" + (f"_s{TRAIN_SEED}" if NEW_SEED is not None else "")
if "RUN_DIR" in globals() and os.path.basename(str(RUN_DIR)).replace("_dry", "") != RUN_NAME:
    raise RuntimeError(f"결과 폴더(RUN_DIR={RUN_DIR})가 이미 정해졌습니다 → 이 셀을 모델 로드 셀보다 앞에 두고 1번 셀부터 다시 실행하세요 (다른 실행 폴더를 덮어쓰지 않도록 멈춤)")
RES_PRESET = TRAIN_PRESET
MAKE_SUBMISSION, TEST_PRESET, TEST_TTA = True, "up2.0", False
EVAL_PRESETS, EVAL_TTA = ["up2.0"], False
print(f"따라 할 멤버: {RECIPE_FROM} → 새 실행 폴더: {RUN_NAME}{'_dry' if DRY_RUN else ''}  (bf16, 검증 1,000문항 제외하고 학습)")
for _k in _COPY: print(f"  {_k:<16} = {globals()[_k]!r}" + ("" if _k in _cfg else "  (기록 없음 → 기본값)"))
print(f"  유효 배치         = {MICRO_BS} × {GRAD_ACCUM} = {MICRO_BS * GRAD_ACCUM} (기록 {_ss}) | 끝나면 검증 up2.0 → test 추론 {TEST_PRESET}")
print("  원래 멤버와 바뀐 것: bf16" + "".join(f" · {c}" for c in _changed))
_prev = os.path.join(OUT_DIR, RUN_NAME)
if os.path.isdir(_prev) and any(f.startswith("preds_final_") and "_test__" in f for f in os.listdir(_prev)):
    print(f"ℹ️ {RUN_NAME}에는 이미 test 예측이 있습니다 (끝난 실행). 다시 돌릴 필요가 없습니다.")'''

cells = list(nb.cells)
cells[0] = nbformat.v4.new_markdown_cell(INTRO)
cells[2] = nbformat.v4.new_code_cell(CFG)
assert "drive.mount" in cells[6].source and cells[6].cell_type == "code"
new = [nbformat.v4.new_markdown_cell("## 3-1. 레시피 복사 (RECIPE_FROM과 같은 설정 → bf16)\n기존 멤버의 `train_log.json` 설정을 가져와 덮어쓰고, 무엇을 가져왔는지 보여 줍니다. **값이 원래 멤버와 같은지, '바뀐 것' 줄이 의도대로인지** 한 번 보고 넘어가세요."),
       nbformat.v4.new_code_cell(RECIPE)]
cells = cells[:7] + new + cells[7:]
for c in cells:
    if c.cell_type == "code": c.outputs, c.execution_count = [], None
    c["id"] = uuid.uuid4().hex[:8]
out = nbformat.v4.new_notebook(); out.metadata = nb.metadata; out.cells = cells
for c in out.cells:
    if c.cell_type == "code": compile("\n".join((l[: len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l) for l in c.source.split("\n")), "<cell>", "exec")
nbformat.validate(out); nbformat.write(out, OUT)
print("저장:", OUT, "| 셀", len(cells))
