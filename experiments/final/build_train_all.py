# -*- coding: utf-8 -*-
"""통합 학습 노트북(01_train_vlm.ipynb) 빌더.
기반: HCX 학습 노트북(transformers 계열 판별·HCX 처리·InternVL·보조 학습 포함)의 셀을 그대로 쓰고,
      비전 LoRA · 이어학습(INIT_ADAPTER) · OCR 프롬프트 · 레시피 표 · dev 의사 라벨 셀을 더한다.
사용: python3 build_train_all.py [출력 경로]"""
import json, sys, os, re, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipes_table import RECIPES_SRC

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/final/pkg/01_train_vlm.ipynb"
S = "/home/user/vqa/final/src"
def hc(i): return open(f"{S}/hcx_{i:02d}_code.txt", encoding="utf-8").read()
def rep(s, old, new, cnt=1):
    n = s.count(old); assert n == cnt, (n, old[:100]); return s.replace(old, new)

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})

# ───────────────────────────── 0. 소개 ─────────────────────────────
md(r"""
# VQA 학습 노트북 (통합판) · LoRA 파인튜닝 → 검증 → test 추론

이 노트북 하나로 제출에 쓴 **학습 모델을 모두** 다시 만들 수 있습니다. **1-1번 셀의 `RECIPE`에 실행 이름을 넣고** 위에서부터 실행하면, 그 실행과 같은 설정으로 학습 → 검증 → test 추론까지 하고 `OUT_DIR/<실행 이름>/`에 저장합니다.
(Flash-Next 학습은 `02_train_flashnext_runpod.ipynb`, 학습 없는 zero-shot 모델은 `03_*` 노트북입니다.)

**지원 모델 계열** — 프로세서 종류로 자동 판별합니다. 모델별 처리는 `MODEL_NOTES.md`에 정리했습니다.

| 계열 | 모델 예 | 해상도 조절 | 모델별 처리 |
|---|---|---|---|
| Qwen3.5 / 3.6 / 3.8 · Qwen3-VL | `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-35B-A3B` | 이미지 픽셀 범위 (`min/max_pixels`) | 선형 어텐션 커널(`flash-linear-attention`) 설치, 생각 모드 끔 |
| Gemma 4 | `google/gemma-4-31B-it` | 이미지당 소프트 토큰 수 | HF 라이선스 동의 + 토큰 |
| InternVL3.5 | `OpenGVLab/InternVL3_5-38B-HF` | 448px 타일 개수 | — |
| HyperCLOVA X SEED Think | `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | 이미지 픽셀 범위 | transformers 개발 버전, 생각 모드 끔 + 빈 생각 블록 닫기 (7-1번 셀) |

**핵심 설계**
- **정답 글자만 학습**: 추론 때와 똑같은 프롬프트를 넣고, 마지막 위치(답이 나올 자리)의 로짓에만 교차 엔트로피를 겁니다. 학습과 추론의 입력이 완전히 같습니다.
- **채점**: 답 자리에서 a·b·c·d(대소문자·앞 공백 변형 포함) 토큰 확률을 모아 네 보기 확률로 정규화합니다. 네 글자 확률 합(`letter_mass`)이 1에 가까워야 정상입니다.
- **선지 셔플 증강**: epoch마다 보기 순서를 무작위로 섞고 정답 글자를 그에 맞춰 바꿉니다(자리 편향 제거).
- **고정 검증셋**: train에서 정답 글자별 250문항씩 뽑은 1,000문항(시드 42, md5 `df05e4bc…`). 모든 실행이 같은 문항으로 비교됩니다.
- **끊겨도 이어서**: `SAVE_EVERY` 스텝마다 어댑터·옵티마이저·진행 상태를 저장합니다. 같은 셀들을 다시 실행하면 이어서 학습합니다.

**켜고 끄는 옵션** (1번 셀, 레시피가 덮어씀)

| 옵션 | 뜻 |
|---|---|
| `TRAIN_PRESET` | 학습 해상도 (`native` ≈ 원본, `up2.0` ≈ 2배) — 모델 계열별 값은 `RES_PRESETS*` |
| `USE_4BIT` | `"auto"`: 20B 이상 모델은 4bit NF4(QLoRA, 비전 타워·투영층·lm_head는 양자화 안 함) / `False`: bf16 |
| `VISION_LORA` | 비전 블록·투영층의 선형층에도 LoRA (학습률 = `LR x VIS_LR_SCALE`) |
| `USE_OCR` | 이미지 속 글자(OCR 결과)를 프롬프트 앞에 넣음 (`00b_ocr_paddle.ipynb` 결과 필요) |
| `AUX_GEN_CSV` | 보기 없는 단답형 보조 학습 (`00_data_prep.ipynb`가 만든 파일) |
| `INIT_ADAPTER` · `PLAN_EPOCH_OFFSET` | 기존 어댑터에서 이어학습 · 문항 순서 선택 |
| `EXTRA_TRAIN_CSV` | 추가 학습 데이터 (dev 의사 라벨, 12-2번 셀이 만듦) |
| `TRAIN_ON_FULL` (1-1번 `FULL_DATA`) | 검증 1,000문항까지 합쳐 6,714문항 학습 (제출용. 검증 점수는 의미 없음) |
| `TRAIN_SEED` · `IMG_AUG` · `TRAIN_PRESETS` | 학습 시드 · 밝기/대비 증강 · 다해상도 학습 |

**실행 순서**: ① `DRY_RUN = True`(기본)로 위에서부터 끝까지 실행 → 흐름·메모리·예상 시간 확인 (결과는 `<실행 이름>_dry` 폴더) → ② `DRY_RUN = False`로 바꿔 1번 셀부터 다시 실행 (본 학습).
HyperCLOVA X는 transformers 개발 버전을 설치하므로 **새 세션**에서 실행하세요.

**결과 파일** (`OUT_DIR/<실행 이름>/`, 조합 노트북이 이 이름으로 찾습니다)
- `adapter_ep<N>/` (LoRA 어댑터) · `train_log.json` (학습 설정·epoch별 검증 정확도)
- `preds_final_ep<N>__<해상도>__1000.csv` (검증 예측) · `preds_final_ep<N>_test__<해상도>__6714.csv` (test 예측)
- 열: `id, pred, p_a, p_b, p_c, p_d, letter_mass, img_tokens` · 단독 제출 파일 `submission_<실행>_final_ep<N>_<해상도>.csv`
""")

# ───────────────────────────── 1. 기본 설정 ─────────────────────────────
md("## 1. 기본 설정\n1-1번 셀에서 `RECIPE`를 고르면 그 실행의 값으로 덮어씁니다. `RECIPE = None`이면 여기 값 그대로 새 실험을 합니다.")
code(r'''
RUN_NAME   = "my_run"          # 실험 이름 = 저장 폴더 이름 (RECIPE를 고르면 레시피 이름으로 바뀜)
MODEL_ID   = "Qwen/Qwen3.5-27B"
DRY_RUN    = True              # True: 3스텝만 학습해 전체 흐름 점검 (본 학습 전 필수) / False: 본 학습

# ── 학습 ──
TRAIN_PRESET = "native"        # 학습 해상도 (아래 RES_PRESETS의 이름)
EPOCHS     = 1
MAX_STEPS  = None              # 숫자를 넣으면 그 스텝에서 학습을 끝냄. None이면 EPOCHS만큼
LR         = 1e-4
MICRO_BS   = 2                 # GPU에 한 번에 올리는 문항 수. 학습 중 OOM이면 자동으로 절반 + GRAD_ACCUM 2배 (유효 배치 유지)
GRAD_ACCUM = 8                 # 유효 배치 = MICRO_BS x GRAD_ACCUM = 16
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
WARMUP_RATIO = 0.05            # cosine 스케줄의 warmup 비율
SHUFFLE_OPTIONS = True         # 선지 셔플 증강
TRAIN_ON_FULL = False          # True: 검증셋까지 합쳐 6,714문항으로 학습 (1-1번 셀의 FULL_DATA가 켬)
VISION_LORA  = False           # True: 비전 블록·merger(투영층)의 선형층에도 LoRA (언어 LoRA는 그대로)
VIS_LR_SCALE = 0.2             # 비전 쪽 LoRA 학습률 = LR x 이 값
INIT_ADAPTER = None            # 이 어댑터 폴더에서 이어학습 (옵티마이저·스케줄러는 새로). 예: f"{OUT_DIR}/q35_27b_r16_native/adapter_ep1"
PLAN_EPOCH_OFFSET = 0          # 1이면 원 학습의 '2번째 epoch' 문항·보기 순서로 학습 (이어학습용)
EPOCH_END_EVAL = True          # False: epoch 끝의 학습 해상도 검증 평가 생략 (어댑터 저장은 그대로)
LOG_EVERY, EVAL_EVERY, SAVE_EVERY = 10, 100, 50    # 옵티마이저 스텝 단위
QUICK_EVAL_N = 300             # 학습 도중 빠른 점검용 검증 문항 수 (epoch 끝에는 1,000개 전체 평가)
TRAIN_SEED = 42                # 학습 순서·선지 셔플·LoRA 초기화 시드 (같은 레시피로 두 번째 모델을 만들 때 43 등)
IMG_AUG = False                # True: 학습 이미지에만 밝기·대비·채도·감마·JPEG 화질을 흔듦 (글자·위치는 그대로)
TRAIN_PRESETS = None           # 예: ["native", "up1.5", "up2.0"] → 마이크로 배치마다 학습 해상도를 무작위로 (다해상도 학습)
EXTRA_TRAIN_CSV = None         # 추가 학습 데이터 CSV (열: id,path,question,a,b,c,d,answer). 검증 문항은 자동 차단

# ── 프롬프트: OCR 글자 (00b_ocr_paddle.ipynb 결과) ──
USE_OCR       = False          # True: 이미지 속 글자 줄을 질문 앞에 넣음 (학습·검증·test 모두 같은 형식)
OCR_TAG       = "pvl16_spotting"   # OUT_DIR/ocr_paddle/<OCR_TAG>/ocr_text.csv
OCR_MAX_CHARS = 1000           # 넣을 글자 수 상한. 넘으면 질문·보기와 글자가 많이 겹치는 줄부터 남김
OCR_POS_TAGS  = True           # 줄마다 대략의 위치 "(위 왼쪽)"

# ── 보조 학습: 보기 없는 단답형 (00_data_prep.ipynb가 만든 파일) ──
AUX_GEN_CSV = None             # 열: id,path,question,answer_text. None이면 끔
AUX_GEN_EVERY = 5              # 객관식 마이크로 배치 N개마다 보조 마이크로 배치 1개 (4면 문항 비중 80:20)
AUX_GEN_WEIGHT = 1.0           # 보조 손실(정답 문자열 토큰 평균 교차 엔트로피) 가중치
AUX_MAX_ANS_TOKENS = 24        # 이보다 긴 정답 문자열은 보조 학습에서 제외
GEN_PROMPT_VERSION = "v1_en_sys"   # 보조 학습 지시문: "v1_en_sys"(영어) | "v2_ko_sys"(한국어)

# ── 평가 / 제출 ──
EVAL_PRESETS = ["up2.0"]       # 학습이 끝난 뒤 이 해상도들로 검증 1,000문항 평가
EVAL_TTA     = False           # 가장 좋았던 해상도에서 선지 순열 4종 평균(TTA)의 효과도 측정
MAKE_SUBMISSION = True         # 학습·검증 뒤 이어서 test 추론 → 제출 파일
TEST_PRESET  = "up2.0"         # test 추론 해상도 ("auto": 검증이 가장 좋았던 해상도)
TEST_TTA     = False
ZEROSHOT_CHECK_N = 0           # >0이면 학습 전(LoRA 0)에 검증 앞 N문항을 채점해 기본 모델 실력 확인

# ── 경로 · 하드웨어 ──
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
USE_4BIT   = "auto"            # "auto": VRAM 30GB 미만·bf16 미지원·20B 이상 모델이면 4bit NF4(QLoRA), 그 외 bf16 LoRA / False: bf16
BATCH_SIZE = "auto"            # 평가용 배치
SEED, VAL_SIZE = 42, 1000      # 검증 분할용 — 공통값이므로 바꾸지 마세요
TRAIN_LIMIT = None             # 디버그용: 학습 샘플 수 제한
SKIP_MISSING_IMAGES = False
HF_TOKEN   = ""                # Gemma처럼 라이선스 승인이 필요한 모델일 때 (비워 두면 Colab 보안 비밀 'HF_TOKEN')

RES_PRESETS = {                # Qwen·HCX 계열: 이미지 픽셀 수 범위 (min_pixels, max_pixels)
    "baseline384": (384 * 384, 384 * 384),
    "mid":         (256 * 28 * 28, 512 * 28 * 28),
    "native":      (256 * 28 * 28, 1280 * 28 * 28),
    "up1.5":       (1_550_000, 2_100_000),
    "up2.0":       (2_700_000, 3_400_000),
}
# Gemma 4: 이미지당 '소프트 토큰' 수 (280 ≈ 원본, 1120 ≈ 1.9배)
RES_PRESETS_GEMMA4 = {"baseline384": 70, "mid": 140, "native": 280, "up1.5": 560, "up2.0": 1120}
# InternVL: 448px 타일 개수 (타일 1개 = 이미지 토큰 256개 + 전체 축소본 1타일). 720x960 사진 기준 up2.0(12타일) ≈ 1.9배
RES_PRESETS_INTERNVL = {"baseline384": 1, "mid": 4, "native": 6, "up1.5": 9, "up2.0": 12, "up2.5": 16}
RES_PRESET = TRAIN_PRESET
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # GPU 메모리 단편화 완화 (torch import 전에 설정)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")   # 허깅페이스 고속 다운로드
''')

# ───────────────────────────── 1-1. 레시피 ─────────────────────────────
md(r"""
## 1-1. 레시피 — 제출에 쓴 실행을 이름으로 고르기
- 아래 `RECIPES`의 값은 각 실행의 `train_log.json`(학습 기록)에서 그대로 옮긴 것입니다. 정밀도(`USE_4BIT`)는 기록에 없어 실행 이름·학습 노트북 기준으로 적었습니다.
- `SYNC_FROM_LOG = True`면 `OUT_DIR/<RECIPE>/train_log.json`이 있을 때 **기록 값을 다시 한 번 대조**해서 다른 값이 있으면 기록 쪽으로 맞추고 출력합니다.
- 이미 끝난 실행 폴더(제출 파일이 있음)에는 덮어쓰지 않고 멈춥니다. 같은 설정으로 다시 만들어 비교하려면 `RUN_SUFFIX = "_repro"`처럼 붙이세요.
- 학습이 중간에 끊긴 폴더(체크포인트만 있음)는 이어서 학습합니다.
- `REUSE_FINISHED = True`: 끝난 실행을 **학습 없이** 불러와 11번 이후(평가·test·dev 의사 라벨)만 다시 돌립니다. 저장된 예측 파일은 그대로 쓰고(덮어쓰지 않음), `REPRO_CHECK_N`문항은 저장된 어댑터로 새로 채점해 저장된 예측과 같은지 확인합니다.
""")
code(r'''
RECIPE        = "q35_27b_r16_native_bf16"   # 아래 RECIPES의 이름. None이면 1번 셀 설정 그대로 (새 실험)
FULL_DATA     = False      # True: 같은 레시피를 검증 1,000문항까지 합친 6,714문항으로 학습 → 실행 이름 끝에 "_full"
SYNC_FROM_LOG = True       # True: OUT_DIR/<RECIPE>/train_log.json이 있으면 기록 값과 대조해 맞춤
RUN_SUFFIX    = ""         # 예: "_repro" → 원래 실행과 다른 폴더에 다시 만들기
REUSE_FINISHED = False     # True: 끝난 실행 폴더의 어댑터를 불러와 학습 없이 평가·test·dev 의사 라벨만 (예측 파일은 이어쓰기 규칙대로 재사용)
REPRO_CHECK_N  = 100       # REUSE_FINISHED일 때 검증 앞 N문항을 저장된 어댑터로 새로 채점해 저장된 예측과 대조 (0이면 생략)

''' + RECIPES_SRC.strip("\n") + r'''

import os, json
LOG_SYNCED = {}
if RECIPE:
    assert RECIPE in RECIPES, f"RECIPES에 없는 이름입니다: {RECIPE}\n목록: {list(RECIPES)}"
    for _k, _v in RECIPES[RECIPE].items(): globals()[_k] = _v
    RUN_NAME = RECIPE
    _lp = os.path.join(OUT_DIR, RECIPE, "train_log.json")
    _log = json.load(open(_lp, encoding="utf-8")) if os.path.exists(_lp) else {}
    if SYNC_FROM_LOG and _log:
        _cfg = dict(_log.get("config") or {})
        _SYNC = ["MODEL_ID", "TRAIN_PRESET", "EPOCHS", "MAX_STEPS", "LR", "LORA_R", "LORA_ALPHA", "SHUFFLE_OPTIONS", "TRAIN_SEED", "TRAIN_ON_FULL",
                 "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "AUX_GEN_CSV", "AUX_GEN_EVERY", "AUX_GEN_WEIGHT", "VISION_LORA", "VIS_LR_SCALE",
                 "INIT_ADAPTER", "PLAN_EPOCH_OFFSET", "GEN_PROMPT_VERSION", "OCR_TAG", "OCR_MAX_CHARS", "OCR_POS_TAGS"]
        for _k in _SYNC:
            if _k in _cfg and _cfg[_k] != globals().get(_k):
                if _k in ("OCR_TAG", "OCR_MAX_CHARS", "OCR_POS_TAGS") and not str(_cfg.get("PROMPT_VERSION", "")).startswith("ocr_"): continue
                LOG_SYNCED[_k] = (globals().get(_k), _cfg[_k]); globals()[_k] = _cfg[_k]
        if "PROMPT_VERSION" in _cfg and str(_cfg["PROMPT_VERSION"]).startswith("ocr_") != bool(USE_OCR):
            LOG_SYNCED["USE_OCR"] = (USE_OCR, not USE_OCR); USE_OCR = not USE_OCR
        if "STEP_SAMPLES" in _cfg and int(_cfg["STEP_SAMPLES"]) != MICRO_BS * GRAD_ACCUM:
            _ss = int(_cfg["STEP_SAMPLES"]); assert _ss % MICRO_BS == 0, f"유효 배치 {_ss}를 MICRO_BS={MICRO_BS}로 나눌 수 없습니다"
            LOG_SYNCED["GRAD_ACCUM"] = (GRAD_ACCUM, _ss // MICRO_BS); GRAD_ACCUM = _ss // MICRO_BS
        assert int(_cfg.get("SEED", 42)) == 42, f"검증 분할 시드가 공통값(42)이 아닙니다: {_cfg.get('SEED')}"
        _unsup = {k: _cfg[k] for k in ("LOSS_MODE", "SAVE_FRACS") if _cfg.get(k) not in (None, "full")}
        if _unsup: print("⚠️ 이 노트북에 없는 학습 설정이 기록에 있습니다 (결과가 다를 수 있음):", _unsup)
        print(f"학습 기록 대조: {_lp} → " + (f"다른 값 {len(LOG_SYNCED)}개를 기록대로 맞춤" if LOG_SYNCED else "레시피와 모두 같음 ✅"))
        for _k, (_a, _b) in LOG_SYNCED.items(): print(f"  {_k:<18} {_a!r} → {_b!r}")
    elif SYNC_FROM_LOG:
        print(f"ℹ️ 학습 기록이 없습니다 ({_lp}) → 레시피 값 그대로 씁니다")
    if FULL_DATA:
        assert not TRAIN_ON_FULL, f"{RECIPE}는 이미 전체 데이터 학습 레시피입니다"
        _ea = {int(k): v for k, v in (_log.get("epoch_acc") or {}).items()}
        _best = max(_ea, key=lambda e: (_ea[e], e)) if _ea else None
        if _best is not None and _best != int(EPOCHS):
            raise RuntimeError(f"{RECIPE}는 EPOCHS={EPOCHS}로 학습하고 검증으로 epoch {_best}를 골랐습니다. "
                               "전체 데이터에서는 검증으로 epoch를 고를 수 없어 멈춥니다 → EPOCHS를 그 값으로 바꾸고 다시 실행하세요")
        if MAX_STEPS:
            _old = int(MAX_STEPS); MAX_STEPS = int(round(_old * 6714 / 5714))
            print(f"⚠️ 원래 실행은 MAX_STEPS={_old}에서 끊었습니다 → 데이터가 늘어난 비율(6,714/5,714)만큼 {MAX_STEPS}로 늘립니다")
        TRAIN_ON_FULL = True; RUN_NAME = RECIPE + "_full"
        EVAL_PRESETS, EVAL_TTA, MAKE_SUBMISSION, TEST_PRESET, TEST_TTA = ["up2.0"], False, True, "up2.0", False
    RUN_NAME += RUN_SUFFIX
    RES_PRESET = TRAIN_PRESET
    print(f"\n레시피 {RECIPE}: {RECIPE_NOTES.get(RECIPE, '')}")
else:
    print("RECIPE = None → 1번 셀 설정 그대로 (새 실험). 레시피 목록:")
    for _n, _t in RECIPE_NOTES.items(): print(f"  {_n:<30} {_t}")
_prev = os.path.join(OUT_DIR, RUN_NAME)
_done = os.path.isdir(_prev) and os.path.exists(os.path.join(_prev, "train_log.json")) and any(f.startswith("adapter_ep") for f in os.listdir(_prev))
if REUSE_FINISHED:
    assert not DRY_RUN and _done, f"REUSE_FINISHED: 끝난 실행 폴더(train_log.json + adapter_ep*)가 없습니다: {_prev} (DRY_RUN = False여야 함)"
    print(f"↩️ 끝난 실행을 학습 없이 다시 씀: {_prev}")
elif not DRY_RUN and os.path.isdir(_prev) and any(f.startswith("submission_") and f.endswith(".csv") for f in os.listdir(_prev)):
    raise RuntimeError(f"{_prev}는 이미 끝난 실행입니다 (제출 파일 있음). 덮어쓰지 않도록 멈춥니다 → 다시 만들어 비교하려면 RUN_SUFFIX = \"_repro\", "
                       "저장된 어댑터로 평가·test·dev 의사 라벨만 다시 돌리려면 REUSE_FINISHED = True")
_SHOW = ["RUN_NAME", "MODEL_ID", "DRY_RUN", "TRAIN_PRESET", "EPOCHS", "MAX_STEPS", "LR", "MICRO_BS", "GRAD_ACCUM", "USE_4BIT", "TRAIN_SEED", "TRAIN_ON_FULL",
         "VISION_LORA", "USE_OCR", "AUX_GEN_CSV", "AUX_GEN_EVERY", "INIT_ADAPTER", "PLAN_EPOCH_OFFSET", "EPOCH_END_EVAL", "GEN_PROMPT_VERSION",
         "EXTRA_TRAIN_CSV", "EVAL_PRESETS", "MAKE_SUBMISSION", "TEST_PRESET"]
print("최종 설정:"); [print(f"  {_k:<18} = {globals()[_k]!r}") for _k in _SHOW]
''')

# ───────────────────────────── 2. 설치 ─────────────────────────────
md("## 2. 라이브러리 설치 (버전 고정)\nHyperCLOVA X만 transformers 개발 버전(검증한 커밋)을 쓰고, 나머지는 transformers 5.17.0입니다. 모델 계열을 바꿀 때는 **세션을 다시 시작**하고 처음부터 실행하세요.")
code(r'''
import re
_HCX = "hyperclova" in str(MODEL_ID).lower()
# HyperCLOVA X 비전 모델은 transformers 정식 5.17에 없고 개발 버전(main)에만 있음 → 검증한 커밋으로 고정
TF_PIN = ("transformers @ git+https://github.com/huggingface/transformers.git@98d39824ed30e684e5122d04a2d9564efffc4965" if _HCX
          else "transformers==5.17.0")
print("transformers:", TF_PIN)
!pip -q install "{TF_PIN}" "accelerate==1.15.0" "bitsandbytes==0.50.2" "peft==0.21.0"
# Colab에 기본 설치된 구버전 torchao가 있으면 peft 0.21이 LoRA를 붙이는 순간 ImportError → 쓰지 않으므로 제거
!pip -q uninstall -y torchao
# Qwen3.5 이후의 선형 어텐션용 Triton 커널. 없으면 순수 PyTorch 대체 구현으로 돌아 OOM·저속
if re.search(r"Qwen3\.[5-9]", MODEL_ID):      # Qwen3.5 / 3.6 / 3.8 — 같은 선형 어텐션 구조(model_type qwen3_5, qwen3_5_moe)
    !pip -q install "flash-linear-attention==0.5.2"
''')

# ───────────────────────────── 3. 데이터 ─────────────────────────────
md("## 3. 데이터 준비 (Drive의 대회 zip → `DATA_DIR`)")
code(hc(6))

# ───────────────────────────── 4. import ─────────────────────────────
md("## 4. 공통 import · CSV 로드")
code(hc(9))

# ───────────────────────────── 5. 분할 ─────────────────────────────
md("## 5. 고정 검증셋 분할 (모든 실행 공통)")
code(rep(hc(11), "train.csv 원본/SEED/VAL_SIZE를 확인하세요.", "train.csv 원본/SEED/VAL_SIZE를 확인하세요."))

# ───────────────────────────── 6. 프롬프트 ─────────────────────────────
md(r"""
## 6. 프롬프트 (+ OCR 글자 선택)
- system: 한 글자로만 답하라는 지시 (영어 한 줄). user: 이미지 → 질문 → `(a)~(d)` 보기 → "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
- `USE_OCR = True`면 질문 앞에 `[이미지 속 글자 — OCR 자동 인식 결과 …]` 블록이 붙습니다. 보기 순서와 무관한 블록이라 선지를 섞어도 같습니다.
- system 역할을 못 푸는 템플릿은 7-1번 셀이 자동으로 판별해 지시문을 문자열이나 user 메시지 앞에 넣습니다.
""")
P = hc(13)
P = rep(P, 'PROMPT_VERSION = "baseline_v1"\n', 'PROMPT_VERSION = f"ocr_v2_{OCR_TAG}" if USE_OCR else "baseline_v1"      # 학습 기록에 남는 프롬프트 이름\n')
P = rep(P, 'LETTERS = "abcd"\n', r'''LETTERS = "abcd"

# ── OCR 결과 불러오기 (USE_OCR일 때만): 00b_ocr_paddle.ipynb가 만든 OUT_DIR/ocr_paddle/<OCR_TAG>/ocr_text.csv ──
def _clean_line(x):                                          # OCR이 생성 루프에 빠진 흔적 정리
    y = re.sub(r"(.)\1{5,}", r"\1" * 6, x)                   # 같은 글자가 6번 넘게 이어지면 6번으로
    y = re.sub(r"(.{2,12}?)\1{3,}", r"\1\1\1", y)            # 짧은 조각이 4번 넘게 반복되면 3번으로
    return y.strip()
def _clean_lines(lines):
    out, seen = [], {}
    for t, x in lines:
        y = _clean_line(x)
        if not y or len(y) < 0.4 * len(x): continue            # 대부분이 반복이던 줄은 버림
        seen[y] = seen.get(y, 0) + 1
        if seen[y] > 3: continue                                # 같은 글자 줄은 3번까지
        out.append((t, y))
    return out
def _norm(s): return re.sub(r"[\s\W_]+", "", str(s).lower())
def _bg(s): return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())
OCR_HEADER = "[이미지 속 글자 — OCR 자동 인식 결과. 틀리거나 빠진 글자가 있을 수 있음]"
OCR_LINES, OCR_INFO = {}, {}
if USE_OCR:
    OCR_FILE = os.path.join(OUT_DIR, "ocr_paddle", OCR_TAG, "ocr_text.csv")
    assert os.path.exists(OCR_FILE), f"OCR 결과가 없습니다: {OCR_FILE} → 00b_ocr_paddle.ipynb를 먼저 끝까지 실행하세요"
    _ocr = pd.read_csv(OCR_FILE, keep_default_na=False)
    _raw_lines = {p: [(t or None, str(x)) for t, x in json.loads(j)] for p, j in zip(_ocr.path, _ocr.lines_json)}
    OCR_LINES = {p: _clean_lines(v) for p, v in _raw_lines.items()}
    _need = {"train": list(train_df.path)}
    if MAKE_SUBMISSION: _need["test"] = list(test_df.path)
    if EXTRA_TRAIN_CSV: _need["추가 학습 데이터"] = list(pd.read_csv(EXTRA_TRAIN_CSV).path)
    for _k, _ps in _need.items():
        _miss = [p for p in _ps if p not in OCR_LINES]
        assert not _miss, f"OCR 결과가 없습니다: {_k} 이미지 {len(_miss)}장 (예: {_miss[:3]}) → 00b_ocr_paddle.ipynb의 SPLITS·샤드 완료 여부를 확인하세요"
    _info_fp = os.path.join(os.path.dirname(OCR_FILE), "ocr_info.json")
    OCR_INFO = json.load(open(_info_fp)) if os.path.exists(_info_fp) else {}
    _nc = pd.to_numeric(_ocr.n_chars, errors="coerce").fillna(0)
    print(f"OCR {OCR_TAG}: {len(OCR_LINES):,}장 | {OCR_INFO.get('OCR_MODEL_ID', '?')} · {OCR_INFO.get('OCR_TASK', '?')} | "
          f"이미지당 글자 수 중앙값 {_nc.median():.0f} · {OCR_MAX_CHARS}자 초과 {(_nc > OCR_MAX_CHARS).mean() * 100:.1f}%")

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
    return OCR_HEADER + "\n" + ("\n".join(fmt) if fmt else "(인식된 글자 없음)")
''')
P = rep(P, '    user_text = build_mc_prompt(str(row["question"]), opts)\n',
        '    user_text = build_mc_prompt(str(row["question"]), opts)\n'
        '    if USE_OCR: user_text = ocr_block(row) + "\\n\\n" + user_text          # OCR 블록 → 질문·보기·답 지시\n')
code(P)

# ───────────────────────────── 7. 모델 (7-1 프로세서·모델별 처리 / 7-2 모델·LoRA) ─────────────────────────────
M = hc(15)
i_split = M.index("t0 = time.time()\n")
M1, M2 = M[:i_split], M[i_split:]
M1 = rep(M1, '''import shutil
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME + ("_dry" if DRY_RUN else ""))''', '''PRECISION = "nf4" if use_4bit else str(DTYPE).replace("torch.", "")      # 학습 기록에 남김

import shutil
RUN_DIR = os.path.join(OUT_DIR, RUN_NAME + ("_dry" if DRY_RUN else ""))''')
M1 = rep(M1, '''IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)''', '''# ═══════════ 모델별 처리 (MODEL_NOTES.md) ═══════════
# 1) 생각(추론) 모드 끄기: 템플릿 변수 이름이 enable_thinking(Qwen·Gemma) 또는 thinking(HyperCLOVA X)
# 2) system 내용 형식: 목록형 system을 못 풀고 글자 그대로 찍는 템플릿은 문자열로, 그래도 안 되면 user 메시지 앞에 붙임
# 3) 빈 생각 블록 닫기: 템플릿이 '<think>\\n'으로 끝나면(HyperCLOVA X) '\\n</think>\\n\\n'을 붙여 첫 토큰에서 바로 답하게 함
IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)''')
md(r"""
## 7-1. 장치·정밀도 · 프로세서 · 모델별 프롬프트 보정
이 셀이 모델마다 다른 처리를 정합니다(자세한 내용은 `MODEL_NOTES.md`).
- **생각 모드 끄기**: 템플릿 변수 `enable_thinking=False`(Qwen·Gemma) / `thinking=False`(HyperCLOVA X).
- **system 형식 판별**: 목록형 system 내용이 풀리지 않고 글자 그대로 찍히는 템플릿이면 문자열로, 그래도 안 되면 user 메시지 앞에 붙입니다.
- **빈 생각 블록 닫기**: HyperCLOVA X 템플릿은 생각 블록을 연 채(`<think>\n`) 끝나므로 `\n</think>\n\n`을 붙여 첫 토큰이 곧 답 글자가 되게 합니다.
- 확인할 것: `프롬프트 미리보기(끝)` — Qwen은 `</think>\n\n`, HyperCLOVA X는 `<think>\n\n</think>\n\n`으로 끝나야 합니다. 첫 학습 로그의 loss가 1보다 한참 작아야 정상입니다(20 이상이면 답 자리가 틀린 것).
""")
code(M1)

md("## 7-2. 모델 로드 + LoRA 부착 (비전 LoRA · 이어학습 선택)")
M2 = rep(M2, '''    print("언어 모델 경로 이름이 달라('language_model' 없음) 비전·투영 모듈을 뺀 선형층에 LoRA를 붙입니다")
''', r'''    print("언어 모델 경로 이름이 달라('language_model' 없음) 비전·투영 모듈을 뺀 선형층에 LoRA를 붙입니다")
# 비전 쪽 모듈 판별: 'proj' = merger·투영층, 'tower' = 비전 인코더 블록, None = 언어 모델 등
_VIS_TOWER = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder)(\.|$)")
_VIS_PROJ = re.compile(r"(^|\.)(merger|multi_modal_projector|mm_projector|embed_vision|projector)(\.|$)")
def _vis_kind(name):
    if "language_model" in name or "lm_head" in name: return None
    if _VIS_PROJ.search(name): return "proj"
    if _VIS_TOWER.search(name): return "tower"
    return None
VIS_TARGETS = {"tower": [], "proj": []}
if VISION_LORA:
    _mods = dict(base.named_modules())
    for _n, _m in _mods.items():
        _k = _vis_kind(_n)
        if _k and isinstance(_m, torch.nn.Linear) and "patch_embed" not in _n:
            VIS_TARGETS[_k].append(_n)
    assert VIS_TARGETS["tower"], "비전 블록의 선형층을 찾지 못했습니다 → 이 모델의 비전 모듈 이름을 확인하세요 (_VIS_TOWER)"
    _vis_types = sorted({type(_mods[_n]).__name__ for _n in VIS_TARGETS["tower"] + VIS_TARGETS["proj"]})
    print(f"비전 LoRA 대상: 비전 블록 {len(VIS_TARGETS['tower'])}개 + merger/투영 {len(VIS_TARGETS['proj'])}개 | 층 종류 {_vis_types}")
    print("  예:", VIS_TARGETS["tower"][:4], VIS_TARGETS["proj"][:2])
    assert not any("4bit" in t or "8bit" in t for t in _vis_types), "비전 선형층이 양자화돼 있습니다(bf16이어야 함) → llm_int8_skip_modules를 확인하세요"
    from transformers.modeling_layers import GradientCheckpointingLayer
    _vblk = next((_m for _n, _m in _mods.items() if _vis_kind(_n) == "tower" and isinstance(_m, GradientCheckpointingLayer)), None)
    print("비전 블록 그래디언트 체크포인팅:", "켜짐" if (_vblk is not None and _vblk.gradient_checkpointing) else "꺼짐/해당 없음")
    _lm_names = [_n for _n in _mods if re.fullmatch(LORA_TARGETS, _n)]
    assert _lm_names, "언어 모델 LoRA 대상을 찾지 못했습니다"
    LORA_TARGETS = _lm_names + VIS_TARGETS["tower"] + VIS_TARGETS["proj"]      # 정확한 모듈 이름 목록 (PEFT가 이름 그대로 매칭)
    del _mods
''')
M2 = rep(M2, '''RESUMING = os.path.exists(os.path.join(CKPT_DIR, "trainer_state.pt"))
if RESUMING:
    model = PeftModel.from_pretrained(base, os.path.join(CKPT_DIR, "adapter"), is_trainable=True)
    print("↩️ 체크포인트에서 어댑터를 불러왔습니다:", CKPT_DIR)
else:''', '''RESUMING = os.path.exists(os.path.join(CKPT_DIR, "trainer_state.pt"))
INIT_ADAPTER_SHA = None
def _file_sha(p):
    _h = hashlib.sha256()
    with open(p, "rb") as f:
        for _b in iter(lambda: f.read(1 << 24), b""): _h.update(_b)
    return _h.hexdigest()[:16]
if RESUMING:
    model = PeftModel.from_pretrained(base, os.path.join(CKPT_DIR, "adapter"), is_trainable=True)
    print("↩️ 체크포인트에서 어댑터를 불러왔습니다:", CKPT_DIR)
elif INIT_ADAPTER:                                                       # 기존 어댑터에서 이어학습 (옵티마이저·스케줄러는 9번 셀에서 새로 만듦)
    assert os.path.isfile(os.path.join(INIT_ADAPTER, "adapter_model.safetensors")), f"시작 어댑터가 없습니다: {INIT_ADAPTER}"
    model = PeftModel.from_pretrained(base, INIT_ADAPTER, is_trainable=True)
    print("↪️ 기존 어댑터에서 이어학습:", INIT_ADAPTER)
else:''')
M2 = rep(M2, '''model.print_trainable_parameters()''', '''if INIT_ADAPTER:
    INIT_ADAPTER_SHA = _file_sha(os.path.join(INIT_ADAPTER, "adapter_model.safetensors"))
    print("시작 어댑터 지문:", INIT_ADAPTER_SHA)
model.print_trainable_parameters()''')
M2 = rep(M2, '''n_lora_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and any(k in n for k in ("visual", "vision", "siglip")))
print(f"LoRA가 붙은 층: {n_lora}개 (비전 타워 {n_lora_vis}개 — 0이어야 정상)")
assert n_lora > 0 and n_lora_vis == 0''', '''n_lora_vis = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A") and (_vis_kind(n) or any(k in n for k in ("visual", "vision", "siglip"))))
print(f"LoRA가 붙은 층: {n_lora}개 (비전 쪽 {n_lora_vis}개 — " + ("비전 LoRA 켬)" if VISION_LORA else "0이어야 정상)"))
assert n_lora > 0 and ((n_lora_vis > 0) if VISION_LORA else (n_lora_vis == 0))
LORA_TARGET_COUNTS = {"language": n_lora - n_lora_vis, "vision": n_lora_vis}
_lc = ((_log.get("config") or {}).get("LORA_TARGET_COUNTS") if RECIPE and "_log" in globals() else None)
if _lc: print("LoRA 층 수가 학습 기록과 " + ("같음 ✅" if _lc == LORA_TARGET_COUNTS else f"다름 ⚠️ 기록 {_lc} · 지금 {LORA_TARGET_COUNTS}"))''')
code(M2)

# ───────────────────────────── 8. 채점 ─────────────────────────────
md("## 8. 채점 함수 (답 자리 로짓 → 네 보기 확률) + 선지 순열 TTA")
code(hc(17))

# ───────────────────────────── 9. 학습 준비 ─────────────────────────────
md("## 9. 학습 준비: 배치 구성 · 손실 · 보조 학습 · 옵티마이저 · 체크포인트\n옵티마이저는 언어 LoRA(`LR`)와 비전 LoRA(`LR x VIS_LR_SCALE`) 두 그룹입니다(비전 LoRA를 켰을 때).")
PR = hc(19)
PR = rep(PR, '''GEN_INSTRUCT = ("You are a helpful visual question answering assistant. "
                "Read the text in the image and answer with the exact words only. No explanation.")''', '''GEN_INSTRUCT = ("You are a helpful visual question answering assistant. "
                "Read the text in the image and answer with the exact words only. No explanation.")
if GEN_PROMPT_VERSION == "v2_ko_sys":
    GEN_INSTRUCT = ("당신은 이미지 속 글자를 읽고 질문에 답하는 도우미입니다. "
                    "선택지는 주어지지 않습니다. 이미지에 적힌 표현 그대로 짧게 답하고, 설명하지 마세요.")''')
PR = rep(PR, '''params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)''', '''_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
VIS_PARAMS = [p for n, p in _named if _vis_kind(n)]
LM_PARAMS = [p for n, p in _named if not _vis_kind(n)]
params = LM_PARAMS + VIS_PARAMS
assert bool(VIS_PARAMS) == bool(VISION_LORA), "비전 LoRA 설정과 학습 파라미터가 맞지 않습니다"
_groups = [{"params": LM_PARAMS, "lr": LR}] + ([{"params": VIS_PARAMS, "lr": LR * VIS_LR_SCALE}] if VIS_PARAMS else [])
optimizer = torch.optim.AdamW(_groups, lr=LR, weight_decay=0.0)
print(f"학습 파라미터: 언어 {sum(p.numel() for p in LM_PARAMS):,} (LR {LR:g})" + (f" | 비전 {sum(p.numel() for p in VIS_PARAMS):,} (LR {LR * VIS_LR_SCALE:g})" if VIS_PARAMS else ""))''')
PR = rep(PR, '''"TRAIN_ON_FULL", "SEED", "TRAIN_SEED", "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "AUX_GEN_CSV", "AUX_GEN_EVERY", "AUX_GEN_WEIGHT"]}}''',
         '''"TRAIN_ON_FULL", "SEED", "TRAIN_SEED", "EXTRA_TRAIN_CSV", "IMG_AUG", "TRAIN_PRESETS", "AUX_GEN_CSV", "AUX_GEN_EVERY", "AUX_GEN_WEIGHT",
          "VISION_LORA", "VIS_LR_SCALE", "INIT_ADAPTER", "INIT_ADAPTER_SHA", "PLAN_EPOCH_OFFSET", "GEN_PROMPT_VERSION",
          "PROMPT_VERSION", "OCR_TAG", "OCR_MAX_CHARS", "OCR_POS_TAGS", "LORA_DROPOUT", "LORA_TARGET_COUNTS", "PRECISION"]}}''')
PR = rep(PR, '''    _defaults = {"TRAIN_SEED": 42, "MAX_STEPS": None, "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None, "AUX_GEN_CSV": None, "AUX_GEN_EVERY": 5, "AUX_GEN_WEIGHT": 1.0}
    same = all(ck["state"]["config"].get(k, _defaults.get(k)) == v for k, v in state["config"].items())''',
         '''    _defaults = {"TRAIN_SEED": 42, "MAX_STEPS": None, "EXTRA_TRAIN_CSV": None, "IMG_AUG": False, "TRAIN_PRESETS": None, "AUX_GEN_CSV": None, "AUX_GEN_EVERY": 5, "AUX_GEN_WEIGHT": 1.0,
                 "VISION_LORA": False, "VIS_LR_SCALE": 0.2, "INIT_ADAPTER": None, "INIT_ADAPTER_SHA": None, "PLAN_EPOCH_OFFSET": 0, "GEN_PROMPT_VERSION": "v1_en_sys",
                 "PROMPT_VERSION": "baseline_v1"}     # 예전 체크포인트에 없는 항목의 기본값 (기록도 기본값도 없는 항목은 비교하지 않음)
    same = all(ck["state"]["config"].get(k, _defaults.get(k, v)) == v for k, v in state["config"].items())''')
PR = rep(PR, """    print(f"↩️ {state['global_step']}/{total_steps} 스텝부터 이어서 학습합니다")""", """    print(f"↩️ {state['global_step']}/{total_steps} 스텝부터 이어서 학습합니다")
if REUSE_FINISHED:                                   # 학습하지 않음: 끝난 실행의 기록(epoch별 검증 정확도)으로 11번 셀이 어댑터를 고름
    _old = json.load(open(os.path.join(RUN_DIR, "train_log.json"), encoding="utf-8"))
    state["epoch_acc"] = {int(k): v for k, v in (_old.get("epoch_acc") or {}).items()}
    state["global_step"] = total_steps
    print(f"↩️ 학습 없이 끝난 실행을 씁니다 (기록된 epoch 정확도 {state['epoch_acc']})")""")
code(PR)

# ───────────────────────────── 10. 학습 루프 ─────────────────────────────
md("## 10. 학습 루프\n이어학습(`INIT_ADAPTER`)이면 먼저 시작 어댑터가 원래 모델과 같은 답을 내는지(검증 앞 40문항, 97.5% 이상) 확인합니다.")
L = hc(21)
L = rep(L, '''if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
model.train(); optimizer.zero_grad(set_to_none=True)''', '''if INIT_ADAPTER and not RESUMING and not REUSE_FINISHED:   # 이어학습 시작점 재현 확인: 원 어댑터의 저장된 검증 예측과 같아야 정상
    _ref_p = os.path.join(os.path.dirname(INIT_ADAPTER.rstrip("/")), "preds_final_ep1__up2.0__1000.csv")
    if os.path.exists(_ref_p):
        _ref = pd.read_csv(_ref_p).drop_duplicates("id", keep="last").set_index("id")
        set_resolution("up2.0"); model.eval()
        _chk = val_df.head(16 if DRY_RUN else 40)
        _pc, _ = predict_df(_chk, "init_check", resume=False, verbose=False); _pc = _pc.set_index("id").loc[_chk.id]
        _agree = float((_pc.pred.values == _ref.loc[_chk.id, "pred"].values).mean())
        _dmax = float(np.abs(_pc[["p_a", "p_b", "p_c", "p_d"]].values - _ref.loc[_chk.id, ["p_a", "p_b", "p_c", "p_d"]].values).max())
        print(f"🔎 이어학습 시작점 재현: 답 일치 {_agree * 100:.1f}% | 확률 최대 차 {_dmax:.4f} ({len(_chk)}문항, up2.0)")
        assert _agree >= 0.975, "시작 어댑터가 원 모델과 다르게 동작합니다 → INIT_ADAPTER 경로와 모델·정밀도 설정을 확인하세요"
        set_resolution(TRAIN_PRESET)
    else:
        print("⚠️ 시작 어댑터의 검증 예측 파일이 없어 재현 확인을 건너뜁니다:", _ref_p)
if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
model.train(); optimizer.zero_grad(set_to_none=True)''')
L = rep(L, "    if epoch != plan_epoch: order, perms_all = epoch_plan(epoch); plan_epoch = epoch",
        "    if epoch != plan_epoch: order, perms_all = epoch_plan(epoch + PLAN_EPOCH_OFFSET); plan_epoch = epoch")
L = rep(L, '''    scaler.unscale_(optimizer)
''', '''    scaler.unscale_(optimizer)
    if VISION_LORA and step == step0:
        _gB = [float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and _vis_kind(n) and "lora_B" in n]
        _gA = [float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and _vis_kind(n) and "lora_A" in n]
        print(f"🔎 첫 스텝 비전 LoRA 그래디언트: B 합 {sum(_gB):.3e} ({len(_gB)}개) | A 합 {sum(_gA):.3e} — B가 0보다 크면 정상 (처음부터 학습할 땐 첫 스텝 A가 0이어도 정상)")
        assert sum(_gB) > 0, "비전 LoRA에 그래디언트가 흐르지 않습니다"
''')
L = rep(L, '''        el = time.time() - t_start; done = step + 1 - step0
''', '''        el = time.time() - t_start; done = step + 1 - step0; _lrs = "/".join(f"{x:.1e}" for x in scheduler.get_last_lr())
''')
L = rep(L, 'f"lr {scheduler.get_last_lr()[0]:.2e} | gnorm', 'f"lr {_lrs} | gnorm')
L = rep(L, '''    if end_of_epoch and not DRY_RUN:
        acc, _ = evaluate(val_df, f"ep{epoch + 1}", resume=False, verbose=False)
        state["epoch_acc"][epoch + 1] = acc
        model.save_pretrained(os.path.join(RUN_DIR, f"adapter_ep{epoch + 1}"))''', '''    if end_of_epoch and not DRY_RUN:
        if EPOCH_END_EVAL:
            acc, _ = evaluate(val_df, f"ep{epoch + 1}", resume=False, verbose=False)
            state["epoch_acc"][epoch + 1] = acc
        model.save_pretrained(os.path.join(RUN_DIR, f"adapter_ep{epoch + 1}"))''')
L = rep(L, '''if DEVICE == "cuda": print(f"최대 VRAM 사용:''', '''if VISION_LORA and not REUSE_FINISHED:
    _wB = sum(float(p.detach().float().norm()) for n, p in model.named_parameters() if _vis_kind(n) and "lora_B" in n)
    print(f"🔎 비전 LoRA B 가중치 크기 합 {_wB:.3e} (0이면 비전 쪽이 학습되지 않은 것)")
    assert _wB > 0, "비전 LoRA 가중치가 처음 그대로입니다"
if DEVICE == "cuda": print(f"최대 VRAM 사용:''')
L = rep(L, "(+평가 약 15분)", "(+검증 평가·test 추론 시간은 별도)")
L = rep(L, '''json.dump({"epoch_acc": state["epoch_acc"], "history": state["history"], "config": state["config"]},''',
        '''if not REUSE_FINISHED: json.dump({"epoch_acc": state["epoch_acc"], "history": state["history"], "config": state["config"], "recipe": RECIPE},''')
code(L)

# ───────────────────────────── 11. 최종 평가 ─────────────────────────────
md("## 11. 최종 평가: 가장 좋은 epoch의 어댑터 → 해상도별 검증 점수 + 오답 리포트\n`DRY_RUN`에서는 저장한 어댑터를 다시 불러와 보고, test 추론 경로도 몇 문항으로 점검합니다.")
E = hc(23)
E = rep(E, '''                     f"<div><b>{r.id}</b><p>{_html.escape(str(r.question))}</p><ul>{opts}</ul></div></div>")''',
        '''                     f"<div><b>{r.id}</b><p>{_html.escape(str(r.question))}</p><ul>{opts}</ul>"
                     + (f"<pre style='white-space:pre-wrap;font-size:12px;background:#f5f5f5;padding:6px'>{_html.escape(ocr_block(r))}</pre>" if USE_OCR else "")
                     + "</div></div>")''')
E = rep(E, '''if not DRY_RUN and state["epoch_acc"]:
    best_ep = max(state["epoch_acc"], key=lambda e: (state["epoch_acc"][e], e))     # 동률이면 더 오래 학습한 쪽
    print("epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()}, "→ 사용할 어댑터: epoch", best_ep)''',
        '''if not DRY_RUN:
    best_ep = max(state["epoch_acc"], key=lambda e: (state["epoch_acc"][e], e)) if state["epoch_acc"] else n_epochs   # 동률이면 더 오래 학습한 쪽 · epoch 끝 평가를 껐으면 마지막 어댑터
    print("epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()}, "→ 사용할 어댑터: epoch", best_ep)''')
E = rep(E, "for preset in (EVAL_PRESETS if not DRY_RUN else [TRAIN_PRESET]):",
        "for preset in (EVAL_PRESETS if not DRY_RUN else list(dict.fromkeys([TRAIN_PRESET] + list(EVAL_PRESETS)))):")
E = rep(E, '''if TRAIN_ON_FULL: print("⚠️ TRAIN_ON_FULL=True: 검증셋을 학습에 썼으므로 위 점수는 의미가 없습니다 (제출용 모델).")
set_resolution(TRAIN_PRESET)''', '''if TRAIN_ON_FULL: print("⚠️ TRAIN_ON_FULL=True: 검증셋을 학습에 썼으므로 위 점수는 의미가 없습니다 (제출용 모델).")
if DRY_RUN:
    _ad = os.path.join(CKPT_DIR, "adapter")
    assert os.path.isdir(_ad), "DRY_RUN 체크포인트 어댑터가 저장되지 않았습니다"
    _w = load_peft_weights(_ad, device="cpu"); _vk = [k for k in _w if _vis_kind(k)]
    _res = set_peft_model_state_dict(model, _w); _unexp = list(getattr(_res, "unexpected_keys", None) or [])
    print(f"🔎 저장된 어댑터 다시 불러오기: 키 {len(_w)}개 (비전 {len(_vk)}개) | 예상 밖 키 {len(_unexp)}개")
    assert not _unexp and (bool(_vk) == bool(VISION_LORA)), "어댑터 저장/재로드 점검 실패"
    set_resolution(TEST_PRESET if TEST_PRESET != "auto" else EVAL_PRESETS[-1])
    _pt, _ = predict_df(test_df.head(2 * BS), "dry_test", resume=False, verbose=False)
    print(f"🔎 test 추론 경로 점검({RES_PRESET}): {len(_pt)}문항 | letter_mass 평균 {_pt.letter_mass.mean():.3f} | 이미지 토큰 평균 {_pt.img_tokens.mean():.0f}")
    assert len(_pt) == min(2 * BS, len(test_df)) and _pt.letter_mass.mean() > 0.9, "test 추론 경로 이상 (letter_mass가 낮음 → 7-1번 셀의 프롬프트 끝을 확인하세요)"
    print("✅ DRY_RUN 점검 모두 통과")
if REUSE_FINISHED and REPRO_CHECK_N:                  # 저장된 어댑터로 새로 채점 → 저장된 검증 예측과 대조 (재현성)
    _pre = "up2.0" if "up2.0" in EVAL_PRESETS else EVAL_PRESETS[0]
    _sp = os.path.join(RUN_DIR, f"preds_{FINAL_TAG}__{_pre}__{len(val_df)}.csv")
    if os.path.exists(_sp):
        set_resolution(_pre); _chk = val_df.head(int(REPRO_CHECK_N))
        _new, _ = predict_df(_chk, "repro_check", resume=False, verbose=False)
        _old = pd.read_csv(_sp).drop_duplicates("id", keep="last").set_index("id").loc[_chk.id]; _new = _new.set_index("id").loc[_chk.id]
        _ag = float((_new.pred.values == _old.pred.values).mean()); _dm = float(np.abs(_new[["p_a", "p_b", "p_c", "p_d"]].values - _old[["p_a", "p_b", "p_c", "p_d"]].values).max())
        REPRO = {"n": len(_chk), "agree": _ag, "max_prob_diff": _dm}
        print(f"🔎 재현 대조 ({_pre}, 검증 앞 {len(_chk)}문항): 저장된 예측과 답 일치 {_ag * 100:.1f}% · 확률 최대 차 {_dm:.4f} "
              "(같은 GPU 종류면 거의 0, GPU·라이브러리가 다르면 작은 차이는 정상)")
    else:
        print("⚠️ 재현 대조할 저장된 검증 예측이 없습니다:", _sp)
set_resolution(TRAIN_PRESET)''')
code(E)

# ───────────────────────────── 12. test ─────────────────────────────
md("## 12. test 추론 → 제출 파일 (이어하기 지원)")
code(hc(25))

# ───────────────────────────── 12-2. dev 의사 라벨 ─────────────────────────────
md(r"""
## 12-2. (선택) dev 의사 라벨 만들기 — `*_dev` 레시피용
dev 문항(정답 없음, 교육생 응답 5개)을 이 실행의 모델로 채점해, **모델 확신도 ≥ `MIN_CONF`**이고 **응답 다수결 표 수 ≥ `MIN_VOTES`**이며 **모델 답 = 다수결 답**인 문항만 학습용 CSV로 저장합니다.
`gemma4_31b_r16_up20` 레시피에서 `MAKE_DEV_PSEUDO = True`로 실행하면 `dev_pseudo__gemma4_31b_r16_up20__up2.0__c0.8_v3_agree.csv`가 생기고, `*_dev` 레시피가 이 파일을 `EXTRA_TRAIN_CSV`로 씁니다.
""")
code(r'''
MAKE_DEV_PSEUDO = False                      # True: dev 채점 → 의사 라벨 CSV
DEV_PRESET, MIN_CONF, MIN_VOTES, REQUIRE_AGREE = "up2.0", 0.8, 3, True
if MAKE_DEV_PSEUDO and not DRY_RUN:
    P = ["p_a", "p_b", "p_c", "p_d"]
    assert dev_df is not None and len(dev_df), "dev.csv가 없습니다"
    for c in ["question", "a", "b", "c", "d"]: dev_df[c] = dev_df[c].astype(str)
    resp_cols = [c for c in dev_df.columns if c.startswith("answer")]         # answer1 ~ answer5 (교육생 응답)
    assert not (set(dev_df.id) & set(val_df.id)) and not (set(dev_df.id) & set(train_df.id)), "dev와 train/검증 id가 겹칩니다"
    set_resolution(DEV_PRESET); model.eval()
    pred, _ = predict_df(dev_df, "dev")                                     # RUN_DIR/preds_dev__{DEV_PRESET}__{len(dev)}.csv (이어하기 지원)
    m = dev_df.merge(pred, on="id", how="inner")
    def vote(row):
        v = [str(row[c]).strip().lower() for c in resp_cols]; v = [x for x in v if x in LETTERS]
        if not v: return None, 0
        cnt = {L: v.count(L) for L in set(v)}; top = max(cnt.values()); winners = [L for L, k in cnt.items() if k == top]
        return (winners[0] if len(winners) == 1 else None), top            # 동점이면 다수결 없음
    vv = [vote(r) for _, r in m.iterrows()]
    m["plural"] = [w for w, _ in vv]; m["n_votes"] = [k for _, k in vv]
    m["conf"] = m[P].max(axis=1)
    agree = m.pred == m.plural
    print(f"dev {len(m):,}문항 | 다수결 있음 {int(m.plural.notna().sum()):,} | 모델 = 다수결 {int(agree.sum()):,} ({agree.mean() * 100:.1f}%)")
    print("응답 다수결 표수별 모델 일치율:", {int(k): f"{agree[m.n_votes == k].mean() * 100:.0f}% (n={int((m.n_votes == k).sum())})" for k in sorted(m.n_votes.unique()) if k > 0})
    print("모델 확신도 분포:", {f"≥{t}": int((m.conf >= t).sum()) for t in (0.5, 0.7, 0.8, 0.9, 0.95)})
    keep = (m.conf >= MIN_CONF) & (m.n_votes >= MIN_VOTES)
    if REQUIRE_AGREE: keep &= agree
    out = m.loc[keep, ["id", "path", "question", "a", "b", "c", "d"]].copy(); out["answer"] = m.loc[keep, "pred"].values
    op = os.path.join(OUT_DIR, f"dev_pseudo__{RUN_NAME}__{DEV_PRESET}__c{MIN_CONF}_v{MIN_VOTES}{'_agree' if REQUIRE_AGREE else ''}.csv")
    out.to_csv(op, index=False)
    print(f"\n채택 {len(out):,}문항 / {len(m):,} | 정답 분포 {out.answer.value_counts().sort_index().to_dict()}\n저장: {op}")
    set_resolution(TRAIN_PRESET)
else:
    print("dev 의사 라벨: 끔 (MAKE_DEV_PSEUDO = False)" if not MAKE_DEV_PSEUDO else "DRY_RUN → 건너뜀")
''')

# ───────────────────────────── 13. 요약 ─────────────────────────────
md("## 13. 결과 요약")
SU = hc(27)
SU = rep(SU, '''print("RUN:", RUN_NAME, "| epoch별 검증 정확도:", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})''',
         '''print("RUN:", RUN_NAME, "| 레시피:", RECIPE, "| epoch별 검증 정확도(학습 해상도):", {e: round(a * 100, 2) for e, a in state["epoch_acc"].items()})
print(f"모델 {MODEL_ID} | 정밀도 {PRECISION} | 학습 해상도 {TRAIN_PRESET} | 유효 배치 {STEP_SAMPLES} | LR {LR:g} | 비전 LoRA {VISION_LORA} | OCR {USE_OCR} | "
      f"보조 학습 {'끔' if not AUX_GEN_CSV else os.path.basename(AUX_GEN_CSV)} | 이어학습 {'없음' if not INIT_ADAPTER else INIT_ADAPTER} | 전체 데이터 {TRAIN_ON_FULL}")
if "summary" in globals(): print("최종 검증:", summary)''')
code(SU)

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    s = c["source"]; lines = s.split("\n"); c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 구문 점검 (!줄은 pass로)
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("\n".join(re.sub(r"^(\s*)!.*$", r"\1pass", l) for l in "".join(c["source"]).split("\n")))
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
