# 새 모델 후보 조사 + stage2_newfam 노트북 (2026-09-27)

> 목적: 앙상블에 넣을, 기존 멤버와 다른 문항에서 틀리는 강한 모델 찾기.
> 기존 계열: Qwen3.5-27B · Qwen3.8-27B · Qwen3.5-9B · Qwen3-VL-8B · Gemma4-31B · HCX-SEED-Think-32B · InternVL3.5-38B(95.7, 제외).
> 조건: HF 공개 가중치 · transformers 5.17에서 기본 지원 · G4 96GB / A100 80GB에서 bf16 LoRA 가능.

## 1. 후보 순위
| 순위 | 모델 | 크기 / GPU | 근거 | 기존과의 차이 | 노트북 |
|---|---|---|---|---|---|
| 1 | `meta-models/Muse-Glimmer-30B` (2026-08, Apache-2.0, 게이트 없음) | dense 약 30B (비전 1.8B 포함), bf16 60GB → A100·G4 | 같은 표: CharXiv-R 78.8 / OmniDocBench 75.8 / MMMU-Pro 74 (Gemma4-31B 77.7/72.5/73, Qwen3.6-27B 78.4/77.8/75) | 완전히 새 계열(메타 비전 인코더 + 새 토크나이저). 한국어는 공식 평가 없음 → zero-shot 먼저 | `stage2_newfam` (전용 처리 추가) |
| 2 | `Qwen/Qwen3-VL-32B-Instruct` 검증용 학습 | 33B, bf16 66GB | OCRBench v2 68.4 (EXAONE 표 1위, Qwen3.5-27B 67.3). KRETA 90.3 (Qwen3.5-27B 96.5) — 단 8B 형제가 LoRA 후 우리 검증 96.5 | Qwen3.5와 다른 세대. 지금은 `_full`만 있어 검증 확률이 없음 | 기존 stage2 그대로 |
| 3 | `Qwen/Qwen3.6-35B-A3B` (2026-04) | MoE 35B/활성 3B, bf16 70GB → G4 전용 | CC-OCR 81.9 · OmniDocBench 89.9 (Qwen3.5-27B 81.0 · 88.9). 3.5-35B-A3B OCRBench 91.0 (27B 89.4) | Qwen3.5 계보 → 겹침 가능성 큼. 전문가 층은 3D 파라미터라 LoRA는 어텐션·공유 전문가에만 붙음 | 그대로 (qwen3_5_moe) |
| 4 | `LGAI-EXAONE/EXAONE-4.5-33B` | 33B, bf16 66GB | Qwen3.5-27B보다 전부 낮음: OCRBench v2 63.2 vs 67.3, KRETA 91.9 vs 96.5, K-Viscuit 80.1 vs 84.0, KMMMU 42.7 vs 51.7 | LLM·비전 모두 새 계열 | 그대로 (Qwen2-VL식 프로세서). 라이선스 NC |
| 5 | `google/gemma-4-26B-A4B-it` | MoE 25.2B/활성 3.8B, 52GB | MMMU-Pro 73.8 (31B 76.9), OmniDocBench 편집거리 0.149 (31B 0.131) | G와 같은 계열 | 그대로 |
| 6 | `Qwen/Qwen3.6-27B` | 27B dense | OCRBench 89.4 (3.5-27B와 같음) | q35와 같은 구조, q38이 q35보다 낮았던 전례 | 그대로 |

### 제외
- 너무 큼: Qwen3.5-122B-A10B(OCRBench 92.1 1위지만 bf16 244GB, 전문가 층이 3D 파라미터라 bitsandbytes 4bit 불가), Qwen3.8-Flash-Next 180B, Step-3.7-Flash 198B, GLM-5.3-Flash 320B, Mistral-Small-4 119B, GLM-4.6V 106B, SKT A.X K2 688B, Inkling 975B.
- 커스텀 코드·구버전: Ovis2.6-30B-A3B (trust_remote_code, transformers 4.57).
- 비공개: Qwen4-Exp (transformers 5.17에 구조만 있음, 체크포인트 401).
- 작거나 약함: VARCO-VISION-2.0-14B, A.X-4.0-VL-Light 7B, HCX-SEED-Omni-8B, Gemma-4-12B, GLM-4.6V-Flash 9B, North-Micro-Vision 2.4B.
- 다른 축(참고): PaddleOCR-VL-1.6(0.9B, 한국어 포함 109개 언어, OmniDocBench v1.6 1위)으로 글자를 뽑아 프롬프트에 넣는 방법. 멤버 재학습이 필요하고 진짜 모델 실수가 6개뿐이라 이득 여지 작음.

## 2. `stage2_newfam.ipynb`
기존 `stage2_qwen35_27b_bf16.ipynb`(= src_stage2)와 학습·평가·제출·이어하기가 같고, 추가한 것은 다음과 같다.
- **Muse Glimmer 처리**
  - 이 모델의 프로세서는 이미지를 확대하지 않고 토큰 상한(4,096)만 둔다. 그래서 Qwen과 같은 픽셀 범위(`RES_PRESETS`)로 28 배수 크기에 직접 맞춘 뒤 넘긴다. up2.0 상한은 모델 한도 321만 픽셀로 제한.
  - 생성 프롬프트가 `<|start|>assistant`로 끝나고 다음 토큰에서 추론 채널과 답 채널이 갈린다. 그래서 `<|message|>`를 붙인 뒤의 글자를 채점·학습한다.
  - `reasoning_strength="low"`를 넘긴다.
  - 템플릿이 오늘 날짜를 넣으므로 `current_date`를 고정해 재현성을 지킨다(`TEMPLATE_DATE`).
- **학습 전 zero-shot 점검** (`ZERO_SHOT_N=300`, `ZERO_SHOT_ONLY=True` 기본)
  - LoRA가 0인 상태에서 검증 앞 300문항을 up2.0으로 채점한다.
  - Drive에 다른 실행의 `preds_zeroshot*.csv`(예: E5의 Q35)가 있으면 같은 문항 정확도를 함께 출력한다.
  - letter_mass가 0.9 미만이면 경고한다.
  - 참고 기록: Q35 제로샷 96.1 → 학습 97.2 (1,000문항) · Gemma4 97.67 → 98.33 (앞 300문항).
- **실행 순서**
  1. `ZERO_SHOT_ONLY=True`로 전체 실행
  2. `ZERO_SHOT_ONLY=False, DRY_RUN=True`로 전체 실행
  3. `DRY_RUN=False`로 본 학습 (1의 결과 재사용)
- **E14 호환:** 결과 파일 이름은 기존과 같다(`preds_final_ep1__up2.0__1000.csv`). zero-shot 파일은 E14 목록에 잡히지 않는다.
- **점검:** 실제 transformers 5.17 구조의 작은 무작위 모델(Muse Glimmer, EXAONE 4.5, 실제 프로세서)로 CPU에서 13개 시나리오를 돌려 모두 통과했다.
  - 시나리오: zero-shot만 · 이어서 학습 · DRY_RUN · 중간 중단 → 이어하기 · 토큰 상한 · zero-shot 없음.
  - 검사 항목: 프롬프트 끝, 고정 날짜, 이미지 토큰 수(독립 계산), zero-shot 확률 = 기본 모델 직접 채점(차이 1e-7), 제출 파일.
  - Qwen3.6-35B-A3B(qwen3_5_moe)·Gemma4 26B-A4B(MoE)는 모델 단위로 점검했다: 로드, LoRA 부착(비전 0), logits_to_keep 순전파·역전파.

## 출처
- https://huggingface.co/meta-models/Muse-Glimmer-30B · https://huggingface.co/blog/muse-glimmer
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B · https://huggingface.co/Qwen/Qwen3.6-27B · https://llm-stats.com/benchmarks/ocrbench
- https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B
- https://huggingface.co/google/gemma-4-26B-A4B-it · https://huggingface.co/google/gemma-4-12B-it
- https://codersera.com/blog/qwen-3-8-model-lineup-2026/ · https://huggingface.co/docs/transformers/model_doc/qwen4_exp
- https://huggingface.co/stepfun-ai/Step-3.7-Flash · https://huggingface.co/zai-org/GLM-5.3-Flash · https://huggingface.co/thinkingmachines/Inkling
- https://huggingface.co/AIDC-AI/Ovis2.6-30B-A3B · https://blog.roboflow.com/best-open-source-ocr-models/
- https://www.ddaily.co.kr/page/view/2026081217071457815 · https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct

## 3. zero-shot 점검 결과 (2026-09-27 02시, 검증 앞 300문항 · up2.0, 팀 실행)
| 실행 | zero-shot | letter_mass | 판단 |
|---|---|---|---|
| Qwen3.5-27B (E5 폴더, 기준) | 96.33% | ~1 | — |
| Qwen3.6-35B-A3B (`q36_35ba3b_r16_native_bf16`) | 96.33% | — | Q35와 같음 → 학습 후보 (G4) |
| HCX-32B | 95.67% | — | 학습 후 97.0 |
| Qwen3.8-Flash-Next IQ3_XXS | 95.33% | — | 학습 불가(180B) |
| InternVL3.5-38B | 94.33% | — | 학습 후 95.7 |
| A.X-4.0-VL-Light 8B (native) | 93.00% | — | — |
| EXAONE-4.5-33B | 85.67% (이전 dry 84.67·85.00) | 0.994 | **접음**: 측정은 정상. 학습 이득(+1.1~1.4%p)을 더해도 강한 풀 기준(~96%)에 한참 못 미침 |
| Muse-Glimmer-30B | 28.00% | 0.003 | **측정 무효**: 답 자리 첫 토큰에 글자를 거의 안 둠 → 생성 기반 zero-shot으로 다시 잼 (stage2_newfam 갱신) |
- zero-shot → 학습 후는 대략 +1.1~1.4%p였다 (Q35 96.33→97.2, HCX 95.67→97.0, IVL 94.33→95.7. zero-shot은 앞 300문항, 학습 후는 1,000문항 기준). 그래서 학습 후 97%를 바라보려면 zero-shot이 대략 95.5% 이상이어야 한다.
