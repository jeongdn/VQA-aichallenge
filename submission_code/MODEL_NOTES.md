# 모델별 처리 정리 (MODEL_NOTES)

모든 모델은 같은 방식으로 답합니다: **답 자리 한 토큰의 로짓**에서 a·b·c·d(대소문자·앞 공백 변형 포함) 확률을 모아 네 보기 확률로 정규화합니다.
그래서 모델마다 "프롬프트가 끝난 바로 다음 토큰이 답 글자가 되도록" 맞추는 처리가 필요하고, 그 처리를 여기 모았습니다.
네 글자 확률 합(`letter_mass`)이 1에 가까우면 답 자리가 맞은 것입니다. 모든 노트북이 이 값을 기록하고, 낮으면 멈춥니다.

## 1. 공통 (01_train_vlm.ipynb)
| 항목 | 내용 | 코드 위치 |
|---|---|---|
| 프롬프트 | system: "Answer using exactly one letter among a, b, c, or d. No explanation." / user: 이미지 → 질문 → `(a)~(d)` 보기 → "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요." | 6번 셀 `build_messages` |
| 생각 모드 | 템플릿 변수 `enable_thinking=False` (Qwen·Gemma) | 7-1번 셀 `TEMPLATE_KW` |
| system 형식 판별 | 목록형 system 내용이 풀리지 않고 글자 그대로 찍히면 문자열로(`SYSTEM_AS_STRING`), 그래도 안 되면 지시문을 user 메시지 앞에 붙임 | 7-1번 셀 |
| 학습 손실 | 답 자리 로짓의 전체 어휘 교차 엔트로피, 목표는 정답 소문자 토큰 하나 | 9번 셀 `train_micro_step` |
| 채점 | `logsumexp`로 변형 토큰 확률을 합친 뒤 네 보기 softmax | 8번 셀 `score_rows` |
| 4bit | `USE_4BIT="auto"`: 20B 이상이면 NF4(이중 양자화, bf16 계산). **비전 타워·투영층·lm_head는 양자화하지 않음** (`llm_int8_skip_modules`) | 7-1번 셀 |
| LoRA 대상 | 언어 모델의 `q/k/v/o_proj`, `gate/up/down_proj`, Qwen3.5 선형 어텐션의 `in_proj_qkv`·`in_proj_z`·`out_proj` (r16 · α32 · dropout 0.05) | 7-2번 셀 `LORA_TARGETS` |
| 비전 LoRA | `VISION_LORA=True`면 비전 블록·merger의 `nn.Linear`(패치 임베딩 제외)를 이름으로 골라 추가, 학습률 `LR x 0.2`. 첫 스텝 B 그래디언트 > 0, 학습 후 B 가중치 > 0을 확인 | 7-2 · 9 · 10번 셀 |
| 입력-grad 훅 | PEFT가 거는 "입력 임베딩 grad 요구" 훅이 transformers 5.x에서 비전 타워까지 역전파 그래프에 넣음 → 비재진입 체크포인팅에서는 불필요하므로 제거 | 7-2번 셀 |
| NVRTC | 일부 Colab 이미지에서 GPU 정수 `prod`가 런타임 컴파일 라이브러리를 못 찾음 → 라이브러리를 직접 불러오고, 안 되면 정수 `prod`만 CPU로 (결과 동일) | 4번 셀 `_fix_nvrtc` |

## 2. 계열별
### Qwen3.5 · Qwen3.6(MoE) · Qwen3.8 (model_type `qwen3_5`, `qwen3_5_moe`)
- 선형 어텐션(Gated DeltaNet) 층이 있어 `flash-linear-attention==0.5.2`의 Triton 커널이 필요합니다. 없으면 순수 PyTorch 구현으로 돌아 OOM·저속이라 설치 셀이 자동으로 설치하고, 7-2번 셀이 커널 사용 여부를 출력합니다.
- 해상도는 픽셀 범위(`min_pixels`, `max_pixels`)입니다. native = 20만~100만 픽셀, up2.0 = 270만~340만 픽셀.
- Qwen3.6-35B-A3B(MoE)도 같은 코드로 학습합니다(bf16, G4 96GB).

### Gemma 4 (31B, 26B-A4B)
- 해상도는 이미지당 소프트 토큰 수(`max_soft_tokens`: 280 ≈ 원본, 1120 ≈ 1.9배). 프로세서에 샘플별 이미지 **목록**을 넘겨야 해서 `[[im] for im in images]`로 감쌉니다.
- HF 라이선스 동의 + 토큰(`HF_TOKEN`)이 필요합니다.
- 학습 해상도 up2.0에서 가장 좋아, 31B 레시피는 모두 up2.0 학습(유효 배치 16 = 1 × 16)입니다.

### InternVL3.5-38B (HF 형식)
- 해상도는 448px 타일 개수(`crop_to_patches=True`, `max_patches`: native 6 · up2.0 12, 타일당 이미지 토큰 256 + 전체 축소본 1타일).
- 비전은 `vision_tower`, 투영은 `multi_modal_projector` 경로입니다.

### HyperCLOVA X SEED Think 32B — 출력 제한용 처리
이 모델은 그대로 두면 답 자리에서 a~d가 아니라 생각 블록을 쓰기 시작합니다. 세 가지를 맞춰 답 자리를 고정했습니다.
1. **transformers 개발 버전**: 비전 모델(HyperCLOVA X + Qwen2.5-VL 비전 인코더)이 정식 5.17에 없어 검증한 커밋 `98d39824ed30e684e5122d04a2d9564efffc4965`로 고정 설치합니다(2번 셀이 MODEL_ID를 보고 자동 선택). 계열을 바꿀 때는 세션을 새로 시작합니다.
2. **생각 모드 끄기**: 이 템플릿은 변수 이름이 `thinking`이라 `TEMPLATE_KW = {"thinking": False}`를 넘깁니다(`enable_thinking`이 아님).
3. **빈 생각 블록 닫기 (`CLOSE_THINK`)**: 생각을 꺼도 생성 프롬프트가 `<think>\n`으로 끝납니다. 끝이 이렇게 열려 있으면 `\n</think>\n\n`을 붙여 닫아서, 다음 토큰이 곧 답 글자가 되게 합니다(`_fix_prompt`, 학습·검증·test 모두 같은 입력).
- 확인: 7-1번 셀 `프롬프트 미리보기(끝)`이 `<think>\n\n</think>\n\n`으로 끝나야 하고, 첫 학습 로그의 loss가 1보다 한참 작아야 합니다(24 근처면 답 자리가 틀린 것 → 즉시 중단).
- 이 처리로 zero-shot 300문항 95.67% · letter_mass 1.000을 확인한 뒤 학습했습니다(학습 후 검증 97.0%).
- 커스텀 코드 모델용 대비(`TRUST_REMOTE_CODE`): 이미지-텍스트 자동 클래스로 안 되면 `AutoModelForCausalLM`, sdpa 미지원이면 eager, 언어 모델 경로 이름이 다르면 비전·투영 모듈을 뺀 선형층 정규식으로 LoRA.

## 3. Flash-Next 학습 (02_runpod_flashnext_bf16.ipynb, RunPod B300 1장)
- **공통 보정 (NaN 방지)**: 배치 안 짧은 문항의 패딩 자리 질의는 볼 수 있는 키가 없습니다(마스크 행이 전부 가려짐). B300(CUDA 13)의 cuDNN 어텐션은 이런 행의 역전파에서 NaN을 냅니다. 학습 중에만 이 행을 "모든 키를 보게" 풀어 넘깁니다. 진짜 토큰의 행은 그대로라 손실·기울기가 같습니다(작은 모델 3종 대조 차이 0). 평가·예측은 바꾸지 않습니다.
- **QSA 인덱서 벡터화**: transformers 5.17의 희소 어텐션 토큰 선택이 질의 토큰마다 파이썬 루프라 문항당 수십 초 → 같은 계산을 한 번에. 선택 마스크는 원래 구현과 같습니다(작은 모델 대조).
- **n-gram 임베딩 표(95.4GiB)**: 128조각을 한꺼번에 이어 붙이지 않고 하나씩 복사해 RAM 사용을 줄이고, GPU 1장이면 이 표만 CPU RAM에 둡니다(순전파마다 조회 한 번이라 속도 영향 작음).
- 학습·평가 방식(분할·프롬프트·손실·선지 셔플·LoRA·up2.0 평가)은 01번과 같고, 결과 파일 이름도 같습니다.

## 4. zero-shot (03a · 03b, vLLM 0.30.0)
- 프롬프트·분할·해상도(up2.0)·글자 변형 합산은 학습 모델과 같습니다. 답 자리 **상위 64개 로그확률**에서 a~d를 모으고, 없는 글자는 0(`n_missing` 기록).
- 이미지 크기를 transformers 5.17 `smart_resize`와 같게 미리 맞춰 넘깁니다(검증 3장에서 두 경로의 픽셀·격자가 같은지 확인, 다르면 미리 맞추기를 끔).
- Qwen3.5-122B-A10B는 GPTQ-Int4 사전 양자화 체크포인트(`moe_wna16`)를 씁니다(직접 양자화하지 않음).
- **B300(sm_103)**: pod의 nvcc가 `compute_103a`를 몰라 FlashInfer 즉석 컴파일이 멈춥니다 → `attention_config={"backend": "TRITON_ATTN"}`, `moe_backend="triton"`, `enable_flashinfer_autotune=False`. `pandas`·`nvidia-cutlass`는 없으면 설치합니다.
- **Colab G4 (Blackwell SM12x)**: 선형 어텐션 커널은 `gdn_prefill_backend="triton"`, `VLLM_USE_FLASHINFER_SAMPLER=0`(채점은 그리디 1토큰이라 무관). vLLM이 가중치를 받을 때 진행 막대를 숨겨서 `snapshot_download`로 먼저 받습니다.
- 결과는 학습 모델과 같은 이름(`preds_final_ep1__up2.0__1000.csv` 등)으로 저장하고, `train_log.json`에 `ZERO_SHOT: true`를 남깁니다.

## 5. 판정 모델 (04_runpod_xcheck_any.ipynb)
- 대상: 기준 모델과 보조 모델의 답이 갈린 문항만(06번 6-1번 셀이 정답을 보지 않고 목록을 만듦).
- 프롬프트: 이미지 + 질문 + 보기 + 두 후보 답(모델 A/B = 기준 답 / 보조 모델 최다 대안, 자리는 문항 id 해시로 무작위) + 교차 검증 지시(근거의 모순 확인 → 흔한 착각 배제 → 단계별 결론, 마지막 줄 `정답: (x)`). 문구는 결과를 보기 전에 고정했습니다.
- 생성: 문항당 표본 5개, temperature 0.7 · top_p 0.8 · top_k 20, 최대 1,200토큰, 생각 모드 끔. 마지막 `정답:`을 읽고, 읽은 표본이 3개 미만이거나 최다 표가 동점이면 기권.
- 저장: 목록 밖 문항은 기준 답(표 없음)으로 채운 학습 모델과 같은 형식 + `judged`·`vote`·`n_valid` 열, `train_log.json`에 `XCHECK: true`, `XCHECK_ANCHOR`.
- **HyperCLOVA X를 판정 모델로 쓸 때 (vLLM 0.30)**
  - HF 저장소 main이 transformers 네이티브 형식(`hyperclovax_vision_v2`)으로 바뀌어, transformers 5.17과 vLLM 0.30이 설정 파일을 읽지 못합니다.
  - 그래서 vLLM 0.30 자체 시험과 같은 원격 코드판 커밋 `a6cdfd3464d1b767259cad23e164eaf39d3e3960`(`trust_remote_code=True`)으로 고정합니다. `MODEL_PRESETS`의 `REVISION`이 받기·로드·기록에 모두 같은 커밋으로 들어갑니다.
  - 첫 공개 커밋(`f5e4236`)은 쓰지 않습니다. 원격 코드가 `hyperclovax` 설정을 `exist_ok` 없이 등록하는데, 5.17에는 이 이름이 이미 있어서 오류가 납니다.
  - 이 커밋의 템플릿은 생각 끄기 변수가 `skip_reasoning`입니다. 그래서 `{"skip_reasoning": True}`를 넘기고, 그래도 `<think>`로 열린 채 끝나면 닫습니다.
  - 5번 셀이 두 가지를 확인한 뒤에 진행합니다: 이미지 자리(`<|IMAGE_PAD|>`)가 1개인지, 프롬프트가 닫힌 생각 블록(`</think>\n\n`)으로 끝나는지.

## 6. 쓰지 않은 모델 (참고)
- EXAONE 4.5 33B(zero-shot 85.7%), Muse Glimmer 30B(답 채널 구조가 달라 별도 처리 필요), GLM-5.3-Flash(92.0%), A.X-4.0-VL-Light(93.0%) — 괄호는 검증 앞 300문항 zero-shot: 한국어 글자 읽기가 약하거나 조합에 보탬이 없어 제출 조합에 넣지 않았습니다.
