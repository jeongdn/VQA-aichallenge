# E19b (제안 번호) — RunPod: Qwen3.8-Flash-Next(B300 1장) · GLM-5.3-Flash(B300 2장, FP8) LoRA (2026-09-27 밤)

## 결정
- 122B 대신 **Qwen3.8-Flash-Next**를 골랐다. 두 번째 새 멤버로는 같은 Qwen 계열인 122B 대신 **GLM-5.3-Flash**를 골랐다(앙상블 다양성).
- GLM-5.3(753B, `glm_moe_dsa`)은 텍스트 전용이라 제외했다.
- GLM-5.3-Flash 사양: 320B(활성 18B), GLM-5 계열 첫 멀티모달, MIT 라이선스, 공식 가중치 FP8(62파일 약 331GB), 공식 언어 en·zh.
- 레시피는 두 모델 모두 stage2와 같다: 분할 md5, baseline_v1 프롬프트, LoRA r16/α32, native 학습 1 epoch, 유효 배치 16, up2.0 평가, zero-shot 300문항 게이트 93%.

## 노트북 (한 빌더 `build_runpod.py`, 0번 셀 `MODEL_ID`로 전환)
| 파일 | 모델 | 실행 이름 | pod |
|---|---|---|---|
| `runpod_flashnext_bf16.ipynb` | Qwen/Qwen3.8-Flash-Next | `q38flash_r16_native_bf16` | B300 1장 (RAM 188GB) |
| `runpod_glm53flash_fp8.ipynb` | zai-org/GLM-5.3-Flash | `glm53flash_r16_native_fp8w` | B300 2장 (RAM 1006GB) |
| (같은 노트북) | Qwen/Qwen3.5-122B-A10B | `q35_122b_a10b_r16_native_bf16` | B300 1장 |

- `runpod_bridge_colab.ipynb`의 B 단계가 `RUN_NAME="auto"`로 바뀌었다. 저장소 `vqa_runs/` 아래의 실행을 전부 Drive로 받는다.
- 0번 셀
  - RAM은 cgroup 한도 기준으로 보고, 다운로드로 쌓인 파일 캐시는 사용량에서 뺀다.
  - 드라이버·CUDA 기준을 GPU별로 나눴다: B300 575/12.9, B200 570/12.8, H100·H200 525/12.1. 그래서 H200 pod(CUDA 12.8)도 된다.
  - 필요 디스크는 이미 받은 가중치를 빼고 계산한다.
- 2번 셀: 필요 CUDA를 GPU 능력치로 정한다. 드라이버가 575 미만이면 cu128 휠을 쓴다. import 점검에 `qwen4_exp`·`glm5_next`·FP8 모듈을 넣었다.
- 7-0번 셀: 공통 보정(`sdpa_patch.py`, 아래 "첫 실제 실행" 절) + Flash-Next 보정 + GLM 보정.
- 10번 셀: 기울기 NaN 안전장치(아래 절).
- 가이드: 별도 문서 "RunPod Flash-Next 학습 가이드"(이 저장소에는 없음)(9절 GLM-5.3-Flash, "문제가 생기면" 표).

## Flash-Next — transformers 5.17 그대로는 안 되는 두 곳 (`flash_patch.py`, 7-0번 셀)
- **크기**
  - 로드되는 파라미터는 177.4B(MTP 제외), bf16 330.4GiB다.
  - 그중 n-gram 표가 51.2B(95.4GiB)다. 2번째 층 PLE 하나이고, 체크포인트에는 128조각으로 들어 있다.
  - 나머지는 235.1GiB다.
- **QSA 인덱서**
  - 원래 코드는 질의 토큰마다 파이썬 루프를 돈다. 3천여 토큰이면 문항당 수십 초라 학습이 불가능하다.
  - 벡터화 버전으로 바꿨다. 작은 모델 대조 결과:
    - 동점을 없앤 점수에서는 선택 마스크가 완전히 같다.
    - 실제 relu에서는 점수 0인 동점의 순서만 다르다(742행 중 13행). 고르는 토큰 수는 같다.
    - 모델 로짓 차이는 0이다.
    - CPU S=512에서 35배 빠르다.
- **n-gram 표**
  - transformers는 128조각을 CPU에서 한꺼번에 cat하면서 RAM을 약 190GiB 쓴다.
  - 조각은 `key_mapping`으로 불러오기에서 빼고, 빈 표(95GiB 무작위 초기화는 생략)에 조각별로 복사한다. 값은 기본 불러오기와 같다.
  - GPU 1장이면 표를 CPU RAM에 둔다. device_map에서 이 표만 빼고 나머지를 GPU0 하나에 올리므로 accelerate 훅이 붙지 않는다.
  - GPU 2장 이상이면 `device_map="auto"`로 표까지 GPU에 올린다.

## GLM-5.3-Flash — FP8 가중치 그대로 학습 (`glm_patch.py` + `glm_nb_edits.py`)
- **구조**
  - 45층: KDA 선형 어텐션 34층 + DSA(MLA + 희소 인덱서) 11층
  - 전문가 288개(top-8) + 공유 전문가 1개, 처음 3층은 dense
  - mHC, 비전 24층(patch 14 · merge 2)
  - 로드 313.9B. bf16이면 584.7GiB라 B300 2장(536GiB)·B200 3장(537GiB)에 안 들어간다.
- **FP8 상주**
  - 공식 FP8을 그대로 GPU 2장에 올린다(약 305GiB).
  - FP8Linear·FP8Experts의 forward는 "블록 스케일로 bf16 복원 → F.linear / bf16 grouped_mm"으로 바꿨다. 복원 계산은 transformers `Fp8Dequantize`와 같다.
  - transformers의 FP8 층은 추론 커널이라 역전파가 없고, 활성값까지 FP8로 양자화한다.
  - 작은 모델 대조(`dequantize=True`로 복원한 모델과 비교): 로짓·손실·LoRA 기울기 차이 0.
- **LoRA**
  - 기존 정규식에 `q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj`를 더했다.
  - PEFT가 LoRA를 기반 층 dtype(FP8)으로 바꿨다가 float32로 되돌린다. 그래서 FP8 층 위 lora_A만 kaiming 규칙으로 다시 초기화한다(시드 고정).
- **입력**
  - 템플릿은 `[gMASK]<sop>`로 시작하고, 생성 프롬프트가 `<|assistant|><think>`로 끝난다. 생각을 끄는 옵션은 없어서 `reasoning_effort="low"`를 준다.
  - 8번 셀이 답 자리 후보 3개를 검증 16문항으로 채점해, letter_mass가 가장 큰 것을 고른다. 후보: `</think>`, `</think>\n`, `<think>` 떼고 `</think>`.
  - 토크나이저가 `[gMASK]<sop>`를 또 붙이므로 `add_special_tokens=False`로 넣는다.
  - GLM 이미지 프로세서는 이미지를 키우지 않고 패딩만 한다. 그래서 `fit_pixels`로 28 배수 크기에 맞추고 `min/max_image_tokens`를 넓게 둔다. 토큰 수 = h·w/784(up2.0 약 4.3천).
- **확인한 transformers 구현**: 인덱서는 벡터화돼 있고(`@torch.no_grad`), 전문가는 `use_experts_implementation`(grouped_mm)이다. KDA는 fla 0.5.2의 `ops.kda`를 쓴다. DSA 마스크는 top-k 인덱스로 만든 bool 마스크이고 `ALL_ATTENTION_FUNCTIONS`의 sdpa를 거친다(패딩 질의 행은 전부 False → 공통 보정 대상).
- **무해한 차이**: KDA `forget_gate.f_a/f_b`는 FP8Linear로 만들어지지만 체크포인트 가중치가 bf16이다. 이름 정규화 때문이고, 원래 forward도 보통 행렬곱이다.

## pod 비교 (직접 확인한 사양)
| pod | VRAM | RAM | Flash-Next | GLM-5.3-Flash |
|---|---|---|---|---|
| B300 1장 | 288GB | 188GB | ✅ (n-gram 표는 RAM) | ✗ |
| B300 2장 | 576GB | 1006GB | ✅ | ✅ FP8 (권장, $15.78/시간) |
| B200 3장 | 540GB | 849GB | — | ✅ FP8 |
| H200 SXM 6장 | 846GB | 1506GB | — | ✅ (CUDA 12.8, 느리고 비쌈) |

## 첫 실제 실행 (Flash-Next, B300 1장, 9/27 밤) — 학습 NaN의 원인과 수정
- **DRY_RUN 증상**
  - zero-shot은 정상이었다: 24문항 100%, letter_mass 0.993.
  - 학습은 첫 스텝부터 깨졌다: step 1 손실 0.0461에 gnorm nan. LoRA가 NaN으로 바뀌어 2·3스텝 손실도 nan이 됐고, 빠른 검증에서 "로짓에 NaN/Inf"로 멈췄다.
  - 스텝 시간은 약 10초였다.
- **진단** (같은 커널, LoRA를 초기값으로 되돌린 뒤 역전파 한 번)
  | 조건 | NaN 기울기 | 손실 |
  |---|---|---|
  | ① 2문항 (왼쪽 패딩 있음) | 588/600 (가장 위 층 47 `q_proj`) | 0.0397 |
  | ② 1문항씩 (패딩 없음) | 0/600 | 0.0488 · 0.0173 |
  | ③ 2문항 + 전부 가려진 행 풀기 | 0/600 | 0.0397 (①과 같음) |
  | ④ 2문항 + cuDNN SDPA 끔 | 0/600 | 0.0437 |
- **원인**
  - 왼쪽 패딩 자리의 질의 행은 볼 키가 하나도 없다(마스크 행이 전부 False).
  - torch cu130의 cuDNN SDPA는 이런 행을 순전파에서는 멀쩡히 넘기지만, 역전파에서 NaN을 낸다.
  - transformers 4.x는 이런 행을 미리 풀어 주는 처리(`_unmask_unattended`)가 있었지만, 5.x에서 빠졌다(`allow_torch_fix`는 효과 없음).
  - GLM(DSA 마스크)·122B도 같은 함수를 거치므로 같은 위험이 있다.
- **수정** (세 노트북 공통)
  - `sdpa_patch.py` → 7-0번 셀 "공통 보정". `ALL_ATTENTION_FUNCTIONS["sdpa"]`를 감싸, **기울기 계산 중일 때만** 전부 가려진 행을 모두 True로 바꾼다.
    - float 마스크는 전부 min이거나 -inf인 행을 0으로 바꾼다.
    - 평가·예측(기울기 없음)은 마스크를 그대로 넘긴다.
  - 10번 셀 안전장치
    - gnorm이 유한하지 않으면 cuDNN SDPA를 끄고 같은 스텝을 다시 계산한다.
    - 그래도 NaN이면 옵티마이저 스텝을 건너뛴다. 스케줄러는 진행하고, `state.history`에 `nan_skip`을 남긴다.
    - 3번이면 FloatingPointError로 멈춘다.
- **시험** (CPU, 모두 통과)
  - `test_sdpa_patch.py`: 작은 Flash-Next·GLM(FP8+보정)·122B, 왼쪽 패딩 + 이미지 + LoRA + 체크포인팅.
    - `cudnn_nan_emu.py`로 cuDNN 버그를 흉내 내면 NaN이 재현된다: 38/50·46/58·38/50. 1문항이면 NaN이 없다.
    - 보정을 넣으면 NaN 0이고, 손실·LoRA 기울기가 보통 SDPA와 차이 0이다.
    - 평가 로짓은 보정 전후가 비트 단위로 같다.
  - `run_runpod2.py … nanguard`: 버그 흉내 + 보정이면 DRY_RUN이 통과한다. 보정을 빼면 3스텝을 건너뛰고 멈추며, LoRA는 유한한 채로 남는다.
  - 실제 Jupyter 커널(`EMU_CUDNN=1 run_kernel_runpod2.py`): Flash-Next DRY_RUN → 본 실행 → 올리기 통과.
  - 기존 흐름 시험(flash·glm·122b의 env·install·flow)도 다시 통과했다.
- **이후 진행**
  - Flash-Next는 DRY_RUN을 다시 하지 않고 새 노트북(`DRY_RUN=False`)으로 본 실행에 들어갔다.
  - zero-shot 300문항은 **96.67%**, letter_mass 0.990이었다. 같은 300문항에서 Gemma4-31B zero-shot은 97.67%였다.
  - 첫 학습 로그(step 10/357)는 loss 0.1448, train_acc 95.0%, gnorm 0.64, 9.7s/step이었다. NaN 없이 학습이 돌았다.
  - DRY_RUN의 100%는 앞 24문항만 본 값이다. 실력 96.7%에서 24/24가 나올 확률은 약 44%다.
  - GLM-5.3-Flash는 실제 GPU가 처음이라 DRY_RUN부터 한다. 한때 B300이 동나 B200 3장을 검토했지만, 결국 **B300 2장**을 잡았다.
- **볼륨 디스크가 가득 참** (본 실행 25스텝 체크포인트 무렵)
  - df 기준 676G/750G였다. `/workspace/hf`가 336G(가이드대로 터미널에서 `hf download`), `/workspace/.cache`가 337G였다.
  - 가중치가 두 벌이었다. pod 터미널(로그인 셸)에는 HF 캐시 경로가 미리 정해져 있다. 그래서 가이드의 `export HF_HOME=/workspace/hf`만으로는 `hf download`가 `/workspace/.cache/…`로 갔다.
  - Jupyter 커널에는 그 설정이 없다. 그래서 노트북은 `/workspace/hf`를 보고 한 벌을 더 받았다.
  - GLM pod(B300 2장, 볼륨 644GB)에서도 같은 일이 생겼다. 터미널로 받은 328GB가 숨은 폴더에 있었다. 0번 셀은 '받은 양 0/331GB'로 보고, 공간 부족(남은 316GB < 필요 346GB)으로 멈췄다.
    - 다시 받지 말고 `models--zai-org--GLM-5.3-Flash` 폴더를 `/workspace/hf/hub/`로 `mv`하도록 안내했다. 같은 디스크라 즉시 끝나고, 스냅샷의 상대 심볼릭 링크도 그대로 유효하다.
  - 볼륨을 750GB로 늘렸고, 그 과정에서 pod가 재시작됐다. 25스텝 체크포인트는 정상이라 거기서부터 이어서 학습한다.
  - 수정: 0번 셀이 `HF_HOME`·`HF_HUB_CACHE`·`HF_XET_CACHE`를 `/workspace/hf`로 **강제**하고 `HUGGINGFACE_HUB_CACHE`를 지운다. 다른 캐시에 같은 모델이 있으면 크기와 함께 알려만 준다.
  - 가이드의 터미널 명령도 `export HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf/hub`로 바꿨다. 터미널의 미리 정해진 경로를 덮어쓰기 위해서다.
  - 시험: 0번 셀 시나리오에 "HF_HOME·HUGGINGFACE_HUB_CACHE가 미리 정해진 pod + 그 캐시에 같은 모델 337GB"를 더했다. flash·glm·122b 모두 통과.
  - GLM은 결국 B300 2장 pod를 잡았다. 이 노트북으로 한다.
- **GLM-5.3-Flash 결과: zero-shot 게이트에서 멈춤 → 접음** (9/28 0시 반)
  - DRY_RUN 학습 스텝을 건너뛰고 바로 본 실행에 들어갔다. DRY_RUN zero-shot은 24문항 95.83%였다.
  - 본 실행 zero-shot(검증 앞 300문항, up2.0)은 **92.00%**, letter_mass 1.000이었다. 게이트 93% 미만이라 학습 전에 멈췄다.
  - 답 자리 형식은 완벽하므로 고장이 아니라 실력으로 판단했다. 공식 지원 언어가 en·zh라 한국어 글자를 잘 못 읽는다.
  - 같은 300문항 zero-shot 비교: Gemma4-31B 97.67 · Flash-Next 96.67 · Q35 96.33 · HCX 95.67 · InternVL 94.33 · **GLM 92.00**.
  - 학습 이득 +0.9~1.4%p를 더해도 93~93.5%로 예상된다. 조합 멤버(96.8~97.6)보다 3%p 넘게 낮고, 96% 미만 멤버는 ens_search에서 보탬이 없었다.
  - 권고: 학습하지 않는다. GLM pod(B300 2장)는 Terminate한다.
  - 게이트를 낮춰 학습하려면: 커널에서 `ZERO_SHOT_MIN_ACC = 0.90` 후 10번 셀부터 끝까지 실행한다. zero-shot은 저장돼 있어 재사용된다.
- **정상 출력 (걱정하지 않아도 되는 것)**
  - 7번 셀의 LOAD REPORT에 `ngram_skipped_shard_0…127 | UNEXPECTED`와 `ngram_embedding.weight | MISSING`이 뜬다.
    - n-gram 표를 일부러 빼고 불러온 뒤 조각별로 채우기 때문이다.
    - 다른 가중치가 빠졌으면 셀이 assert로 멈춘다.
  - 10번 셀의 `causal_conv1d … falling back`: 작은 합성곱 하나만 PyTorch 구현으로 돈다. 결과는 같고, DRY_RUN에서 잰 스텝 시간에 이미 들어가 있다.

## 점검 (CPU, 작은 모델 + 가짜 HF 허브)
- 흐름 시험(`run_runpod2.py flash|glm|122b`) 전부 통과. 순서:
  1. 브리지 A
  2. DRY_RUN
  3. zero-shot 게이트
  4. 중단 → 이어하기
  5. 올리기
  6. 브리지 B
  7. ens_search가 새 멤버로 찾음
- GLM은 시험에서도 FP8 그대로 불러와 보정 경로를 탔다.
- 0번 셀 시나리오 통과
  - Flash-Next 12개: RAM 188·120·100GB, cgroup v1, B300 2장, B200 2장, H100, 드라이버, 볼륨(이미 받은 가중치 포함), 컨테이너
  - GLM 7개: B300 2장, H200 6장, B200 3장, B300 1장, B200 2장, 드라이버, 볼륨
- 2번 셀 시나리오 6개 통과(H200 cu128 포함)
- 실제 Jupyter 커널: Flash-Next DRY_RUN → 본 실행 → 올리기 통과(cuDNN 버그 흉내를 켠 상태 포함)
- NaN 시험: `test_sdpa_patch.py`, `run_runpod2.py … nanguard`(위 절)
- 시험하지 못한 것: 실제 GPU에서의 GLM 속도·메모리, KDA fla 커널의 B300 역전파, 실제 GLM 체크포인트의 이름 정규화. 모두 GLM DRY_RUN에서 드러난다.
