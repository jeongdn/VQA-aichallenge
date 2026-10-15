# 기준 모델 교체 규칙: Flash-Next 기준 + 보조 모델 + 판정 모델 (2026-09-28 새벽)

## 방향
- 기준 모델 하나의 답을 두고, 기준 모델이 불확실하고 보조 모델들이 같은 다른 답을 가리킬 때만 뒤집는 방식을 시험한다.
  - 기준 모델: Flash-Next `q38flash_r16_native_bf16` (검증 96.9, 공개 0.97110)
- ea0e9f(공개 0.97736)는 따로 남겨 둔다. 이미 여러 모델이 들어가 있으므로 기준 모델로는 쓰지 않는다.
- 보조 모델 = 학습한 멤버(계열별 최고) + zero-shot 대형 모델.
- 판정 모델 = 교차 검증 프롬프트. 기준과 보조 모델의 답이 갈린 문항만 다시 따져 보게 한다.
  - 여러 모델을 쓰되 약한 모델은 넣지 않는다. 이유:
    - 대상이 가장 어려운 문항이다.
    - 프롬프트가 대안을 보여 주므로, 약한 모델 여럿이 같은 오답에 '합의'할 위험이 있다.
  - 판정 모델 목록과 규칙은 결과를 보기 전에 고정한다. 돌린 판정 모델은 모두 넣는다(골라 빼면 과적합).

## 판정 모델 후보 (zero-shot 앞 300문항 기준)
- 1순위 Gemma-4-31B-it (97.67): Qwen과 다른 계열
- 2순위 Qwen3.5-122B-A10B GPTQ (검증 1,000에서 96.0)
  - 09-28 06시: 가중치 로드 중 메모리 부족이 반복돼 Qwen3.8-27B로 바꿔 돌림
- 3순위 HCX-SEED-Think-32B (95.67): 네이버·한국어. vLLM 0.30에서 쓰는 법은 아래 'HCX 판정 수정'
- 선택: InternVL3.5-38B (94.33), Qwen3.5-27B·Qwen3.6-35B-A3B (96.33, 122B와 같은 계열)
- 제외
  - GLM-5.3-Flash (92.00, 한국어 약함)
  - GLM-4.6V (단독 성능이 낮음)
  - 9B 이하 소형

## 파일
- `runpod_zs_any.ipynb` (RunPod B300)
  - `MODEL_ID`만 바꿔 zero-shot으로 채점한다(vLLM 0.30, 글자 확률).
  - 기본 `VAL_N=1000`: 팀 공통 검증 1,000 + test 6,714.
  - 끝 셀에서 HF 비공개 저장소 `vqa_runs/zs_<모델>_up20`에 올린다. Colab 브리지 B 단계로 Drive에 받는다.
  - B300 오류 수정(09-28 새벽)
    - 증상: pod의 nvcc가 sm_103을 몰라 FlashInfer 즉석 컴파일에서 `Unsupported gpu architecture 'compute_103a'`로 멈췄다.
    - 수정: `attention_config={"backend": "TRITON_ATTN"}`, `enable_flashinfer_autotune=False`, `MOE_BACKEND="triton"`. 설정 이름은 vLLM 0.30.0 소스에서 확인했다.
    - pandas·nvidia-cutlass는 없으면 설치한다.
- `runpod_xcheck_any.ipynb` (RunPod B300, 판정 모델) · HCX용으로 MODEL_ID만 맞춘 복사본 `runpod_xcheck_hcx.ipynb`
  - HF 저장소에서 `xcheck/xcheck_targets.csv`(판정 목록)를 받는다.
  - 대상 문항만 E20과 같은 한국어 '교차 검증 프롬프트'로 묻는다.
    - 모델 A/B 자리 = {기준 답, 보조 모델 최다 대안}. 어느 쪽이 A인지는 문항 id 해시로 무작위다.
  - 생성 설정: 표본 5개, T 0.7, top_p 0.8, top_k 20, 최대 1200토큰, 생각 모드 끔.
  - 다수결로 이 판정 모델의 한 표를 정한다. 표본 3개 미만이거나 최다 표가 동점이면 기권.
  - E14 형식으로 저장한다(목록 밖은 기준 답 = 표 없음, `judged`/`vote`/`n_valid` 열 포함).
    - `train_log`: `XCHECK=True`, `MODEL_ID="xcheck:<모델>"`, `XCHECK_ANCHOR`, `TARGETS_TAG`, `REVISION`, `SYSTEM_AS_STRING`
  - `vqa_runs/xc_<모델>_up20`에 올린다.
  - 이어하기: 목록이 바뀌면 A/B 짝이 달라진 문항과 새 대상만 새로 판정한다.
  - 주의: `xc_` 실행은 anchor_rescue 전용이다. ens_search를 다시 돌릴 때는 빼야 한다.
  - 동시 실행: B300 한 장에서 노트북 두 개는 권하지 않는다(09-28 06시 질문).
    - vLLM이 시작할 때 GPU의 `GPU_MEM_UTIL`(0.92)만큼을 잡으므로, 하나가 돌고 있으면 두 번째는 시작하지 못한다.
    - 굳이 하려면 둘 다 시작 전에 0.45로 둬야 한다.
    - pod 메모리 여유도 작고, GPU를 나눠 쓰는 것이라 합쳐도 크게 빨라지지 않는다 → 순서대로 돌린다.
- `anchor_rescue.ipynb` (Colab CPU)
  - 뒤집기 규칙: 기준 확신 < CONF, 마진 < MARGIN, 표 AGREE개 이상이 같은 다른 보기를 고를 때만 뒤집는다.
  - 검증 절차: 5-fold 교차 적합(온도 보정은 fold마다), 부트스트랩 1,000번, 문항 유형별, 문턱값 일관성.
  - 채택 기준: 순이득 ≥ +10, 살림 ≥ 2×망침, 부트스트랩 ≥ 95%, 문턱값 일관.
  - zero-shot 실행(검증 ≥ 95.5)은 학습 멤버와 같은 계열이어도 넣는다.
  - 09-28 추가
    - 6-1번 셀: 판정 목록(보조 모델만으로 정함, 정답 안 봄)을 저장하고 HF에 올린다.
    - 판정 실행을 한 표씩 넣는다. 다른 기준으로 만든 판정은 뺀다.
    - AGREE 격자를 2~표 수(최대 8)로 넓혔다.
    - 조율 비교에서 판정 모델은 뺀다.
    - 판정 모델 성적표를 보인다.

## HCX 판정 수정 (09-28 06:40)
- 증상: 실제 pod의 9번 셀 `LLM(...)`에서 `ValidationError ... model type hyperclovax_vision_v2 but Transformers does not recognize this architecture`.
- 원인
  - HF 저장소 main이 transformers 네이티브 형식(`hyperclovax_vision_v2`)으로 바뀌었다. transformers 5.17·vLLM 0.30은 이 형식을 모른다.
  - vLLM v0.30.0 자체 시험(`tests/models/registry.py`)은 이 모델을 `trust_remote_code=True`, `revision="a6cdfd3464d1b767259cad23e164eaf39d3e3960"`(원격 코드판, 'quickfix' 커밋)로 고정해 쓴다. 그 시험의 transformers는 5.16.1이다.
  - 첫 공개 커밋 `f5e4236`은 쓰면 안 된다. 원격 코드가 `AutoConfig.register("hyperclovax")`를 `exist_ok` 없이 부르는데, 5.17에는 `hyperclovax`가 이미 있다.
  - 이전 노트북에서는 5번 셀 AutoProcessor가 main에서 토크나이저만 불러와서 조용히 지나갔다. 그래서 해상도 인자도 빠졌다(`mm_processor_kwargs {}`).
- 생각 모드 문제도 같이 발견: 이 커밋 템플릿의 생각 끄기 변수는 `skip_reasoning`이다. 생성 프롬프트는 `<|im_start|>assistant\n<think>\n`으로 끝난다. 이전 노트북은 `enable_thinking`만 찾아서 생각 블록이 열린 채 생성했을 것이다.
- 수정 (`experiments/xcheck/hcx_patch.py`를 빌더가 불러 씀)
  - `MODEL_PRESETS`의 HCX 항목에 `REVISION`을 고정한다. 5번 셀(프로세서)·8번 셀(`model_info`·`snapshot_download`)·9번 셀(`revision`·`tokenizer_revision`·`code_revision`)·기록에 모두 같은 커밋이 들어간다.
  - 고정을 지운 경우: 5번 셀이 main 로드 실패나 이미지 처리기 없는 프로세서를 보면 스스로 같은 커밋으로 고정한다.
  - `_think_kw`: `enable_thinking` → 없으면 `thinking`·`skip_reasoning` 중 템플릿에 있는 것. `_close_think`: 생각을 끄는데 `<think>`로 열린 채 끝나면 닫는다.
  - system 목록형 content가 글자 그대로 찍히면 문자열로 넣는다(HCX 템플릿은 목록형을 풀어서 해당 없음).
  - HCX 점검(모델 받기 전에 멈춤): `<|IMAGE_PAD|>` 1개, 프롬프트 끝 `</think>\n\n`.
- 점검 (CPU): HF에서 받은 HCX 템플릿을 실제 5.17 프로세서에 넣어 시험했다. 가짜 vLLM에는 HCX 이미지 자리 변환을 넣었다.
  - hcx: 커밋이 모든 곳에 들어가고, 생성 인자는 `{"skip_reasoning": True}`다. 프롬프트 130개 모두 생각 블록이 닫히고 이미지 자리가 1개다. system은 글자로 들어간다. E14·재현·올리기가 통과했고, 해상도 인자도 들어간다.
  - hcxfb: 고정을 지워도 5번 셀이 스스로 고정한다.
  - 기존 main·resume·listchange·nolist: 모두 통과.
- 남은 불확실: 실제 원격 코드(processing_vlm.py)와 5.17의 호환은 직접 못 봤다. vLLM 0.30 시험이 5.16.1로 통과하는 커밋이라 문제없을 것으로 본다. 첫 묶음 점검에서 확인한다.

## 실행 순서
1. B300: zero-shot 모델들 (`runpod_zs_any`)
2. Colab: 브리지 B → `anchor_rescue`
   - 판정 없이 1차 결과가 나오고, 판정 목록이 HF에 올라간다.
   - zero-shot을 다 넣은 뒤에 목록을 만들어야 한다. 보조 모델이 바뀌면 목록이 바뀌어 판정을 새 대상만큼 더 돌려야 한다.
3. B300: `runpod_xcheck_any`를 `MODEL_ID` Gemma → 122B(→ Qwen3.8-27B) → HCX 순서로, 매번 Restart & Run All (동시 실행 말고 순서대로)
4. Colab: 브리지 B → `anchor_rescue` 다시 (판정 모델 포함) → 제출 파일

## 점검 (CPU, 가짜 vLLM·가짜 HF)
- runpod_zs_any
  - main: B300 우회 설정이 LLM에 들어감
  - notest: test 없이 올리려 하면 멈춤
- runpod_xcheck_any
  - main
    - 대상만 요청(검증 80 + test 50)
    - A/B 무작위, 설계한 표 결과, 기권
    - 성적 줄 설계값, E14, train_log, 재현, 올리기
  - resume: 중단 후 이어하기, 중복 없음
  - listchange: 바뀐 5 + 새 3문항만 새로 판정
  - nolist: 모델 받기 전에 멈춤
  - hcx / hcxfb: 위 'HCX 판정 수정'
- anchor_rescue
  - nocal / cal / manual / zsfam / noanchor
  - judge: 판정 2명이 JF 20 살림·JH 3 망침, 설계값 72/11. 다른 기준의 판정은 뺌. 목록 바뀜 경고
  - integ(종단): anchor_rescue 목록 → 판정 노트북 → anchor_rescue가 한 표로 잡음, 태그 일치
- 실제 GPU에서 확인하지 못한 것
  - Triton 어텐션 속도
  - Gemma·HCX 템플릿의 생성 형식(첫 묶음 점검에서 확인)
  - HCX 원격 코드판의 실제 로드(위 '남은 불확실')

## 출처
- vLLM v0.30.0 tests/models/registry.py (HCX 커밋 고정): https://github.com/vllm-project/vllm/blob/v0.30.0/tests/models/registry.py
- vLLM PR #56335 (HCX 시험 모델 커밋 고정): https://github.com/vllm-project/vllm/pull/56335
- vLLM issue #56397 (hyperclovax_vision_v2 인식 불가): https://github.com/vllm-project/vllm/issues/56397
- HCX 저장소: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B · 토론 #15(quickfix): https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B/discussions/15
