# 텍스트 이미지 VQA (4지선다) — 제출 코드

이미지 + 한국어 질문 + 보기 4개를 받아 a~d 중 하나를 고르는 과제입니다.
허깅페이스 사전 학습 **시각-언어 모델(VLM)** 여러 개를 LoRA로 파인튜닝하고, 학습하지 않은 대형 모델(zero-shot)을 더한 뒤, **고정 검증셋 1,000문항**으로만 조합 규칙을 정해 test 답을 만들었습니다.
추론은 모두 로컬 GPU(transformers · vLLM)에서 했고 외부 API는 쓰지 않았습니다.

## 1. 실행 순서

| 순서 | 노트북 | 환경 | 하는 일 | 결과 (`vqa_runs/…`) |
|---|---|---|---|---|
| 0 | `00_data_prep.ipynb` | Colab CPU | 데이터 풀기·점검, 고정 검증 분할 확인, 보조 학습 데이터 | `valid_ids.csv`, `aux/aux_open_v1.csv` |
| 0b | `00b_ocr_paddle.ipynb` | Colab GPU | PaddleOCR-VL로 이미지 속 글자 읽기 (OCR 레시피용) | `ocr_paddle/<태그>/ocr_text.csv` |
| 1 | `01_train_vlm.ipynb` | Colab A100 80GB · G4 96GB | **학습 모델 전부**: `RECIPE`에 실행 이름을 넣고 실행 → LoRA 학습·검증·test 예측 | `<실행>/` |
| 2 | `02_runpod_flashnext_bf16.ipynb` | RunPod B300 | Qwen3.8-Flash-Next(초대형) bf16 LoRA 학습 | `q38flash_r16_native_bf16/` |
| 3a | `03a_zs122b_vllm_colab.ipynb` | Colab G4 | Qwen3.5-122B-A10B(GPTQ-Int4) zero-shot | `q35_122b_a10b_gptq_zs/` |
| 3b | `03b_runpod_zs_any.ipynb` | RunPod B300 | 아무 모델 zero-shot (`MODEL_ID`만 바꿈, 예: Gemma 4 31B) | `zs_<모델>_up20/` |
| 4 | `04_runpod_xcheck_any.ipynb` | RunPod B300 | 판정 모델: 답이 갈린 문항만 교차 검증 프롬프트로 다시 묻기 | `xc_<모델>_up20/` |
| 5 | `05_ens_search.ipynb` | Colab CPU | 멤버 조합 검색 (전수·그리디·좌표 상승, 중첩 CV) → 제출 파일 + 설정 JSON | `ens_search/` |
| 6 | `06_anchor_rescue.ipynb` | Colab CPU | 기준 모델 + 보조 모델 교체 규칙 (5겹 교차 적합·부트스트랩) → 제출 파일 + 설정 JSON | `anchor_rescue/` |
| 7 | `07_final_check.ipynb` | Colab CPU | 고른 제출 파일의 형식 점검 + 설정 JSON만으로 **독립 코드가 다시 만들어 대조** | — |
| — | `util_runpod_bridge_colab.ipynb` | Colab CPU | Drive ↔ HF 비공개 저장소 ↔ RunPod 파일 옮기기 | — |

- RunPod에는 Google Drive가 붙지 않아 데이터·결과를 **HF 비공개 데이터셋 저장소**로 옮겼습니다(브리지 노트북 A 단계: 데이터 올리기, B 단계: 결과 받기). 저장소는 비공개로만 만들고 대회 뒤 지웁니다(C 단계).
- 모델별 처리(HyperCLOVA X 답 자리 고정, Gemma·InternVL 해상도, Flash-Next 보정, vLLM 설정)는 `MODEL_NOTES.md`에 정리했습니다.

## 2. 제출 파일 만들기 (최종 CSV)
1. 멤버 예측이 `vqa_runs/<실행>/`에 모두 있어야 합니다 (1~4번, 또는 이미 만든 결과).
2. **조합**: `05_ens_search.ipynb` — 검색 후보나 직접 넣은 가중치로 저장하면 `ens_search/submission_<이름>_<해시>.csv`와 같은 이름의 `.json`이 생깁니다.
   - 공개 최고 조합 `ea0e9f`: `gemma4_31b_r16_up20_bf16 + gemma4_31b_r16_up20_ocr_bf16 + ivl35_38b_r16_native + q35_27b_r16_native_bf16 + q38_27b_r16_native`, 같은 가중치 확률 평균 (검증 98.0%, 공개 0.97736).
3. **교체 규칙**: `06_anchor_rescue.ipynb` — 기준 모델 답을 네 조건(보조 모델 불일치 · 기준 확신 < CONF · 1·2위 차 < MARGIN · 보조 모델 AGREE개 이상이 같은 다른 보기)을 모두 만족할 때만 바꿉니다. `anchor_rescue/submission_anchor_*.csv` + `.json`.
4. **점검**: `07_final_check.ipynb`의 `CANDIDATES`에 고른 파일을 넣고 실행 → 형식 ✅, 재현 "같음" ✅를 확인합니다.
- 설정 JSON에는 멤버 예측 파일 경로·md5, 가중치·온도, 규칙 문턱값이 모두 들어 있어, 같은 파일로 언제 다시 만들어도 같은 CSV가 나옵니다(난수 없음).

## 3. 재현성 — 세 단계
| 단계 | 방법 | 기대 결과 |
|---|---|---|
| ① 저장된 예측 → 제출 CSV | `07_final_check` (또는 05번 13번 셀, 06번 재실행) | **완전히 같음** (결정적 계산) |
| ② 저장된 어댑터 → 예측 | `01_train_vlm`에서 `RECIPE` + `REUSE_FINISHED = True` → `REPRO_CHECK_N`문항을 새로 채점해 저장된 예측과 대조 | 같은 GPU 종류면 답 일치 ~100% |
| ③ 처음부터 다시 학습 | `01_train_vlm`에서 `RECIPE` + `RUN_SUFFIX = "_repro"` | 시드·순서·설정이 같아 거의 같은 모델 (GPU 커널의 비결정성 때문에 답이 소수 다를 수 있음) |
- 레시피 값은 각 실행의 `train_log.json`에서 그대로 옮겼고, `SYNC_FROM_LOG = True`면 실행할 때 기록과 한 번 더 대조합니다.
- 모든 노트북이 검증 분할 md5(`df05e4bc19d27ebceeef2570672149c2`)를 확인하고 다르면 멈춥니다.

## 4. 환경 · 라이브러리 (노트북 설치 셀이 고정 설치)
| 용도 | 버전 |
|---|---|
| 학습·평가 (01) | transformers 5.17.0 (HyperCLOVA X만 개발 버전 커밋 `98d39824ed`), accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.0, flash-linear-attention 0.5.2 (Qwen3.5 이후), torch는 Colab 기본 |
| zero-shot·판정 (03·04) | vLLM 0.30.0 |
| OCR (00b) | PaddleOCR-VL (노트북 설치 셀 참고) |
| GPU | Colab A100 80GB, Colab G4 96GB, RunPod B300 288GB (Flash-Next 학습·zero-shot·판정) |

## 5. 데이터 사용
- 학습: train 6,714문항 중 **검증 1,000문항을 뺀 5,714문항**(정답 글자별 250문항씩 시드 42로 고정 분할). `_full` 레시피만 검증까지 합쳐 6,714문항(제출용, 검증 점수는 의미 없음).
- 증강: 매 epoch 보기 순서 섞기(정답 글자도 함께 바꿈). 선택으로 밝기·대비·JPEG 화질 흔들기(`IMG_AUG`, 기본 끔).
- 보조 학습(`aux_open_v1.csv`): 학습 문항 중 보기 없이도 답이 정해지는 글자 읽기 문항만, 정답 선지 글자를 답으로 (보기 없는 단답형).
- dev: 정답이 없어, 모델 확신 ≥ 0.8이고 교육생 응답 다수결(3표 이상)과 모델 답이 같은 문항만 의사 라벨로 (`*_dev` 레시피, 대회 규칙상 dev 증강 허용).
- OCR 글자(`*_ocr_bf16` 레시피): 대회 이미지를 PaddleOCR-VL로 읽은 결과. 외부 데이터는 쓰지 않았습니다.
- test는 추론에만 썼고, 정답을 추정해 학습하지 않았습니다.

## 6. 학습 모델 (01번 `RECIPES`) — 검증 1,000문항 · up2.0 정확도
| 실행 | 모델 | 정밀도 | 학습 해상도 | 특이 사항 | 검증 % |
|---|---|---|---|---|---|
| `gemma4_31b_r16_up20` | Gemma 4 31B | NF4 | up2.0 | — | 97.6 |
| `gemma4_31b_r16_up20_bf16` | Gemma 4 31B | bf16 | up2.0 | — | 97.5 |
| `gemma4_31b_r16_up20_ocr_bf16` | Gemma 4 31B | bf16 | up2.0 | OCR 글자 프롬프트 | 97.7 |
| `gemma4_31b_r16_up20_dev` | Gemma 4 31B | NF4 | up2.0 | + dev 의사 라벨 | 97.4 |
| `gemma4_26ba4b_r16_up20_bf16` | Gemma 4 26B-A4B (MoE) | bf16 | native | — | 96.9 |
| `q35_27b_r16_native` | Qwen3.5-27B | NF4 | native | — | 97.2 |
| `q35_27b_r16_native_bf16` | Qwen3.5-27B | bf16 | native | — | 97.1 |
| `q35_27b_r16_native_ocr_bf16` | Qwen3.5-27B | bf16 | native | OCR 글자 프롬프트 | 96.9 |
| `q35_27b_r16_review1_up20_vis` | Qwen3.5-27B | NF4 | up2.0 | 비전 LoRA | 97.3 |
| `q35_27b_r16_e4aux20_native` | Qwen3.5-27B | NF4 | native | native 어댑터에서 0.5 epoch 이어학습 + 보조 학습 20% | 97.6 |
| `q35_9b_r16_native` | Qwen3.5-9B | bf16 | native | 2 epoch | 96.4 |
| `q36_35ba3b_r16_native_bf16` | Qwen3.6-35B-A3B (MoE) | bf16 | native | — | 97.0 |
| `q38_27b_r16_native` | Qwen3.8-27B | NF4 | native | — | 96.8 |
| `q38_27b_r16_native_43` | Qwen3.8-27B | NF4 | native | 학습 시드 43 | 96.6 |
| `q38_27b_r16_native_bf16` | Qwen3.8-27B | bf16 | native | — | 96.3 |
| `q3vl_8b_r16_native` | Qwen3-VL-8B | bf16 | native | 2 epoch 중 검증으로 1 epoch 선택 | 96.5 |
| `hcx32b_r16_native` | HyperCLOVA X SEED Think 32B | NF4 | native | 생각 블록 닫기 (MODEL_NOTES) | 97.0 |
| `hcx32b_r16_native_bf16` | HyperCLOVA X SEED Think 32B | bf16 | native | — | 96.5 |
| `ivl35_38b_r16_native` | InternVL3.5-38B | NF4 | native (6타일) | — | 95.7 |
| `q38flash_r16_native_bf16` | Qwen3.8-Flash-Next | bf16 | native | 02번 노트북 (RunPod) | 96.9 |
| `*_full` | (위 레시피) | bf16 | — | 검증 포함 6,714문항 학습 (제출 섞기용) | — |

공통: LoRA r16 · α32 · dropout 0.05, 유효 배치 16, LR 1e-4(이어학습 2e-5), cosine + warmup 5%, 1 epoch(표시한 것 제외), 평가·test 해상도 up2.0.
zero-shot: `q35_122b_a10b_gptq_zs` 96.0 · `zs_gemma_4_31b_it_up20` 97.0 (학습 없음).

## 7. 결과 폴더 규칙
`vqa_runs/<실행>/`
- `train_log.json` — 학습 설정(`config`)·epoch별 검증 정확도(학습 해상도 기준)·레시피 이름. zero-shot은 `ZERO_SHOT: true`, 판정은 `XCHECK: true`.
- `preds_final_ep<N>__up2.0__1000.csv` (검증) · `preds_final_ep<N>_test__up2.0__6714.csv` (test) — 열 `id, pred, p_a, p_b, p_c, p_d, letter_mass, img_tokens`.
- `adapter_ep<N>/` — LoRA 어댑터 (01·02번).
- 조합 노트북(05·06)은 이 이름 규칙으로 멤버를 자동으로 찾습니다. 폴더 이름 조각에 `full`이 있거나 검증 99% 초과인 실행은 검증을 학습에 쓴 것으로 보고 검색·규칙 선택에서 뺍니다.

## 8. 실험 번호 (노트북 설명에 나오는 이름)
| 번호 | 내용 |
|---|---|
| E1~E5, D3500 | Qwen3.5-27B 학습 변형(학습률·epoch·LoRA 세기·보조 학습·데이터 양) — 앙상블 이득 없음 |
| E7 | 보기별 예/아니오 검증 · E9 선지 순열 TTA · E10 애매한 문항 고해상도 재채점 — 불채택 |
| E8 | 검증 포함 전체 데이터 재학습(`_full`) |
| E13 | bf16 재학습 (4bit와 답 99% 같음) |
| E14 | 검증만으로 조합 고르기 — 이때 정한 멤버 파일 형식을 모든 노트북이 씀("E14 형식") |
| E15 | 전 멤버 조합 탐색 (05번의 전신) |
| E20 | 불확실 문항 교차 검증 (04번 판정 모델의 전신) |
