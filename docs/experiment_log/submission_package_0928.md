# 제출 패키지 (코드 + 문서) — 2026-09-28 새벽

## 최종 제출 (확정)
- 1번: `submission__1__ea0e9f.csv` — 후보 A 조합(Gemma 4 bf16 · Gemma 4 OCR · InternVL3.5 · Qwen3.5 bf16 · Qwen3.8, 같은 가중치 확률 평균). 검증 98.0%, 공개 0.97736 (3,281/3,357).
- 2번: `submission_anchor__q38flash_r16_native_bf16_505d49.csv` — 후보 B 교체 규칙(Flash-Next 기준 + 보조 모델 표). 공개 0.97378 (3,269/3,357). 최종 점수는 두 제출 중 높은 쪽이라 방식이 다른 둘째 칸으로 냄.
- 솔루션 설명 PPT `A084_VQA_solution.pptx`(19장, 발표 메모 포함)의 4·12·13·14장, 개발 과정 문서(`docs/development_process.md`) 1·6장에 두 파일을 최종 제출로 표시함.

## 산출물
- 코드 zip: `submission_code_0928.zip` (md5 `eb1172427c067bf471188263ffe362ff`), 폴더 `submission_code/`
  - `00_data_prep` (CPU: 데이터 점검·분할 md5·보조 학습 데이터 `aux_open_v1.csv` — 실제 train.csv로 3,744문항 확인)
  - `00b_ocr_paddle` · `01_train_vlm` (통합 학습) · `02_runpod_flashnext_bf16` · `03a_zs122b_vllm_colab` · `03b_runpod_zs_any` · `04_runpod_xcheck_any` · `05_ens_search` · `06_anchor_rescue` · `07_final_check` (새로 만듦) · `util_runpod_bridge_colab`
  - `README.md` (실행 순서·재현 3단계·환경·데이터 사용·멤버 표·실험 번호), `MODEL_NOTES.md` (HCX 답 자리 고정 등 모델별 처리)
- 제출 문서: "텍스트 이미지 VQA — 데이터 처리 및 모델 개발 과정" (`docs/development_process.md`), 1~8장 모두 채움 (6장에 멤버 검증 정확도 차트 + 제출 비교 표)
- 판정 프롬프트 문구(`[작전 지시: …]`)는 재현을 위해 코드에 그대로 둠.

## 01_train_vlm (통합 학습 노트북)
- HCX 학습 노트북(계열 판별·HCX 처리·InternVL·보조 학습)을 바탕으로 비전 LoRA(build_nb7), 이어학습(INIT_ADAPTER·PLAN_EPOCH_OFFSET·EPOCH_END_EVAL·GEN_PROMPT_VERSION), OCR 프롬프트, dev 의사 라벨(12-2번 셀), 레시피 표를 합침.
- 레시피 26개: 팀 학습 기록 `_recipes_all.json`의 train_log config를 그대로 옮김. 정밀도(USE_4BIT)는 기록에 없어 이름·노트북 기준(`*_bf16`·E8 `_full` = bf16, 나머지 20B 이상 = NF4 auto).
- `SYNC_FROM_LOG`: Drive의 train_log와 다시 대조해 다른 값은 기록대로. `FULL_DATA`: `_full` 만들기(epoch 선택 불일치면 멈춤, MAX_STEPS 비율 조정). `RUN_SUFFIX`: 다른 폴더에 재학습. 끝난 실행 폴더는 덮어쓰지 않고 멈춤.
- `REUSE_FINISHED`: 끝난 실행을 학습 없이 불러와 평가·test·dev 의사 라벨만 (예측 파일·train_log md5 그대로 확인), `REPRO_CHECK_N`문항 재채점 대조.
- 7번 셀을 7-1(프로세서·모델별 프롬프트 보정: 생각 모드·system 형식·빈 생각 블록 닫기)과 7-2(모델·LoRA)로 나눔.
- LoRA 층 수 기록(`LORA_TARGET_COUNTS`)·정밀도(`PRECISION`)·레시피 이름을 train_log에 남기고, 기록과 층 수를 대조.
- CPU 시험(작은 Qwen3.5 + 가짜 프로세서, `experiments/final/run_train_all.py`) 9개 모두 통과: dry · base · vis · e4 · ocr · sync · fulldata · dev(REUSE) · resume(끊었다 이어도 확률 차 0).

## 07_final_check (새 노트북)
- 형식 점검 + 설정 JSON만으로 독립 코드가 다시 만들어 대조 (ens_search/manual JSON: 가중 평균·온도·full mix, 멤버 md5 / anchor JSON: 네 조건 교체 규칙) + 제출끼리 차이 + 선택: 다수결 재구성.
- 시험: ens_search 저장본 6개, anchor 저장본 4개(판정·온도 보정 포함) 모두 "같음", 멤버 파일 변조 감지.

## 판정 모델 진행 (0928 05:40~)
- 122B GPTQ 판정이 B300에서 가중치 14/39에서 OOM(남은 10MB)을 냄. 처음 두 번은 다른 커널이 GPU를 92% 잡은 것으로 보였음. 모든 커널을 정리한 뒤에도 반복되면 로더 문제로 보고, 판정 모델을 `Qwen/Qwen3.8-27B`로 바꾸기로 함(프리셋 없이 기본값으로 동작, RUN_NAME `xc_qwen3_8_27b_up20`).
- anchor_rescue 기존 결과: `a12cf3`(보조 6개) 살림 7·망침 3·순 +4·부트스트랩 88% → 불채택, `19f20f` 순 +2 → 불채택.
