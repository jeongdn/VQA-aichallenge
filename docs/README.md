# 문서

- [`development_process.md`](development_process.md) — 대회에 낸 추가 문서 "데이터 처리 및 모델 개발 과정" (1~8장: 개요, 데이터, 학습, 추론, 앙상블, 결과, 재현, 배운 점)
- `experiment_log/` — 대회 중에 실험마다 남긴 기록 (날짜순)

| 문서 | 내용 |
|---|---|
| [`E_series_results_0926.md`](experiment_log/E_series_results_0926.md) | Qwen3.5-27B 학습 변형 E1~E5 · D3500 결과와 검증 오답 23개의 구조 |
| [`handoff_E_series_0927.md`](experiment_log/handoff_E_series_0927.md) | 위 실험의 자세한 정리 (판정 규칙, 체크포인트별 결과, 나왔던 제안) |
| [`merged_status_0927.md`](experiment_log/merged_status_0927.md) | 실험 번호 통합표(E1~E16), E14 조합, 오답 분석, 결정 사항 |
| [`model_survey_0927.md`](experiment_log/model_survey_0927.md) | 새 모델 후보 조사와 zero-shot 점검 결과 |
| [`zs122b_0927.md`](experiment_log/zs122b_0927.md) | Qwen3.5-122B-A10B GPTQ-Int4 zero-shot (vLLM) 설계와 점검 |
| [`manual_ens_0927.md`](experiment_log/manual_ens_0927.md) | 직접 가중치 앙상블 노트북 |
| [`ocr_experiment_0927.md`](experiment_log/ocr_experiment_0927.md) | PaddleOCR-VL 글자 프롬프트 실험과 판정 |
| [`import_old_0927.md`](experiment_log/import_old_0927.md) | 옛 예측 파일을 실행 폴더로 옮기기 |
| [`ens_search_0927.md`](experiment_log/ens_search_0927.md) | 조합 검색 노트북과 실제 결과 (공개 최고 `ea0e9f`) |
| [`runpod_122b_0927.md`](experiment_log/runpod_122b_0927.md) | RunPod B300에서 122B bf16 LoRA 준비 |
| [`runpod_flash_glm_0927.md`](experiment_log/runpod_flash_glm_0927.md) | Qwen3.8-Flash-Next · GLM-5.3-Flash LoRA, 학습 NaN 원인과 수정 |
| [`e20_xcheck_0928.md`](experiment_log/e20_xcheck_0928.md) | E7-b · E6 · 보기 글자 신호, E20 교차 검증 설계 |
| [`anchor_zs_runpod_0928.md`](experiment_log/anchor_zs_runpod_0928.md) | 기준 모델 교체 규칙 · zero-shot · 판정 모델, HCX 판정 수정 |
| [`submission_package_0928.md`](experiment_log/submission_package_0928.md) | 제출 패키지 구성과 최종 제출 2개 |

- 공개 점수의 문항 수는 점수 × 3,357입니다.
- 기록 속 "Drive", "`vqa_runs/…`"는 팀 Google Drive의 결과 폴더입니다. 이 저장소에는 없습니다.
- 기록은 그때그때 쓴 것이라, 뒤 문서에서 바뀐 판단이 있을 수 있습니다. 최종 결과는 `development_process.md`가 기준입니다.
