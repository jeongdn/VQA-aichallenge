# 실험 노트북 · 빌더 · CPU 점검 스크립트

대회 기간에 만든 실험 노트북과, 노트북을 만들고 점검한 스크립트를 작업 폴더 구조 그대로 모았습니다.
최종 제출에 쓴 노트북은 `../submission_code/`에 정리돼 있고, 여기 있는 것은 그 과정의 기록입니다.

## 폴더 읽는 법
- `<폴더>/out/*.ipynb` — 실제로 Colab · RunPod에서 돌린 노트북입니다. 대부분 같은 폴더의 `build_*.py`가 만들었습니다.
- `build_*.py` — 노트북 빌더. 기반 노트북(`src_stage2.ipynb` 등)의 셀을 읽어 바꾼 뒤 `out/`에 씁니다.
- `run_*.py`, `test_*.py`, `make_fixtures*.py`, `make_tiny*.py`, `fake*.py`, `fake_vllm/` — CPU 점검용입니다. 작은 무작위 모델·가짜 vLLM·가짜 HF 허브로 노트북 코드 셀을 끝까지 돌려 결과를 독립 계산과 대조했습니다.
- 스크립트 안의 경로는 작업 환경 기준입니다.
  - `/home/user/vqa/` = 이 `experiments/` 폴더
  - `/tmp/vqa_scratch/`, `/tmp/vqa_uploads/`, `/tmp/vqa_sample_images/` = 점검용 임시 폴더 (대회 CSV·예시 이미지를 두던 곳)
- 대회 데이터(CSV·이미지)는 이 저장소에 없습니다. 점검 스크립트를 다시 돌리려면 대회 데이터를 직접 받아 위 경로에 두어야 합니다.
- 대회 데이터 압축 파일의 비밀번호는 `ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"`로 바꿔 두었습니다. 실행할 때 대회 안내의 비밀번호로 바꿔 넣으세요.
- 노트북의 실행 결과(출력)는 지웠습니다. 학습한 어댑터·예측 파일은 팀 Drive에 있고 이 저장소에는 없습니다.

## 폴더별 내용
실험 번호는 `../docs/experiment_log/merged_status_0927.md`의 통합 번호입니다. 폴더·파일 이름은 옛 번호 그대로라 다를 수 있습니다.

| 폴더 | 실험 | 내용 | 결과 · 기록 문서 |
|---|---|---|---|
| `src_stage2.ipynb` | — | 팀의 기본 stage2 학습 노트북 (대부분 빌더의 출발점) | — |
| `e1/` | E16 | NF4로 학습한 Qwen3.5-27B 어댑터를 bf16 모델에 얹어 다시 채점 | 검증 97.3% (원래 97.2%) · `merged_status_0927.md` |
| `e5/` | E10 | 애매한 문항만 up3.0·up4.0으로 다시 채점 | 멤버 단위 이득 없음 · 불채택 |
| `e7/` | E7 | 불확실 문항의 보기별 예/아니오 검증 (Gemma · Qwen3.8 zero-shot) | Gemma 검증 +2, Qwen3.8 −1 |
| `e7b/` | E7-b | 채점해 둔 E7 점수를 조합 `ea0e9f` 위에 다시 얹기 (CPU) | 검증 98.0 → 97.7 · 불채택 · `e20_xcheck_0928.md` |
| `e8/` | E8 | 검증 포함 전체 데이터(6,714문항) 재학습 `_full` + 교체 | 공개 −4 · 불채택 |
| `e9/` | E9 | 불확실 문항만 보기 순서 TTA4로 다시 채점 | 공개 하락 · 불채택 |
| `e10/` | E15 | 전 멤버 조합 탐색 (방법 5개, 중첩 CV) | 공개 0.97467 · 불채택 |
| `e11/` | E11 | E15 전체 평균 위 결합기 | 취소 |
| `e13/` | E13 | bf16 재학습 멤버로 자리 바꾸기 판정 (CPU) | 쓰지 않음 (E14로 대체) |
| `e14/` | E14 | 검증만으로 조합 고르기 (중첩 CV 10겹 × 5회 · 1-표준오차 규칙) + 검증 오답 보고서 | 중첩 CV 97.58% · 이때 정한 예측 파일 형식("E14 형식")을 이후 모든 노트북이 씀 |
| `bf16/` | E13 | Qwen3.5 · Qwen3.8 · Gemma 4 · HCX bf16 재학습 노트북 | 이득 없음 (Q35 −1, Q38 −5, 4bit와 답 99% 같음) |
| `clone/` | — | 기존 멤버 레시피를 그대로 복사해 bf16으로 학습 (시드 변경 옵션) | — |
| `hcx/` | E9·E10 HCX판 | HCX에도 TTA · 고해상도 재채점 (`e9e5_hcx`, 중단) + HCX 학습 원본 노트북 2개 | — |
| `newfam/` | — | 새 계열 학습 노트북 `stage2_newfam` (학습 전 zero-shot 점검 포함) | `model_survey_0927.md` |
| `ocr/` | E17 | PaddleOCR-VL로 이미지 글자를 뽑아 프롬프트에 넣고 재학습 | Gemma 97.7% · Qwen3.5 96.9%, 일관된 이득 없음 · `ocr_experiment_0927.md` |
| `zs122b/` | E18 | Qwen3.5-122B-A10B GPTQ-Int4 zero-shot (vLLM, Colab G4) | 검증 96.0% · `zs122b_0927.md` |
| `zsany/` | — | 아무 모델 zero-shot (`MODEL_ID`만 바꿈, vLLM) | `anchor_zs_runpod_0928.md` |
| `runpod/` | E19 · E19b | RunPod B300 LoRA 노트북 (Qwen3.5-122B bf16, Qwen3.8-Flash-Next, GLM-5.3-Flash FP8, Qwen3.5-397B FP8) + Colab ↔ RunPod 브리지 | Flash-Next 검증 96.9% · 공개 0.97110, GLM은 zero-shot 92.0%로 접음 · `runpod_122b_0927.md`, `runpod_flash_glm_0927.md` |
| `e20/` | E20 | 조합이 불확실한 문항만 교차 검증 프롬프트로 다시 묻기 (122B) | 판정 모델(04번)의 전신 · `e20_xcheck_0928.md` |
| `xcheck/` | — | 판정 모델 노트북 `runpod_xcheck_any` + HCX 보정 `hcx_patch.py` | `anchor_zs_runpod_0928.md` |
| `deliver/` | — | HCX용으로 `MODEL_ID`만 맞춘 판정 노트북 | `anchor_zs_runpod_0928.md` |
| `anchor/` | — | 기준 모델 + 보조 모델 교체 규칙 (5겹 교차 적합 · 부트스트랩) | `a12cf3` 순이득 +4, `19f20f` +2 · 둘 다 불채택 |
| `ens_search/` | — | 조합 검색 (전수 · 그리디 · 좌표 상승, 중첩 CV) | 전수 1위 · 확률 = `ea0e9f` 공개 0.97736 · `ens_search_0927.md` |
| `manual_ens/` | — | 직접 가중치를 바꿔 보는 앙상블 패널 (CPU) | `manual_ens_0927.md` |
| `import_old/` | — | 옛 노트북이 만든 예측 파일을 실행 폴더 형식으로 옮기기 | `import_old_0927.md` |
| `final/` | — | 제출 패키지 빌더: `build_train_all.py` → `01_train_vlm`, `build_small_nbs.py` → `00_data_prep`·`07_final_check`, `run_train_all.py`(CPU 시험 9개), `src/`(셀 원본), `_recipes_all.json`(학습 기록 설정) | `submission_package_0928.md` |

- Qwen3.5-27B 학습 변형 E1~E5 · D3500의 노트북은 따로 만들어 이 폴더에 없습니다. 결과는 `E_series_results_0926.md`, `handoff_E_series_0927.md`에 있습니다.
- 빌더가 쓰는 `final/pkg/`는 이 저장소의 `../submission_code/`입니다.
