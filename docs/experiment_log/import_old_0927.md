# 옛 예측 파일 → 실행 폴더 (import_old_preds.ipynb) — 0927

## 배경
- `vqa_runs` 바로 아래에 옛 노트북이 만든 예측 파일이 있다. 파일 이름은 `preds_<모델>__<해상도>__4bit__valid1000.csv` / `…__test6714.csv` 꼴이다.
  - 모두 2026-09-22에 만들어졌고, 환경 기록은 `env_Qwen3.5-27B.json`(A100 40GB, torch 2.11, transformers 5.17)이다.
- 앙상블 노트북(E14 · manual_ensemble · ens_search · final_pick)은 `vqa_runs/<실행>/preds_final_ep<N>__<해상도>__1000.csv` 형식만 찾기 때문에 이 파일들을 못 봤다.
- 열 형식은 같다: `id, pred, p_a~p_d, letter_mass, img_tokens`.
- Qwen3.5-27B 옛 파일(02:14 생성)은 `q35_27b_r16_native` 학습이 아직 100스텝이던 때 나왔다. 그래서 따로 학습한 모델로 보인다. 복사본인지는 노트북이 확인한다.
- `q3vl_32b_r16_native` 폴더는 비어 있다. Qwen3-VL-32B 비-full 예측은 옛 파일로만 남아 있다.

## 옮기는 대상 (3개)
| 새 실행 이름 | 원래 파일 | 해상도 | 계열 |
|---|---|---|---|
| `q35_27b_old_up15` | `preds_Qwen3.5-27B__up1.5__4bit__valid1000/test6714.csv` | up1.5 | Qwen3.5-27B |
| `q35_27b_old_up20` | `preds_Qwen3.5-27B__up2.0__4bit__…` | up2.0 | Qwen3.5-27B |
| `q3vl_32b_old_up15` | `preds_Qwen3-VL-32B__up1.5__4bit__…` | up1.5 | `q3vl_32b_r16_native_full`의 MODEL_ID |

- 쓸 수 없는 옛 파일
  - 검증만 있음(test 없음): Qwen3-VL-32B up2.0, Qwen3.5-9B bf16 native·up1.5
  - 검증 일부만 있음: gemma valid300, Qwen3-VL-8B valid50

## 노트북이 하는 일
- **옮기기 전 확인** — 하나라도 걸리면 옮기지 않는다.
  1. 검증 문항이 팀 공통 1,000문항과 정확히 같은지 (id의 `.jpg` 유무는 맞춰 줌). `valid_ids.csv`와도 대조한다.
  2. test 6,714문항이 모두 있는지
  3. 확률이 정상인지 (합이 1이 아니면 네 보기로 다시 나눔)
  4. 검증 90~99% 범위인지 (99% 초과는 검증 누출 의심)
  5. 기존 실행의 복사본이 아닌지: 검증 답 일치 99.9% 이상이면서 확률 차이 0.001 이하면 복사본으로 보고 건너뜀. 가장 비슷한 기존 실행 3개도 보여 준다.
- **옮긴 결과**: `vqa_runs/<새 이름>/`에 다음 파일을 만든다. 원래 파일은 그대로 둔다.
  - `preds_final_ep1__<해상도>__1000.csv`
  - `preds_final_ep1_test__<해상도>__6714.csv`
  - `train_log.json`: MODEL_ID, TRAIN_ON_FULL=False, 원래 파일 경로와 md5
- **다시 실행해도 안전함**: 이미 있는 폴더는 덮어쓰지 않는다(`OVERWRITE=False`).
- 옮긴 뒤 앙상블 노트북을 다시 실행하면 자동으로 멤버가 된다.

## 점검 (CPU, 가짜 Drive)
- 옛 형식 파일 6가지로 시험했다.
  - 새 모델, 복사본, id에 .jpg 없음 + 확률 합 0.9, 다른 분할, test 없음, 누출
  - 기대대로 2개는 옮겨지고 4개는 거부됐다.
- 옮긴 파일을 원래 파일과 대조했다: 확률 값이 같고, id 순서가 맞고, 합을 다시 나눈 것도 맞다.
- 다시 실행해도 폴더를 덮어쓰지 않았다.
- ens_search와 manual_ensemble 모두 새 실행 2개를 멤버로 잡았고, 계열과 해상도도 맞았다. ens_search는 검색까지 끝까지 돌았다.
