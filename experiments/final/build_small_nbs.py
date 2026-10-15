# -*- coding: utf-8 -*-
"""00_data_prep.ipynb (CPU) · 07_final_check.ipynb (CPU) 빌더"""
import json, os, re, ast
PKG = "/home/user/vqa/final/pkg"
AUX = json.load(open("/home/user/vqa/final/src/aux_blocks.json", encoding="utf-8"))

def nb_of(cells, gpu=False):
    out = []
    for kind, t in cells:
        t = t.strip("\n"); lines = t.split("\n")
        c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code": c.update({"execution_count": None, "outputs": []})
        out.append(c)
    md = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    if gpu: md.update({"accelerator": "GPU"})
    return {"cells": out, "metadata": md, "nbformat": 4, "nbformat_minor": 0}

def save(nb, name):
    p = os.path.join(PKG, name); json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for c in nb["cells"]:
        if c["cell_type"] == "code": ast.parse("\n".join(re.sub(r"^(\s*)!.*$", r"\1pass", l) for l in "".join(c["source"]).split("\n")))
    print("저장:", p, "| 셀", len(nb["cells"]))

# ═════════════════════════ 00_data_prep ═════════════════════════
DP = [("markdown", r"""
# 00 · 데이터 준비 (CPU, 1~3분)

모든 노트북이 같이 쓰는 데이터를 준비하고 점검합니다. GPU가 필요 없습니다.

| 순서 | 하는 일 | 결과 |
|---|---|---|
| 1 | Drive의 대회 zip을 `DATA_DIR`에 풀기 | `train.csv`, `test.csv`, `dev.csv`, `sample_submission.csv`, 이미지 폴더 |
| 2 | 데이터 점검 · **고정 검증 분할** 확인 | `OUT_DIR/valid_ids.csv` (검증 1,000문항 id) |
| 3 | **보기 없는 단답형 보조 학습 데이터** 만들기 | `OUT_DIR/aux/aux_open_v1.csv` (`q35_27b_r16_e4aux20_native` 레시피가 씀) |

- **고정 검증 분할**: train 6,714문항에서 정답 글자(a~d)별 250문항씩, 시드 42로 뽑은 1,000문항입니다. 모든 학습·조합 노트북이 같은 분할(md5 `df05e4bc…`)을 쓰고, 다르면 멈춥니다. 나머지 5,714문항으로 학습합니다.
- **보조 학습 데이터**: 학습 문항 5,714개 중 **보기를 가려도 답이 정해지고, 답이 이미지 글자를 그대로 읽는 것**(가격·시간·전화·상호·층·문구 등)만 골라, 정답 선지의 글자를 그대로 답으로 씁니다.
  빼는 것: 보기가 있어야 성립하는 질문(다음 중·고르기), 부정형, '모두/해당 없음' 류 정답, 색상, 의미·목적·추론, 개수·계산, 20자 넘는 답, 글자 읽기가 아닌 질문.
- 다른 데이터 파일: OCR 글자는 `00b_ocr_paddle.ipynb`(GPU), dev 의사 라벨은 `01_train_vlm.ipynb`의 12-2번 셀이 만듭니다.
- 외부 데이터는 쓰지 않았습니다. dev는 정답이 없고 교육생 응답 5개만 있어, 응답이 합의되고 모델도 확신하는 문항만 의사 라벨로 썼습니다(`*_dev` 레시피).
"""),
("markdown", "## 1. 설정"),
("code", r'''
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
AUX_OUT      = f"{OUT_DIR}/aux/aux_open_v1.csv"      # 01_train_vlm의 q35_27b_r16_e4aux20_native 레시피가 읽는 경로
EXPECTED_SPLIT_MD5 = "df05e4bc19d27ebceeef2570672149c2"
SEED, VAL_SIZE = 42, 1000
MAX_ANS_CHARS = 20                                    # 보조 학습: 이보다 긴 정답 글자는 제외 (짧게 읽어 쓰는 답만)
'''),
("markdown", "## 2. 압축 풀기 · 데이터 점검 · 고정 검증 분할"),
("code", r'''
import os, re, json, hashlib, subprocess
import numpy as np, pandas as pd
try:
    from google.colab import drive; drive.mount("/content/drive")
except ImportError:
    pass
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    print("압축 해제 중... (몇 분 걸립니다)")
    r = subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-800:]
os.makedirs(OUT_DIR, exist_ok=True)
tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv")); te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
dv = pd.read_csv(os.path.join(DATA_DIR, "dev.csv")) if os.path.exists(os.path.join(DATA_DIR, "dev.csv")) else None
tr["answer"] = tr["answer"].astype(str).str.strip().str.lower()
assert set(tr.answer) <= set("abcd") and list(te.id) == list(ss.id), "정답 글자 또는 test/sample_submission id 순서 이상"
print(f"train {len(tr):,} · test {len(te):,} · dev {0 if dv is None else len(dv):,} | 정답 분포 {tr.answer.value_counts().sort_index().to_dict()}")
for name, df in [("train", tr), ("test", te)] + ([("dev", dv)] if dv is not None else []):
    miss = [p for p in df.path if not os.path.exists(os.path.join(DATA_DIR, p))]
    print(f"  {name}: 이미지 없음 {len(miss)}개" + (f" (예: {miss[:3]})" if miss else ""))
if dv is not None:
    rc = [c for c in dv.columns if c.startswith("answer")]
    top = dv[rc].apply(lambda r: max([list(r.astype(str).str.lower()).count(k) for k in "abcd"]), axis=1)
    print(f"  dev 응답 열 {rc} | 문항별 최다 표 수 분포 {top.value_counts().sort_index().to_dict()} (정답 없음 → 의사 라벨은 모델 확신과 응답 합의가 모두 필요)")

# 고정 검증 분할 (모든 노트북과 같은 코드)
rng = np.random.RandomState(SEED); vi = []
for L in "abcd":
    idx = np.where(tr.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[: VAL_SIZE // 4].tolist())
vi = sorted(vi); val_ids = tr.id.iloc[vi].tolist()
md5 = hashlib.md5(",".join(val_ids).encode()).hexdigest()
print(f"\n검증 {len(val_ids)} / 학습 {len(tr) - len(val_ids)} | split md5 {md5} " + ("✅ 공통 분할과 같음" if md5 == EXPECTED_SPLIT_MD5 else "⚠️ 공통 분할과 다름 → train.csv 원본을 확인하세요"))
assert md5 == EXPECTED_SPLIT_MD5
pd.DataFrame({"id": val_ids}).to_csv(os.path.join(OUT_DIR, "valid_ids.csv"), index=False)
print("저장:", os.path.join(OUT_DIR, "valid_ids.csv"))
'''),
("markdown", "## 3. 보기 없는 단답형 보조 학습 데이터 (`aux_open_v1.csv`)"),
("code", AUX["AUX_BUILD"]),
("markdown", "## 4. 표본 40개 확인\n보기 없이는 답이 정해지지 않거나, 이미지 글자와 다른 표현이 많이 보이면 규칙(2번 셀의 정규식)을 확인하세요."),
("code", AUX["AUX_SHOW"].replace("E4 처리군을 돌리기 전에 알려 주세요.", "보조 학습 레시피를 돌리기 전에 규칙을 확인하세요.")),
]
save(nb_of(DP), "00_data_prep.ipynb")

# ═════════════════════════ 07_final_check ═════════════════════════
FC = [("markdown", r"""
# 07 · 최종 제출 파일 점검 (CPU)

고른 제출 파일(CSV)마다 다음을 확인합니다. GPU가 필요 없습니다.
1. **형식**: `id, answer` 두 열, `sample_submission.csv`와 같은 6,714문항·같은 순서, 답은 a~d 한 글자, 빈칸 없음.
2. **재현**: 같은 이름의 설정 JSON(조합 노트북이 저장)이 있으면, JSON에 적힌 멤버 예측 파일과 규칙만으로 **이 노트북의 독립 코드가 test 답을 다시 만들어** 저장된 CSV와 같은지 봅니다.
   - `05_ens_search` / 수동 조합 JSON: 멤버별 test 확률 → (온도 보정) → 가중 평균 → 최댓값. 멤버 파일의 md5가 JSON 기록과 같은지도 봅니다.
   - `06_anchor_rescue` JSON: 기준 모델 확률(온도 보정) + 보조 모델 표 → 네 조건(불일치·확신도·마진·다수 동의)을 모두 만족할 때만 뒤집기.
3. **비교**: 고른 파일끼리 test 답이 다른 문항 수 (공개 리더보드는 test의 약 절반이라 다른 문항의 절반쯤이 공개 점수에 반영됨).
4. (선택) **다수결 파일**: 입력 CSV 여러 개의 다수결을 다시 만들어 저장된 파일과 비교 (동률은 알파벳 앞 글자).
"""),
("markdown", "## 1. 설정 — 제출할 파일 경로 (OUT_DIR 기준)"),
("code", r'''
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
CANDIDATES = {                                    # 이름: OUT_DIR 기준 CSV 경로 (같은 이름의 .json이 있으면 재현 점검)
    "제출 1": "ens_search/submission_my_search_ea0e9f.csv",
    "제출 2": None,                               # 예: "anchor_rescue/submission_anchor_…csv"
}
MAJORITY = None    # 선택: {"saved": "ensembles/submission_majority_vote.csv", "inputs": ["ensembles/a.csv", "ensembles/b.csv", "ensembles/c.csv"]}
N_PUBLIC = 3357    # 공개 리더보드 문항 수 (점수 × 3,357 = 맞힌 문항 수)
'''),
("markdown", "## 2. 데이터 (CSV만)"),
("code", r'''
import os, re, json, math, hashlib, subprocess
import numpy as np, pandas as pd
try:
    from google.colab import drive; drive.mount("/content/drive")
except ImportError:
    pass
if not os.path.exists(os.path.join(DATA_DIR, "sample_submission.csv")):
    r = subprocess.run(["7z", "e", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP, "sample_submission.csv", "test.csv", "-r"], capture_output=True, text=True)
    assert os.path.exists(os.path.join(DATA_DIR, "sample_submission.csv")), r.stdout[-500:] + r.stderr[-500:]
SS = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv")); TEST_IDS = SS.id.tolist(); LETTERS = "abcd"
print(f"sample_submission {len(SS):,}문항")
def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def P_of(rel):
    d = pd.read_csv(os.path.join(OUT_DIR, rel)).drop_duplicates("id", keep="last").set_index("id")
    miss = [i for i in TEST_IDS if i not in d.index]; assert not miss, f"{rel}: test 문항 {len(miss)}개 없음"
    return d.loc[TEST_IDS, ["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)
def temp_scale(p, T):
    q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def answers_of(path):
    s = pd.read_csv(path).set_index("id").reindex(TEST_IDS)["answer"].astype(str).str.strip().str.lower()
    return s.map({L: i for i, L in enumerate(LETTERS)}).to_numpy()
'''),
("markdown", "## 3. 점검 함수 (형식 · 조합 재현 · 교체 규칙 재현)"),
("code", r'''
def check_format(path):
    d = pd.read_csv(path, dtype=str); probs = []
    if list(d.columns) != ["id", "answer"]: probs.append(f"열 이름 {list(d.columns)} (id, answer여야 함)")
    if len(d) != len(SS): probs.append(f"행 수 {len(d)} (≠ {len(SS)})")
    elif list(d.id) != TEST_IDS: probs.append("id 순서가 sample_submission과 다름")
    if "answer" in d:
        bad = d.answer.isna() | ~d.answer.astype(str).isin(list(LETTERS))
        if bad.any(): probs.append(f"a~d가 아닌 답 {int(bad.sum())}개")
    return probs

def rebuild_ens(info):
    # 05_ens_search / 수동 조합 JSON: {"cfg": {"weights", "method", "temp", "full"}, "members": {실행: {"test", "T", "test_md5", "full_test"}}}
    c, mem = info["cfg"], info["members"]; runs = list(c["weights"]); notes = []
    Ps = []
    for r in runs:
        m = mem[r]; fp = os.path.join(OUT_DIR, m["test"])
        if "test_md5" in m and md5file(fp) != m["test_md5"]: notes.append(f"{r} test 파일 md5가 JSON 기록과 다름")
        P = P_of(m["test"])
        if c.get("full", "none") != "none" and "full_test" in m:
            Pf = P_of(m["full_test"]); P = (P + Pf) / 2 if c["full"] == "mix" else Pf
        Ps.append(temp_scale(P, m["T"]) if c.get("temp") == "val" else P)
    w = np.array([float(c["weights"][r]) for r in runs]); w = w / w.sum()
    if c["method"] == "prob": S = sum(wi * p for wi, p in zip(w, Ps))
    elif c["method"] == "logprob": S = sum(wi * np.log(np.clip(p, 1e-6, 1.0)) for wi, p in zip(w, Ps))
    else: S = sum(wi * np.eye(4)[p.argmax(1)] for wi, p in zip(w, Ps)) + 1e-3 * sum(wi * p for wi, p in zip(w, Ps))
    desc = " + ".join(f"{r}×{c['weights'][r]:g}" for r in runs) + f" | {c['method']} · 온도 {c.get('temp', 'none')} · full {c.get('full', 'none')}"
    return S.argmax(1), desc, notes

def rebuild_anchor(meta):
    # 06_anchor_rescue JSON: 기준 모델 확률(온도 보정) + 보조 모델 표 → 네 조건을 모두 만족할 때만 다른 보기로
    rule, files = meta["rule"], meta["files"]
    PA = P_of(files[meta["anchor"]]["test"])
    PA = temp_scale(PA, float(rule["T_anchor"]))                               # 보정 안 한 실행은 T = 1.0으로 기록됨 (06번과 같은 계산)
    RP = np.stack([P_of(files[r]["test"]) for r in meta["rescuers"]])          # (보조 모델 수, 문항, 4)
    n = PA.shape[0]; a = PA.argmax(1)
    votes = np.zeros((n, 4)); psum = RP.sum(0)
    for m in range(RP.shape[0]): votes[np.arange(n), RP[m].argmax(1)] += 1
    s = np.sort(PA, 1); pmax, marg = s[:, -1], s[:, -1] - s[:, -2]
    v = votes.copy(); v[np.arange(n), a] = -1                                   # 기준 답을 뺀 다른 보기 중 가장 많은 표 (동률은 확률 합)
    alt = (v + 1e-3 * psum / max(psum.max(), 1e-9)).argmax(1); cnt = v[np.arange(n), alt]
    flip = (cnt >= rule["AGREE"]) & (pmax < rule["CONF"]) & (marg < rule["MARGIN"])
    desc = (f"기준 {meta['anchor']} · 보조 {len(meta['rescuers'])}개 (판정 {len(meta.get('judges') or [])}) | CONF < {rule['CONF']} · MARGIN < {rule['MARGIN']} · "
            f"AGREE ≥ {rule['AGREE']} · T {float(rule['T_anchor']):.2f} | 뒤집음 {int(flip.sum())} (JSON 기록 {meta.get('test_flips')})")
    notes = [] if meta.get("test_flips") in (None, int(flip.sum())) else ["뒤집은 문항 수가 JSON 기록과 다름"]
    return np.where(flip, alt, a), desc, notes
'''),
("markdown", "## 4. 후보별 점검"),
("code", r'''
ANS, ROWS = {}, []
for name, rel in CANDIDATES.items():
    if not rel: continue
    p = os.path.join(OUT_DIR, rel)
    if not os.path.exists(p): print(f"❌ {name}: 파일 없음 {p}"); continue
    probs = check_format(p); a = answers_of(p); ANS[name] = a
    row = {"후보": name, "파일": rel, "형식": "✅" if not probs else "❌ " + "; ".join(probs), "md5": md5file(p)[:12],
           "답 분포": dict(zip(LETTERS, np.bincount(a[~np.isnan(a)].astype(int), minlength=4).tolist())) if not probs else "-"}
    jp = p[:-4] + ".json"
    if os.path.exists(jp):
        info = json.load(open(jp, encoding="utf-8"))
        kind = "교체 규칙" if "anchor" in info else ("조합" if "cfg" in info else None)
        if kind:
            re_a, desc, notes = (rebuild_anchor if kind == "교체 규칙" else rebuild_ens)(info)
            same = int((re_a != a).sum())
            row["재현"] = ("✅ 같음" if same == 0 else f"❌ {same}문항 다름") + (" · ⚠️ " + "; ".join(notes) if notes else "")
            print(f"▶ {name} [{kind}] {desc}")
            if kind == "조합" and info.get("val_acc") is not None: print(f"   JSON 기록: 검증 {info['val_acc']:.2f}% · 오답 {info.get('val_err')}")
            if kind == "교체 규칙": print(f"   JSON 기록: 교차 적합 살림 {info['cv']['rescue']} · 망침 {info['cv']['harm']} · 채택 {info['cv']['adopt']}")
        else: row["재현"] = "JSON 형식 모름"
    else: row["재현"] = "설정 JSON 없음 (재현 점검 생략)"
    ROWS.append(row)
pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 220)
print(pd.DataFrame(ROWS).to_string(index=False))
names = list(ANS)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        d = int((ANS[names[i]] != ANS[names[j]]).sum())
        print(f"\n{names[i]} vs {names[j]}: test 답이 다른 문항 {d}개 → 공개 리더보드에서는 약 {d / 2:.0f}문항이 갈림 (운만으로 ±{2 * math.sqrt(max(d, 1) / 2):.0f}문항 정도는 흔함)")
'''),
("markdown", "## 5. (선택) 다수결 파일 다시 만들기"),
("code", r'''
if MAJORITY:
    A = np.stack([answers_of(os.path.join(OUT_DIR, f)) for f in MAJORITY["inputs"]], 1)
    cnt = np.stack([(A == k).sum(1) for k in range(4)], 1); maj = cnt.argmax(1)                # 동률(셋 다 다름)이면 알파벳 앞 글자
    sv = answers_of(os.path.join(OUT_DIR, MAJORITY["saved"]))
    print(f"다수결 {len(MAJORITY['inputs'])}개 입력 → 저장된 {MAJORITY['saved']}와 " + ("완전히 같음 ✅" if (maj == sv).all() else f"{int((maj != sv).sum())}문항 다름 ❌"))
else:
    print("MAJORITY = None → 건너뜀")
'''),
]
save(nb_of(FC), "07_final_check.ipynb")
