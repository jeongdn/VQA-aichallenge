# -*- coding: utf-8 -*-
"""옛 예측 파일(vqa_runs 바로 아래 preds_<모델>__<해상도>__<정밀도>__valid1000/test6714.csv)을
표준 실행 폴더(vqa_runs/<이름>/preds_final_ep1__<해상도>__1000.csv + _test__…__6714.csv + train_log.json)로 옮기는 노트북 빌더.
옮기기 전에 확인: 검증 문항 = 팀 공통 분할, test 6,714 다 있음, 확률 형식, 검증 누출 의심, 기존 실행의 복사본인지.
사용: python3 build_import_old.py [출력 경로]"""
import sys, json, uuid, os, io, contextlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/vqa/import_old/out/import_old_preds.ipynb"
SP = "/tmp/vqa_scratch"
_n14 = {}
_argv = sys.argv; sys.argv = ["build_e14.py", f"{SP}/_e14_tmp_i.ipynb"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("/home/user/vqa/e14/build_e14.py", encoding="utf-8").read(), _n14)
sys.argv = _argv
def rep(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)
DATA = _n14["_n8"]["DATA"]
DATA = rep(DATA, 'ENS_DIR, E8_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR), os.path.join(OUT_DIR, E8_SUBDIR)', 'ENS_DIR = os.path.join(OUT_DIR, ENSEMBLE_SUBDIR)')
DATA = rep(DATA, 'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,} | 제출 파일 폴더: {ENS_DIR}")',
           'print(f"✅ 데이터 OK | 검증 {len(VAL_IDS):,} (팀 공통 분할) | test {len(TEST_IDS):,}")')
HELPERS = _n14["_n8"]["HELPERS"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n") + "\n"})

md("""
# 옛 예측 파일을 실행 폴더로 옮기기 (CPU)

`vqa_runs` 바로 아래에 있는 옛 노트북의 예측 파일(`preds_<모델>__<해상도>__4bit__valid1000.csv` / `…__test6714.csv`)은 앙상블 노트북들이 찾지 못합니다. 이 노트북이 이 파일들을 표준 실행 폴더로 옮깁니다.
- 옮긴 뒤 **ens_search · manual_ensemble · E14 · final_pick을 다시 실행하면 자동으로 멤버로 들어갑니다.**
- 원래 파일은 건드리지 않고 복사만 합니다.

**옮기기 전에 확인하는 것** (하나라도 걸리면 그 파일은 옮기지 않음)
1. 검증 파일의 문항이 팀 공통 검증 1,000문항과 정확히 같은가.
   - 다르면 다른 분할로 학습한 모델이라, 우리 검증 문항을 학습에 썼을 수 있습니다.
2. test 파일에 6,714문항이 다 있는가.
3. 확률 열(`p_a`~`p_d`)이 있고 값이 정상인가.
4. 검증이 99%를 넘지 않는가. 넘으면 검증 누출을 의심합니다.
5. 기존 실행의 복사본이 아닌가.
   - 검증 답이 99.9% 이상 같고 확률도 거의 같으면 같은 모델로 보고 건너뜁니다. 같은 모델을 두 번 넣으면 표만 두 번 세게 됩니다.
   - 가장 비슷한 기존 실행 3개와의 일치율도 보여 줍니다.
""")

md("## 1. 설정")
code(r'''
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
ENSEMBLE_SUBDIR = "ensembles"
SEED, VAL_SIZE = 42, 1000
N_PUBLIC = 3357

SRC_SUBDIR = ""        # 옛 파일이 있는 폴더 (OUT_DIR 기준). 지금은 vqa_runs 바로 아래라 ""
# (새 실행 이름, 검증 파일, test 파일, 해상도, 기반 모델 ID, 같은 계열의 기존 실행 — 있으면 그 train_log의 MODEL_ID를 그대로 씀)
IMPORTS = [
    ("q35_27b_old_up15", "preds_Qwen3.5-27B__up1.5__4bit__valid1000.csv", "preds_Qwen3.5-27B__up1.5__4bit__test6714.csv", "up1.5", "Qwen/Qwen3.5-27B", "q35_27b_r16_native"),
    ("q35_27b_old_up20", "preds_Qwen3.5-27B__up2.0__4bit__valid1000.csv", "preds_Qwen3.5-27B__up2.0__4bit__test6714.csv", "up2.0", "Qwen/Qwen3.5-27B", "q35_27b_r16_native"),
    ("q3vl_32b_old_up15", "preds_Qwen3-VL-32B__up1.5__4bit__valid1000.csv", "preds_Qwen3-VL-32B__up1.5__4bit__test6714.csv", "up1.5", "Qwen/Qwen3-VL-32B-Instruct", "q3vl_32b_r16_native_full"),
]
MIN_VAL_ACC = 90.0     # 이보다 낮으면 고장으로 보고 옮기지 않음
SUSPECT_ACC = 99.0     # 이보다 높으면 검증 누출 의심 → 옮기지 않음
DUP_AGREE = 99.9       # 기존 실행과 검증 답 일치율(%)이 이 이상이고
DUP_MAXDIFF = 1e-3     #   확률 차이가 이 이하면 복사본으로 보고 건너뜀
ADD_DUPLICATES = False # True면 복사본이어도 옮김 (보통 False)
OVERWRITE = False      # True면 이미 있는 같은 이름 폴더를 덮어씀
''')

md("## 2. 데이터 (CSV만 꺼냄) · 팀 공통 검증 분할")
code(DATA)

md("## 3. 예측 파일 읽기 함수 (앙상블 노트북들과 같음)")
code(HELPERS)

md("## 4. 기존 실행의 검증 예측 모으기 (복사본 확인용)")
code(r'''
EXIST = {}
for run in sorted(os.listdir(OUT_DIR)):
    d = os.path.join(OUT_DIR, run)
    if not os.path.isdir(d): continue
    for pre in sorted({m.group(2) for f in os.listdir(d) if (m := re.match(r"^preds_final_ep(\d+)__(.+)__1000\.csv$", f))}):
        try: EXIST[(run, pre)] = load(run, pre, "val")
        except Exception: pass
print(f"기존 실행의 검증 예측 {len(EXIST)}개 (실행 {len({r for r, _ in EXIST})}개)")
_vf = os.path.join(OUT_DIR, "valid_ids.csv")
if os.path.exists(_vf):
    _v = pd.read_csv(_vf); _col = "id" if "id" in _v.columns else _v.columns[0]; _ids = set(_v[_col].astype(str))
    _same = _ids == set(VAL_IDS) or {i + ".jpg" for i in _ids} == set(VAL_IDS)
    print(f"vqa_runs/valid_ids.csv ({len(_ids)}개): 팀 공통 검증 분할과 " + ("같음 ✅" if _same else "다름 ⚠️ — 옛 노트북이 다른 분할을 썼을 수 있음"))
''')

md("## 5. 파일마다 확인")
code(r'''
LETTERS4 = ["p_a", "p_b", "p_c", "p_d"]
def read_old(path, ids):
    d = pd.read_csv(path)
    miss = [c for c in ["id"] + LETTERS4 if c not in d.columns]
    if miss: return None, f"열 없음 {miss} (있는 열: {list(d.columns)[:12]})"
    d["id"] = d["id"].astype(str); want = set(ids)
    if d["id"].isin(want).mean() < 0.5:                                        # id 모양이 다르면 .jpg 붙이기/떼기
        for fix in (lambda s: s + ".jpg", lambda s: s[:-4] if s.endswith(".jpg") else s):
            alt = d["id"].map(fix)
            if alt.isin(want).mean() > 0.99: d["id"] = alt; break
    d = d.drop_duplicates("id", keep="last").set_index("id")
    missing, extra = [i for i in ids if i not in d.index], sorted(set(d.index) - want)
    if missing or extra: return None, f"문항이 다름: 없는 문항 {len(missing)}개 (예: {missing[:3]}) · 더 있는 문항 {len(extra)}개 (예: {extra[:3]})"
    d = d.loc[ids]; P = d[LETTERS4].to_numpy(float)
    if not np.isfinite(P).all() or (P < 0).any(): return None, "확률에 NaN/음수가 있음"
    s = P.sum(1)
    if np.abs(s - 1).max() > 1e-3:
        if (s <= 0).any(): return None, "확률 합이 0인 문항이 있음"
        d[LETTERS4] = P / s[:, None]; print(f"    (확률 합이 1이 아닌 문항이 있어 네 보기로 다시 나눔: 합 {s.min():.3f}~{s.max():.3f})")
    if "pred" not in d.columns: d["pred"] = [LETTERS[i] for i in d[LETTERS4].to_numpy().argmax(1)]
    return d, None

READY, _rows = [], []
for name, fv, ft, pre, mid, like in IMPORTS:
    pv, pt = os.path.join(OUT_DIR, SRC_SUBDIR, fv), os.path.join(OUT_DIR, SRC_SUBDIR, ft)
    row = {"새 실행": name, "해상도": pre}
    print(f"▶ {name} ← {fv}")
    def _no(short, detail): row["판정"] = "❌ " + short; _rows.append(row); print(f"   ❌ {short} — {detail}")
    if not (os.path.exists(pv) and os.path.exists(pt)):
        _no("파일 없음", " · ".join(f for f, ok in ((pv, os.path.exists(pv)), (pt, os.path.exists(pt))) if not ok)); continue
    dv, why = read_old(pv, VAL_IDS)
    if dv is None: _no("검증 파일 문제", why); continue
    dt, why = read_old(pt, TEST_IDS)
    if dt is None: _no("test 파일 문제", why); continue
    P = dv[LETTERS4].to_numpy(float); acc = float((P.argmax(1) == Y_VAL).mean() * 100)
    row.update({"검증(%)": round(acc, 2), "NLL": round(nll(P, Y_VAL), 4), "letter_mass": round(float(dv["letter_mass"].mean()), 4) if "letter_mass" in dv.columns else None})
    sims = sorted(((float((P.argmax(1) == Q.argmax(1)).mean() * 100), float(np.abs(P - Q).max()), r, p) for (r, p), Q in EXIST.items() if r != name),   # 이미 옮긴 자기 폴더는 빼고
                  key=lambda t: (-t[0], t[1], t[2], t[3]))           # 답 일치율 높은 순 → 확률 차이 작은 순
    row["가장 비슷한 기존 실행 (검증 답 일치%)"] = " · ".join(f"{r}@{p} {a:.1f}" for a, _, r, p in sims[:3])
    dup = [(r, p) for a, dmax, r, p in sims if a >= DUP_AGREE and dmax <= DUP_MAXDIFF]
    if acc < MIN_VAL_ACC: row["판정"] = f"❌ 검증 < {MIN_VAL_ACC}% (고장)"
    elif acc > SUSPECT_ACC: row["판정"] = f"❌ 검증 > {SUSPECT_ACC}% (검증 누출 의심)"
    elif dup and not ADD_DUPLICATES: row["판정"] = f"⏭ 복사본 ({dup[0][0]}@{dup[0][1]}와 같은 예측) → 건너뜀"
    else:
        fam = mid
        lg = (read_log(like).get("config") or {}) if like else {}
        if lg.get("MODEL_ID"): fam = str(lg["MODEL_ID"])
        READY.append((name, fv, ft, pre, fam, dv, dt, pv, pt, acc))
        row["판정"] = ("✅ 이미 옮겨져 있음" if os.path.isdir(os.path.join(OUT_DIR, name)) else "✅ 옮길 수 있음") + (" (복사본이지만 ADD_DUPLICATES=True)" if dup else "")
        row["계열 (MODEL_ID)"] = fam
    _rows.append(row); print("   " + row["판정"] + f" | 검증 {acc:.2f}%")
print()
print(pd.DataFrame(_rows).to_string(index=False))
''')

md("## 6. 실행 폴더로 옮기기")
code(r'''
import datetime
def md5file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
DONE = []
for name, fv, ft, pre, fam, dv, dt, pv, pt, acc in READY:
    d = os.path.join(OUT_DIR, name)
    if os.path.exists(d) and os.listdir(d) and not OVERWRITE:
        print(f"⏭ {name}: 폴더가 이미 있음 → 그대로 둠 (다시 만들려면 OVERWRITE=True)"); DONE.append(name); continue
    os.makedirs(d, exist_ok=True)
    cols = ["id", "pred"] + LETTERS4 + [c for c in ("letter_mass", "img_tokens") if c in dv.columns]
    dv.reset_index()[[c for c in cols if c in dv.reset_index().columns]].to_csv(os.path.join(d, f"preds_final_ep1__{pre}__1000.csv"), index=False)
    dt.reset_index()[[c for c in cols if c in dt.reset_index().columns]].to_csv(os.path.join(d, f"preds_final_ep1_test__{pre}__{len(TEST_IDS)}.csv"), index=False)
    log = {"config": {"MODEL_ID": fam, "TRAIN_ON_FULL": False, "PRECISION": "4bit", "EVAL_PRESET": pre,
                      "IMPORTED_FROM": {"val": os.path.join(SRC_SUBDIR, fv), "test": os.path.join(SRC_SUBDIR, ft)}},
           "epoch_acc": {"1": round(acc / 100, 4)},
           "imported": {"when": datetime.datetime.now().isoformat(timespec="seconds"), "val_md5": md5file(pv), "test_md5": md5file(pt),
                        "note": "옛 노트북 예측 파일을 import_old_preds.ipynb로 옮김 (예측값은 그대로, 문항 순서만 맞춤)"}}
    json.dump(log, open(os.path.join(d, "train_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    DONE.append(name); print(f"✅ {name}/ 만듦: preds_final_ep1__{pre}__1000.csv · preds_final_ep1_test__{pre}__{len(TEST_IDS)}.csv · train_log.json")
''')

md("## 7. 앙상블 노트북이 읽는 방식으로 다시 읽어 확인")
code(r'''
_CACHE.clear()
for name in DONE:
    pre = next(p for n, _, _, p, *_ in IMPORTS if n == name)
    fv_, why_v = member_file(name, pre, "val"); ft_, why_t = member_file(name, pre, "test")
    assert fv_ and ft_, f"{name}: 표준 이름으로 못 찾음 ({why_v} / {why_t})"
    Pv, Pt = load(name, pre, "val"), load(name, pre, "test")
    fam = (read_log(name).get("config") or {}).get("MODEL_ID")
    print(f"✅ {name}@{pre}: 검증 {(Pv.argmax(1) == Y_VAL).mean() * 100:.2f}% · test {len(Pt):,}문항 · 계열 {fam}")
print("\n다음: ens_search.ipynb / manual_ensemble.ipynb를 처음부터 다시 실행하면 이 실행들이 멤버로 들어갑니다 (4번 셀 '멤버 N개'가 늘어남).")
''')

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]: c["id"] = uuid.uuid4().hex[:8]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("저장:", OUT, "| 셀", len(cells), "| 코드 셀", sum(c["cell_type"] == "code" for c in cells))
