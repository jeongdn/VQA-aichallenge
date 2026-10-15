# -*- coding: utf-8 -*-
"""import_old_preds.ipynb 시험 (CPU) + 옮긴 뒤 ens_search / manual_ensemble이 멤버로 잡는지까지 확인.
가짜 Drive(E14 시험용)에 옛 형식 파일을 만든다:
  - Qwen3.5-27B up2.0: q35_27b_r16_native를 조금 흔든 예측 (새 모델) → 옮겨져야 함
  - Qwen3.5-27B up1.5: q35_27b_r16_native 예측을 그대로 복사 → 복사본으로 건너뛰어야 함
  - Qwen3-VL-32B up1.5: q3vl_8b를 흔든 예측, id에 .jpg 없음 → id 맞춰서 옮겨져야 함 (계열은 q3vl_32b_r16_native_full의 MODEL_ID)
  - 다른 분할(검증 문항 10개 바뀜) → 거부 / test 파일 없음 → 거부 / 검증 누출(99.7%) → 거부"""
import sys, os, json, shutil, io, contextlib, glob
import numpy as np, pandas as pd
SP = "/tmp/vqa_scratch"
ROOT = f"{SP}/importtest"; RUNS = f"{ROOT}/vqa_runs"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
shutil.copytree(f"{SP}/e14test/vqa_runs", RUNS); os.symlink(f"{SP}/e14test/content", f"{ROOT}/content")

def nb_cells(path): return ["".join(c["source"]) for c in json.load(open(path, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
def run(g, name, src, show=True):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): exec(compile(src, f"<{name}>", "exec"), g)
    except Exception:
        print(buf.getvalue()[-3000:]); raise
    if show: print(f"\n━━━━ [{name}] ━━━━\n{buf.getvalue()}")
    return buf.getvalue()

IMP = nb_cells("/home/user/vqa/import_old/out/import_old_preds.ipynb")
g = {"__name__": "__main__"}
run(g, "1 설정", IMP[0], show=False)
g.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip"})
run(g, "2 데이터", IMP[1]); run(g, "3 함수", IMP[2], show=False)
VAL_IDS, TEST_IDS, Y = g["VAL_IDS"], g["TEST_IDS"], g["Y_VAL"]
LET = "abcd"; rng = np.random.RandomState(7)

def src(run_, pre, split):
    f = f"{RUNS}/{run_}/preds_final_ep1{'_test' if split == 'test' else ''}__{pre}__{1000 if split == 'val' else 6714}.csv"
    return pd.read_csv(f)
def shake(df, s=0.5):
    P = df[["p_a", "p_b", "p_c", "p_d"]].to_numpy() * np.exp(rng.normal(0, s, (len(df), 4))); P /= P.sum(1, keepdims=True)
    out = df.copy(); out[["p_a", "p_b", "p_c", "p_d"]] = P; out["pred"] = [LET[i] for i in P.argmax(1)]; return out
# (1) 새 모델 — Qwen3.5-27B up2.0
shake(src("q35_27b_r16_native", "up2.0", "val")).to_csv(f"{RUNS}/preds_Qwen3.5-27B__up2.0__4bit__valid1000.csv", index=False)
shake(src("q35_27b_r16_native", "up2.0", "test")).to_csv(f"{RUNS}/preds_Qwen3.5-27B__up2.0__4bit__test6714.csv", index=False)
# (2) 복사본 — Qwen3.5-27B up1.5 (행 순서만 섞음)
src("q35_27b_r16_native", "up2.0", "val").sample(frac=1, random_state=1).to_csv(f"{RUNS}/preds_Qwen3.5-27B__up1.5__4bit__valid1000.csv", index=False)
src("q35_27b_r16_native", "up2.0", "test").to_csv(f"{RUNS}/preds_Qwen3.5-27B__up1.5__4bit__test6714.csv", index=False)
# (3) Qwen3-VL-32B — id에 .jpg 없음, 확률 합이 1이 아님(letter mass처럼) → 다시 나눔
v = shake(src("q3vl_8b_r16_native", "up2.0", "val"), 0.8); v["id"] = v["id"].str.replace(".jpg", "", regex=False); v[["p_a", "p_b", "p_c", "p_d"]] *= 0.9
t = shake(src("q3vl_8b_r16_native", "up2.0", "test"), 0.8); t["id"] = t["id"].str.replace(".jpg", "", regex=False)
v.to_csv(f"{RUNS}/preds_Qwen3-VL-32B__up1.5__4bit__valid1000.csv", index=False); t.to_csv(f"{RUNS}/preds_Qwen3-VL-32B__up1.5__4bit__test6714.csv", index=False)
# (4) 다른 분할
bad = shake(src("q38_27b_r16_native", "up2.0", "val")); others = [i for i in pd.read_csv(f"{ROOT}/content/train.csv").id if i not in set(VAL_IDS)]
bad.loc[:9, "id"] = others[:10]; bad.to_csv(f"{RUNS}/preds_X__up2.0__4bit__valid1000.csv", index=False)
shake(src("q38_27b_r16_native", "up2.0", "test")).to_csv(f"{RUNS}/preds_X__up2.0__4bit__test6714.csv", index=False)
# (5) test 없음
shake(src("q38_27b_r16_native", "up2.0", "val")).to_csv(f"{RUNS}/preds_Y__up2.0__4bit__valid1000.csv", index=False)
# (6) 검증 누출
lk = src("q38_27b_r16_native", "up2.0", "val").set_index("id").loc[VAL_IDS].reset_index(); P = np.full((1000, 4), 0.01); P[np.arange(1000), Y] = 0.97; P[:3] = np.roll(P[:3], 1, 1)
lk[["p_a", "p_b", "p_c", "p_d"]] = P; lk.to_csv(f"{RUNS}/preds_Z__up2.0__4bit__valid1000.csv", index=False)
shake(src("q38_27b_r16_native", "up2.0", "test")).to_csv(f"{RUNS}/preds_Z__up2.0__4bit__test6714.csv", index=False)
pd.DataFrame({"id": VAL_IDS}).to_csv(f"{RUNS}/valid_ids.csv", index=False)

g["IMPORTS"] = g["IMPORTS"] + [("x_badsplit", "preds_X__up2.0__4bit__valid1000.csv", "preds_X__up2.0__4bit__test6714.csv", "up2.0", "Qwen/Qwen3.8-27B", ""),
                               ("y_notest", "preds_Y__up2.0__4bit__valid1000.csv", "preds_Y__up2.0__4bit__test6714.csv", "up2.0", "Qwen/Qwen3.8-27B", ""),
                               ("z_leak", "preds_Z__up2.0__4bit__valid1000.csv", "preds_Z__up2.0__4bit__test6714.csv", "up2.0", "Qwen/Qwen3.8-27B", "")]
o4 = run(g, "4 기존 예측", IMP[3]); assert "valid_ids.csv (1000개): 팀 공통 검증 분할과 같음 ✅" in o4
o5 = run(g, "5 확인", IMP[4])
judge = {r["새 실행"]: r["판정"] for r in [dict(zip(["새 실행"], [n])) | {"판정": j} for n, j in []]} if False else None
rows = {n: [l for l in o5.splitlines() if l.startswith("▶ " + n)] for n, *_ in g["IMPORTS"]}
ready = [r[0] for r in g["READY"]]
assert ready == ["q35_27b_old_up20", "q3vl_32b_old_up15"], ready
assert "복사본 (q35_27b_r16_native@up2.0와 같은 예측)" in o5 and "문항이 다름" in o5 and "파일 없음" in o5 and "검증 누출 의심" in o5
assert "확률 합이 1이 아닌 문항이 있어" in o5
o6 = run(g, "6 옮기기", IMP[5]); o7 = run(g, "7 다시 읽기", IMP[6])
for name, sfile, pre, fam in [("q35_27b_old_up20", "preds_Qwen3.5-27B__up2.0__4bit", "up2.0", "Qwen/Qwen3.5-27B"),
                              ("q3vl_32b_old_up15", "preds_Qwen3-VL-32B__up1.5__4bit", "up1.5", "Qwen/Qwen3-VL-32B-Instruct")]:
    d = f"{RUNS}/{name}"; files = sorted(os.listdir(d))
    assert files == sorted([f"preds_final_ep1__{pre}__1000.csv", f"preds_final_ep1_test__{pre}__6714.csv", "train_log.json"]), files
    lg = json.load(open(f"{d}/train_log.json", encoding="utf-8")); assert lg["config"]["MODEL_ID"] == fam and lg["config"]["TRAIN_ON_FULL"] is False
    new_v = pd.read_csv(f"{d}/preds_final_ep1__{pre}__1000.csv"); assert list(new_v.id) == list(VAL_IDS)
    old_v = pd.read_csv(f"{RUNS}/{sfile}__valid1000.csv"); old_v["id"] = [i if i.endswith(".jpg") else i + ".jpg" for i in old_v.id]
    ov = old_v.set_index("id").loc[VAL_IDS][["p_a", "p_b", "p_c", "p_d"]].to_numpy(); ov = ov / ov.sum(1, keepdims=True)
    assert np.allclose(new_v[["p_a", "p_b", "p_c", "p_d"]].to_numpy(), ov, atol=1e-12)
    new_t = pd.read_csv(f"{d}/preds_final_ep1_test__{pre}__6714.csv"); assert list(new_t.id) == list(TEST_IDS)
assert not os.path.exists(f"{RUNS}/q35_27b_old_up15") and not os.path.exists(f"{RUNS}/x_badsplit") and not os.path.exists(f"{RUNS}/z_leak")
print("✅ 옮기기: 새 모델 2개 폴더 생성(표준 이름·id 순서·확률 그대로·합 다시 나눔·계열) · 복사본/다른 분할/test 없음/누출 4개 거부")
# 다시 실행 → 이미 있음
g2 = {"__name__": "__main__"}; run(g2, "1", IMP[0], show=False); g2.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip"})
for i, nm in enumerate(["2", "3", "4", "5"], 1): run(g2, nm, IMP[i], show=False)
o6b = run(g2, "6 again", IMP[5], show=False); assert o6b.count("폴더가 이미 있음") == 2, o6b
o5b = g2  # 두 번째 실행에서는 q35_27b_old_up20이 기존 실행으로 잡혀서, 자기 자신과 비교해 복사본 판정이 나는지 확인
print("✅ 다시 실행해도 폴더를 덮어쓰지 않음")

# ── ens_search · manual_ensemble이 멤버로 잡는지 ──
for nbname, path in [("ens_search", "/home/user/vqa/ens_search/out/ens_search.ipynb"), ("manual_ensemble", "/home/user/vqa/manual_ens/out/manual_ensemble.ipynb")]:
    C = nb_cells(path); h = {"__name__": "__main__"}
    run(h, "1", C[0], show=False); h.update({"DATA_DIR": f"{ROOT}/content", "OUT_DIR": RUNS, "DRIVE_ZIP": "/nonexistent.zip"})
    for i in range(1, 5): run(h, str(i + 1), C[i], show=False)
    mem = h["MEMBERS"] if "MEMBERS" in h else h["NAMES"]
    assert "q35_27b_old_up20" in mem and "q3vl_32b_old_up15" in mem, mem
    fam = h["FAMR"]; assert fam["q35_27b_old_up20"] == "Qwen/Qwen3.5-27B" and fam["q3vl_32b_old_up15"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert h["PRE"]["q3vl_32b_old_up15"] == "up1.5" and h["PRE"]["q35_27b_old_up20"] == "up2.0"
    print(f"✅ {nbname}: 새 실행 2개가 멤버로 들어감 (멤버 {len(mem)}개) · 계열·해상도 맞음")
    if nbname == "ens_search":
        for i in range(5, 9): run(h, str(i + 1), C[i], show=False)
        cands = h["CANDS"]; print(f"   검색 끝: 후보 {len(cands)}개 · 풀 {h['M_S']}개 (코드 {h['CODE']['q35_27b_old_up20']}, {h['CODE']['q3vl_32b_old_up15']})")
print("\n✅✅ import_old_preds 시험 통과")
