# -*- coding: utf-8 -*-
"""E7-b 시험용 Drive 구조 (설계한 확률 → 기대 결과를 미리 앎)
- clean4 멤버 5개(검증 = P0) + E7 점수(clean4 1·2위 차 < 0.8인 문항만) + e7_info 기록(고침 3 · 새로 틀림 1 · +2 · 바뀐 답 4)
- ea0e9f 멤버 5개(같은 가중치 확률 평균 = P1 / test = PT, q38 bf16은 _full mix) + ens_search 설정 JSON + 저장된 제출 CSV
- 추가 조합: c0ffee(확률+온도), l0gpr0(로그확률+_full mix), v0te00(다수결 → 건너뜀) — 멤버는 무작위 확률
기대: ea0e9f 검증 고침 3 · 새로 틀림 1 · +2 · 바뀐 답 4 · 섞음 14 · 점수 없음 2 / test 바뀐 답 30 · 불확실 95 · 섞음 85 · 점수 없음 7"""
import os, json, shutil, hashlib
import numpy as np, pandas as pd
ROOT = "/tmp/vqa_scratch/e7btest"
U = "/tmp/vqa_uploads"
if os.path.exists(ROOT): shutil.rmtree(ROOT)
DATA, RUNS = f"{ROOT}/content", f"{ROOT}/vqa_runs"; os.makedirs(DATA); os.makedirs(RUNS)
for src, dst in [("6247e8ae-train.csv", "train.csv"), ("847ba753-test.csv", "test.csv"), ("eb905257-sample_submission.csv", "sample_submission.csv")]:
    shutil.copy(f"{U}/{src}", f"{DATA}/{dst}")
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
train["answer"] = train["answer"].astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(train.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val = train.iloc[sorted(vi)].reset_index(drop=True)
assert hashlib.md5(",".join(val.id).encode()).hexdigest() == "df05e4bc19d27ebceeef2570672149c2"
y = val.answer.map("abcd".index).to_numpy(); N, NT = len(val), len(test)
NEGP = r"아닌|않는|없는|틀린|잘못"
NEGV = val.question.astype(str).str.contains(NEGP).to_numpy(); NEGT = test.question.astype(str).str.contains(NEGP).to_numpy()
R = np.random.RandomState(7)

def row(top, p_top, sec, p_sec):
    p = np.zeros(4); p[top] = p_top; p[sec] = p_sec; rest = [k for k in range(4) if k not in (top, sec)]
    p[rest] = (1 - p_top - p_sec) / len(rest); return p
def conf(g): return row(g, 0.97, (g + 1) % 4, 0.01)

# ── 검증 설계 ──
nonneg = R.permutation(np.where(~NEGV)[0]); negi = R.permutation(np.where(NEGV)[0]); c = 0
def take(n):
    global c; out = nonneg[c:c + n]; c += n; return [int(i) for i in out]
S = {"A": take(2), "B": take(1), "C": take(1), "Dm": take(2), "E": take(2), "F": take(10), "G": [int(negi[0])]}
P0 = np.stack([conf(y[i]) for i in range(N)]); P1 = P0.copy()
S_VAL = {}                      # id 위치 → 보기별 s (E7 점수)
def s_pref(k, hi=3.0): s = np.full(4, -hi); s[k] = hi; return s
for i in S["A"]:  w = (y[i] + 1) % 4; P0[i] = row(w, 0.45, y[i], 0.40); P1[i] = row(y[i], 0.95, w, 0.02); S_VAL[i] = s_pref(y[i])
for i in S["B"]:  w = (y[i] + 1) % 4; P0[i] = row(w, 0.45, y[i], 0.40); P1[i] = row(w, 0.45, y[i], 0.40); S_VAL[i] = s_pref(y[i])
for i in S["C"]:  w = (y[i] + 1) % 4; P0[i] = row(y[i], 0.50, w, 0.40); P1[i] = row(y[i], 0.50, w, 0.40); S_VAL[i] = s_pref(w)
for i in S["Dm"]: w = (y[i] + 1) % 4; P1[i] = row(w, 0.45, y[i], 0.40)                                   # clean4 확신 → E7 점수 없음
for i in S["E"]:  w = (y[i] + 1) % 4; P0[i] = row(y[i], 0.75, w, 0.15); P1[i] = row(w, 0.45, y[i], 0.40); S_VAL[i] = s_pref(y[i])
for i in S["F"]:  w = (y[i] + 1) % 4; P0[i] = row(y[i], 0.50, w, 0.40); P1[i] = row(y[i], 0.50, w, 0.40); S_VAL[i] = s_pref(y[i])
for i in S["G"]:  w = (y[i] + 1) % 4; P0[i] = row(w, 0.45, y[i], 0.40); P1[i] = row(w, 0.45, y[i], 0.40); S_VAL[i] = s_pref(y[i])
m0 = np.sort(P0, 1); m0 = m0[:, -1] - m0[:, -2]
assert set(np.where(m0 < 0.8)[0]) == set(S_VAL), "E7 점수는 clean4 차이 < 0.8인 문항에만"

# ── test 설계 (ea0e9f 조합 = PT, q38 bf16은 _full과 섞어야 PT가 됨) ──
yt = R.randint(0, 4, NT)                      # '조합 1위' 보기 (정답은 모름)
PT = np.stack([conf(yt[i]) for i in range(NT)])
tnon = R.permutation(np.where(~NEGT)[0]); tneg = R.permutation(np.where(NEGT)[0]); ct = 0
def ttake(n):
    global ct; out = tnon[ct:ct + n]; ct += n; return [int(i) for i in out]
T = {"TC": ttake(30), "TA": ttake(50), "TM": ttake(7), "TF": ttake(5), "TS": ttake(20), "TN": [int(i) for i in tneg[:3]]}
S_TEST = {}
for i in T["TC"]: sec = (yt[i] + 1) % 4; PT[i] = row(yt[i], 0.45, sec, 0.40); S_TEST[i] = s_pref(sec)          # E7이 2위를 강하게 → 바뀜
for i in T["TA"]: sec = (yt[i] + 1) % 4; PT[i] = row(yt[i], 0.50, sec, 0.40); S_TEST[i] = s_pref(yt[i])        # E7 동의
for i in T["TM"]: sec = (yt[i] + 1) % 4; PT[i] = row(yt[i], 0.45, sec, 0.40)                                   # 점수 없음
for i in T["TF"]: sec = (yt[i] + 1) % 4; PT[i] = row(yt[i], 0.45, sec, 0.40); S_TEST[i] = s_pref(yt[i])        # _full을 안 섞으면 답이 뒤집힘
for i in T["TS"]: S_TEST[i] = s_pref((yt[i] + 2) % 4)                                                          # 조합 확신 → 안 섞임
for i in T["TN"]: sec = (yt[i] + 1) % 4; PT[i] = row(yt[i], 0.45, sec, 0.40); S_TEST[i] = s_pref(sec)          # 부정형 → 안 섞음
DELTA = np.zeros((NT, 4))
for i in T["TF"]: sec = (yt[i] + 1) % 4; DELTA[i, yt[i]] = -0.3; DELTA[i, sec] = +0.3

def save_preds(path, ids, Pm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": ids, "pred": ["abcd"[i] for i in Pm.argmax(1)], "p_a": Pm[:, 0], "p_b": Pm[:, 1], "p_c": Pm[:, 2], "p_d": Pm[:, 3],
                  "letter_mass": 0.999, "img_tokens": 4}).to_csv(path, index=False)
VF, TF = "preds_final_ep1__up2.0__1000.csv", f"preds_final_ep1_test__up2.0__{NT}.csv"
for r in ["q35_27b_r16_native", "q38_27b_r16_native", "gemma4_31b_r16_up20_dev", "gemma4_31b_r16_up20", "hcx32b_r16_native"]:
    save_preds(f"{RUNS}/{r}/{VF}", val.id, P0); save_preds(f"{RUNS}/{r}/{TF}", test.id, np.stack([conf(k) for k in yt]))
EA = ["gemma4_31b_r16_up20_bf16", "gemma4_31b_r16_up20_ocr_bf16", "ivl35_38b_r16_native", "q35_27b_r16_native_bf16", "q38_27b_r16_native_bf16"]
for r in EA[:4]: save_preds(f"{RUNS}/{r}/{VF}", val.id, P1); save_preds(f"{RUNS}/{r}/{TF}", test.id, PT)
save_preds(f"{RUNS}/{EA[4]}/{VF}", val.id, P1); save_preds(f"{RUNS}/{EA[4]}/{TF}", test.id, PT + DELTA)
save_preds(f"{RUNS}/q38_27b_r16_native_full/{TF}", test.id, PT - DELTA)
# 추가 조합용 무작위 멤버
RND = {}
for r in ["x_temp_a", "x_temp_b", "x_log_a", "x_log_b", "x_log_full"]:
    RND[r] = (R.dirichlet(np.ones(4) * 0.7, N), R.dirichlet(np.ones(4) * 0.7, NT))
    if r != "x_log_full": save_preds(f"{RUNS}/{r}/{VF}", val.id, RND[r][0])
    save_preds(f"{RUNS}/{r}/{TF}", test.id, RND[r][1])

def md5file(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def temp_scale(p, T): q = np.clip(p, 1e-12, 1.0) ** (1.0 / T); return q / q.sum(1, keepdims=True)
def combine(Ps, w, method):
    w = np.asarray(w, float); w = w / w.sum()
    if method == "prob": return sum(wi * p for wi, p in zip(w, Ps))
    if method == "logprob":
        s = sum(wi * np.log(np.clip(p, 1e-6, 1.0)) for wi, p in zip(w, Ps)); s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    return sum(wi * np.eye(4)[p.argmax(1)] for wi, p in zip(w, Ps)) + 1e-3 * sum(wi * p for wi, p in zip(w, Ps))
def rd(r, f): return pd.read_csv(f"{RUNS}/{r}/{f}").set_index("id")
def P_file(path, ids): return pd.read_csv(path).set_index("id").loc[ids][["p_a", "p_b", "p_c", "p_d"]].to_numpy(float)

def make_base(sub, name, tag, weights, method, temp, full, Ts=None, fulls=None):
    # ens_search save()와 같은 모양의 JSON + 제출 CSV (답은 이 스크립트가 따로 계산)
    d = f"{RUNS}/{sub}"; os.makedirs(d, exist_ok=True); mem = {}; PsV, PsT = [], []
    for r in weights:
        fv, ft = f"{RUNS}/{r}/{VF}", f"{RUNS}/{r}/{TF}"
        Pv, Pt = P_file(fv, val.id), P_file(ft, test.id)
        mem[r] = {"family": "fx/" + r, "preset": "up2.0", "T": (Ts or {}).get(r, 1.0), "val_acc": float((Pv.argmax(1) == y).mean() * 100),
                  "val": os.path.relpath(fv, RUNS), "test": os.path.relpath(ft, RUNS), "val_md5": md5file(fv), "test_md5": md5file(ft)}
        if full != "none" and r in (fulls or {}):
            ff = f"{RUNS}/{fulls[r]}/{TF}"; mem[r].update({"full_test": os.path.relpath(ff, RUNS), "full_test_md5": md5file(ff)})
            Pf = P_file(ff, test.id); Pt = (Pt + Pf) / 2 if full == "mix" else Pf
        if temp == "val": Pv, Pt = temp_scale(Pv, mem[r]["T"]), temp_scale(Pt, mem[r]["T"])
        PsV.append(Pv); PsT.append(Pt)
    w = [weights[r] for r in weights]; SV, ST = combine(PsV, w, method), combine(PsT, w, method)
    safe = name.replace(" · ", "_").replace(" ", "_"); fn = f"submission_{safe}_{tag}.csv"
    at = ST.argmax(1); pd.DataFrame({"id": test.id, "answer": ["abcd"[i] for i in at]}).to_csv(f"{d}/{fn}", index=False)
    info = {"name": name, "file": fn, "cfg": {"weights": {r: float(v) for r, v in weights.items()}, "method": method, "temp": temp, "full": full},
            "val_acc": float((SV.argmax(1) == y).mean() * 100), "val_err": int((SV.argmax(1) != y).sum()), "val_nll": None, "ref": "clean4_qg_hcx",
            "members": mem, "T_GRID": "T70", "created": "2026-09-27T11:00:00", "test_dist": {}}
    json.dump(info, open(f"{d}/{fn[:-4]}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return SV, ST, f"{d}/{fn}"
SV1, ST1, EA_CSV = make_base("ens_search", "전수 1위 · 확률", "ea0e9f", {r: 1 for r in EA}, "prob", "none", "mix", fulls={"q38_27b_r16_native_bf16": "q38_27b_r16_native_full"})
assert np.allclose(SV1, P1) and np.allclose(ST1, PT) and (ST1.argmax(1) == PT.argmax(1)).all()
_nomix = combine([P_file(f"{RUNS}/{r}/{TF}", test.id) for r in EA], [1] * 5, "prob").argmax(1)
assert int((_nomix != ST1.argmax(1)).sum()) == len(T["TF"]), "TF: _full을 안 섞으면 답이 뒤집혀야 함"
make_base("ens_search", "좌표 상승 · 확률+온도", "c0ffee", {"x_temp_a": 2.5, "x_temp_b": 0.5, "gemma4_31b_r16_up20_bf16": 1}, "prob", "val", "mix",
          Ts={"x_temp_a": 1.7, "x_temp_b": 0.4, "gemma4_31b_r16_up20_bf16": 1.2})
make_base("manual_ens", "manual log", "l0gpr0", {"x_log_a": 1, "x_log_b": 1.5}, "logprob", "none", "mix", fulls={"x_log_b": "x_log_full"})
make_base("manual_ens", "manual vote", "v0te00", {"x_log_a": 1, "x_log_b": 1}, "vote", "none", "none")
# 제출 파일 폴더 (clean4 등, 읽기만)
os.makedirs(f"{RUNS}/ensembles", exist_ok=True)
pd.DataFrame({"id": test.id, "answer": ["abcd"[k] for k in yt]}).to_csv(f"{RUNS}/ensembles/submission_clean4_qg_hcx.csv", index=False)

# ── E7 점수 파일 (중복 행: 앞의 틀린 값은 뒤의 값으로 덮여야 함 / test 한 문항은 보기 3개만 → 점수 없음) ──
E7D = f"{RUNS}/e7_yn_gemma_4_31b_it"; os.makedirs(E7D)
def yn_rows(S_, ids, dup_first=None, partial=None):
    rows = []
    for i, s in S_.items():
        for k in range(4):
            if partial is not None and i == partial and k == 3: continue
            sg = 1 / (1 + np.exp(-s[k])); rows.append({"id": ids[i], "opt": k, "s": float(s[k]), "p_yes": 0.95 * sg, "p_no": 0.95 * (1 - sg), "yn_mass": 0.95, "img_tokens": 1120})
    if dup_first is not None:
        bad = [{"id": ids[dup_first], "opt": k, "s": float(-v), "p_yes": 0.5, "p_no": 0.45, "yn_mass": 0.95, "img_tokens": 1120} for k, v in enumerate(S_[dup_first])]
        rows = bad + rows
    return pd.DataFrame(rows)
_part = T["TM"][0]; S_TEST_W = dict(S_TEST); S_TEST_W[_part] = s_pref(yt[_part])          # TM 한 문항: 보기 3개만 채점
yn_rows(S_VAL, list(val.id), dup_first=S["B"][0]).to_csv(f"{E7D}/yn_val__up2.0.csv", index=False)
yn_rows(S_TEST_W, list(test.id), dup_first=T["TC"][0], partial=_part).to_csv(f"{E7D}/yn_test__up2.0.csv", index=False)
json.dump({"model_id": "google/gemma-4-31B-it", "precision": "bf16", "prefill": "", "preset": "up2.0", "base": "clean4_qg_hcx", "tau": 0.5, "alpha": 0.25,
           "exclude_neg": True, "val_up2.0": {"primary": {"fixed": 3, "broken": 1, "net": 2, "D": 4}, "verdict": "보류 — 1차 가설 순이득 +2", "yn_mass": 0.95}},
          open(f"{E7D}/e7_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
EXP_TEST = PT.argmax(1).copy()
for i in T["TC"]: EXP_TEST[i] = (yt[i] + 1) % 4
json.dump({"S": S, "T": T, "val_ids": list(val.id), "test_ids": list(test.id), "exp_test": EXP_TEST.tolist(), "ea_csv": EA_CSV},
          open(f"{ROOT}/fixture.json", "w"), ensure_ascii=False)
print("E7-b fixtures OK |", {k: len(v) for k, v in S.items()}, {k: len(v) for k, v in T.items()})
