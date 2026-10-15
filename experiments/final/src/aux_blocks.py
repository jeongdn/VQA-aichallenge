AUX_CFG = r'''
DRIVE_ZIP    = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ZIP_PASSWORD = "YOUR_ZIP_PASSWORD"
DATA_DIR     = "/content"
OUT_DIR      = "/content/drive/MyDrive/vqa_runs"
AUX_OUT      = f"{OUT_DIR}/aux/aux_open_v1.csv"      # E4 처리군 노트북의 AUX_GEN_CSV와 같은 경로
EXPECTED_SPLIT_MD5 = "df05e4bc19d27ebceeef2570672149c2"
MAX_ANS_CHARS = 20                                    # 이보다 긴 정답 글자는 제외 (짧게 읽어 쓰는 답만)
'''
AUX_BUILD = r'''
import os, re, hashlib, subprocess
import numpy as np, pandas as pd
try:
    from google.colab import drive; drive.mount("/content/drive")
except ImportError:
    pass
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    r = subprocess.run(["7z", "e", "-y", f"-p{ZIP_PASSWORD}", f"-o{DATA_DIR}", DRIVE_ZIP, "train.csv", "-r"], capture_output=True, text=True)
    assert os.path.exists(os.path.join(DATA_DIR, "train.csv")), r.stdout[-500:] + r.stderr[-500:]
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), dtype={"id": str})
for c in ["question", "a", "b", "c", "d"]: df[c] = df[c].astype(str)
df["answer"] = df["answer"].astype(str).str.strip().str.lower()
rng = np.random.RandomState(42); vi = []
for L in "abcd":
    idx = np.where(df.answer.values == L)[0]; rng.shuffle(idx); vi.extend(idx[:250].tolist())
val_ids = df.id.iloc[sorted(vi)].tolist()
assert hashlib.md5(",".join(val_ids).encode()).hexdigest() == EXPECTED_SPLIT_MD5, "검증 분할이 팀 공통과 다릅니다"
tr = df[~df.id.isin(set(val_ids))].copy(); assert len(tr) == 5714
tr["answer_text"] = [str(r[r["answer"]]).strip() for _, r in tr.iterrows()]

# 보기가 없으면 성립하지 않거나, 이미지에 적힌 글자를 그대로 쓰는 답이 아닌 문항은 뺌
NEED_OPT = r"다음\s*중|보기\s*중|중\s*(?:옳|맞|올바|알맞|적절|해당)|옳은\s*것|맞는\s*것|올바른\s*것|알맞은\s*것|적절한\s*것|고르|선택지|선택하"
NEG      = r"아닌|않은|않는|않을|없는|틀린|잘못|제외"
OPT_META = r"모두|전부|해당\s*없|알\s*수\s*없|위의|모든|all of the above|none of|보기"
COLOR    = r"색|컬러|빛깔"
MEANING  = r"의미|뜻|나타내|상징|목적|용도|이유|왜|어떤\s*(?:종류|업종|곳|장소)|무엇을\s*하는|추론|추측|예상|분위기|느낌"
CALC     = r"몇\s*(?:개|명|번째|가지|대|장|병|잔|배)|개수|합계|합은|총\s|총액|차이|더하|빼|계산|평균|얼마나\s*(?:더|많|적)|할인된|거스름"
KEEP_Q   = r"가격|얼마|시간|몇\s*시|영업|운영|전화|번호|연락처|이름|상호|가게|매장|간판|적혀|적힌|쓰여|쓰인|써\s*있|표시|문구|글자|글씨|날짜|요일|주소|메뉴|제목|브랜드|층"
q, a = tr.question, tr.answer_text
opts_unique = tr.apply(lambda r: len({str(r[k]).strip() for k in "abcd"}) == 4, axis=1)
masks = {
    "보기가 있어야 성립(다음 중·고르기)": q.str.contains(NEED_OPT),
    "부정형(아닌·없는)":               q.str.contains(NEG),
    "정답이 '모두/해당 없음' 류":        a.str.contains(OPT_META, case=False),
    "색상":                            q.str.contains(COLOR),
    "의미·목적·추론":                    q.str.contains(MEANING),
    "개수·계산":                        q.str.contains(CALC),
    f"긴 답(>{MAX_ANS_CHARS}자)":        ~a.str.len().between(1, MAX_ANS_CHARS),
    "선지 중복":                        ~opts_unique,
    "글자 읽기 문항 아님":               ~(q.str.contains(KEEP_Q) | a.str.contains(r"\d")),
}
drop = np.zeros(len(tr), bool); rows = []
for k, mk in masks.items():
    n = int((mk.values & ~drop).sum()); rows.append({"제외 이유(순서대로)": k, "문항": n}); drop |= mk.values
print(pd.DataFrame(rows).to_string(index=False))
aux = tr.loc[~drop, ["id", "path", "question", "answer_text"]].reset_index(drop=True)
assert not (set(aux.id) & set(val_ids)), "검증 문항이 섞였습니다"
print(f"
보조학습 문항: {len(aux):,} / 5,714 ({len(aux) / 5714 * 100:.0f}%)")
os.makedirs(os.path.dirname(AUX_OUT), exist_ok=True)
aux.to_csv(AUX_OUT, index=False)
aux.sample(min(40, len(aux)), random_state=0).to_csv(AUX_OUT.replace(".csv", "_sample40.csv"), index=False)
print("저장:", AUX_OUT, "| 표본 40개:", AUX_OUT.replace(".csv", "_sample40.csv"))
kinds = {"가격": r"가격|얼마", "시간·날짜": r"시간|몇\s*시|영업|운영|날짜|요일", "전화": r"전화|번호|연락처", "이름·상호": r"이름|상호|가게|매장|간판|브랜드|제목", "층": r"층"}
print("유형(겹칠 수 있음):", {k: int(aux.question.str.contains(p).sum()) for k, p in kinds.items()})
'''
AUX_SHOW = r'''
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
print(aux.sample(min(40, len(aux)), random_state=0)[["id", "question", "answer_text"]].to_string(index=False))
print("
→ 이상한 문항(보기 없이는 답이 정해지지 않거나, 이미지 글자와 다른 표현)이 여러 개 보이면 E4 처리군을 돌리기 전에 알려 주세요.")
'''

