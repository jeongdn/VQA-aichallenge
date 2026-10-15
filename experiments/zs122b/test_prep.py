# -*- coding: utf-8 -*-
"""노트북 6번 셀(미리 줄이기)을 실제 5.17 이미지 프로세서와 대조: 실제 사진(업로드 30장) + 극단 비율·아주 작은/큰 합성 이미지.
경로 A = 원본 → 프로세서(up2.0 범위) (stage2와 같은 경로)
경로 B = 노트북 prep_image → 프로세서(vLLM 쪽 넓은 범위)
두 경로의 pixel_values·격자가 완전히 같아야 하고, 이미지 토큰 수 = (h/32)·(w/32)여야 함."""
import os, glob, json, math
import numpy as np, pandas as pd
from PIL import Image
from transformers import AutoProcessor

SP = "/tmp/vqa_scratch/zs122b"
U = "/tmp/vqa_uploads"
NB = "/home/user/vqa/zs122b/out/zs122b_vllm.ipynb"
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
CFG, IMG_CELL = cells[0], cells[5]
assert "def prep_image" in IMG_CELL

imgdir = f"{SP}/prep_imgs"; os.makedirs(imgdir, exist_ok=True)
paths = sorted(glob.glob(f"{U}/*.jpg")) + sorted(glob.glob(f"{U}/*.png"))
rng = np.random.RandomState(0)
for (w, h) in [(40, 72), (5000, 30), (30, 5000), (6000, 4000), (4032, 3024), (1, 150), (900, 899), (2048, 1536), (1920, 1080), (111, 7777)]:
    a = rng.randint(0, 255, (h, w, 3), dtype=np.uint8); p = f"{imgdir}/syn_{w}x{h}.png"
    Image.fromarray(a).save(p); paths.append(p)
print("이미지", len(paths))

g = {"__name__": "__main__"}
exec(compile(CFG, "<cfg>", "exec"), g)
proc = AutoProcessor.from_pretrained(f"{SP}/tiny_proc")
from PIL import ImageOps
def load_image(rel):
    im = Image.open(rel); im = ImageOps.exif_transpose(im); return im.convert("RGB")
g.update({"processor": proc, "load_image": load_image, "math": math, "Image": Image, "np": np,
          "val_df": pd.DataFrame({"id": [os.path.basename(p) for p in paths[:3]], "path": paths[:3]})})
exec(compile(IMG_CELL, "<img>", "exec"), g)
assert g["PRE_RESIZE"] is True, "노트북 점검 셀이 미리 줄이기를 껐음"
IP, MIN_PX, MAX_PX, FACTOR, VK = proc.image_processor, g["MIN_PX"], g["MAX_PX"], g["FACTOR"], g["VLLM_MM_KW"]
bad, rows = [], []
for p in paths:
    im0 = load_image(p)
    try:
        A = IP(images=[im0], min_pixels=MIN_PX, max_pixels=MAX_PX, return_tensors="pt")
    except ValueError as e:                                  # 비율 200 초과 — stage2도 같은 오류 (노트북도 같은 오류를 내야 함)
        try: g["prep_image"](p); bad.append((p, "prep는 통과했는데 프로세서는 오류"))
        except ValueError: rows.append((os.path.basename(p), im0.size, "비율>200: 둘 다 오류 (같음)"))
        continue
    im1, (h, w) = g["prep_image"](p)
    B = IP(images=[im1], **VK, return_tensors="pt")
    same = A["image_grid_thw"].tolist() == B["image_grid_thw"].tolist() and A["pixel_values"].shape == B["pixel_values"].shape
    diff = float((A["pixel_values"] - B["pixel_values"]).abs().max()) if same else float("inf")
    ntok = int(A["image_grid_thw"][0].prod()) // 4
    ok = same and diff == 0.0 and ntok == (h // FACTOR) * (w // FACTOR) and MIN_PX <= h * w <= int(MAX_PX * 1.1) and im1.size == (w, h)
    rows.append((os.path.basename(p)[-28:], im0.size, (w, h), ntok, diff))
    if not ok: bad.append((p, same, diff, ntok, (h, w)))
for r in rows: print(" ", r)
print("불일치:", bad)
assert not bad
print("✅ 미리 줄이기 = 5.17 프로세서 경로와 픽셀까지 같음 (", len(rows), "장 )")
