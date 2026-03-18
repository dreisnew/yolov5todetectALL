import cv2, os
from string import ascii_uppercase

# ─── CONFIG ────────────────────────────────────────────────────
INPUT_DIR       = r"C:\Users\Andrea Chiang\Downloads\LABELS FOR M"
OUTPUT_DIR      = r"C:\Users\Andrea Chiang\Downloads\LABELS FOR M\new"
MARGIN          = 10                 # px from edges
LETTER_BOX_FRAC = 0.15               # box height = 10% of image height
BOX_PAD_FRAC    = 0.02               # padding = 2% of image height
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────────────────────────

# 1) Collect all images
imgs = []
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.png','.jpg','jpeg','tif','tiff')):
            imgs.append((root, f))
# sort by filename (or change key if you want folder-order)
imgs.sort(key=lambda x: x[1])

# pre‐measure letter "A" at scale=1 to help compute scale dynamically
# instead of “_, (base_w, base_h), base_bl = …”
(base_w, base_h), base_bl = cv2.getTextSize("A", FONT, 1.0, 1)


# 2) Process & stamp
for idx, (root, fname) in enumerate(imgs):
    # pick letter: 0→A, 1→B, ... 25→Z; wrap around if >26
    letter = ascii_uppercase[idx % len(ascii_uppercase)]

    # load image
    img_path = os.path.join(root, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W = img.shape[:2]
    BOX_PAD = int(H * BOX_PAD_FRAC)
    desired_box_h = H * LETTER_BOX_FRAC

    # compute fontScale so box height ≈ desired_box_h
    fontScale = (desired_box_h - base_bl - 2*BOX_PAD) / base_h
    fontScale = max(fontScale, 0.1)
    thickness = max(1, int(fontScale))

    # re‐measure at this scale
    (tw, th), baseline = cv2.getTextSize(letter, FONT, fontScale, thickness)

    # compute box coords
    box_w = tw + 2*BOX_PAD
    box_h = th + baseline + 2*BOX_PAD
    x2 = W - MARGIN
    y2 = H - MARGIN
    x1 = x2 - box_w
    y1 = y2 - box_h

    # draw white box + black letter
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), -1)
    text_x = x1 + BOX_PAD
    text_y = y1 + BOX_PAD + th
    cv2.putText(img, letter, (text_x, text_y),
                FONT, fontScale, (0,0,0), thickness,
                cv2.LINE_AA)

    # save into mirrored folder structure
    rel = os.path.relpath(root, INPUT_DIR)
    outdir = os.path.join(OUTPUT_DIR, rel)
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, fname), img)

print("✅ Done—stamped images A, B, C, … in order!") 

# import os, glob
# import pandas as pd
# from sklearn.metrics import confusion_matrix

# # 1) point these at your actual folders
# GT_LABEL_DIR   = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_BBs_80-10-10_split_with_augemented\data_80-10-10_augmented\labels\test"            # ground-truth .txt files
# PRED_LABEL_DIR = r"C:\Users\Andrea Chiang\Downloads\python\yolo_thesis\yolov5\runs\detect\with aug\ALL_HEM_80-10-10_with_augments_100Ev5lconf70\labels"             # YOLO’s predicted .txt files

# y_true = []
# y_pred = []

# for gt_file in glob.glob(f"{GT_LABEL_DIR}/*.txt"):
#     img_id = os.path.basename(gt_file)[:-4]
#     # read GT labels
#     with open(gt_file) as f:
#         for line in f:
#             cls_true = int(float(line.split()[0]))  # <— note float→int cast
#             # read YOLO’s predicted file (if it exists)
#             pred_path = os.path.join(PRED_LABEL_DIR, img_id + ".txt")
#             if os.path.exists(pred_path):
#                 with open(pred_path) as pf:
#                     cls_pred = int(float(pf.readline().split()[0]))
#                     y_true.append(cls_true)
#                     y_pred.append(cls_pred)
#             else:
#                 # YOLO missed this image → skip it
#                 continue

# # build 2×2 confusion matrix for just ALL & HEM
# labels = [0, 1]                                # 0=ALL, 1=HEM
# names  = ["ALL", "HEM"]
# cm = confusion_matrix(y_true, y_pred, labels=labels)
# cm_df = pd.DataFrame(
#     cm,
#     index=[f"Actual {n}" for n in names],
#     columns=[f"Predicted {n}" for n in names]
# )
# print("Without threshold optimization\n")
# print(cm_df.to_string())

# import numpy as np
# row_sums = cm.sum(axis=1, keepdims=True)
# cm_norm = cm / row_sums
# print(cm_norm)

