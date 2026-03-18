#region just mAP
# import os
# import sys
# import json
# import numpy as np
# import tifffile
# from pycocotools import mask as mask_utils
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# # ─── CONFIG ────────────────────────────────────────────────────────────────────
# GT_ROOT     = r"C:\Users\Andrea Chiang\Downloads\For Testing Instance Masks"
# PRED_ROOT   = r"C:\Users\Andrea Chiang\Downloads\cellpose_for_testing"
# GT_SUFFIX   = "_instmask.tif"    # your ground-truth instance masks
# PRED_SUFFIX = "_wbc_mask.tif"    # Cellpose’s per-cell instance output

# # ─── BUILD FUNCTIONS ───────────────────────────────────────────────────────────
# def build_coco_annotations(root, suffix, is_pred=False):
#     """
#     Walk 'root' for files ending with 'suffix'. For each:
#       - Read mask to get height/width.
#       - Add an entry to 'images'.
#       - Split into per-instance labels and encode each as RLE.
#     Returns (images_list, annotations_list).
#     """
#     images = []
#     annots = []
#     ann_id = 1
#     img_map = {}  # key -> (img_id, full_path)

#     # 1) Collect images + dimensions
#     for dp, _, files in os.walk(root):
#         for fn in files:
#             if not fn.endswith(suffix):
#                 continue
#             full = os.path.join(dp, fn)
#             rel  = os.path.relpath(full, root)  # e.g. "ALL_resized/23BM66/Snap_011_instmask.tif"
#             mask_arr = tifffile.imread(full)
#             h, w = mask_arr.shape[:2]

#             img_id = len(images) + 1
#             images.append({
#                 "id":        img_id,
#                 "file_name": rel,
#                 "height":    h,
#                 "width":     w
#             })
#             key = rel[:-len(suffix)]
#             img_map[key] = (img_id, full)

#     # 2) Build annotations
#     for key, (img_id, full) in img_map.items():
#         mask = tifffile.imread(full).astype(np.uint16)
#         for inst in np.unique(mask):
#             if inst == 0:
#                 continue
#             binm = (mask == inst).astype(np.uint8)
#             rle  = mask_utils.encode(np.asfortranarray(binm))
#             rle["counts"] = rle["counts"].decode("ascii")
#             bbox = mask_utils.toBbox(rle).tolist()
#             area = int(mask_utils.area(rle))

#             annot = {
#                 "id":           ann_id,
#                 "image_id":     img_id,
#                 "category_id":  1,
#                 "segmentation": rle,
#                 "bbox":         bbox,
#                 "area":         area,
#                 "iscrowd":      0,
#             }
#             if is_pred:
#                 annot["score"] = 1.0
#             annots.append(annot)
#             ann_id += 1

#     return images, annots

# def remap_and_filter(images, annots):
#     """
#     Keep only annotations whose image_id is in 'images',
#     then remap image IDs to contiguous 1..N.
#     """
#     valid_ids = {img["id"] for img in images}
#     annots[:] = [a for a in annots if a["image_id"] in valid_ids]

#     old2new = {}
#     for new_id, img in enumerate(images, start=1):
#         old2new[img["id"]] = new_id
#         img["id"] = new_id
#     for a in annots:
#         a["image_id"] = old2new[a["image_id"]]

# # ─── MAIN ──────────────────────────────────────────────────────────────────────
# # 1) Build GT and prediction sets
# gt_images,   gt_annots   = build_coco_annotations(GT_ROOT,   GT_SUFFIX,   is_pred=False)
# pred_images, pred_annots = build_coco_annotations(PRED_ROOT, PRED_SUFFIX, is_pred=True)

# # 2) Keep only common images (by stripping suffix)
# gt_keys   = {img["file_name"][:-len(GT_SUFFIX)] for img in gt_images}
# pred_keys = {img["file_name"][:-len(PRED_SUFFIX)] for img in pred_images}
# common    = gt_keys & pred_keys
# if not common:
#     print("❌ No matching GT↔pred pairs found. Check your roots & suffixes.", file=sys.stderr)
#     sys.exit(1)

# gt_images   = [img for img in gt_images   if img["file_name"][:-len(GT_SUFFIX)]   in common]
# pred_images = [img for img in pred_images if img["file_name"][:-len(PRED_SUFFIX)] in common]

# # 3) Filter annotations and remap IDs
# remap_and_filter(gt_images, gt_annots)
# remap_and_filter(pred_images, pred_annots)

# # Sanity checks
# if not gt_images or not gt_annots:
#     print("❌ No GT data remain after filtering.", file=sys.stderr)
#     sys.exit(1)
# if not pred_images or not pred_annots:
#     print("❌ No prediction data remain after filtering.", file=sys.stderr)
#     sys.exit(1)

# # 4) Write COCO JSONs
# coco_gt = {
#     "info":        {},
#     "licenses":    [],
#     "images":      gt_images,
#     "annotations": gt_annots,
#     "categories":  [{"id":1, "name":"cell"}],
# }
# with open("gt.json", "w") as f:
#     json.dump(coco_gt, f)

# # dt.json must be a JSON array of pred annotations only
# with open("dt.json", "w") as f:
#     json.dump(pred_annots, f)

# # 5) Run COCOeval for instance segmentation
# coco    = COCO("gt.json")
# coco_dt = coco.loadRes("dt.json")
# evaluator = COCOeval(coco, coco_dt, iouType="segm")
# evaluator.evaluate()
# evaluator.accumulate()
# evaluator.summarize()

#region another just mAP
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imread
from skimage.measure import label, regionprops
from glob import glob

# 📌 Function to compute instance-aware mAP
def compute_instance_iou_map(gt_mask, pred_mask, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    gt_labels = label(gt_mask)
    pred_labels = label(pred_mask)

    gt_instances = [region.coords for region in regionprops(gt_labels)]
    pred_instances = [region.coords for region in regionprops(pred_labels)]

    if len(gt_instances) == 0 and len(pred_instances) == 0:
        return 1.0
    if len(gt_instances) == 0 or len(pred_instances) == 0:
        return 0.0

    iou_matrix = np.zeros((len(gt_instances), len(pred_instances)))
    for i, gt_coords in enumerate(gt_instances):
        gt_set = set(map(tuple, gt_coords))
        for j, pred_coords in enumerate(pred_instances):
            pred_set = set(map(tuple, pred_coords))
            intersection = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    ap_list = []
    for thresh in iou_thresholds:
        matched_gt = set()
        matched_pred = set()
        tp = 0
        for i in range(len(gt_instances)):
            for j in range(len(pred_instances)):
                if i in matched_gt or j in matched_pred:
                    continue
                if iou_matrix[i, j] >= thresh:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    tp += 1
        fp = len(pred_instances) - tp
        fn = len(gt_instances) - tp
        precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        ap_list.append(precision)

    return np.mean(ap_list)

# === Paths
pred_root = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\ML\Folder Run Results3"
gt_root = r"C:\Users\Andrea Chiang\Downloads\For Testing Instance Masks"
output_csv = "omnipose_mAP_only.csv"

results = []
pred_mask_paths = sorted(glob(os.path.join(pred_root, "**", "*_instance.tif"), recursive=True))

for i, pred_path in enumerate(pred_mask_paths, 1):
    rel_path = os.path.relpath(pred_path, pred_root)
    gt_path = os.path.join(gt_root, rel_path.replace("_instance.tif", "_instmask.tif"))

    print(f"🔍 [{i}/{len(pred_mask_paths)}] Computing mAP for: {rel_path}")

    if not os.path.exists(gt_path):
        print(f"⚠️  Skipping — GT not found: {rel_path}")
        continue

    pred_mask = imread(pred_path)
    gt_mask = imread(gt_path)

    if pred_mask.shape != gt_mask.shape:
        print(f"⚠️  Skipping — Shape mismatch: {rel_path}")
        continue

    try:
        map_val = compute_instance_iou_map(gt_mask, pred_mask)
        parts = rel_path.split(os.sep)
        results.append({
            "Patient": parts[-2],
            "Image": parts[-1].replace("_wbc_instance.tif", ""),
            "mAP": map_val
        })
        print(f"✅ Done: mAP = {map_val:.4f}\n")
    except Exception as e:
        print(f"❌ Error on {rel_path}: {e}\n")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved mAP evaluation results to {output_csv}")

# Summary
average_map = df["mAP"].mean()
print(f"\n📋 Average mAP: {average_map:.4f}")

# Plot
plt.figure(figsize=(6, 5))
sns.boxplot(data=df[["mAP"]])
plt.title("Omnipose mAP Distribution")
plt.tight_layout()
plt.savefig("omnipose_mAP_boxplot.png")
print("📊 Saved mAP boxplot to omnipose_mAP_boxplot.png")

#region mAP
# import os
# import numpy as np
# from tifffile import imread
# from glob import glob
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # === CONFIGURATION ===
# GT_ROOT = r"C:\Users\Andrea Chiang\Downloads\For Testing Instance Masks"
# PRED_ROOT = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\ML\Folder Run Results3"
# IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
# GT_SUFFIX = "_instmask.tif"
# PRED_SUFFIX = "_instance.tif"

# # === FAST IOU MATRIX ===
# def fast_iou_matrix(gt_label, pred_label):
#     gt_ids = np.unique(gt_label)
#     pred_ids = np.unique(pred_label)
#     gt_ids = gt_ids[gt_ids != 0]
#     pred_ids = pred_ids[pred_ids != 0]
#     if len(gt_ids) == 0 or len(pred_ids) == 0:
#         return np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
#     iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
#     for i, gid in enumerate(gt_ids):
#         gt_mask = gt_label == gid
#         for j, pid in enumerate(pred_ids):
#             pred_mask = pred_label == pid
#             inter = np.logical_and(gt_mask, pred_mask).sum()
#             union = np.logical_or(gt_mask, pred_mask).sum()
#             iou[i, j] = inter / union if union else 0.0
#     return iou

# # === COMPUTE mAP PER IMAGE ===
# def compute_map(gt_mask, pred_mask):
#     iou_mat = fast_iou_matrix(gt_mask, pred_mask)
#     if iou_mat.size == 0:
#         return 1.0 if gt_mask.max() == pred_mask.max() == 0 else 0.0
#     ap_list = []
#     for thr in IOU_THRESHOLDS:
#         matches = iou_mat >= thr
#         tp = np.count_nonzero(matches.any(axis=1))
#         fp = matches.shape[1] - np.count_nonzero(matches.any(axis=0))
#         precision = tp / (tp + fp + 1e-8)
#         ap_list.append(precision)
#     return float(np.mean(ap_list))

# # === PROCESS ONE PAIR ===
# def process_pair(gt_path):
#     rel = os.path.relpath(gt_path, GT_ROOT)
#     pred_path = os.path.join(PRED_ROOT, rel.replace(GT_SUFFIX, PRED_SUFFIX))
#     if not os.path.exists(pred_path):
#         return rel, None
#     gt = imread(gt_path)
#     pred = imread(pred_path)
#     score = compute_map(gt, pred)
#     return rel, score

# # === MAIN BATCH LOOP ===
# def main():
#     gt_paths = glob(os.path.join(GT_ROOT, "**", f"*{GT_SUFFIX}"), recursive=True)
#     results = []
#     with ProcessPoolExecutor() as executor:
#         futures = {executor.submit(process_pair, p): p for p in gt_paths}
#         for fut in as_completed(futures):
#             rel, score = fut.result()
#             if score is None:
#                 print(f"❌ Missing prediction for: {rel}")
#             else:
#                 results.append((rel, round(score, 4)))
#                 print(f"✅ {rel}: mAP = {score:.4f}")
#     # Save results
#     df = pd.DataFrame(results, columns=["Image", "mAP"])
#     output_csv = os.path.join(PRED_ROOT, "instance_map_results.csv")
#     df.to_csv(output_csv, index=False)
#     print(f"\n📊 Mean mAP: {df['mAP'].mean():.4f}")
#     print(f"🗂 Results saved to: {output_csv}")

# if __name__ == "__main__":
#     main()