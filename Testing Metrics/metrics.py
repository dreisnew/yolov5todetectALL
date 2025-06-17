# #region preapre masks 

# import os
# import cv2
# import numpy as np

# # === CONFIG ===
# BASE_DIR = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_OG"
# OUT_MASKS_ROOT = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\cellpose_wbc_GT_masks"
# os.makedirs(OUT_MASKS_ROOT, exist_ok=True)

# # Traverse each group like REMISSIONS_resized24BMHA
# for root, dirs, files in os.walk(BASE_DIR):
#     for file in files:
#         if not file.endswith(".txt"):
#             continue

#         # === Construct Paths ===
#         label_path = os.path.join(root, file)
#         rel_path = os.path.relpath(label_path, BASE_DIR)

#         # Expect structure like: REMISSIONS_resized24BMHA\labels\train\Snap_001.txt
#         parts = rel_path.split(os.sep)
#         if len(parts) < 4 or parts[-3] != "labels":
#             continue  # skip malformed or irrelevant folders

#         group = parts[0]                  # REMISSIONS_resized24BMSD
#         label_or_images = parts[1]        # 'labels'
#         subfolder = parts[2]              # 'train'
#         img_name = os.path.splitext(file)[0] + ".jpg"
#         image_path = os.path.join(BASE_DIR, group, "images", subfolder, img_name)


#         if not os.path.exists(image_path):
#             print(f"⚠️ Image not found: {image_path}")
#             continue

#         # === Load image and label ===
#         image = cv2.imread(image_path)
#         height, width = image.shape[:2]
#         mask = np.zeros((height, width), dtype=np.uint8)

#         with open(label_path, 'r') as f:
#             lines = f.readlines()

#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 6:
#                 continue

#             coords = list(map(float, parts[1:]))  # skip class
#             if len(coords) % 2 != 0:
#                 continue

#             points = []
#             for i in range(0, len(coords), 2):
#                 x = int(coords[i] * width)
#                 y = int(coords[i + 1] * height)
#                 points.append([x, y])
#             points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
#             cv2.fillPoly(mask, [points], 255)

#         # === Save binary mask ===
#         save_dir = os.path.join(OUT_MASKS_ROOT, group, "masks")
#         os.makedirs(save_dir, exist_ok=True)
#         out_path = os.path.join(save_dir, img_name.replace(".jpg", ".tif"))
#         cv2.imwrite(out_path, mask)

# print("✅ All masks generated and saved.")


#region metrics using masks all pics
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tifffile import imread
# from sklearn.metrics import (
#     jaccard_score, f1_score, precision_score, recall_score,
#     confusion_matrix
# )
# from skimage.metrics import hausdorff_distance
# from glob import glob

# # 📁 Base directories
# pred_dir = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\4th\MW_wbc"
# gt_base  = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\GT_masks_renamed"

# results = []

# # cellpose
# # pred_mask_paths = sorted(glob(os.path.join(pred_dir, "**", "*_pred.tif"), recursive=True))

# #omnipose
# pred_mask_paths = sorted(glob(os.path.join(pred_dir, "**", "*_wbc_mask.tif"), recursive=True))


# for pred_path in pred_mask_paths:
#     path_parts = pred_path.split(os.sep)
#     grandparent = path_parts[-3]  # e.g., REMISSIONS_resized
#     parent = path_parts[-2]       # e.g., 24BMMA
#     patient_folder = os.path.join(grandparent, parent)
#     pred_filename = os.path.basename(pred_path)
    
#     #cellpose
#     # img_name = pred_filename.replace("_pred.tif", ".tif")

#     #omnipose
#     img_name = pred_filename.replace("_wbc_mask.tif", ".tif")

#     gt_path = os.path.join(gt_base, grandparent, parent, "masks", img_name)

#     print(f"\n🔍 Checking: {img_name}")
#     print(f"   PRED: {pred_path}")
#     print(f"   GT:   {gt_path}")

#     if not os.path.exists(gt_path):
#         print("   ❌ GT mask not found. Skipping.")
#         continue

#     pred_mask = imread(pred_path).astype(bool)
#     gt_mask = imread(gt_path).astype(bool)

#     if pred_mask.shape != gt_mask.shape:
#         print(f"   ⚠️ Shape mismatch: pred {pred_mask.shape}, gt {gt_mask.shape}")
#         continue

#     pred_flat = pred_mask.flatten()
#     gt_flat = gt_mask.flatten()

#     try:
#         iou = jaccard_score(gt_flat, pred_flat)
#         dice = f1_score(gt_flat, pred_flat)
#         precision = precision_score(gt_flat, pred_flat)
#         recall = recall_score(gt_flat, pred_flat)
#         hausdorff = hausdorff_distance(gt_mask, pred_mask)

#         tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()
#         accuracy = (tp + tn) / (tp + tn + fp + fn)
#         specificity = tn / (tn + fp)

#         results.append({
#             "Patient": patient_folder,
#             "Image": img_name.replace(".tif", ""),
#             "IoU": iou,
#             "Dice": dice,
#             "Precision": precision,
#             "Recall": recall,
#             "Hausdorff": hausdorff,
#             "Accuracy": accuracy,
#             "Specificity": specificity,
#             "TP": tp,
#             "TN": tn,
#             "FP": fp,
#             "FN": fn
#         })

#         print("   ✅ Metrics computed.")

#     except Exception as e:
#         print(f"   ❌ Metric computation failed: {e}")
#         continue

# # 🧾 Save results to CSV
# df = pd.DataFrame(results)
# df.to_csv("MW_allpics_metrics.csv", index=False)
# print("✅ Saved evaluation results to MW_allpics_metrics.csv")

# # 📊 Summary and interpretation
# if df.empty:
#     print("🚫 No results — check image paths, mask sizes, or predictions.")
#     exit()

# average_metrics = df[["IoU", "Dice", "Precision", "Recall", "Hausdorff", "Accuracy", "Specificity"]].mean()

# def interpret(metric, value):
#     if metric == "Hausdorff":
#         if value < 5: return "Excellent boundary match"
#         elif value < 10: return "Acceptable boundary match"
#         else: return "Poor boundary match"
#     else:
#         if value > 0.9: return "Excellent"
#         elif value > 0.7: return "Good"
#         elif value > 0.5: return "Fair"
#         else: return "Poor"

# print("\n📋 Average Metrics and Interpretation:")
# for metric, value in average_metrics.items():
#     print(f"{metric}: {value:.4f} → {interpret(metric, value)}")

# # 📈 Visualize with boxplot
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df[["IoU", "Dice", "Precision", "Recall", "Hausdorff", "Accuracy", "Specificity"]])
# plt.title("Segmentation Metrics Distribution")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("MW_allpics_metric_boxplot_full.png")
# print("📊 Saved boxplot to MW_allpics_metric_boxplot_full.png")


#region  renaming yolo folder
# import os
# import shutil
# from glob import glob

# # === CONFIG ===
# PRED_ROOT = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\cellpose_wbc_with_masks"
# GT_ORIGINAL_ROOT = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\GT_masks"
# GT_NEW_ROOT = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\GT_masks_renamed"

# # === Find all _pred.tif files ===
# pred_mask_paths = glob(os.path.join(PRED_ROOT, "**", "*_pred.tif"), recursive=True)

# for pred_path in pred_mask_paths:
#     path_parts = pred_path.split(os.sep)
#     if len(path_parts) < 3:
#         continue

#     group = path_parts[-3]  # REMISSIONS_resized
#     patient = path_parts[-2]  # 24BMMA2
#     img_name = os.path.basename(pred_path).replace("_pred.tif", ".tif")

#     # Original GT folder name (concatenated version)
#     old_gt_folder = os.path.join(GT_ORIGINAL_ROOT, group + patient, "masks")
#     old_gt_path = os.path.join(old_gt_folder, img_name)

#     # New GT folder structure (group/patient/masks/img.tif)
#     new_gt_folder = os.path.join(GT_NEW_ROOT, group, patient, "masks")
#     new_gt_path = os.path.join(new_gt_folder, img_name)

#     if not os.path.exists(old_gt_path):
#         print(f"❌ GT not found: {old_gt_path}")
#         continue

#     os.makedirs(new_gt_folder, exist_ok=True)
#     shutil.copy2(old_gt_path, new_gt_path)
#     print(f"✅ Moved: {img_name} → {new_gt_path}")

# print("\n🎉 GT folder restructuring complete. Use this as your `gt_base` in the metrics script.")

#region metrics for testing MW
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tifffile import imread
# from glob import glob
# from sklearn.metrics import (
#     jaccard_score, f1_score, precision_score, recall_score,
#     confusion_matrix
# )
# from skimage.metrics import hausdorff_distance
# import re
# # ─── CONFIG ────────────────────────────────────────────────────────────────
# gt_folder = r"C:\Users\Andrea Chiang\Downloads\For Testing"
# pred_dir  = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\4th\MW_wbc"

# results = []

# # ─── 1) Find every GT .tif under all subfolders ────────────────────────────
# gt_paths = sorted(glob(os.path.join(gt_folder, "**", "*.tif"), recursive=True))

# for gt_path in gt_paths:
#     # 1) compute rel path and raw subfolder
#     rel       = os.path.relpath(gt_path, gt_folder)
#     raw_sub   = os.path.basename(os.path.dirname(gt_path))
#     img_name  = os.path.basename(gt_path)
#     mask_name = img_name.replace(".tif", "_wbc_mask.tif")

#     # 2) split raw_sub into (root, patient)
#     m = re.match(r"(.*resized)(.+)", raw_sub)
#     if not m:
#         print(f"❌ Unexpected GT folder name '{raw_sub}', skipping.")
#         continue

#     root_folder, patient = m.group(1), m.group(2)
#     # now root_folder="REMISSIONS_resized", patient="24BMSD"

#     # 3) build the pred path under the same two-level tree
#     pred_path = os.path.join(pred_dir, root_folder, patient, mask_name)
#     if not os.path.exists(pred_path):
#         print(f"❌ Missing prediction for {rel}, skipping.")
#         continue


#     print(f"\n🔍 Checking: {rel}")
#     print(f"   GT:   {gt_path}")
#     print(f"   PRED: {pred_path}")

#     # ─── 3) Load & binarize ─────────────────────────────────────────────────
#     gt_mask   = imread(gt_path).astype(bool)
#     pred_mask = imread(pred_path).astype(bool)

#     # ─── 4) Shape check ────────────────────────────────────────────────────
#     if gt_mask.shape != pred_mask.shape:
#         print(f"   ⚠️ Shape mismatch: GT {gt_mask.shape}, PRED {pred_mask.shape}")
#         continue

#     # ─── 5) Compute metrics ───────────────────────────────────────────────
#     gt_flat, pred_flat = gt_mask.ravel(), pred_mask.ravel()
#     iou       = jaccard_score(gt_flat, pred_flat)
#     dice      = f1_score(gt_flat, pred_flat)
#     precision = precision_score(gt_flat, pred_flat)
#     recall    = recall_score(gt_flat, pred_flat)
#     hausdorff = hausdorff_distance(gt_mask, pred_mask)
#     tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()
#     accuracy    = (tp + tn) / (tp + tn + fp + fn)
#     specificity = tn / (tn + fp)
#     subdir = f"{root_folder}/{patient}"  

#     results.append({
#         "Subfolder":   subdir,
#         "Image":       img_name.replace(".tif",""),
#         "IoU":         iou,
#         "Dice":        dice,
#         "Precision":   precision,
#         "Recall":      recall,
#         "Hausdorff":   hausdorff,
#         "Accuracy":    accuracy,
#         "Specificity": specificity,
#         "TP": tp, "TN": tn, "FP": fp, "FN": fn
#     })
#     print("   ✅ Metrics computed.")

# # ─── 6) Save results ─────────────────────────────────────────────────────
# df = pd.DataFrame(results)
# csv_out = "MW_allpics_metrics_GTdriven.csv"
# df.to_csv(csv_out, index=False)
# print(f"\n✅ Saved all metrics to {csv_out}")

# # ─── 7) (Optional) Summary & Boxplot ─────────────────────────────────────
# if not df.empty:
#     avg = df[["IoU","Dice","Precision","Recall","Hausdorff","Accuracy","Specificity"]].mean()
#     def interpret(m, v):
#         if m=="Hausdorff":
#             return "Excellent" if v<5 else "Acceptable" if v<10 else "Poor"
#         else:
#             return "Excellent" if v>0.9 else "Good" if v>0.7 else "Fair" if v>0.5 else "Poor"
#     print("\n📋 AVERAGE METRICS:")
#     for m, v in avg.items():
#         print(f"  {m}: {v:.3f} → {interpret(m,v)}")

#     plt.figure(figsize=(12,6))
#     df[["IoU","Dice","Precision","Recall","Hausdorff","Accuracy","Specificity"]].boxplot()
#     plt.xticks(rotation=45)
#     plt.title("Segmentation Metrics Distribution")
#     plt.tight_layout()
#     plt.savefig("MW_allpics_metric_boxplot_full.png")
#     print("📊 Saved boxplot as MW_allpics_metric_boxplot_full.png")
# else:
#     print("🚫 No data to summarize.")

#region cellpose
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from glob import glob
from sklearn.metrics import (
    jaccard_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from skimage.metrics import hausdorff_distance

# ─── CONFIG ────────────────────────────────────────────────────────────────
gt_folder = r"C:\Users\Andrea Chiang\Downloads\For Testing"
pred_dir  = (
    r"C:\Users\Andrea Chiang\Downloads\python\thesis"
    r"\Image Processing Models Testing\Unet\dataset\test"
    r"\results\omnipose_2nd_try_wbc"
)

# regex to split "REMISSIONS_resized24BMSD" → ("REMISSIONS_resized", "24BMSD")
split_pat = re.compile(r"(.*resized)(.+)")

results = []

# 1) gather all GT .tif under every subfolder
gt_paths = sorted(glob(os.path.join(gt_folder, "**", "*.tif"), recursive=True))

for gt_path in gt_paths:
    # raw subfolder name (e.g. "REMISSIONS_resized24BMSD")
    raw_sub = os.path.basename(os.path.dirname(gt_path))
    # image stem (e.g. "Snap_015")
    stem = os.path.splitext(os.path.basename(gt_path))[0]

    # split into ("REMISSIONS_resized", "24BMSD")
    m = split_pat.match(raw_sub)
    if not m:
        print(f"❌ Unexpected GT folder '{raw_sub}', skipping.")
        continue
    root_folder, patient = m.groups()

    # look only in the matching pred sub-folder
    pred_sub = os.path.join(pred_dir, root_folder, patient)
    pattern  = os.path.join(pred_sub, f"{stem}_wbc_mask.tif")
    matches  = glob(pattern)

    if len(matches) != 1:
        print(f"❌ Found {len(matches)} matches for '{stem}_wbc_mask.tif' in {pred_sub}, skipping.")
        continue

    pred_path = matches[0]
    print(f"\n🔍 Checking {root_folder}\\{patient}\\{stem}")
    print(f"   GT:   {gt_path}")
    print(f"   PRED: {pred_path}")

    # load & binarize
    gt_mask   = imread(gt_path).astype(bool)
    pred_mask = imread(pred_path).astype(bool)

    if gt_mask.shape != pred_mask.shape:
        print(f"⚠️ Shape mismatch GT{gt_mask.shape} vs PRED{pred_mask.shape}, skipping.")
        continue

    # compute metrics
    gt_flat, pred_flat = gt_mask.ravel(), pred_mask.ravel()
    iou       = jaccard_score(gt_flat, pred_flat)
    dice      = f1_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat)
    recall    = recall_score(gt_flat, pred_flat)
    hausdorff = hausdorff_distance(gt_mask, pred_mask)
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)

    results.append({
        "Subfolder":   f"{root_folder}/{patient}",
        "Image":       stem,
        "IoU":         iou,
        "Dice":        dice,
        "Precision":   precision,
        "Recall":      recall,
        "Hausdorff":   hausdorff,
        "Accuracy":    accuracy,
        "Specificity": specificity,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    })
    print("   ✅ Metrics computed.")

# ─── END OF LOOP ────────────────────────────────────────────────────────────

# 2) save CSV
df = pd.DataFrame(results)
csv_out = "omnipose_metrics.csv"
df.to_csv(csv_out, index=False)
print(f"\n✅ Saved metrics to {csv_out}")

# 3) average metrics & boxplot
if not df.empty:
    avg = df[["IoU","Dice","Precision","Recall","Hausdorff","Accuracy","Specificity"]].mean()
    print("\n📋 AVERAGE METRICS:")
    for m, v in avg.items():
        if m == "Hausdorff":
            tag = "Excellent" if v < 5 else "Acceptable" if v < 10 else "Poor"
        else:
            tag = "Excellent" if v > 0.9 else "Good" if v > 0.7 else "Fair" if v > 0.5 else "Poor"
        print(f"  {m}: {v:.3f} → {tag}")

    plt.figure(figsize=(12,6))
    df[["IoU","Dice","Precision","Recall","Hausdorff","Accuracy","Specificity"]].boxplot()
    plt.xticks(rotation=45)
    plt.title("Segmentation Metrics Distribution")
    plt.tight_layout()
    plt.savefig("omnipose_boxplot.png")
    print("📊 Boxplot saved to omnipose_boxplot_.png")
else:
    print("🚫 No data to summarize.")




