# # import os
# # import shutil

# # # === CONFIG ===
# # path_list_file = r"C:\Users\Andrea Chiang\Downloads\stardist_test_image_paths.txt"
# # gt_mask_root   = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\GT_masks"
# # dest_root      = r"C:\Users\Andrea Chiang\Downloads\For Testing"

# # # === Read image paths ===
# # with open(path_list_file, 'r') as f:
# #     image_paths = [line.strip() for line in f if line.strip()]

# # # === Copy corresponding mask files while keeping folder structure ===
# # for img_rel_path in image_paths:
# #     # Convert Snap_XXX.jpg → Snap_XXX.tif
# #     img_name = os.path.basename(img_rel_path).replace(".jpg", ".tif")

# #     # Get subfolder (e.g., ALL_resized23BM66)
# #     subfolder = os.path.dirname(img_rel_path)

# #     # Full mask path from GT directory
# #     mask_path = os.path.join(gt_mask_root, subfolder, "masks", img_name)

# #     # Destination folder (preserve subfolder structure)
# #     dest_subfolder = os.path.join(dest_root, subfolder)
# #     os.makedirs(dest_subfolder, exist_ok=True)

# #     # Destination full path
# #     dest_path = os.path.join(dest_subfolder, img_name)

# #     # Copy if exists
# #     if os.path.exists(mask_path):
# #         shutil.copy2(mask_path, dest_path)
# #         print(f"✅ Copied: {mask_path} → {dest_path}")
# #     else:
# #         print(f"❌ Not found: {mask_path}")


# import os
# import cv2
# import numpy as np
# from tifffile import imread, imwrite

# def yolo_to_instance_mask(image_path, yolo_txt_path, out_mask_path, start_id=1):
#     """
#     Read an image to get H×W, parse YOLO .txt (class, x_center, y_center, w, h),
#     and write a uint16 TIFF where each box is a unique instance ID.
#     """
#     # 1) Load image for dimensions
#     if image_path.lower().endswith(('.tif', '.tiff')):
#         img = imread(image_path)
#     else:
#         img = cv2.imread(image_path)
#     H, W = img.shape[:2]

#     # 2) Create blank uint16 mask
#     mask = np.zeros((H, W), dtype=np.uint16)

#     # 3) Read YOLO .txt, ignore extra columns
#     with open(yolo_txt_path, 'r') as f:
#         lines = [line.strip().split() for line in f if line.strip()]

#     inst_id = start_id
#     for parts in lines:
#         # Only take the first five tokens: cls, x_c, y_c, w, h
#         cls, x_c, y_c, w, h = parts[:5]
#         x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

#         # 4) Convert normalized coords to pixel-space rectangle
#         x1 = int((x_c - w/2) * W)
#         y1 = int((y_c - h/2) * H)
#         x2 = int((x_c + w/2) * W)
#         y2 = int((y_c + h/2) * H)

#         # 5) Clamp to image bounds
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(W, x2), min(H, y2)

#         # 6) Fill that rectangle in the mask with this instance ID
#         mask[y1:y2, x1:x2] = inst_id
#         inst_id += 1

#     # 7) Save the mask TIFF
#     os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
#     imwrite(out_mask_path, mask)
#     print(f"  ✓ Saved: {out_mask_path}")

# def batch_convert_root(BASE_ROOT, OUT_ROOT, exts=('.jpg','.png','.tif','.tiff')):
#     """
#     For each subfolder under BASE_ROOT (e.g. REMISSIONS_resized24BMHA/...):
#       subfolder/
#         images/train/*.ext
#         labels/train/*.txt
#     Outputs each instance mask as:
#       OUT_ROOT/subfolder/<image_basename>_instmask.tif
#     """
#     for sub in sorted(os.listdir(BASE_ROOT)):
#         subpath = os.path.join(BASE_ROOT, sub)
#         img_dir = os.path.join(subpath, 'images', 'train')
#         txt_dir = os.path.join(subpath, 'labels', 'train')
#         if not (os.path.isdir(img_dir) and os.path.isdir(txt_dir)):
#             continue

#         print(f"\nProcessing dataset: {sub}")
#         for fname in sorted(os.listdir(img_dir)):
#             if not fname.lower().endswith(exts):
#                 continue
#             base, _ = os.path.splitext(fname)
#             img_path = os.path.join(img_dir, fname)
#             txt_path = os.path.join(txt_dir, base + '.txt')
#             if not os.path.exists(txt_path):
#                 print(f"  ! Skipping {fname}: no label found")
#                 continue

#             out_subdir = os.path.join(OUT_ROOT, sub)
#             out_mask   = os.path.join(out_subdir, f"{base}_instmask.tif")
#             yolo_to_instance_mask(img_path, txt_path, out_mask)

# if __name__ == "__main__":
#     # ─── EDIT THESE TWO PATHS ────────────────────────────────────────────────
#     BASE_ROOT = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_OG"
#     OUT_ROOT  = r"C:\Users\Andrea Chiang\Downloads\YOLO_Instance_Masks"
#     # ─────────────────────────────────────────────────────────────────────────

#     batch_convert_root(BASE_ROOT, OUT_ROOT)

#region get instance masks
import os
from skimage.io import imread, imsave
from skimage.measure import label

# === CONFIGURATION ===
binary_root = r"C:\Users\Andrea Chiang\Downloads\For Testing"
output_root = r"C:\Users\Andrea Chiang\Downloads\For Testing Instance Masks"

# Walk through all binary .tif files (excluding *_instmask.tif)
for dirpath, _, filenames in os.walk(binary_root):
    for fname in filenames:
        if fname.endswith(".tif") and not fname.endswith("_instmask.tif"):
            # Full path of binary mask
            full_path = os.path.join(dirpath, fname)
            
            # Load and binarize
            binary_mask = imread(full_path)
            binary_mask = (binary_mask > 0)

            # Convert to instance mask
            instance_mask = label(binary_mask)

            # Build destination path
            rel_path = os.path.relpath(full_path, binary_root)
            out_rel = rel_path.replace(".tif", "_instmask.tif")
            out_path = os.path.join(output_root, out_rel)

            # Create destination folder if needed
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save instance mask
            imsave(out_path, instance_mask.astype("uint16"))
            print(f"✅ Saved: {out_path}")

