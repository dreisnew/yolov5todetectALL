import os
import shutil

# === CONFIG ===
path_list_file = r"C:\Users\Andrea Chiang\Downloads\stardist_test_image_paths.txt"
gt_mask_root   = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\GT_masks"
dest_root      = r"C:\Users\Andrea Chiang\Downloads\For Testing"

# === Read image paths ===
with open(path_list_file, 'r') as f:
    image_paths = [line.strip() for line in f if line.strip()]

# === Copy corresponding mask files while keeping folder structure ===
for img_rel_path in image_paths:
    # Convert Snap_XXX.jpg → Snap_XXX.tif
    img_name = os.path.basename(img_rel_path).replace(".jpg", ".tif")

    # Get subfolder (e.g., ALL_resized23BM66)
    subfolder = os.path.dirname(img_rel_path)

    # Full mask path from GT directory
    mask_path = os.path.join(gt_mask_root, subfolder, "masks", img_name)

    # Destination folder (preserve subfolder structure)
    dest_subfolder = os.path.join(dest_root, subfolder)
    os.makedirs(dest_subfolder, exist_ok=True)

    # Destination full path
    dest_path = os.path.join(dest_subfolder, img_name)

    # Copy if exists
    if os.path.exists(mask_path):
        shutil.copy2(mask_path, dest_path)
        print(f"✅ Copied: {mask_path} → {dest_path}")
    else:
        print(f"❌ Not found: {mask_path}")
