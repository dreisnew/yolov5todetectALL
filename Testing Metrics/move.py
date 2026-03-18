# import os
# import shutil
# from glob import glob

# # === CONFIGURATION ===
# source_root = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\cellpose_wbc_with_masks"
# reference_root = r"C:\Users\Andrea Chiang\Downloads\For Testing"
# output_root = r"C:\Users\Andrea Chiang\Downloads\cellpose_for_testing"
# log_path = r"C:\Users\Andrea Chiang\Downloads\cellpose_match_log.txt"

# # Step 1: Gather full relative path to *_instmask.tif (without suffix)
# ref_core_paths = set()
# for root, _, files in os.walk(reference_root):
#     for fname in files:
#         if fname.lower().endswith("_instmask.tif"):
#             full_path = os.path.join(root, fname)
#             rel_path = os.path.relpath(full_path, reference_root)
#             core_path = rel_path.replace("_instmask.tif", "")  # e.g., ALL_resized\23BM66\Snap_002
#             ref_core_paths.add(core_path)

# # Step 2: Match full path to *_wbc_mask.tif
# matched = 0
# with open(log_path, "w", encoding="utf-8") as log:
#     log.write(f"🔍 Found {len(ref_core_paths)} *_instmask.tif relative paths.\n\n")

#     all_mask_paths = glob(os.path.join(source_root, "**", "*_wbc_mask.tif"), recursive=True)

#     for full_path in all_mask_paths:
#         rel_path = os.path.relpath(full_path, source_root)
#         core_path = rel_path.replace("_wbc_mask.tif", "")

#         if core_path in ref_core_paths:
#             dest_path = os.path.join(output_root, rel_path)
#             os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#             shutil.copy2(full_path, dest_path)
#             matched += 1
#             log.write(f"✅ Match: {core_path} → {rel_path}\n")

#     log.write(f"\n📦 Total matched and copied: {matched}\n")

# print(f"✅ DONE. Copied {matched} masks — see log here: {log_path}")


import os
import shutil
from glob import glob

# === CONFIGURATION ===
source_root = r"C:\Users\Andrea Chiang\Downloads\python\thesis\Image Processing Models Testing\Unet\dataset\test\results\omnipose_instance"
reference_root = r"C:\Users\Andrea Chiang\Downloads\For Testing"
output_root = r"C:\Users\Andrea Chiang\Downloads\omnipose_for_testing"
log_path = r"C:\Users\Andrea Chiang\Downloads\omnipose_match_log.txt"

# Step 1: Gather full relative path to *_instmask.tif (without suffix)
ref_core_paths = set()
for root, _, files in os.walk(reference_root):
    for fname in files:
        if fname.lower().endswith("_instmask.tif"):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, reference_root)
            core_path = rel_path.replace("_instmask.tif", "")  # e.g., ALL_resized\23BM66\Snap_002
            ref_core_paths.add(core_path)

# Step 2: Match *_wbc_instance.tif from Omnipose against those core paths
matched = 0
with open(log_path, "w", encoding="utf-8") as log:
    log.write(f"🔍 Found {len(ref_core_paths)} *_instmask.tif relative paths.\n\n")

    all_mask_paths = glob(os.path.join(source_root, "**", "*_wbc_instance.tif"), recursive=True)

    for full_path in all_mask_paths:
        rel_path = os.path.relpath(full_path, source_root)
        core_path = rel_path.replace("_wbc_instance.tif", "")  # Match Snap_xxx part

        if core_path in ref_core_paths:
            dest_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(full_path, dest_path)
            matched += 1
            log.write(f"✅ Match: {core_path} → {rel_path}\n")

    log.write(f"\n📦 Total matched and copied: {matched}\n")

print(f"✅ DONE. Copied {matched} masks — see log here: {log_path}")
