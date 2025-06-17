#region split 80-20
# import os
# import random
# import shutil

# BASE_DIR = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING"
# VAL_SPLIT = 0.2

# for group in os.listdir(BASE_DIR):  # ALL_resized, REMISSIONS_resized
#     group_path = os.path.join(BASE_DIR, group)
#     if not os.path.isdir(group_path):
#         continue

#     for subfolder in os.listdir(group_path):
#         folder_path = os.path.join(group_path, subfolder)
#         if not os.path.isdir(folder_path):
#             continue

#         print(f"Processing {group}/{subfolder}...")
#         img_train_dir = os.path.join(folder_path, "images", "train")
#         lbl_train_dir = os.path.join(folder_path, "labels", "train")
#         img_val_dir   = os.path.join(folder_path, "images", "val")
#         lbl_val_dir   = os.path.join(folder_path, "labels", "val")

#         os.makedirs(img_val_dir, exist_ok=True)
#         os.makedirs(lbl_val_dir, exist_ok=True)

#         log_path = os.path.join(folder_path, "val_split_log.txt")
#         with open(log_path, "w") as log_file:
#             img_files = [f for f in os.listdir(img_train_dir) if f.lower().endswith(('.jpg', '.png'))]
#             val_sample = random.sample(img_files, int(len(img_files) * VAL_SPLIT))

#             for img_name in val_sample:
#                 lbl_name = os.path.splitext(img_name)[0] + ".txt"
#                 shutil.move(os.path.join(img_train_dir, img_name), os.path.join(img_val_dir, img_name))
#                 lbl_src = os.path.join(lbl_train_dir, lbl_name)
#                 lbl_dst = os.path.join(lbl_val_dir, lbl_name)
#                 if os.path.exists(lbl_src):
#                     shutil.move(lbl_src, lbl_dst)

#                 log_file.write(f"{img_name}\n")

#         print(f"Moved {len(val_sample)} → logged in val_split_log.txt")

#region unsplit
# import os
# import shutil

# BASE_DIR = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING"

# for group in os.listdir(BASE_DIR):  # ALL_resized, REMISSIONS_resized, etc.
#     group_path = os.path.join(BASE_DIR, group)
#     if not os.path.isdir(group_path):
#         continue

#     for subfolder in os.listdir(group_path):
#         folder_path = os.path.join(group_path, subfolder)
#         if not os.path.isdir(folder_path):
#             continue

#         log_path = os.path.join(folder_path, "val_split_log.txt")
#         if not os.path.exists(log_path):
#             print(f"No log for {group}/{subfolder}, skipping.")
#             continue

#         img_train = os.path.join(folder_path, "images", "train")
#         img_val   = os.path.join(folder_path, "images", "val")
#         lbl_train = os.path.join(folder_path, "labels", "train")
#         lbl_val   = os.path.join(folder_path, "labels", "val")

#         moved_count = 0
#         with open(log_path, "r") as f:
#             for line in f:
#                 img_name = line.strip()
#                 if not img_name:
#                     continue

#                 # move image back
#                 src_img = os.path.join(img_val, img_name)
#                 dst_img = os.path.join(img_train, img_name)
#                 if os.path.exists(src_img):
#                     shutil.move(src_img, dst_img)

#                 # move label back
#                 lbl_name = os.path.splitext(img_name)[0] + ".txt"
#                 src_lbl = os.path.join(lbl_val, lbl_name)
#                 dst_lbl = os.path.join(lbl_train, lbl_name)
#                 if os.path.exists(src_lbl):
#                     shutil.move(src_lbl, dst_lbl)

#                 moved_count += 1

#         # remove the log file when done
#         os.remove(log_path)
#         print(f"{group}/{subfolder}: moved {moved_count} files back to train and removed log.")

#region split 80-10-10
# import os, random, shutil

# BASE_DIR   = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_BBs_split_with_augemented"
# TRAIN_FRAC = 0.8
# VAL_FRAC   = 0.1

# # ─── NEW global counters ────────────────────────────────────────────────
# grand_total = grand_train = grand_val = grand_test = 0

# for group in os.listdir(BASE_DIR):
#     group_path = os.path.join(BASE_DIR, group)
#     if not os.path.isdir(group_path):
#         continue

#     for subfolder in os.listdir(group_path):
#         folder_path = os.path.join(group_path, subfolder)
#         if not os.path.isdir(folder_path):
#             continue

#         print(f"Processing {group}/{subfolder}...")

#         # target dirs
#         img_dirs = {
#             "train": os.path.join(folder_path, "images", "train"),
#             "val":   os.path.join(folder_path, "images", "val"),
#             "test":  os.path.join(folder_path, "images", "test"),
#         }
#         lbl_dirs = {
#             k: d.replace("images", "labels") for k, d in img_dirs.items()
#         }
#         for d in (*img_dirs.values(), *lbl_dirs.values()):
#             os.makedirs(d, exist_ok=True)

#         # gather originals (they all start out in images/train)
#         all_images = [
#             f for f in os.listdir(img_dirs["train"])
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ]

#         N      = len(all_images)
#         n_val  = int(N * VAL_FRAC)
#         n_test = int(N * (1 - TRAIN_FRAC - VAL_FRAC))  # 0.1

#         random.shuffle(all_images)
#         val_images   = set(all_images[:n_val])
#         test_images  = set(all_images[n_val : n_val + n_test])
#         train_images = set(all_images[n_val + n_test :])

#         for split, img_set in [("val", val_images),
#                                ("test", test_images),
#                                ("train", train_images)]:
#             for img_name in img_set:
#                 shutil.move(os.path.join(img_dirs["train"], img_name),
#                             os.path.join(img_dirs[split],  img_name))

#                 lbl_name = os.path.splitext(img_name)[0] + ".txt"
#                 src_lbl  = os.path.join(lbl_dirs["train"], lbl_name)
#                 dst_lbl  = os.path.join(lbl_dirs[split],  lbl_name)
#                 if os.path.exists(src_lbl):
#                     shutil.move(src_lbl, dst_lbl)

#         # per-folder log
#         with open(os.path.join(folder_path, "split_log.txt"), "w") as log:
#             log.write(f"Total: {N}\n")
#             log.write(f"Train: {len(train_images)}\n")
#             log.write(f"Val:   {len(val_images)}\n")
#             log.write(f"Test:  {len(test_images)}\n")

#         print(f"→ train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

#         # ─── accumulate ────────────────────────────────────────────────
#         grand_total += N
#         grand_train += len(train_images)
#         grand_val   += len(val_images)
#         grand_test  += len(test_images)

# # ─── final summary ─────────────────────────────────────────────────────
# print("\n====== GLOBAL SPLIT SUMMARY ======")
# print(f"Total images : {grand_total}")
# print(f"  Train : {grand_train}")
# print(f"  Val   : {grand_val}")
# print(f"  Test  : {grand_test}")
# print("==================================")


#region flatten
import os
import shutil

# --- adjust these paths ---
SPLIT_DIR = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_BBs_8--10-10_split_with_augemented"
DEST_DIR  = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\data_80-10-10_augmented"

# 1) create flat target folders
for data_type in ("images", "labels"):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DEST_DIR, data_type, split), exist_ok=True)

# 2) walk group → subfolder → images/labels splits
for group in os.listdir(SPLIT_DIR):
    group_path = os.path.join(SPLIT_DIR, group)
    if not os.path.isdir(group_path):
        continue

    # inside each group (e.g. ALL_resized), iterate its subfolders (23BM66, 24BMLY, …)
    for sub in os.listdir(group_path):
        sub_path = os.path.join(group_path, sub)
        if not os.path.isdir(sub_path):
            continue

        for data_type in ("images", "labels"):
            for split in ("train", "val", "test"):
                src_dir = os.path.join(sub_path, data_type, split)
                if not os.path.isdir(src_dir):
                    continue

                dst_dir = os.path.join(DEST_DIR, data_type, split)
                for fname in os.listdir(src_dir):
                    src = os.path.join(src_dir, fname)
                    # prefix with both group and subfolder to avoid any name clashes
                    new_name = f"{group}_{sub}_{fname}"
                    dst = os.path.join(dstir := dst_dir, new_name)
                    shutil.copy2(src, dst)

# 3) (optional) print counts to verify
for data_type in ("images", "labels"):
    for split in ("train", "val", "test"):
        cnt = len(os.listdir(os.path.join(DEST_DIR, data_type, split)))
        print(f"{data_type}/{split}: {cnt} files")


