#region augment fr
import os
import cv2
import albumentations as A

# === CONFIG ===
ROOT_DIR = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_BBs_no_split"
OUTPUT_ROOT = ROOT_DIR + "_augmented_plus_original"

# === Define transforms ===
def make_transform(aug_list):
    return A.Compose(
        aug_list,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.0,  # Set to 0 to avoid dropping boxes
            check_each_transform=False  # Prevent over-strict validation
        )
    )

vertical_flip = make_transform([A.VerticalFlip(p=1.0)])
horizontal_flip = make_transform([A.HorizontalFlip(p=1.0)])
both_flip = make_transform([A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0)])

# === Helper to clip bboxes to [0, 1]
# ─── Robust clip: clamp in corner form, then convert back to YOLO ──────────────
def clip_bbox_yolo(box):
    """
    box: [x_center, y_center, w, h]  (all normalized)
    returns new box or None if it collapses to zero area
    """
    x_c, y_c, w, h = box
    # convert to corner form
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2

    # clamp to [0,1]
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    # recompute width/height
    new_w = x_max - x_min
    new_h = y_max - y_min
    if new_w <= 0 or new_h <= 0:
        return None  # box vanished

    new_x_c = x_min + new_w / 2
    new_y_c = y_min + new_h / 2
    return [new_x_c, new_y_c, new_w, new_h]


# === Read and write functions
def read_yolo_labels(label_path):
    boxes, labels = [], []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, bbox = int(parts[0]), list(map(float, parts[1:]))
            boxes.append(bbox)
            labels.append(cls)
    return boxes, labels

def save_yolo_labels(output_path, bboxes, class_labels):
    with open(output_path, 'w') as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join([str(round(b, 6)) for b in box])}\n")

# === Processing loop
for root, _, files in os.walk(ROOT_DIR):
    if r"\images\train" not in root:
        continue

    for fname in files:
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(root, fname)
        label_path = img_path.replace(r"\images\train", r"\labels\train").replace('.jpg', '.txt').replace('.png', '.txt')

        if not os.path.exists(label_path):
            print(f"⚠ Missing label for {fname}")
            continue

        image = cv2.imread(img_path)
        bboxes, class_labels = read_yolo_labels(label_path)
        if not bboxes:
            continue

        # Clip original boxes BEFORE passing to Albumentations
        bboxes = [clip_bbox_yolo(b) for b in bboxes]

        # === Save original ===
        rel_subpath = os.path.relpath(root, ROOT_DIR)
        out_img_dir = os.path.join(OUTPUT_ROOT, rel_subpath)
        out_lbl_dir = out_img_dir.replace(r"\images\train", r"\labels\train")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_img_dir, fname), image)
        save_yolo_labels(os.path.join(out_lbl_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt')), bboxes, class_labels)

        # === Save augmentations ===
        augmentations = {
            '_vflip': vertical_flip,
            '_hflip': horizontal_flip,
            '_vhflip': both_flip
        }

        for suffix, transform in augmentations.items():
            try:
                sample = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = sample['image']
                aug_bboxes = [clip_bbox_yolo(b) for b in sample['bboxes']]
                aug_classes = sample['class_labels']

                if not aug_bboxes:
                    continue

                aug_fname = fname.replace('.jpg', f'{suffix}.jpg').replace('.png', f'{suffix}.png')
                aug_lblname = aug_fname.replace('.jpg', '.txt').replace('.png', '.txt')

                cv2.imwrite(os.path.join(out_img_dir, aug_fname), aug_img)
                save_yolo_labels(os.path.join(out_lbl_dir, aug_lblname), aug_bboxes, aug_classes)
            except Exception as e:
                print(f"❌ Failed to process {fname} with {suffix}: {e}")

print("✅ DONE: Augmented dataset saved.")



#region preview augment
# import os
# import cv2
# import albumentations as A
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # === MODIFY THIS ===
# TEST_IMAGE_PATH = r"C:\Users\Andrea Chiang\Downloads\Finished Annotations\FOR YOLO TRAINING_BBs_no_split\REMISSIONS_resized\24BMLY\images\train\Snap_001.jpg"
# TEST_LABEL_PATH = TEST_IMAGE_PATH.replace(r"\images\train", r"\labels\train").replace(".jpg", ".txt")

# # === SAFE TRANSFORM (NO COLOR SHIFTS) ===
# # transform = A.Compose([
# #     A.HorizontalFlip(p=0.5),
# #     A.VerticalFlip(p=0.3),
# #     A.ShiftScaleRotate(
# #         shift_limit=0.1,
# #         scale_limit=0.2,
# #         rotate_limit=15,
# #         border_mode=cv2.BORDER_REFLECT_101,
# #         p=0.8
# #     )
# # ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# vertical_flip = A.Compose([
#     A.VerticalFlip(p=1.0)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# horizontal_flip = A.Compose([
#     A.HorizontalFlip(p=1.0)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# both_flip = A.Compose([
#     A.VerticalFlip(p=1.0),
#     A.HorizontalFlip(p=1.0)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# # === READ YOLO LABELS ===
# def read_yolo_labels(label_path):
#     boxes, labels = [], []
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             cls, bbox = int(parts[0]), list(map(float, parts[1:]))
#             boxes.append(bbox)
#             labels.append(cls)
#     return boxes, labels

# # === DRAW YOLO BOXES ===
# def draw_yolo_bboxes(ax, image, bboxes, title, color='lime'):
#     ax.set_title(title)
#     ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     for box in bboxes:
#         x_center, y_center, w, h = box
#         x = (x_center - w / 2) * image.shape[1]
#         y = (y_center - h / 2) * image.shape[0]
#         w *= image.shape[1]
#         h *= image.shape[0]
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
#         ax.add_patch(rect)
#     ax.axis('off')

# # === MAIN BLOCK ===
# if not os.path.exists(TEST_IMAGE_PATH) or not os.path.exists(TEST_LABEL_PATH):
#     print("❌ Check image or label path.")
# else:
#     # === Load image and labels ===
#     image = cv2.imread(TEST_IMAGE_PATH)
#     bboxes, class_labels = read_yolo_labels(TEST_LABEL_PATH)

#     # Apply flips
#     v_sample = vertical_flip(image=image, bboxes=bboxes, class_labels=class_labels)
#     v_image = v_sample['image']
#     v_bboxes = v_sample['bboxes']

#     h_sample = horizontal_flip(image=image, bboxes=bboxes, class_labels=class_labels)
#     h_image = h_sample['image']
#     h_bboxes = h_sample['bboxes']

#     both_sample = both_flip(image=image, bboxes=bboxes, class_labels=class_labels)
#     both_image = both_sample['image']
#     both_bboxes = both_sample['bboxes']

#     # === Plot all versions ===
#     # === Plot all versions in a 2x2 grid ===
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))

#     draw_yolo_bboxes(axes[0, 0], image, bboxes, title="Original")
#     draw_yolo_bboxes(axes[0, 1], v_image, v_bboxes, title="Vertical Flip")
#     draw_yolo_bboxes(axes[1, 0], h_image, h_bboxes, title="Horizontal Flip")
#     draw_yolo_bboxes(axes[1, 1], both_image, both_bboxes, title="Vertical + Horizontal Flip")

#     plt.tight_layout()
#     plt.show()

