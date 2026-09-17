import numpy as np
import tifffile
import cc3d


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

GT_PATH = "/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/slab19_hippocampus_chunks/masks/chunk_z5_y36_x30000.tif"
PRED_PATH = "/lustre_scratch/shared_scratch/hillmanLab/hackathonData_2026/KIDS26-Team8/Unet/out/mask.tif"

IOU_THRESHOLD = 0.01


# --------------------------------------------------
# LOAD BINARY MASKS
# --------------------------------------------------

gt = tifffile.imread(GT_PATH) > 0
pred = tifffile.imread(PRED_PATH) > 0

assert gt.shape == pred.shape, "GT and prediction must have the same shape"

print("Volume shape:", gt.shape)


# --------------------------------------------------
# CONNECTED COMPONENTS
# --------------------------------------------------

gt_labels = cc3d.connected_components(gt, connectivity=26)
pred_labels = cc3d.connected_components(pred, connectivity=26)

n_gt = int(gt_labels.max())
n_pred = int(pred_labels.max())

print("GT objects:", n_gt)
print("Predicted objects:", n_pred)


# --------------------------------------------------
# OBJECT SIZES
# --------------------------------------------------

gt_sizes = np.bincount(gt_labels.ravel())
pred_sizes = np.bincount(pred_labels.ravel())


# --------------------------------------------------
# FIND OVERLAPPING GT / PRED OBJECTS
# --------------------------------------------------

# Only look at voxels where both masks contain an object
overlap_mask = (gt_labels > 0) & (pred_labels > 0)

gt_overlap = gt_labels[overlap_mask].astype(np.int64)
pred_overlap = pred_labels[overlap_mask].astype(np.int64)

# Encode each (GT label, prediction label) pair as one integer
pair_ids = gt_overlap * (n_pred + 1) + pred_overlap

unique_pairs, intersection = np.unique(pair_ids, return_counts=True)

gt_ids = unique_pairs // (n_pred + 1)
pred_ids = unique_pairs % (n_pred + 1)


# --------------------------------------------------
# COMPUTE IoU
# --------------------------------------------------

union = (
    gt_sizes[gt_ids]
    + pred_sizes[pred_ids]
    - intersection
)

ious = intersection / union

print("Number of overlapping object pairs:", len(ious))

if len(ious) > 0:
    print("Maximum IoU:", ious.max())
    print("Top 20 IoUs:")
    print(np.sort(ious)[-20:][::-1])

# --------------------------------------------------
# ONE-TO-ONE MATCHING
# --------------------------------------------------

# Highest-IoU pairs considered first
order = np.argsort(ious)[::-1]

matched_gt = set()
matched_pred = set()
matches = []

for idx in order:

    iou = ious[idx]

    if iou < IOU_THRESHOLD:
        break

    gt_id = int(gt_ids[idx])
    pred_id = int(pred_ids[idx])

    if gt_id not in matched_gt and pred_id not in matched_pred:

        matched_gt.add(gt_id)
        matched_pred.add(pred_id)

        matches.append((gt_id, pred_id, iou))


# --------------------------------------------------
# METRICS
# --------------------------------------------------

TP = len(matches)
FP = n_pred - TP
FN = n_gt - TP

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

mean_iou = (
    np.mean([m[2] for m in matches])
    if matches
    else 0
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("Object-wise results")
print("-------------------")
print(f"GT nuclei:        {n_gt}")
print(f"Pred nuclei:      {n_pred}")
print(f"Matched nuclei:   {TP}")
print(f"False positives:  {FP}")
print(f"False negatives:  {FN}")
print()
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1:               {f1:.4f}")
print(f"Mean matched IoU: {mean_iou:.4f}")