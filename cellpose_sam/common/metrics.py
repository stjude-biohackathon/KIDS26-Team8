# Object-wise instance-segmentation metrics.
#
# Adapted from Caitlin Freeman's segmentation_benchmark/metrics.py
# (branch origin/caitlinfree-segmentation-benchmarking, commit 3c2efcb),
# which itself generalizes the greedy one-to-one IoU-matching approach in
# Peter Simko's Unet/compare_pred_masks.py. Kept as a small, dependency-free
# module (no CLI/class machinery) so it's easy to audit and reuse directly
# against instance-labeled predictions without rebinarizing them.

import numpy as np


def compare_instances(predicted, ground_truth, iou_thresholds=(0.01, 0.5)):
    """Score one instance-labeled prediction against one instance-labeled
    ground truth. Predictions are NOT rebinarized/relabeled here, so a
    method that correctly separates touching objects gets credit for it.

    Returns a dict with threshold-independent stats plus one nested dict
    per entry in `iou_thresholds` (keys like "iou_0.01", "iou_0.5") holding
    true_positives/false_positives/false_negatives/precision/recall/f1/
    mean_matched_iou/panoptic_quality.
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match ground-truth "
            f"shape {ground_truth.shape}"
        )

    predicted_count = int(predicted.max(initial=0))
    ground_truth_count = int(ground_truth.max(initial=0))
    predicted_foreground = predicted > 0
    ground_truth_foreground = ground_truth > 0
    intersection = int(np.count_nonzero(predicted_foreground & ground_truth_foreground))
    predicted_voxels = int(np.count_nonzero(predicted_foreground))
    ground_truth_voxels = int(np.count_nonzero(ground_truth_foreground))
    union = predicted_voxels + ground_truth_voxels - intersection

    dice = (
        1.0
        if predicted_voxels + ground_truth_voxels == 0
        else 2.0 * intersection / (predicted_voxels + ground_truth_voxels)
    )
    semantic_iou = 1.0 if union == 0 else intersection / union

    overlaps = _overlap_rows(predicted, ground_truth, predicted_count, ground_truth_count)

    result = {
        "num_predicted_masks": predicted_count,
        "num_ground_truth_masks": ground_truth_count,
        "count_error": predicted_count - ground_truth_count,
        "absolute_count_error": abs(predicted_count - ground_truth_count),
        "predicted_foreground_voxels": predicted_voxels,
        "ground_truth_foreground_voxels": ground_truth_voxels,
        "dice": dice,
        "semantic_iou": semantic_iou,
    }

    for threshold in iou_thresholds:
        matches = _greedy_matches(overlaps, threshold)
        true_positives = len(matches)
        false_positives = predicted_count - true_positives
        false_negatives = ground_truth_count - true_positives
        precision = _safe_ratio(true_positives, predicted_count)
        recall = _safe_ratio(true_positives, ground_truth_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        matched_ious = [iou for _, iou in matches.values()]
        mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        pq_denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
        panoptic_quality = (
            float(np.sum(matched_ious)) / pq_denominator if pq_denominator else 1.0
        )
        result[f"iou_{threshold}"] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_matched_iou,
            "panoptic_quality": panoptic_quality,
        }

    return result


def _overlap_rows(predicted, ground_truth, predicted_count, ground_truth_count):
    predicted_areas = np.bincount(
        predicted.ravel().astype(np.int64), minlength=predicted_count + 1
    )
    ground_truth_areas = np.bincount(
        ground_truth.ravel().astype(np.int64), minlength=ground_truth_count + 1
    )

    both_foreground = (predicted > 0) & (ground_truth > 0)
    if not np.any(both_foreground):
        return []

    predicted_values = predicted[both_foreground].astype(np.int64)
    ground_truth_values = ground_truth[both_foreground].astype(np.int64)
    encoded = ground_truth_values * (predicted_count + 1) + predicted_values
    pairs, intersections = np.unique(encoded, return_counts=True)

    overlaps = []
    for pair, pair_intersection in zip(pairs, intersections):
        ground_truth_label = int(pair // (predicted_count + 1))
        predicted_label = int(pair % (predicted_count + 1))
        pair_union = (
            predicted_areas[predicted_label]
            + ground_truth_areas[ground_truth_label]
            - pair_intersection
        )
        overlaps.append((predicted_label, ground_truth_label, float(pair_intersection / pair_union)))
    return overlaps


def _greedy_matches(overlaps, iou_threshold):
    matches = {}
    used_ground_truth = set()
    for predicted_label, ground_truth_label, pair_iou in sorted(
        overlaps, key=lambda row: row[2], reverse=True
    ):
        if pair_iou < iou_threshold:
            break
        if predicted_label in matches or ground_truth_label in used_ground_truth:
            continue
        matches[predicted_label] = (ground_truth_label, pair_iou)
        used_ground_truth.add(ground_truth_label)
    return matches


def _safe_ratio(numerator, denominator):
    if denominator:
        return numerator / denominator
    return 1.0 if numerator == 0 else 0.0
