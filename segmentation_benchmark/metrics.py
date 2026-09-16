import math

import numpy as np
from skimage import measure


def compare_instances(predicted, ground_truth, iou_threshold=0.5):
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

    if predicted_voxels + ground_truth_voxels == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / (predicted_voxels + ground_truth_voxels)
    iou = 1.0 if union == 0 else intersection / union

    overlaps = _overlap_rows(predicted, ground_truth)
    matches = _greedy_matches(overlaps, iou_threshold)
    true_positives = len(matches)
    false_positives = predicted_count - true_positives
    false_negatives = ground_truth_count - true_positives

    precision = _safe_detection_ratio(true_positives, predicted_count)
    recall = _safe_detection_ratio(true_positives, ground_truth_count)
    if precision + recall == 0:
        object_f1 = 0.0
    else:
        object_f1 = 2.0 * precision * recall / (precision + recall)

    matched_ious = [value[1] for value in matches.values()]
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    panoptic_denominator = (
        true_positives + 0.5 * false_positives + 0.5 * false_negatives
    )
    panoptic_quality = (
        float(np.sum(matched_ious)) / panoptic_denominator
        if panoptic_denominator
        else 1.0
    )

    best_matches = {}
    for predicted_label, ground_truth_label, pair_iou in overlaps:
        current = best_matches.get(predicted_label)
        if current is None or pair_iou > current[1]:
            best_matches[predicted_label] = (ground_truth_label, pair_iou)

    metrics = {
        "num_predicted_masks": predicted_count,
        "num_ground_truth_masks": ground_truth_count,
        "count_error": predicted_count - ground_truth_count,
        "absolute_count_error": abs(predicted_count - ground_truth_count),
        "predicted_foreground_voxels": predicted_voxels,
        "ground_truth_foreground_voxels": ground_truth_voxels,
        "foreground_fraction": predicted_voxels / predicted.size,
        "dice": dice,
        "semantic_iou": iou,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": object_f1,
        "mean_matched_iou": mean_matched_iou,
        "panoptic_quality": panoptic_quality,
    }
    return metrics, matches, best_matches


def object_measurements(
    labels,
    image_name,
    method,
    voxel_size=None,
    accepted_matches=None,
    best_matches=None,
):
    voxel_size = tuple(voxel_size) if voxel_size is not None else None
    voxel_volume = float(np.prod(voxel_size)) if voxel_size is not None else None
    accepted_matches = accepted_matches or {}
    best_matches = best_matches or {}
    rows = []

    for region in measure.regionprops(labels):
        voxel_count = int(region.area)
        volume = voxel_count * voxel_volume if voxel_volume is not None else None
        centroid = tuple(float(value) for value in region.centroid)
        bbox = tuple(int(value) for value in region.bbox)
        accepted = accepted_matches.get(region.label)
        best = best_matches.get(region.label)
        rows.append(
            {
                "image": image_name,
                "method": method,
                "mask_label": int(region.label),
                "voxel_count": voxel_count,
                "volume_um3": volume,
                "equivalent_sphere_diameter_um": (
                    (6.0 * volume / math.pi) ** (1.0 / 3.0)
                    if volume is not None and volume > 0
                    else None
                ),
                "centroid_z": centroid[0],
                "centroid_y": centroid[1],
                "centroid_x": centroid[2],
                "centroid_z_um": (
                    centroid[0] * voxel_size[0] if voxel_size is not None else None
                ),
                "centroid_y_um": (
                    centroid[1] * voxel_size[1] if voxel_size is not None else None
                ),
                "centroid_x_um": (
                    centroid[2] * voxel_size[2] if voxel_size is not None else None
                ),
                "bbox_z_min": bbox[0],
                "bbox_y_min": bbox[1],
                "bbox_x_min": bbox[2],
                "bbox_z_max": bbox[3],
                "bbox_y_max": bbox[4],
                "bbox_x_max": bbox[5],
                "matched_ground_truth_label": accepted[0] if accepted else None,
                "matched_iou": accepted[1] if accepted else None,
                "best_ground_truth_label": best[0] if best else None,
                "best_iou": best[1] if best else None,
            }
        )
    return rows


def _overlap_rows(predicted, ground_truth):
    predicted_count = int(predicted.max(initial=0))
    ground_truth_count = int(ground_truth.max(initial=0))
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
        overlaps.append(
            (
                predicted_label,
                ground_truth_label,
                float(pair_intersection / pair_union),
            )
        )
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


def _safe_detection_ratio(numerator, denominator):
    if denominator:
        return numerator / denominator
    return 1.0 if numerator == 0 else 0.0
