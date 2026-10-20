import numpy as np

from nuclei_stage1.evaluate import evaluate_one, parse_z_range


def blob(shape, lo, hi, v, vol=None):
    vol = np.zeros(shape, np.int32) if vol is None else vol
    vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = v
    return vol


def test_perfect_prediction():
    gt = blob((20, 40, 40), (2, 2, 2), (10, 12, 12), 1)
    blob(gt.shape, (2, 20, 20), (10, 30, 30), 2, gt)
    m = evaluate_one(gt.copy(), gt, 0.25, 0.15)
    assert m["f1"] == 1.0 and m["mean_matched_iou"] == 1.0 and m["split_gt_nuclei"] == 0


def test_split_and_false_positive_are_counted():
    gt = blob((20, 40, 40), (2, 2, 2), (10, 12, 22), 1)
    pred = blob(gt.shape, (2, 2, 2), (10, 12, 12), 1)
    blob(gt.shape, (2, 2, 12), (10, 12, 22), 2, pred)      # the one GT nucleus predicted as two halves
    blob(gt.shape, (12, 30, 30), (18, 38, 38), 3, pred)    # an object with no GT behind it
    m = evaluate_one(pred, gt, 0.25, 0.15)
    assert m["split_gt_nuclei"] == 1 and m["extra_false_positives"] >= 1


def test_parse_z_range():
    assert parse_z_range("1:127", 128) == (1, 127)
    assert parse_z_range(None, 50) == (0, 50)
