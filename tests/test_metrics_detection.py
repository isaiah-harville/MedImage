import numpy as np
import pytest

from medimage.metrics.detection import DetectionEvaluator, box_iou


def _random_scene(seed, num_images=25, num_classes=3):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(num_images):
        n_gt = int(rng.integers(0, 5))
        xy = rng.uniform(0, 200, size=(n_gt, 2))
        wh = rng.uniform(10, 60, size=(n_gt, 2))
        gt = np.hstack([xy, xy + wh])
        gt_labels = rng.integers(0, num_classes, size=n_gt)
        # Detections: jittered copies of most GT boxes, plus random false positives.
        keep = rng.random(n_gt) < 0.8
        jitter = gt[keep] + rng.normal(0, 6, size=(int(keep.sum()), 4))
        n_fp = int(rng.integers(0, 4))
        fp_xy = rng.uniform(0, 200, size=(n_fp, 2))
        fp = np.hstack([fp_xy, fp_xy + rng.uniform(10, 60, size=(n_fp, 2))])
        boxes = np.vstack([jitter, fp])
        boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)
        labels = np.concatenate([gt_labels[keep], rng.integers(0, num_classes, size=n_fp)])
        scores = rng.random(len(boxes))
        images.append((boxes, scores, labels, gt, gt_labels))
    return images


def _pycocotools_map(images, num_classes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k + 1, "name": str(k)} for k in range(num_classes)],
    }
    dets = []
    for image_id, (boxes, scores, labels, gt_boxes, gt_labels) in enumerate(images, start=1):
        gt["images"].append({"id": image_id, "width": 300, "height": 300})
        for box, label in zip(gt_boxes, gt_labels, strict=True):
            w, h = box[2] - box[0], box[3] - box[1]
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [float(box[0]), float(box[1]), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
        for box, score, label in zip(boxes, scores, labels, strict=True):
            dets.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "score": float(score),
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                }
            )
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    evaluator = COCOeval(coco, coco.loadRes(dets), "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats


@pytest.mark.parametrize("seed", range(3))
def test_map_matches_pycocotools(seed, capsys):
    pytest.importorskip("pycocotools")
    images = _random_scene(seed)
    evaluator = DetectionEvaluator(num_classes=3)
    matched = [evaluator.match_image(*image) for image in images]
    ours = evaluator.compute(matched)
    stats = _pycocotools_map(images, 3)
    assert ours["map"] == pytest.approx(stats[0], abs=1e-6)
    assert ours["map50"] == pytest.approx(stats[1], abs=1e-6)
    assert ours["map75"] == pytest.approx(stats[2], abs=1e-6)


def test_froc_counts_duplicates_as_ignored():
    evaluator = DetectionEvaluator(num_classes=1, froc_rates=(0.125, 0.5, 1.0))
    lesion_a = np.array([[10, 10, 20, 20]])
    lesion_b = np.array([[50, 50, 70, 70]])
    image_a = evaluator.match_image(
        np.array([[10, 10, 20, 20], [11, 11, 20, 20], [80, 80, 90, 90]]),
        np.array([0.9, 0.8, 0.7]),
        np.zeros(3),
        lesion_a,
        np.zeros(1),
    )
    image_b = evaluator.match_image(lesion_b, np.array([0.6]), np.zeros(1), lesion_b, np.zeros(1))
    assert image_a.froc_outcome.tolist() == [1, -1, 0]
    metrics = evaluator.compute([image_a, image_b])
    # Ranked: 0.9 TP, 0.7 FP (0.5 FP/image), 0.6 TP.
    assert metrics["froc@0.125"] == 0.5
    assert metrics["froc@0.5"] == 1.0
    assert metrics["cpm"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_perfect_detections_score_one():
    evaluator = DetectionEvaluator(num_classes=2)
    gt = np.array([[0, 0, 10, 10], [20, 20, 40, 40]])
    labels = np.array([0, 1])
    matched = [evaluator.match_image(gt, np.array([0.9, 0.8]), labels, gt, labels)]
    metrics = evaluator.compute(matched)
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["cpm"] == pytest.approx(1.0)


def test_box_iou():
    iou = box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10], [20, 20, 30, 30]]))
    assert iou[0, 0] == pytest.approx(50 / 150)
    assert iou[0, 1] == 0
