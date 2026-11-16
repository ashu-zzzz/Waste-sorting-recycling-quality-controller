
from pathlib import Path
from datetime import datetime
import json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
DATA = ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"
BASELINE = EXPERIMENTS / "yolo11n_baseline-2" / "weights" / "best.pt"
CBAM_CANDIDATES = sorted(
    EXPERIMENTS.glob("yolo11n_cbam*/weights/best.pt"),
    key=lambda path: path.stat().st_mtime,
    reverse=True
)

IMGSZ = 640
BATCH = 8
DEVICE = 0
CONF = 0.001
IOU = 0.7


def evaluate(label, weights, output_dir):
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"\nEvaluating {label}")
    print(f"Weights: {weights}")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=4,
        conf=CONF,
        iou=IOU,
        plots=True,
        project=str(output_dir),
        name=label,
        exist_ok=False,
        verbose=False
    )

    values = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "inference_ms_per_image": float(metrics.speed["inference"]),
        "weights_size_MB": weights.stat().st_size / (1024 * 1024)
    }

    names = model.names
    class_metrics = {}

    for class_id, score in enumerate(metrics.box.maps):
        class_name = names[class_id]
        class_metrics[class_name] = {
            "mAP50-95": float(score),
            "mAP50": float(metrics.box.ap50[class_id])
        }

    print("\nOverall metrics")
    for key, value in values.items():
        print(f"{key}: {value:.6f}")

    print("\nPer-class metrics")
    for class_name, scores in class_metrics.items():
        print(
            f"{class_name}: "
            f"mAP50={scores['mAP50']:.4f}, "
            f"mAP50-95={scores['mAP50-95']:.4f}"
        )

    return {"weights": str(weights), "overall": values, "classes": class_metrics}


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    if not BASELINE.is_file():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE}")

    if not CBAM_CANDIDATES:
        raise FileNotFoundError(
            "No experiments/yolo11n_cbam*/weights/best.pt was found."
        )

    cbam_weights = CBAM_CANDIDATES[0]
    run_dir = EXPERIMENTS / "controlled_validation" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Dataset: {DATA}")
    print(f"CBAM checkpoint selected: {cbam_weights}")
    print(f"Validation outputs: {run_dir}")

    baseline_results = evaluate("baseline", BASELINE, run_dir)
    cbam_results = evaluate("cbam", cbam_weights, run_dir)

    comparison = {}

    for metric in ("precision", "recall", "mAP50", "mAP50-95"):
        baseline_value = baseline_results["overall"][metric]
        cbam_value = cbam_results["overall"][metric]

        comparison[metric] = {
            "baseline": baseline_value,
            "cbam": cbam_value,
            "change_percentage_points": (cbam_value - baseline_value) * 100
        }

    summary = {
        "dataset": str(DATA),
        "split": "val",
        "settings": {
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "conf": CONF,
            "iou": IOU
        },
        "baseline": baseline_results,
        "cbam": cbam_results,
        "comparison": comparison
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nComparison in percentage points")
    for metric, values in comparison.items():
        print(
            f"{metric}: "
            f"baseline={values['baseline'] * 100:.2f}%, "
            f"CBAM={values['cbam'] * 100:.2f}%, "
            f"change={values['change_percentage_points']:+.2f} pp"
        )

    print(f"\nComparison report: {summary_path}")


if __name__ == "__main__":
    main()
