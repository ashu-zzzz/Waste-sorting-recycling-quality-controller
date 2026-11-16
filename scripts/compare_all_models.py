
from pathlib import Path
from datetime import datetime
import json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
DATA = ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"

MODELS = {
    "baseline": EXPERIMENTS / "yolo11n_baseline-2" / "weights" / "best.pt",
    "cbam": EXPERIMENTS / "yolo11n_cbam" / "weights" / "best.pt",
    "p2": EXPERIMENTS / "yolo11n_p2" / "weights" / "best.pt"
}

def evaluate(name, weights, output):
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        conf=0.001,
        iou=0.7,
        max_det=320,
        plots=True,
        project=str(output),
        name=name,
        exist_ok=False,
        verbose=False
    )

    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "inference_ms": float(metrics.speed["inference"]),
        "weights_MB": weights.stat().st_size / (1024 * 1024)
    }

    classes = {}

    for class_id, map_score in enumerate(metrics.box.maps):
        class_name = model.names[class_id]
        classes[class_name] = {
            "mAP50": float(metrics.box.ap50[class_id]),
            "mAP50-95": float(map_score)
        }

    print(f"\n{name.upper()}")
    for metric, value in overall.items():
        print(f"{metric}: {value:.5f}")

    for class_name, values in classes.items():
        print(
            f"{class_name}: "
            f"mAP50={values['mAP50']:.4f}, "
            f"mAP50-95={values['mAP50-95']:.4f}"
        )

    return {"weights": str(weights), "overall": overall, "classes": classes}


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    for name, weights in MODELS.items():
        if not weights.is_file():
            raise FileNotFoundError(f"{name} weights not found: {weights}")

    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    output = EXPERIMENTS / "controlled_all_models" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    results = {}

    for name, weights in MODELS.items():
        results[name] = evaluate(name, weights, output)

    baseline = results["baseline"]["overall"]
    changes = {}

    for name in ("cbam", "p2"):
        changes[name] = {}

        for metric in ("precision", "recall", "mAP50", "mAP50-95"):
            changes[name][metric + "_change_pp"] = (
                results[name]["overall"][metric] - baseline[metric]
            ) * 100

    report = {
        "dataset": str(DATA),
        "split": "val",
        "settings": {
            "imgsz": 640,
            "batch": 8,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 320
        },
        "models": results,
        "changes_vs_baseline_percentage_points": changes
    }

    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCHANGES VS BASELINE (percentage points)")
    for name, values in changes.items():
        print(f"\n{name.upper()}")
        for metric, change in values.items():
            print(f"{metric}: {change:+.3f}")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
