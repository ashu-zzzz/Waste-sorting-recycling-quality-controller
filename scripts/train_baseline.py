
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "experiments"

def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training stopped.")

    print(f"Dataset: {DATA_YAML}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        seed=42,
        deterministic=True,
        patience=15,
        project=str(RUNS_DIR),
        name="yolo11n_baseline",
        exist_ok=False,
        plots=True,
        save=True
    )

    print("\nTraining finished.")
    print(f"Results: {RUNS_DIR / 'yolo11n_baseline'}")
    print(results.results_dict)

if __name__ == "__main__":
    main()
