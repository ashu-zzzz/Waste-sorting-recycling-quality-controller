
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "experiments" / "yolo11n_baseline-2" / "weights" / "best.pt"
CONFIG = ROOT / "waste_cbam.yaml"

def main():
    if not BASELINE.exists():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE}")

    if not CONFIG.exists():
        raise FileNotFoundError(f"CBAM configuration not found: {CONFIG}")

    baseline = YOLO(str(BASELINE))
    custom = YOLO(str(CONFIG))

    baseline_layers = baseline.model.model
    custom_layers = custom.model.model

    baseline_detect = next(
        i for i, layer in enumerate(baseline_layers)
        if isinstance(layer, Detect)
    )

    custom_detect = next(
        i for i, layer in enumerate(custom_layers)
        if isinstance(layer, Detect)
    )

    source_state = baseline.model.state_dict()
    target_state = custom.model.state_dict()
    transferred = 0
    skipped = []

    for key, value in source_state.items():
        target_key = key

        if key.startswith(f"model.{baseline_detect}."):
            target_key = (
                f"model.{custom_detect}."
                + key[len(f"model.{baseline_detect}."):]
            )

        if (
            target_key in target_state
            and target_state[target_key].shape == value.shape
        ):
            target_state[target_key] = value
            transferred += 1
        else:
            skipped.append(key)

    custom.model.load_state_dict(target_state)

    print(f"Baseline detection layer: {baseline_detect}")
    print(f"CBAM detection layer: {custom_detect}")
    print(f"Transferred weight tensors: {transferred}/{len(source_state)}")
    print(f"Unmatched baseline tensors: {len(skipped)}")
    print("Weight transfer check completed.")

if __name__ == "__main__":
    main()
