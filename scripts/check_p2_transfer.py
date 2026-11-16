
from pathlib import Path
import re
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "experiments" / "yolo11n_baseline-2" / "weights" / "best.pt"
CONFIG = ROOT / "waste_p2.yaml"


def main():
    baseline = YOLO(str(BASELINE)).model
    custom = YOLO(str(CONFIG)).model

    source_layers = baseline.model
    target_layers = custom.model

    source_detect = next(
        i for i, layer in enumerate(source_layers)
        if isinstance(layer, Detect)
    )
    target_detect = next(
        i for i, layer in enumerate(target_layers)
        if isinstance(layer, Detect)
    )

    offset = target_detect - source_detect
    source_state = baseline.state_dict()
    target_state = custom.state_dict()

    transferred = 0
    mismatches = []
    missing = []

    for key, value in source_state.items():
        match = re.match(r"model\.(\d+)\.(.*)", key)

        if not match:
            missing.append(key)
            continue

        layer_id = int(match.group(1))
        rest = match.group(2)

        target_id = layer_id if layer_id < 17 else layer_id + offset

        if layer_id == source_detect:
            parts = rest.split(".")
            if (
                len(parts) > 2
                and parts[0] in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
                and parts[1].isdigit()
            ):
                parts[1] = str(int(parts[1]) + 1)
                rest = ".".join(parts)

        target_key = f"model.{target_id}.{rest}"

        if target_key not in target_state:
            missing.append(key)
        elif target_state[target_key].shape != value.shape:
            mismatches.append(
                f"{key}: {tuple(value.shape)} -> "
                f"{tuple(target_state[target_key].shape)}"
            )
        else:
            transferred += 1

    print(f"Baseline detection layer: {source_detect}")
    print(f"P2 detection layer: {target_detect}")
    print(f"Compatible tensors: {transferred}/{len(source_state)}")
    print(f"Missing tensors: {len(missing)}")
    print(f"Shape mismatches: {len(mismatches)}")

    for item in mismatches[:15]:
        print(item)

    if mismatches:
        print("Transfer needs further inspection before training.")
    else:
        print("No incompatible tensor shapes found in the mapped weights.")


if __name__ == "__main__":
    main()

