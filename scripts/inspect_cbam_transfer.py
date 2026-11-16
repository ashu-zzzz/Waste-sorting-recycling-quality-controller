
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "experiments" / "yolo11n_baseline-2" / "weights" / "best.pt"
CONFIG = ROOT / "waste_cbam.yaml"

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

source = baseline.model.state_dict()
target = custom.model.state_dict()

matched = 0
missing = []
mismatched = []

for key, value in source.items():
    target_key = key

    if key.startswith(f"model.{baseline_detect}."):
        target_key = (
            f"model.{custom_detect}."
            + key[len(f"model.{baseline_detect}."):]
        )

    if target_key not in target:
        missing.append(key)
    elif target[target_key].shape != value.shape:
        mismatched.append(
            (key, tuple(value.shape), tuple(target[target_key].shape))
        )
    else:
        matched += 1

print(f"Compatible tensors: {matched}/{len(source)}")
print(f"Missing keys: {len(missing)}")

for key in missing:
    print("MISSING:", key)

print(f"Shape mismatches: {len(mismatched)}")

for key, source_shape, target_shape in mismatched:
    print(
        f"SHAPE: {key} | baseline={source_shape} "
        f"| CBAM={target_shape}"
    )
