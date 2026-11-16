import re
from pathlib import Path
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Conv, DWConv

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "waste_p2.yaml"
DATA = ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"
OUTPUT = ROOT / "experiments" / "p2_preparation"
INITIAL = OUTPUT / "p2_initial_weights.pt"
REPORT = OUTPUT / "preparation_report.txt"
PRETRAINED = "yolo11n.pt"


def main():
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Model configuration not found: {CONFIG}")

    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    if INITIAL.exists():
        raise FileExistsError(
            f"Prepared weights already exist: {INITIAL}. "
            "Keep the existing preparation or choose a new output name."
        )

    config_text = CONFIG.read_text(encoding="utf-8")

    if not re.search(r"^\s*scale\s*:", config_text, re.MULTILINE):
        CONFIG.write_text("scale: n\n" + config_text, encoding="utf-8")

    torch.manual_seed(42)

    source_model = YOLO(PRETRAINED).model
    source_layers = source_model.model

    source_detect_index = next(
        i for i, layer in enumerate(source_layers)
        if isinstance(layer, Detect)
    )

    source_detect = source_layers[source_detect_index]

    if source_detect.legacy:
        class_width = source_detect.cv3[0][0].conv.out_channels
    else:
        class_width = source_detect.cv3[0][0][1].conv.out_channels

    original_init = Detect.__init__

    def adjusted_detect_init(
        self, nc=80, reg_max=16, end2end=False, ch=()
    ):
        original_init(self, nc, reg_max, end2end, ch)

        if not ch or class_width == max(ch[0], min(nc, 100)):
            return

        if self.legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, class_width, 3),
                    Conv(class_width, class_width, 3),
                    nn.Conv2d(class_width, nc, 1)
                )
                for x in ch
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(
                        DWConv(x, x, 3),
                        Conv(x, class_width, 1)
                    ),
                    nn.Sequential(
                        DWConv(class_width, class_width, 3),
                        Conv(class_width, class_width, 1)
                    ),
                    nn.Conv2d(class_width, nc, 1)
                )
                for x in ch
            )

        if getattr(self, "one2one_cv3", None) is not None:
            from copy import deepcopy
            self.one2one_cv3 = deepcopy(self.cv3)

    Detect.__init__ = adjusted_detect_init

    try:
        target_model = YOLO(str(CONFIG)).model
    finally:
        Detect.__init__ = original_init

    target_layers = target_model.model

    source_detect_index = next(
        i for i, layer in enumerate(source_layers)
        if isinstance(layer, Detect)
    )

    target_detect_index = next(
        i for i, layer in enumerate(target_layers)
        if isinstance(layer, Detect)
    )

    target_detect = target_layers[target_detect_index]
    offset = target_detect_index - source_detect_index

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    transferred_keys = set()
    skipped_class_outputs = []
    problems = []

    def mapped_key(key):
        match = re.match(r"model\.(\d+)\.(.*)", key)

        if not match:
            return None

        source_index = int(match.group(1))
        remainder = match.group(2)

        if source_index < 17:
            target_index = source_index
        elif source_index < source_detect_index:
            target_index = source_index + offset
        elif source_index == source_detect_index:
            target_index = target_detect_index
            parts = remainder.split(".")

            if (
                len(parts) > 1
                and parts[0] in {
                    "cv2", "cv3", "one2one_cv2", "one2one_cv3"
                }
                and parts[1].isdigit()
            ):
                parts[1] = str(int(parts[1]) + 1)
                remainder = ".".join(parts)
        else:
            return None

        return f"model.{target_index}.{remainder}"

    for key, value in source_state.items():
        target_key = mapped_key(key)

        if target_key is None or target_key not in target_state:
            problems.append(f"No target tensor for {key}")
            continue

        target_value = target_state[target_key]

        if value.shape != target_value.shape:
            parts = key.split(".")
            is_class_output = (
                ("cv3" in parts or "one2one_cv3" in parts)
                and value.ndim > 0
                and value.shape[0] == source_detect.nc
                and target_value.shape[0] == target_detect.nc
                and value.shape[1:] == target_value.shape[1:]
            )

            if is_class_output:
                skipped_class_outputs.append(key)
                continue

            problems.append(
                f"{key}: {tuple(value.shape)} -> "
                f"{tuple(target_value.shape)}"
            )
            continue

        target_state[target_key] = value
        transferred_keys.add(target_key)

    if problems:
        raise RuntimeError(
            "Weight transfer failed. No prepared checkpoint was saved:\n"
            + "\n".join(problems[:20])
        )

    target_model.load_state_dict(target_state, strict=True)

    uninitialized = sorted(set(target_state) - transferred_keys)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    prepared_state = {
        key: value.detach().cpu()
        for key, value in target_model.state_dict().items()
    }

    torch.save(
        {
            "state_dict": prepared_state,
            "class_width": class_width,
            "config": str(CONFIG),
            "source_checkpoint": PRETRAINED,
            "transferred_tensor_count": len(transferred_keys),
            "source_tensor_count": len(source_state),
            "skipped_class_output_count": len(skipped_class_outputs),
            "uninitialized_tensor_count": len(uninitialized)
        },
        INITIAL
    )

    report = [
        "P2 MODEL PREPARATION REPORT",
        f"Configuration: {CONFIG}",
        f"Dataset: {DATA}",
        f"Pretrained source: {PRETRAINED}",
        f"Baseline detection layer: {source_detect_index}",
        f"P2 detection layer: {target_detect_index}",
        f"Classification width: {class_width}",
        f"Compatible transferred tensors: {len(transferred_keys)}",
        f"Source tensors: {len(source_state)}",
        f"Class-output tensors left initialized: {len(skipped_class_outputs)}",
        f"New or unmatched target tensors: {len(uninitialized)}",
        f"Prepared weights: {INITIAL}",
        "",
        "New P2-specific layers retain their initialization."
    ]

    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()