
from pathlib import Path
from copy import deepcopy
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect, Conv, DWConv

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"
CONFIG = ROOT / "waste_cbam.yaml"
RUNS_DIR = ROOT / "experiments"
PRETRAINED = "yolo11n.pt"

original_detect_init = Detect.__init__


def fixed_detect_init(self, nc=80, reg_max=16, end2end=False, ch=()):
    original_detect_init(self, nc, reg_max, end2end, ch)

    if not ch:
        return

    target_channels = max(ch[0], 80)
    current_channels = max(ch[0], min(nc, 100))

    if target_channels == current_channels:
        return

    if self.legacy:
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, target_channels, 3),
                Conv(target_channels, target_channels, 3),
                nn.Conv2d(target_channels, nc, 1)
            )
            for x in ch
        )
    else:
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, target_channels, 1)),
                nn.Sequential(
                    DWConv(target_channels, target_channels, 3),
                    Conv(target_channels, target_channels, 1)
                ),
                nn.Conv2d(target_channels, nc, 1)
            )
            for x in ch
        )

    if hasattr(self, "one2one_cv3"):
        self.one2one_cv3 = deepcopy(self.cv3)


Detect.__init__ = fixed_detect_init


class CBAMTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        source_model = YOLO(PRETRAINED).model

        source_layers = source_model.model
        target_layers = model.model

        source_detect = next(
            i for i, layer in enumerate(source_layers)
            if isinstance(layer, Detect)
        )
        target_detect = next(
            i for i, layer in enumerate(target_layers)
            if isinstance(layer, Detect)
        )

        source_state = source_model.state_dict()
        target_state = model.state_dict()

        transferred = 0
        skipped = []
        unexpected = []

        source_prefix = f"model.{source_detect}."
        target_prefix = f"model.{target_detect}."

        for key, value in source_state.items():
            target_key = key

            if key.startswith(source_prefix):
                target_key = target_prefix + key[len(source_prefix):]

            if target_key not in target_state:
                unexpected.append(f"{key}: no matching target tensor")
                continue

            target_value = target_state[target_key]

            if value.shape != target_value.shape:
                expected_class_output = (
                    "cv3" in target_key
                    and value.ndim > 0
                    and target_value.ndim == value.ndim
                    and value.shape[0] == 80
                    and target_value.shape[0] == 6
                    and value.shape[1:] == target_value.shape[1:]
                )

                if expected_class_output:
                    skipped.append(target_key)
                else:
                    unexpected.append(
                        f"{key}: {tuple(value.shape)} -> "
                        f"{tuple(target_value.shape)}"
                    )
                continue

            target_state[target_key] = value
            transferred += 1

        if unexpected:
            raise RuntimeError(
                "Unexpected weight-transfer mismatches. Training stopped:\n"
                + "\n".join(unexpected[:15])
            )

        model.load_state_dict(target_state)

        print(f"Pretrained detection layer: {source_detect}")
        print(f"CBAM detection layer: {target_detect}")
        print(
            f"Transferred pretrained tensors: "
            f"{transferred}/{len(source_state)}"
        )
        print(f"Class-output tensors initialized for six classes: {len(skipped)}")

        return model


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not CONFIG.is_file():
        raise FileNotFoundError(f"CBAM configuration not found: {CONFIG}")

    model = YOLO(str(CONFIG))

    model.train(
        trainer=CBAMTrainer,
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        seed=42,
        deterministic=True,
        patience=15,
        pretrained=False,
        project=str(RUNS_DIR),
        name="yolo11n_cbam",
        exist_ok=False,
        plots=True,
        save=True
    )


if __name__ == "__main__":
    main()
