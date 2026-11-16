from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect, Conv, DWConv

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "waste_p2.yaml"
DATA = ROOT / "GARBAGE CLASSIFICATION" / "rebalanced_dataset" / "data.yaml"
INITIAL = ROOT / "experiments" / "p2_preparation" / "p2_initial_weights.pt"
EXPERIMENTS = ROOT / "experiments"

CLASS_WIDTH = None
PREPARED_STATE = None
ORIGINAL_INIT = Detect.__init__


def install_detect_patch(class_width):
    def adjusted_detect_init(
        self, nc=80, reg_max=16, end2end=False, ch=()
    ):
        ORIGINAL_INIT(self, nc, reg_max, end2end, ch)

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
            self.one2one_cv3 = deepcopy(self.cv3)

    Detect.__init__ = adjusted_detect_init


class P2Trainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(
            cfg=cfg, weights=None, verbose=verbose
        )

        model.load_state_dict(PREPARED_STATE, strict=True)

        print("Prepared P2 weights loaded successfully.")
        print(f"Loaded tensors: {len(PREPARED_STATE)}")

        return model


def main():
    global CLASS_WIDTH, PREPARED_STATE

    if not CONFIG.is_file():
        raise FileNotFoundError(f"Model configuration not found: {CONFIG}")

    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    if not INITIAL.is_file():
        raise FileNotFoundError(
            f"Prepared weights not found: {INITIAL}. "
            "Run prepare_p2.py first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training stopped.")

    prepared = torch.load(
        INITIAL, map_location="cpu", weights_only=True
    )

    CLASS_WIDTH = int(prepared["class_width"])
    PREPARED_STATE = prepared["state_dict"]

    install_detect_patch(CLASS_WIDTH)

    model = YOLO(str(CONFIG))

    print(f"Dataset: {DATA}")
    print(f"Configuration: {CONFIG}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pretrained source: {prepared['source_checkpoint']}")

    model.train(
        trainer=P2Trainer,
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        seed=42,
        deterministic=True,
        patience=15,
        pretrained=False,
        max_det=320,
        project=str(EXPERIMENTS),
        name="yolo11n_p2",
        exist_ok=False,
        plots=True,
        save=True
    )


if __name__ == "__main__":
    main()