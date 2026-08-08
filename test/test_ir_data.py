from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

try:
    from training.dataset.ir_solar import IRSolarDataset
except ImportError as e:
    print("Could not import IRSolarDataset.")
    print("Place this script next to ir_data.py or run it from the repo folder.")
    print(f"Original: {e}")
    sys.exit(2)


class Harness:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def run(self, name, fn) -> None:
        try:
            fn()
        except Exception as e:
            self.failed += 1
            print(f"  [FAIL] {name}")
            print(f"         {type(e).__name__}: {e}")
        else:
            self.passed += 1
            print(f"  [PASS] {name}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print("\n" + "-" * 60)
        print(f"Result: {self.passed}/{total} checks green, {self.failed} red")
        return 0 if self.failed == 0 else 1


def paths_of(ds) -> list[str]:
    return [str(it["image_filepath"]) for it in ds.items]


def labels_of(ds) -> list[int]:
    return [int(it["label"]) for it in ds.items]


def check_non_empty(ds, split) -> None:
    assert len(ds) > 0, f"{split} is empty"
    print(f"         {split}: {len(ds)} Items")


def check_ratio(train_ds, val_ds) -> None:
    n_train, n_val = len(train_ds), len(val_ds)
    assert n_train > n_val, f"train ({n_train}) should be greater than val ({n_val})"
    print(f"         train {n_train} > val {n_val}  (total {n_train + n_val})")


def check_no_leakage(train_ds, val_ds) -> None:
    train_paths = set(paths_of(train_ds))
    val_paths = set(paths_of(val_ds))
    overlap = train_paths & val_paths
    assert not overlap, f"LEAKAGE: {len(overlap)} images in train AND val!"
    assert len(train_paths) == len(train_ds), "Duplicates in train split"
    assert len(val_paths) == len(val_ds), "Duplicates in val split"
    print(f"         train ∩ val = ∅  (no leakage)")


def check_determinism(root) -> None:
    a = IRSolarDataset(root, split="train")
    b = IRSolarDataset(root, split="train")
    assert paths_of(a) == paths_of(b), "two train instances differ (seed?)"
    print(f"         two train instances identical")


def check_labels_binary(ds, split) -> None:
    labels = set(labels_of(ds))
    assert labels <= {0, 1}, f"{split}: labels outside {{0,1}}: {labels}"
    assert labels == {0, 1}, f"{split}: not both classes present: {labels}"
    print(f"         {split}: Labels = {sorted(labels)}")


def check_item_structure(ds, channels, size) -> None:
    item = ds[0]
    assert {"image", "label", "path"}.issubset(item), f"Missing keys: {item.keys()}"

    img = item["image"]
    assert torch.is_tensor(img), f"image not a Tensor: {type(img)}"
    assert img.dtype == torch.float32, f"image dtype {img.dtype}"
    h, w = size
    assert tuple(img.shape) == (channels, h, w), \
        f"image shape {tuple(img.shape)}, expected ({channels},{h},{w})"
    assert torch.isfinite(img).all(), "image NaN/Inf"
    assert img.min() >= 0.0 and img.max() <= 1.0, \
        f"image range [{img.min():.2f},{img.max():.2f}], expected [0,1] (ToTensor)"

    assert item["label"] in (0, 1), f"label {item['label']}"
    assert isinstance(item["path"], str), f"path not a str: {type(item['path'])}"
    print(f"         image {tuple(img.shape)}  range [{img.min():.2f},{img.max():.2f}]")


def check_sample_iteration(ds, split, n=200) -> None:
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step):
        try:
            _ = ds[i]
        except Exception as e:
            raise AssertionError(f"{split}: item {i} crashed -> {type(e).__name__}: {e}") from e
    print(f"         {split}: sample (every {step}th) without crash")


def check_batching(ds, channels, size, split) -> None:
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    b = batch["image"].shape[0]
    h, w = size
    assert tuple(batch["image"].shape) == (b, channels, h, w), \
        f"batch image {tuple(batch['image'].shape)}"
    assert tuple(batch["label"].shape) == (b,), f"batch label {tuple(batch['label'].shape)}"
    print(f"         {split}: Batch ok, image {tuple(batch['image'].shape)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder with images/ + module_metadata.json")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--size", type=int, nargs=2, default=(24, 40), metavar=("H", "W"))
    args = ap.parse_args()
    size = tuple(args.size)

    print(f"root={args.root}  channels={args.channels}  size={size}\n")
    h = Harness()

    ds = {}
    for split in ("train", "val"):
        def build(s=split):
            ds[s] = IRSolarDataset(args.root, split=s)
        h.run(f"instantiate '{split}'", build)

    if "train" in ds and "val" in ds:
        h.run("train non-empty", lambda: check_non_empty(ds["train"], "train"))
        h.run("val non-empty", lambda: check_non_empty(ds["val"], "val"))
        h.run("split ratio (train > val)", lambda: check_ratio(ds["train"], ds["val"]))
        h.run("no leakage (train ∩ val = ∅)", lambda: check_no_leakage(ds["train"], ds["val"]))
        h.run("determinism (fixed seed)", lambda: check_determinism(args.root))
        h.run("train labels binary", lambda: check_labels_binary(ds["train"], "train"))
        h.run("val labels binary", lambda: check_labels_binary(ds["val"], "val"))
        h.run("item structure", lambda: check_item_structure(ds["train"], args.channels, size))
        h.run("train sample iteration", lambda: check_sample_iteration(ds["train"], "train"))
        h.run("batching", lambda: check_batching(ds["train"], args.channels, size, "train"))

    return h.summary()


if __name__ == "__main__":
    sys.exit(main())