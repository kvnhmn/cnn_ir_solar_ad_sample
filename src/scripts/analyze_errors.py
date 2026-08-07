import json
from collections import Counter
from pathlib import Path

import torch
from dataset.ir_solar import IRSolarDataset
from network.ir_net import IRNet


def load_original_classes(root) -> dict[str, str]:
    with open(Path(root) / "module_metadata.json") as f:
        metadata = json.load(f)

    items = {}
    for idx in metadata:
        original_item = metadata[idx]
        items[original_item["image_filepath"]] = original_item["anomaly_class"]

    return items


@torch.no_grad()
def analyze(model, val_ds, device, orig_classes) -> None:
    model.eval()
    false_negatives = []

    tn = fp = fn = tp = 0
    for i in range(len(val_ds)):
        sample = val_ds[i]
        img = sample["image"].unsqueeze(0).to(device)
        true = int(sample["label"])
        pred = model(img).argmax(dim=1).item()

        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
            false_negatives.append(orig_classes[sample["path"]])
        elif true == 1 and pred == 1:
            tp += 1

    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{len(false_negatives)} false negatives")
    print(Counter(false_negatives).most_common())


def main():
    root = "./../../dataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = IRSolarDataset(root, split="val")
    orig_classes = load_original_classes(root)

    model = IRNet(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load("irnet.pt", map_location=device, weights_only=True))
    model.to(device)

    analyze(model, val_ds, device, orig_classes)


if __name__ == "__main__":
    main()