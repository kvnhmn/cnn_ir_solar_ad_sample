import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from core.preprocess import preprocess


class IRSolarDataset(Dataset):

    def __init__(self, root, split="train", val_ratio=0.2, size=(24, 40), seed=42):
        self.root: Path = Path(root)
        self.split = split

        with open(self.root / "module_metadata.json") as f:
            metadata = json.load(f)

        self.items = []
        for idx in metadata:
            original_item = metadata[idx]
            label = 0
            if original_item["anomaly_class"] != "No-Anomaly":
                label = 1

            self.items.append({
                "image_filepath": original_item["image_filepath"],
                "label": label
            })

        random.Random(seed).shuffle(self.items)
        item_len = len(self.items)
        val_len = round(item_len * val_ratio)

        if split == "train":
            self.items = self.items[val_len:]
        else:
            self.items = self.items[:val_len]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(self.root / item["image_filepath"])

        return {
            "image": preprocess(image),
            "label": item["label"],
            "path": item["image_filepath"]
        }


def main():
    root = "./../dataset"

    train_ds = IRSolarDataset(root, split="train")
    val_ds = IRSolarDataset(root, split="val")
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    print("train labels:", Counter(it["label"] for it in train_ds.items))
    print("val labels:  ", Counter(it["label"] for it in val_ds.items))

    sample = train_ds[0]
    print("image:", tuple(sample["image"].shape), "label:", sample["label"])


if __name__ == "__main__":
    main()
