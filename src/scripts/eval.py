import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset.ir_solar import IRSolarDataset
from network.ir_net import IRNet


@torch.no_grad()
def full_evaluation(model, loader, device, class_names=("normal", "anomal")) -> None:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        output = model(images)
        prediction = output.argmax(dim=1)

        all_preds.extend(prediction.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    print(confusion_matrix(all_targets, all_preds))
    print(classification_report(all_targets, all_preds, target_names=class_names))


def main():
    root = "./../../dataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = IRSolarDataset(root, split="val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = IRNet(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load("irnet.pt", map_location=device, weights_only=True))
    model.to(device)

    full_evaluation(model, val_loader, device)


if __name__ == "__main__":
    main()