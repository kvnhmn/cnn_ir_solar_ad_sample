import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.ir_solar import IRSolarDataset
from network.ir_net import IRNet


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        output = model(images)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        output = model(images)
        loss = criterion(output, targets)

        running_loss += loss.item() * images.size(0)
        prediction = output.argmax(dim=1)
        correct += (prediction == targets).sum().item()

    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


def main():
    root = "./../../dataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = IRSolarDataset(root, split="train")
    val_ds = IRSolarDataset(root, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = IRNet(in_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | val-acc {val_acc:.3f}")

    torch.save(model.state_dict(), "irnet.pt")


if __name__ == "__main__":
    main()