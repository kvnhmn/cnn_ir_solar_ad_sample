import torch
import torch.nn.functional as F

from core.ir_net import IRNet

CLASS_NAMES = ("normal", "anomaly")


def load_model(weights_path: str, device: str) -> IRNet:
    model = IRNet(in_channels=1, num_classes=2)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: IRNet, tensor: torch.Tensor, device: str) -> dict:
    x = tensor.unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    label = int(probs.argmax().item())
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "p_anomaly": round(float(probs[1]), 4),
        "confidence": round(float(probs[label]), 4),
    }