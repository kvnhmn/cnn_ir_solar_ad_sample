import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (24, 40)

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    return _transform(image)