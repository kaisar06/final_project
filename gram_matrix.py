import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import vgg19, VGG19_Weights

# === Функция для загрузки изображения ===
def load_image(img_path, size=256):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # добавляем batch размер
    return image

# === Грамм-матрица ===
# === Грамм-матрица ===
def gram_matrix(tensor):
    # Убедимся, что тензор имеет 4 измерения (например, [B, C, H, W])
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # добавим размер батча

    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)  # [C, H*W]
    G = torch.mm(features, features.t())  # [C, C]
    return G / (c * h * w)  # нормализация


# === Загружаем модель VGG и изображение ===

weights = VGG19_Weights.DEFAULT
vgg = vgg19(weights=weights).features.eval()

image = load_image("C:\\project\\pytorch-nst-feedforward\\data\\style-images\\mosaic.jpg")


# Только 1 слой, например relu1_1 (index 2)
layer_index = 2
with torch.no_grad():
    feature_map = vgg[:layer_index + 1](image)  # до слоя включительно
    gram = gram_matrix(feature_map[0])

# === Визуализация грамм-матрицы ===
plt.figure(figsize=(8, 6))
sns.heatmap(gram.cpu().numpy(), cmap='viridis')
plt.title("Gram Matrix")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.show()
