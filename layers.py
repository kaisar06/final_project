import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the image
def load_image(path, size=512):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Convert tensor to displayable image
def im_convert(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    return image

# Get VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.eval()

# Specify the layers you want to extract
selected_layers = {
    '0': 'conv1_1',
    '2': 'relu1_1',
    '5': 'conv2_1',
    '7': 'relu2_1',
    '10': 'conv3_1',
    '12': 'relu3_1',
    '19': 'conv4_1',
    '21': 'relu4_1',
    '28': 'conv5_1',
    '30': 'relu5_1',
}

# Load your image
image_path = "C:\project\pytorch-nst-feedforward\data\content-images\lion.jpg"  # 🔁 Replace with your image path
image = load_image(image_path)

# Send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
vgg = vgg.to(device)

# Pass through layers and collect outputs
features = {}
x = image
for name, layer in vgg._modules.items():
    x = layer(x)
    if name in selected_layers:
        features[selected_layers[name]] = x

# Visualize feature maps (first 5 channels from each layer)
for layer_name, feature_map in features.items():
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Layer: {layer_name}', fontsize=16)
    for i in range(5):
        axs[i].imshow(feature_map[0][i].cpu().detach().numpy(), cmap='viridis')
        axs[i].axis('off')
    plt.show()
