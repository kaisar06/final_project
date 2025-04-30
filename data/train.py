import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from models.definitions.transformer_net import TransformerNet
from utils.utils import prepare_img, save_checkpoint
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt  # <-- Added for plotting

# --- CONFIG ---
style_image_path = "data/style-images/beauty.jpg"
content_dataset_path = "data/dataset"
save_model_path = "models/binaries/beauty_style.pth"
epochs = 10
batch_size = 1
learning_rate = 2e-4
content_weight = 1e0
style_weight = 1e6
image_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Custom Image Dataset ---
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)
                          if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# --- LOAD STYLE IMAGE ---
style_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
style_image = prepare_img(style_image_path, image_size, device)

# --- LOAD CONTENT DATASET ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
train_dataset = CustomImageDataset(content_dataset_path, transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- VGG FOR LOSS ---
vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# --- Define Feature Layers ---
content_layer = '21'  # relu4_2
style_layers = ['0', '5', '10', '19', '28']

def extract_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# --- STYLE FEATURES ---
with torch.no_grad():
    style_features = extract_features(style_image.repeat(batch_size, 1, 1, 1), vgg, style_layers)
    style_grams = {
        l: (f @ f.transpose(-2, -1)).mean(0)
        for l, f in style_features.items()
    }

# --- START TRAINING ---
model = TransformerNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

checkpoint_dir = "models/binaries"
os.makedirs(checkpoint_dir, exist_ok=True)

epoch_losses = []  # <-- Added to track loss

print("Training started...")
for epoch in range(epochs):
    print(f"Total batches: {len(train_loader)}")
    epoch_loss_sum = 0
    batch_count = 0

    for batch_id, x in enumerate(train_loader):
        try:
            print(f"  Batch {batch_id+1} input shape: {x.shape}")
            x = x.to(device)
            stylized = model(x)

            # --- Content loss ---
            features_orig = extract_features(x, vgg, [content_layer])[content_layer]
            features_styled = extract_features(stylized, vgg, [content_layer])[content_layer]
            features_styled_resized = F.interpolate(features_styled, size=features_orig.shape[2:], mode='bilinear', align_corners=False)
            content_loss = content_weight * mse_loss(features_styled_resized, features_orig)

            # --- Style loss ---
            stylized_feats = extract_features(stylized, vgg, style_layers)
            style_loss = 0.0
            for l in style_layers:
                gen_feat = stylized_feats[l]
                style_feat = style_grams[l]

                gen_feat_resized = F.interpolate(gen_feat, size=style_feat.shape[-2:], mode='bilinear', align_corners=False)
                gram = gen_feat_resized @ gen_feat_resized.transpose(-2, -1)
                gram_mean = gram.mean(0)
                style_loss += mse_loss(gram_mean, style_feat)

            style_loss *= style_weight
            total_loss = content_loss + style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_id}, Loss: {total_loss.item():.2f}")

            epoch_loss_sum += total_loss.item()
            batch_count += 1

        except Exception as e:
            print(f"⚠️ Error in batch {batch_id}: {e}")
            continue

    # --- Average epoch loss ---
    avg_epoch_loss = epoch_loss_sum / batch_count if batch_count > 0 else 0
    epoch_losses.append(avg_epoch_loss)
    print(f"✅ Epoch {epoch+1} Average Loss: {avg_epoch_loss:.2f}")

    # --- Save Checkpoint ---
    checkpoint_filename = f"{checkpoint_dir}/model_epoch_{epoch}.pth"
    print(f"Saving checkpoint for epoch {epoch} to {checkpoint_filename}")
    try:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss if 'total_loss' in locals() else None,
        }, filename=checkpoint_filename)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

# --- SAVE FINAL MODEL ---
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
torch.save({'state_dict': model.state_dict()}, save_model_path)
print(f"Model saved to {save_model_path}")

# --- PLOT LOSS ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='blue')
plt.title("Average Total Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_plot.png")
plt.show()
