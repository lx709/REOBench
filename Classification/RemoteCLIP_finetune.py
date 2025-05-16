import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
import open_clip

# === Device Setup ===
device = torch.device('cuda')

# === Normalization ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Transform Function ===
def square_resize_randomcrop(phase, image_resolution=224):
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_resolution),
            transforms.ToTensor(),
            normalize,
        ])

# === Classification Head ===
class CLS_Head(nn.Module):
    def __init__(self, feat_size, cls_num):
        """
        Linear classification head.
        Args:
            feat_size (int): Feature dimension of the backbone output
            cls_num (int): Number of classes
        """
        super(CLS_Head, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, cls_num)
        )

    def forward(self, x):
        return self.cls_head(x)

# === Data Preparation ===
train_path = '/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/train'
test_path = '/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test'

train_dataset = datasets.ImageFolder(root=train_path, transform=square_resize_randomcrop("train"))
test_dataset = datasets.ImageFolder(root=test_path, transform=square_resize_randomcrop("test"))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Load RemoteCLIP Backbone ===
print("Loading RemoteCLIP ViT-B/32...")
model_name = 'ViT-B-32'
ckpt_path = '/opt/data/private/zsy/RS_workspace/RemoteCLIP-ViT-B-32.pt'

model, _, _ = open_clip.create_model_and_transforms(model_name)
checkpoint = torch.load(ckpt_path)
msg = model.load_state_dict(checkpoint, strict=False)
print("Checkpoint loaded:", msg)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
model.eval()
model = model.to(device)

# === Initialize Classification Head ===
model1 = CLS_Head(feat_size=512, cls_num=30).to(device)

# === Optimizer and Scheduler ===
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
num_epochs = 50
val_step = 10
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    model1.train()

    for images, targets in tqdm.tqdm(train_loader, unit_scale=16):
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            features = model.encode_image(images)

        outputs = model1(features)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # === Validation ===
    if epoch % val_step == 0:
        model1.eval()
        val_accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for val_images, val_targets in test_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                val_features = model.encode_image(val_images)
                val_outputs = model1(val_features)

                val_accuracy += (val_outputs.argmax(dim=1) == val_targets).sum().item()
                total_samples += val_targets.size(0)

        print(f"Validation Accuracy: {val_accuracy / total_samples:.4f}")

# === Save Final Head Weights ===
final_path = os.path.join(save_path, f"{num_epochs}_AID_head_RSCLIP_ViT-B-32_model_weights.pth")
torch.save(model1.state_dict(), final_path)
print(f"Training complete. Weights saved to {final_path}")
