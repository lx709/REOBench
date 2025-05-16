import os
import torch
import tqdm
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import open_clip

# === Device Setup ===
device = torch.device('cuda')

# === Normalization ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Transform Function ===
def square_resize_randomcrop(phase, image_resolution):
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size=image_resolution,
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_resolution),
            transforms.ToTensor(),
            normalize,
        ])

# === Classification Head ===
class CLS_Head(nn.Module):
    def __init__(self, feat_size, cls_num):
        super(CLS_Head, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, cls_num),
        )

    def forward(self, x):
        return self.cls_head(x)

# === Data Loading ===
train_transform = square_resize_randomcrop("train", 224)
test_transform = square_resize_randomcrop("test", 224)

train_dataset = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/train',
    transform=train_transform)
test_dataset = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test',
    transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# === Model Initialization ===
print("Loading OpenCLIP ViT-B-32...")
model_name = 'ViT-B-32'
ckpt_path = '/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt'

model, _, _ = open_clip.create_model_and_transforms(model_name)
checkpoint = torch.load(ckpt_path)
msg = model.load_state_dict(checkpoint, strict=False)
print("Loaded checkpoint:", msg)

# Freeze CLIP backbone
for param in model.parameters():
    param.requires_grad = False
model.eval()
model = model.to(device)

# Initialize classification head
model1 = CLS_Head(feat_size=512, cls_num=30).to(device)

# === Optimizer and Scheduler ===
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# === Training ===
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

        acc = val_accuracy / total_samples
        print(f"Validation Accuracy @ Epoch {epoch}: {acc:.4f}")

# === Save Head Weights ===
torch.save(model1.state_dict(), os.path.join(save_path, f"{num_epochs}_AID_head_RS5M_ViT-B-32_model_weights.pth"))
print("Training complete and weights saved.")
