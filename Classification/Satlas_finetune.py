import os
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import satlaspretrain_models
from imagenet_c import corrupt

# === Configurations ===
device = torch.device('cuda')
num_epochs = 100
val_step = 50
batch_size = 16
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# === Load Saved Data ===
train_data = torch.load('data_type/AID_train_data.pth')
test_data = torch.load('data_type/AID_test_data.pth')

# === Load Pretrained Satlas Model ===
print("Loading model...")
weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(
    "Aerial_SwinB_SI", fpn=True,
    head=satlaspretrain_models.Head.CLASSIFY,
    num_categories=30, device='gpu'
)
model = model.to(device)

# === Freeze all but classification head ===
head_list = [
    "head.extra.weight", "head.extra.bias",
    "head.layers.0.0.weight", "head.layers.0.0.bias"
]
for name, param in model.named_parameters():
    if name not in head_list:
        param.requires_grad = False
        print(f"{name} - frozen")

# === Optimizer and Scheduler ===
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    print(f"Starting Epoch {epoch}...")
    model.train()

    for data, target in train_data:
        data, target = data.to(device), target.to(device)

        output, loss = model(data, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # === Validation ===
    if epoch % val_step == 0:
        print("Running validation...")
        model.eval()
        val_accuracy = 0
        total_sample = 0

        with torch.no_grad():
            for val_data, val_target in test_data:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output, val_loss = model(val_data, val_target)

                val_accuracy += (val_output.argmax(dim=1) == val_target).sum().item()
                total_sample += val_target.size(0)

        print(f"Validation Accuracy: {val_accuracy / total_sample:.4f}")

# === Robustness Evaluation: Defocus Blur ===
print("\nRobustness evaluation with defocus blur...")
model.eval()
val_accuracy = 0
total_sample = 0

for val_data, val_target in test_data:
    val_data, val_target = val_data.to(device), val_target.to(device)

    # Apply corruption to each image in the batch
    for i in range(len(val_data)):
        img = val_data[i].cpu().numpy()
        img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
        img = corrupt(img, corruption_name="defocus_blur")
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().cuda() / 255.0
        val_data[i] = img

    val_output, val_loss = model(val_data, val_target)
    val_accuracy += (val_output.argmax(dim=1) == val_target).sum().item()
    total_sample += val_target.size(0)

print(f"Final Robustness Accuracy (defocus blur): {val_accuracy / total_sample:.4f}")

# === Save Model ===
torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_AID_aerisi_model_weights.pth'))
