import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from timm.models.layers import trunc_normal_

from util.pos_embed import interpolate_pos_embed
import models_vit

# === Configuration ===
device = torch.device('cuda')
num_epochs = 100
val_step = 5
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# === Data Transforms ===
train_transformer = Compose([Resize(256), ToTensor()])
test_transform = Compose([Resize(256), ToTensor()])

# === Load Datasets ===
train_imagedata = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/train',
    transform=train_transformer
)
test_imagedata = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test',
    transform=test_transform
)

train_data = DataLoader(dataset=train_imagedata, batch_size=16, shuffle=True)
test_data = DataLoader(dataset=test_imagedata, batch_size=16, shuffle=False)

# === Load Pretrained ViT Model ===
print("Loading ViT model...")
model = models_vit.__dict__["vit_large_patch16"](
    patch_size=16, img_size=256, in_chans=3,
    num_classes=30, drop_path_rate=0.2, global_pool=False,
)

checkpoint = torch.load('/opt/data/private/zsy/RS_workspace/checkpoint_ViT-L_pretrain_fmow_rgb.pth')
checkpoint_model = checkpoint['model']

# Remove incompatible keys
state_dict = model.state_dict()
for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# Load and interpolate position embeddings
interpolate_pos_embed(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)
trunc_normal_(model.head.weight, std=2e-5)
print("Missing keys after loading:", set(msg.missing_keys))

model.to(device)

# === Freeze All Except Head ===
head_list = ["head.weight", "head.bias"]
for name, param in model.named_parameters():
    param.requires_grad = name in head_list
    print(name, "on" if param.requires_grad else "off")

# === Optimizer and Scheduler ===
params_to_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params_to_train, lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}")
    model.train()

    for data, target in train_data:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # === Validation ===
    if epoch % val_step == 0:
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for val_data, val_target in test_data:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                correct += (val_output.argmax(dim=1) == val_target).sum().item()
                total += val_target.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

# === Save Model ===
torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_AID_vit_model.pth'))
