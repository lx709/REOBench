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
from vitae_nc_win_rvsa import ViTAE_NC_Win_RVSA
import tqdm

# === Configurations ===
device = torch.device('cuda')
num_epochs = 100
val_step = 5
batch_size = 16
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# === Load Data ===
train_data = torch.load('data_type/AID_train_data.pth')
test_data = torch.load('data_type/AID_test_data.pth')

# === Initialize and Load Model ===
print("Loading model...")
model = ViTAE_NC_Win_RVSA(
    img_size=256, num_classes=30, drop_path_rate=0.2, use_abs_pos_emb=True,
)
checkpoint = torch.load('vitae-b-checkpoint-1599-transform-no-average.pth')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()

# Remove mismatched keys
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# Positional embedding interpolation and model load
interpolate_pos_embed(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)
print("Missing keys:", set(msg.missing_keys))

# Initialize head weights
trunc_normal_(model.head.weight, std=2e-5)
model = model.to(device)

# === Freeze all layers except classification head ===
head_list = ["head.weight", "head.bias"]
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

    for data, target in tqdm.tqdm(train_data, unit_scale=batch_size):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # === Validation ===
    if epoch % val_step == 0:
        print("Evaluating on validation set...")
        model.eval()
        val_accuracy = 0
        total_sample = 0

        with torch.no_grad():
            for val_data, val_target in tqdm.tqdm(test_data, unit_scale=batch_size):
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_accuracy += (val_output.argmax(dim=1) == val_target).sum().item()
                total_sample += val_target.shape[0]

        print(f"Validation Accuracy: {val_accuracy / total_sample:.4f}")

# === Save Final Model Weights ===
torch.save(model.state_dict(), os.path.join(save_path, f"{epoch}_AID_RVSA_model_weights.pth"))
