import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.transforms import ToTensor, Compose, Resize
import scale_models_vit as models_vit
from timm.models.layers import trunc_normal_
from util.pos_embed_scale import interpolate_pos_embed

# === Configuration ===
device = torch.device('cuda')
num_epochs = 100
val_step = 5
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)
lr = 0.001

# === Load Data ===
train_data = torch.load('data_type/AID_train_data.pth')
test_data = torch.load('data_type/AID_test_data.pth')

# === Load Pretrained ViT Model ===
print("Loading model weights...")
model = models_vit.__dict__["vit_large_patch16"](
    img_size=256,
    num_classes=30,
    drop_path_rate=0.1,
    global_pool=False,
)

checkpoint = torch.load('/opt/data/private/zsy/RVSA/scalemae-vitlarge-800.pth')
checkpoint_model = checkpoint['model']

# Remove incompatible keys
state_dict = model.state_dict()
for k in ["head.weight", "head.bias"]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
    print(f"Removing key pos_embed from pretrained checkpoint")
    del checkpoint_model["pos_embed"]

# Load and interpolate positional embeddings
interpolate_pos_embed(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)
print("Missing keys:", msg.missing_keys)

assert set(msg.missing_keys) == {"head.weight", "head.bias", "pos_embed"}
trunc_normal_(model.head.weight, std=0.01)
model.to(device)

# === Freeze all but classification head ===
head_list = ["head.weight", "head.bias"]
for name, param in model.named_parameters():
    param.requires_grad = name in head_list

# === Optimizer and Scheduler ===
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    print(f"Starting Epoch {epoch}...")
    model.train()

    for data, target in train_data:
        data, target = data.to(device), target.to(device)
        input_res = torch.ones(len(data)).float().to(data.device)

        output = model(data, input_res=input_res)
        loss = criterion(output, target)

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
                input_res = torch.ones(len(val_data)).float().to(val_data.device)

                val_output = model(val_data, input_res=input_res)
                val_accuracy += (val_output.argmax(dim=1) == val_target).sum().item()
                total_sample += val_target.size(0)

        print(f"Validation Accuracy: {val_accuracy / total_sample:.4f}")

# === Save Final Model ===
torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_AID_scalemae_model_weights.pth'))
