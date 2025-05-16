import os
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from vitae_nc_win_rvsa import ViTAE_NC_Win_RVSA

# === Configuration ===
device = torch.device('cuda')
batch_size = 4
num_classes = 30
img_size = 256
model_weights_path = 'weights/100AID_RVSA_model_weights.pth'
test_data_path = '/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test'

# === Load Test Data ===
test_imagedata = datasets.ImageFolder(root=test_data_path, transform=ToTensor())
test_data = DataLoader(dataset=test_imagedata, batch_size=batch_size, shuffle=False)

# === Load Pretrained Model ===
print("Loading model...")
model = ViTAE_NC_Win_RVSA(
    img_size=img_size,
    num_classes=num_classes,
    drop_path_rate=0.2,
    use_abs_pos_emb=True,
)
checkpoint = torch.load(model_weights_path)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")

# === Evaluate on Clean Test Data ===
print("Evaluating on clean test data...")
val_accuracy = 0
total_sample = 0

for val_data, val_target in tqdm.tqdm(test_data, unit_scale=batch_size):
    val_data, val_target = val_data.to(device), val_target.to(device)
    val_output = model(val_data)
    val_accuracy += (val_output.argmax(dim=1) == val_target).sum().item()
    total_sample += val_target.size(0)

final_accuracy = val_accuracy / total_sample
print(f"Final Accuracy on Clean Test Set: {final_accuracy:.4f}")
