import os
import sys
import torch
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import satlaspretrain_models

# === Configuration ===
device = torch.device('cuda')
batch_size = 4
num_classes = 30
model_weights_path = '/opt/data/private/zsy/RS_workspace/weights/100AID_aerisi_model_weights.pth'
test_data_root = '/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test'

# Corruption types used in evaluation
name_of_noise_list = [
    "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
    "brightness_contrast", "clouds", "haze_update_new", "data_gaps",
    "compression_artifacts", "rotate_new", "scale_image", "translate_image_new"
]

# === Load Clean Test Data ===
test_imagedata = datasets.ImageFolder(root=test_data_root, transform=ToTensor())
test_data = DataLoader(dataset=test_imagedata, batch_size=batch_size, shuffle=False)

# === Load Pretrained Satlas Model ===
weights_manager = satlaspretrain_models.Weights()
print("Loading model...")
model = weights_manager.get_pretrained_model(
    "Aerial_SwinB_SI", fpn=True, head=satlaspretrain_models.Head.CLASSIFY,
    num_categories=num_classes, device='gpu'
)
model.load_state_dict(torch.load(model_weights_path))
model = model.to(device)
model.eval()
print(f"Loaded model. Total test batches: {len(test_data)}")

# === Evaluate on Clean Test Data ===
print("Evaluating on clean data...")
val_accuracy = 0
total_sample = 0

for val_data, val_target in test_data:
    val_data, val_target = val_data.to(device), val_target.to(device)
    val_output, _ = model(val_data, val_target)
    val_accuracy += (val_output.argmax(dim=1) == val_target).float().sum().item()
    total_sample += val_target.size(0)

clean_accuracy = val_accuracy / total_sample
print(f"Clean accuracy: {clean_accuracy:.4f}")
