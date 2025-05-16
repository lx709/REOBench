import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import models_vit

# === Configuration ===
device = torch.device('cuda')
batch_size = 4
num_classes = 30
img_size = 256
model_weights_path = 'weights/100AID_satmaepp_model_weights.pth'
output_result_path = 'resultSatMAE++.txt'

name_of_noise_list = [
    "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
    "brightness_contrast", "clouds", "haze_update_new", "data_gaps", "compression_artifacts",
    "rotate_new", "scale_image", "translate_image_new"
]

# === Load Clean Test Data ===
test_imagedata = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test',
    transform=ToTensor()
)
test_data = DataLoader(dataset=test_imagedata, batch_size=batch_size, shuffle=False)

# === Load Model ===
print("Loading model...")
model = models_vit.__dict__["vit_large_patch16"](
    patch_size=16, img_size=img_size, in_chans=3,
    num_classes=num_classes, drop_path_rate=0.2, global_pool=False
)
checkpoint_model = torch.load(model_weights_path)
model.load_state_dict(checkpoint_model, strict=False)
model.to(device)
model.eval()
print("Model loaded successfully.")

# === Evaluate on Clean Test Data ===
print("Evaluating on clean test data...")
val_accuracy = 0
total_sample = 0
for val_data, val_target in test_data:
    val_data, val_target = val_data.to(device), val_target.to(device)
    val_output = model(val_data)
    val_accuracy += (val_output.argmax(dim=1) == val_target).float().sum().item()
    total_sample += val_target.shape[0]
print(f"Clean accuracy: {val_accuracy / total_sample:.4f}")

# === Corruption Robustness Evaluation ===
with open(output_result_path, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f  # Redirect stdout to file

    print(f"Clean accuracy: {val_accuracy / total_sample:.4f}")

    All_Class_Acc = [[0 for _ in range(5)] for _ in range(len(name_of_noise_list))]
    transform = Compose([ToTensor()])

    for cls in range(num_classes):
        for ind, noise_name in enumerate(name_of_noise_list):
            final_accuracy = 0

            for severity in range(1, 6):
                corrupt_path = (
                    f"/opt/data/private/zsy/RS_workspace/corrupted_dataset/AID_corrupted/"
                    f"{cls}/{noise_name}/severity_{severity}"
                )
                imagedata = datasets.ImageFolder(root=corrupt_path, transform=transform)
                prop_imagedata = len(imagedata) / 2000
                eval_loader = DataLoader(imagedata, batch_size=2, shuffle=False)

                eval_accuracy = 0
                total_sample = 0

                print("-" * 50)
                print(f"class: {cls}, noise: {noise_name}, severity: {severity}")

                for eval_data, eval_target in eval_loader:
                    eval_data = eval_data.to(device)
                    eval_target = eval_target.to(device)

                    # Set all targets to the current class (used for robustness evaluation)
                    for i in range(len(eval_target)):
                        eval_target[i] = tensor(cls, device=device)

                    eval_output = model(eval_data)
                    eval_accuracy += (eval_output.argmax(dim=1) == eval_target).float().sum().item()
                    total_sample += eval_target.shape[0]

                acc = eval_accuracy / total_sample
                All_Class_Acc[ind][severity - 1] += acc * prop_imagedata
                final_accuracy += acc / 5

                print(f"Accuracy: {acc:.4f}")
                print("-" * 50)

            print("=" * 50)
            print(f"Final average accuracy for noise '{noise_name}' on class {cls}: {final_accuracy:.4f}")
            print("=" * 50)

    # === Final Summary ===
    print("=" * 50)
    print("Overall accuracy across all classes:")
    print("=" * 50)
    for idx, noise_name in enumerate(name_of_noise_list):
        print(f"{noise_name} from severity 1 to 5:", end=' ')
        for severity in range(5):
            print(f"<< {All_Class_Acc[idx][severity]:.4f} >>", end=' ')
        print()

    sys.stdout = original_stdout  # Restore stdout
