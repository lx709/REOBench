import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose
from timm.models.layers import trunc_normal_
from util.pos_embed_scale import interpolate_pos_embed
import scale_models_vit as models_vit

# === Configuration ===
device = torch.device('cuda')
batch_size = 4
num_classes = 30
img_size = 256
result_file = 'resultScale_MAE.txt'
model_weights_path = '/opt/data/private/zsy/RS_workspace/weights/100AID_scalemae_model_weights.pth'

# === Load Test Data (unused here, but kept for completeness) ===
test_imagedata = datasets.ImageFolder(
    root='/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test',
    transform=ToTensor()
)
test_data = DataLoader(dataset=test_imagedata, batch_size=batch_size, shuffle=False)

# === Load Model ===
print("Loading model weights...")
model = models_vit.__dict__["vit_large_patch16"](
    img_size=img_size,
    num_classes=num_classes,
    drop_path_rate=0.1,
    global_pool=False,
)
checkpoint_model = torch.load(model_weights_path)
msg = model.load_state_dict(checkpoint_model, strict=False)
print("Loaded model:", msg)

model = model.to(device)
model.eval()
print("Model ready for evaluation.")

# === Corruption Types ===
name_of_noise_list = [
    "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
    "brightness_contrast", "clouds", "haze_update_new", "data_gaps", "compression_artifacts",
    "rotate_new", "scale_image", "translate_image_new"
]

# === Accuracy Table: [num_noises x 5 severities] ===
All_Class_Acc = [[0 for _ in range(5)] for _ in range(len(name_of_noise_list))]

# === Evaluate Robustness Under Corruption ===
with open(result_file, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f  # redirect print to file

    transform = Compose([ToTensor()])

    for cls in range(num_classes):
        for noise_idx, name_of_noise in enumerate(name_of_noise_list):
            final_accuracy = 0

            for severity in range(1, 6):
                corrupted_data_root = f"/opt/data/private/zsy/RS_workspace/corrupted_dataset/AID_corrupted/{cls}/{name_of_noise}/severity_{severity}"
                imagedata = datasets.ImageFolder(root=corrupted_data_root, transform=transform)
                prop_imagedata = len(imagedata) / 2000

                eval_loader = DataLoader(imagedata, batch_size=batch_size, shuffle=False)
                eval_accuracy = 0
                total_sample = 0

                print("-" * 50)
                print(f"Class: {cls}, Noise: {name_of_noise}, Severity: {severity}")

                for eval_data, eval_target in eval_loader:
                    eval_data = eval_data.to(device)
                    eval_target = eval_target.to(device)

                    # Force all targets to the current class
                    for i in range(len(eval_target)):
                        eval_target[i] = tensor(cls, device=device)

                    input_res = torch.ones(len(eval_data)).float().to(eval_data.device)
                    eval_output = model(eval_data, input_res=input_res)

                    eval_accuracy += (eval_output.argmax(dim=1) == eval_target).float().sum().item()
                    total_sample += eval_target.shape[0]

                acc = eval_accuracy / total_sample
                All_Class_Acc[noise_idx][severity - 1] += acc * prop_imagedata

                print(f"Accuracy: {acc:.4f}")
                final_accuracy += acc / 5
                print("-" * 50)

            print("=" * 50)
            print(f"Final average accuracy for noise '{name_of_noise}': {final_accuracy:.4f}")
            print("=" * 50)

    # === Print Final Summary ===
    for noise_idx, noise_name in enumerate(name_of_noise_list):
        print(f"{noise_name} from severity 1 to 5:", end=' ')
        for severity in range(5):
            print(f"<< {All_Class_Acc[noise_idx][severity]:.4f} >>", end=' ')
        print()

    sys.stdout = original_stdout  # restore stdout
