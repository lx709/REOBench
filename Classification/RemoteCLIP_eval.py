import os
import torch
import tqdm
import open_clip
import numpy as np
from torch import tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === Device setup ===
device = torch.device('cuda')

# === Normalization ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Resize & Crop Transform ===
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
        super(CLS_Head, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(feat_size, feat_size)
        )

    def forward(self, x):
        return self.cls_head(x)

# === Load Models ===
model1 = CLS_Head(512, 30)
model1.load_state_dict(torch.load("weights/50AID_head_RS5M_ViT-B-32_model_weights.pth"))
model1.to(device)

clip_model_name = 'ViT-B-32'
clip_ckpt_path = '/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt'
model, _, _ = open_clip.create_model_and_transforms(clip_model_name)
checkpoint = torch.load(clip_ckpt_path)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# === Load Clean Test Data ===
test_path = '/opt/data/private/zsy/RS_workspace/AID_dataset_without_corruption/AID/test'
clean_test_data = datasets.ImageFolder(root=test_path, transform=square_resize_randomcrop("test", 224))
clean_test_loader = DataLoader(dataset=clean_test_data, batch_size=16, shuffle=False)

# === Corruptions ===
corruptions = [
    "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
    "brightness_contrast", "clouds", "haze_update_new", "data_gaps",
    "compression_artifacts", "rotate_new", "scale_image", "translate_image_new"
]

# === Evaluation ===
print("Starting evaluation...")

with open('result_ClsHead_RS5M_ViT-B-32.txt', 'w') as f:
    All_Class_Acc = [[0] * 5 for _ in range(len(corruptions))]

    # Evaluate on clean test set
    model1.eval()
    correct = 0
    total = 0
    for images, labels in clean_test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = model.encode_image(images)
            preds = model1(features).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    clean_acc = correct / total
    print(f"Clean Accuracy (OA): {clean_acc:.4f}")
    f.write(f"Clean Accuracy (OA): {clean_acc:.4f}\n\n")

    # Corruption Evaluation
    for cls in range(30):
        print(f"Testing Class {cls}")
        for c_idx, corruption in enumerate(corruptions):
            final_acc = 0
            for severity in range(1, 6):
                corruption_path = f"/opt/data/private/zsy/RS_workspace/corrupted_dataset/AID_corrupted/{cls}/{corruption}/severity_{severity}"
                corrupted_dataset = datasets.ImageFolder(root=corruption_path, transform=square_resize_randomcrop("test", 224))
                corrupted_loader = DataLoader(corrupted_dataset, batch_size=16, shuffle=False)
                prop = len(corrupted_dataset) / 2000

                correct = 0
                total = 0
                for imgs, targets in tqdm.tqdm(corrupted_loader, unit_scale=16):
                    imgs = imgs.to(device)
                    targets = torch.full_like(targets, fill_value=cls, device=device)
                    with torch.no_grad():
                        feats = model.encode_image(imgs)
                        outputs = model1(feats).argmax(dim=1)
                    correct += (outputs == targets).sum().item()
                    total += targets.size(0)

                acc = correct / total
                All_Class_Acc[c_idx][severity - 1] += acc * prop
                final_acc += acc / 5

                f.write(f"{'-'*40}\nClass {cls} | {corruption} | Severity {severity}: {acc:.4f}\n")

            f.write(f"{'='*40}\nFinal Avg Accuracy for {corruption}: {final_acc:.4f}\n{'='*40}\n\n")

    # === Summary ===
    f.write(f"{'='*50}\nOverall Accuracy Summary\n{'='*50}\n")
    for i, corruption in enumerate(corruptions):
        accs = " ".join(f"<< {All_Class_Acc[i][s]:.4f} >>" for s in range(5))
        f.write(f"{corruption} from severity 1 to 5: {accs}\n")
