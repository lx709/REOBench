import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.transforms import ToTensor, Compose, Resize
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# Load train/test datasets
train_data = torch.load('data_type/AID_train_data.pth')
test_data = torch.load('data_type/AID_test_data.pth')

# Training settings
device = torch.device('cuda')
num_epochs = 100
val_step = 5
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# Load pretrained backbone
print("Loading pretrained ViT weights...")
model = models_vit.__dict__["vit_large_patch16"](
    patch_size=16, img_size=256, in_chans=3,
    num_classes=30, drop_path_rate=0.2, global_pool=False
)

checkpoint = torch.load('/opt/data/private/zsy/RS_workspace/pretrain-vit-large-e199.pth')
checkpoint_model = checkpoint['model']

# Remove incompatible keys
state_dict = model.state_dict()
for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

interpolate_pos_embed(model, checkpoint_model)
trunc_normal_(model.head.weight, std=2e-5)
msg = model.load_state_dict(checkpoint_model, strict=False)
print("Missing keys after loading:", set(msg.missing_keys))

model.to(device)

# Freeze all parameters except classification head
head_list = ["head.weight", "head.bias"]
for name, param in model.named_parameters():
    param.requires_grad = name in head_list
    print(name, "on" if param.requires_grad else "off")

# Remove pretrained head
del model.head

# Define a custom classification head
class CLS_Head(nn.Module):
    def __init__(self, feat_size, cls_num):
        super(CLS_Head, self).__init__()
        self.cls_head = nn.Linear(feat_size, cls_num)

    def forward(self, x):
        return self.cls_head(x)

model1 = CLS_Head(1024, 30).to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
criterion = nn.CrossEntropyLoss()

# Training loop
model.eval()
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}")
    model1.train()

    for data, target in train_data:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            features = model(data)
        output = model1(features)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if epoch % val_step == 0:
        model1.eval()
        correct = total = 0
        with torch.no_grad():
            for val_data, val_target in test_data:
                val_data, val_target = val_data.to(device), val_target.to(device)
                features = model(val_data)
                output = model1(features)
                correct += (output.argmax(dim=1) == val_target).sum().item()
                total += val_target.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

# Save final model head weights
torch.save(model1.state_dict(), os.path.join(save_path, f"{epoch}_AID_head_weights.pth"))
