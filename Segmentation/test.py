import open_clip
import torch
model_name = 'ViT-B-32'
ckpt_path = '/opt/data/private/zsy/RS_workspace/RS5M_ViT-B-32.pt'
model, _, _ = open_clip.create_model_and_transforms(model_name)
print(type(model))
checkpoint = torch.load(ckpt_path)
msg = model.load_state_dict(checkpoint, strict=False)
print(type(model))