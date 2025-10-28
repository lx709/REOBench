import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from io import BytesIO
import noise
from scipy.ndimage import gaussian_filter
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms.functional import to_pil_image
import cv2
from skimage.util import random_noise
from datetime import datetime
from tqdm import tqdm
import random
import argparse
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def add_gaussian_noise_batch(image_batch, severity=1):
    # Define sigma levels based on severity
    sigma_levels = [0.04, 0.05, 0.06, 0.07, 0.08]
    sigma = sigma_levels[severity - 1]

    # Convert image batch to tensor and normalize to [0, 1]
    image_batch = torch.stack([torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in image_batch])
    image_batch = image_batch.to(device)

    # Generate Gaussian noise on GPU
    noise = torch.normal(0, sigma, size=image_batch.shape, device=device)

    # Add noise to images and clip to [0, 1] range
    noisy_image_batch = image_batch + noise
    noisy_image_batch = torch.clamp(noisy_image_batch, 0.0, 1.0)

    # Convert back to [0, 255] and to PIL images
    noisy_images = (noisy_image_batch * 255).byte()
    noisy_images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in noisy_images]

    return noisy_images


def add_salt_and_pepper_noise_batch(image_batch, severity=1):
    amount_levels = [0.005, 0.01, 0.02, 0.03, 0.05]
    amount = amount_levels[severity - 1]

    noisy_images = []
    
    for image in image_batch:
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Add salt and pepper noise
        noisy_image = random_noise(image_np, mode='s&p', amount=amount)
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_images.append(Image.fromarray((noisy_image * 255).astype('uint8')))
    
    return noisy_images

def apply_gaussian_blur_batch(image_batch, severity=1):
    kernel_sizes = [3, 5, 7, 9, 11]
    kernel_size = kernel_sizes[severity - 1]

    for i in range(len(image_batch)):
        image_np = np.array(image_batch[i], dtype=np.float32) / 255.0
        blurred_image_np = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
        blurred_image_np = np.clip(blurred_image_np, 0, 1)
        image_batch[i] = Image.fromarray((blurred_image_np * 255).astype('uint8'))

    return image_batch


def apply_motion_blur_batch(image_batch, severity=1):
    kernel_sizes = [2, 4, 6, 8, 10]
    kernel_size = kernel_sizes[severity - 1]

    # Create a horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    for i in range(len(image_batch)):
        # Convert image to NumPy array and normalize to [0, 1]
        image_np = np.array(image_batch[i], dtype=np.float32) / 255.0
        
        # Apply motion blur
        motion_blur_image_np = cv2.filter2D(image_np, -1, kernel)
        
        # Clip the motion blur image to [0, 1] range
        motion_blur_image_np = np.clip(motion_blur_image_np, 0, 1)
        
        # Convert back to PIL image format and replace in image_batch
        image_batch[i] = Image.fromarray((motion_blur_image_np * 255).astype('uint8'))

    return image_batch


def adjust_brightness_contrast_batch(images, severity=1):
    brightness_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    contrast_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    brightness = brightness_levels[severity - 1]
    contrast = contrast_levels[severity - 1]

    # 转换成torch tensor并进行批处理
    images_tensor = torch.stack([torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in images])

    # 将所有图像转移到GPU（如果可用）
    images_tensor = images_tensor.to(device)

    # 应用对比度和亮度调整
    adjusted_images = images_tensor * contrast + brightness

    # 裁剪到[0, 1]范围
    adjusted_images = torch.clamp(adjusted_images, 0.0, 1.0)

    # 将tensor转换回PIL图像
    adjusted_images = (adjusted_images * 255).byte()
    adjusted_images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in adjusted_images]

    return adjusted_images


def generate_perlin_noise_batch(batch_size, height, width, scale=100, seed=None):
    if seed is None:
        seed = random.randint(0, 10000)

    linx = np.linspace(0, width / scale, width, endpoint=False)
    liny = np.linspace(0, height / scale, height, endpoint=False)
    x, y = np.meshgrid(linx, liny)
    base_noise = np.vectorize(lambda i, j: noise.pnoise2(i, j, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=width, repeaty=height, base=seed))(x, y)
    base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())

    # Expand to batch
    noise_batch = np.stack([base_noise for _ in range(batch_size)], axis=0)  # shape: (B, H, W)
    return torch.tensor(noise_batch, dtype=torch.float32)

def add_clouds_batch(image_batch, severity=1, device='cuda'):
    cloud_density_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
    cloud_density = cloud_density_levels[severity - 1]

    # Convert image batch to tensor and move to device
    image_batch = torch.stack([
        torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        for img in image_batch
    ]).to(device)

    B, C, H, W = image_batch.shape

    # Generate Perlin noise
    perlin_noise = generate_perlin_noise_batch(B, H, W, scale=200).to(device)

    # Cloud mask by threshold
    cloud_threshold = 1.0 - cloud_density
    cloud_mask = (perlin_noise > cloud_threshold).float()  # (B, H, W)

    # Gaussian blur using F.gaussian_blur (requires unsqueeze for channel)
    # cloud_mask = F.gaussian_blur(cloud_mask.unsqueeze(1), kernel_size=11, sigma=5).squeeze(1)
    blur = T.GaussianBlur(kernel_size=11, sigma=5)
    cloud_mask = cloud_mask.unsqueeze(1)  # shape: (B, 1, H, W)
    cloud_mask = blur(cloud_mask).squeeze(1)

    # Cloud color (gray)
    cloud_color = torch.full_like(image_batch, 0.9)

    # Blend
    cloudy_image = image_batch * (1 - cloud_mask.unsqueeze(1)) + cloud_color * cloud_mask.unsqueeze(1)
    cloudy_image = torch.clamp(cloudy_image, 0, 1)

    # Convert back to PIL
    cloudy_image = (cloudy_image * 255).byte()
    adjusted_images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in cloudy_image]

    return adjusted_images


def add_haze_batch(image_batch, severity=1):
    # Define intensity levels based on severity
    intensity_levels = [0.2, 0.3, 0.4, 0.5, 0.6]
    intensity = intensity_levels[severity - 1]

    # Convert image batch to tensor and normalize to [0, 1]
    image_batch = torch.stack([torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in image_batch])
    image_batch = image_batch.to(device)

    # Create a haze layer
    haze_layer = torch.ones_like(image_batch, device=device)  # Haze layer with value 1.0

    # Apply haze effect
    hazy_image_batch = image_batch * (1 - intensity) + haze_layer * intensity

    # Clip the resulting images to [0, 1] range
    hazy_image_batch = torch.clamp(hazy_image_batch, 0.0, 1.0)

    # Convert back to [0, 255] and to PIL images
    hazy_images = (hazy_image_batch * 255).byte()
    hazy_images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in hazy_images]

    return hazy_images


def create_data_gaps_batch(image_batch, severity=1):
    lst_stripes = [2, 3, 4, 5, 6]
    lst_width = [3, 4, 5, 6, 7]

    num_stripes = lst_stripes[severity - 1]
    stripe_width = lst_width[severity - 1]
    
    # Convert the image batch to tensor and move to device
    image_batch = torch.stack([torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in image_batch])
    image_batch = image_batch.to(device)

    height, width, channels = image_batch.shape[2], image_batch.shape[3], image_batch.shape[1]

    # Randomly choose an angle in degrees for each image in the batch
    angles_deg = torch.rand(image_batch.shape[0], device=device) * 180
    angles = np.radians(angles_deg.cpu())  # Convert to radians

    # Calculate the spacing between stripes along the perpendicular direction
    diag_length = int(np.sqrt(width ** 2 + height ** 2))
    stripe_spacing = diag_length / num_stripes

    # Create coordinate grids
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
         indexing="xy")

    # Prepare for batch processing
    processed_images = []

    for i in range(image_batch.shape[0]):
        # Calculate the coordinates along the direction perpendicular to the stripes for each image
        coords = x_grid * torch.cos(angles[i]) + y_grid * torch.sin(angles[i])

        # Create the mask for stripes
        mask = (coords % stripe_spacing < stripe_width).float()

        # Apply the mask to the image (set masked areas to 0)
        masked_image = image_batch[i] * (1 - mask)

        processed_images.append(masked_image.cpu())

    # Convert the tensor images back to PIL
    gap_images = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')) for img in processed_images]

    return gap_images


def add_compression_artifacts_batch(image_batch, severity=1):
    quality_levels = [30, 25, 20, 15, 10]
    quality = quality_levels[severity - 1]

    for i in range(len(image_batch)):
        # Convert NumPy array to PIL image
        pil_image = image_batch[i]

        # Apply compression
        output = BytesIO()
        pil_image.save(output, format='JPEG', quality=quality)
        output.seek(0)
        compressed_image = Image.open(output)

        
        image_batch[i] = compressed_image

    return image_batch


parser = argparse.ArgumentParser(description='Example command line arguments')
parser.add_argument('perturbation_type', type=int, help='Specify the perturbation type')
parser.add_argument('--batch_size',type=int,default=1,help='Specify the batch size')
parser.add_argument('--severity',type=int,default=1,help='Specify the severity')
parser.add_argument('--ranges',type=int,default=5,help='Specify the severity ranges')
parser.add_argument('--fused',action='store_true',help='Use for fused noise')
parser.add_argument('--visualize', action='store_true', help='Enable visualization')

args = parser.parse_args()


type_of_noise=[add_gaussian_noise_batch, add_salt_and_pepper_noise_batch, apply_gaussian_blur_batch, 
                apply_motion_blur_batch, adjust_brightness_contrast_batch, add_clouds_batch,
                add_haze_batch, create_data_gaps_batch, add_compression_artifacts_batch]

name_of_noise=["gaussian_noise", "salt_and_pepper_noise", "gaussian_blur",
                "motion_blur", "brightness_contrast", 
                "clouds", "haze", "gaps", "compression_artifacts"]

name_of_noise_fused = ['fused_double_brightness_contrast+clouds', 'fused_double_brightness_contrast+compression_artifacts',
                       'fused_double_clouds+compression_artifacts', 'fused_triple_brightness_contrast+clouds+compression_artifacts']


image_path ='test'
if args.visualize:
    save_prefix="DIOR_corrupted_test_safe_to_delet/"
else:
    save_prefix="DIOR_corrupted_fused/"
# args.batch_size=4

print(f"generating images using {device}, image_path:{image_path}, save:{save_prefix}, batchsize:{args.batch_size}", flush=True)


image_files = [f for f in os.listdir(image_path)]

if args.fused:
    noise_function = type_of_noise[args.perturbation_type-1]
    name_function = name_of_noise[args.perturbation_type-1]
else:
    noise_function_1 = adjust_brightness_contrast_batch
    noise_function_2 = add_clouds_batch
    noise_function_3 = add_compression_artifacts_batch
    name_function = name_of_noise_fused[args.perturbation_type-1]
    
print("generating "+name_function+" data", flush=True)

for j in tqdm(range(args.severity, args.severity+args.ranges)):
    time2=datetime.now()
    print(f"{time2}: generating {name_function} with severity = {j}", flush=True)

    output_path = os.path.join(save_prefix, name_function, str(j))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_files = image_files[i:i + args.batch_size]

        image_list = []
        name_list = []
        for image_name in batch_files:
            save_path = os.path.join(output_path, image_name)
            if os.path.exists(save_path):
                continue  # Skip this image
            image = Image.open(os.path.join(image_path, image_name))
            image_list.append(image)
            name_list.append(image_name)
        
        if not image_list:
            print(f"Skipping {i}, existed batch {batch_files}",flush=True)
            continue  # Skip to the next batch
        
        
        if args.fused:
            # print('generating fused perturbations')
            if args.perturbation_type == 1:
                adjusted_images = noise_function_1(image_list, severity=j)
                adjusted_images = noise_function_2(adjusted_images, severity=j)
            elif args.perturbation_type == 2:
                adjusted_images = noise_function_1(image_list, severity=j)
                adjusted_images = noise_function_3(adjusted_images, severity=j)
            elif args.perturbation_type == 3:
                adjusted_images = noise_function_2(image_list, severity=j)
                adjusted_images = noise_function_3(adjusted_images, severity=j)
            elif args.perturbation_type == 4:
                adjusted_images = noise_function_1(image_list, severity=j)
                adjusted_images = noise_function_2(adjusted_images, severity=j)
                adjusted_images = noise_function_3(adjusted_images, severity=j)
            else:
                print('wrong type')
        else:
            # print('generating single perturbation')
            adjusted_images = noise_function(image_list, severity=j)
                
        for img, image_name in zip(adjusted_images, name_list):
            save_path = os.path.join(output_path, image_name)
            img.save(save_path)  # save the image

    print(f"{datetime.now()}: success generated {name_function} severity = {j}, cost time:{datetime.now()-time2}",flush=True)

print("Finishing generate " + name_function + " data", flush=True)

# usage:
# python image_generation_without_annotations.py [perturbation_type:1-9] [--fused] [--visualize]
#   perturbation_type: Specify the perturbation type (1-9 --> ["gaussian_noise", "salt_and_pepper_noise", "gaussian_blur", "motion_blur", "brightness_contrast", "clouds", "haze", "gaps", "compression_artifacts"])
#   --fused: Only use for fuse noise
#   --visualize: Enable visualization
# example:
# python image_generation_without_annotations.py 1

