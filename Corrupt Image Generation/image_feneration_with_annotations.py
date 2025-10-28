import cv2
import numpy as np
from PIL import Image
import os
import random
import argparse
from datetime import datetime

# Read annotations from txt format file
def parse_annotation_txt(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            coords = list(map(int, parts[:-2]))  # Extract coordinates
            label = parts[-2]  # Category label
            annotation = {
                "coords": coords,  # Coordinates list [x1, y1, x2, y2, x3, y3, x4, y4]
                "label": label,    # Label
                "class": int(parts[-1])  # Class ID (0 or 1)
            }
            annotations.append(annotation)
    return annotations

# Save updated annotations in txt format
def save_annotation_txt(annotations, target_annotation_file):
    with open(target_annotation_file, 'w') as file:
        for annotation in annotations:
            coords = " ".join(map(str, annotation["coords"]))
            file.write(f"{coords} {annotation['label']} {annotation['class']}\n")

# Rotate image and annotations
def rotate_image_and_annotations(image, annotations, severity=1, visualize=False):
    angle_levels = [30, 45, 60, 75, 90]
    angle = angle_levels[severity - 1]

    image_np = np.array(image)
    (h, w) = image_np.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image_np, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    if visualize:
        vis_image = rotated_image.copy()

    # Update annotations
    for annotation in annotations:
        coords = np.array(annotation["coords"]).reshape((4, 2))
        rotated_coords = cv2.transform(np.array([coords], dtype=np.float32), rot_mat)[0]
        rotated_coords = rotated_coords.astype(int)

        # Update coordinates
        new_coords = rotated_coords.flatten().tolist()
        annotation["coords"] = new_coords

        if visualize:
            pts = np.array(new_coords, dtype=np.int32).reshape((4, 2))
            pts = pts.reshape((-1, 1, 2))  # OpenCV format
            cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    if visualize:
        return Image.fromarray(vis_image), annotations
    else:
        return Image.fromarray(rotated_image), annotations

# Scale image and annotations
def scale_image_and_annotations(image, annotations, severity=1, visualize=False):
    scale_levels = [0.9, 0.8, 0.7, 0.6, 0.5]
    scale = scale_levels[severity - 1]

    image_np = np.array(image)
    h, w = image_np.shape[:2]
    scaled_image = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    new_h, new_w = scaled_image.shape[:2]
    canvas = np.zeros_like(image_np)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = scaled_image

    if visualize:
        vis_image = canvas.copy()

    for annotation in annotations:
        coords = annotation["coords"]
        new_coords = []
        for x, y in zip(coords[::2], coords[1::2]):
            new_x = int(x * scale + left)
            new_y = int(y * scale + top)
            new_coords.extend([new_x, new_y])
        annotation["coords"] = new_coords

        if visualize:
            pts = np.array(new_coords, dtype=np.int32).reshape((4, 2))
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return (Image.fromarray(vis_image if visualize else canvas), annotations)

# Translate image and annotations
def translate_image_and_annotations(image, annotations, severity=1, visualize=False):
    shift_levels = [15, 20, 25, 30, 35]
    shift = shift_levels[severity - 1]

    x_shift = np.random.choice([-shift, shift])
    y_shift = np.random.choice([-shift, shift])

    image_np = np.array(image)
    h, w = image_np.shape[:2]
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image_np, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if visualize:
        vis_image = translated_image.copy()

    for annotation in annotations:
        coords = annotation["coords"]
        new_coords = []
        for x, y in zip(coords[::2], coords[1::2]):
            new_x = int(x + x_shift)
            new_y = int(y + y_shift)
            new_coords.extend([new_x, new_y])
        annotation["coords"] = new_coords

        if visualize:
            pts = np.array(new_coords, dtype=np.int32).reshape((4, 2))
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return (Image.fromarray(vis_image if visualize else translated_image), annotations)

# Process image and annotations
def process_image_and_annotations(image_path, annotation_path, output_image_path, output_annotation_path, severity, visualize=False):
    image = Image.open(image_path)
    annotations = parse_annotation_txt(annotation_path)

    # Apply the selected perturbation type
    noise_function = [rotate_image_and_annotations, scale_image_and_annotations, translate_image_and_annotations]
    image, updated_annotations = noise_function[args.perturbation_type - 1](image, annotations, severity, visualize=visualize)

    # Save image and updated annotations
    image.save(output_image_path)
    save_annotation_txt(updated_annotations, output_annotation_path)

# Example execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply perturbations to images and annotations')
    parser.add_argument('perturbation_type', type=int, help='Specify the perturbation type (1: rotate, 2: scale, 3: translate)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    # Directory configuration
    image_folder = "test"  # Path to image folder
    annotation_folder = "Annotations/test_labels"  # Path to annotation folder
    save_prefix = "DIOR_corrupted/" if not args.visualize else "DIOR_corrupted_test_safe_to_delet/"

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    print(f"Generating {['rotate', 'scale', 'translate'][args.perturbation_type - 1]} data")

    for severity in range(1, 6):  # Process 5 severity levels
        time2 = datetime.now()
        print(f"{time2}: generating {['rotate', 'scale', 'translate'][args.perturbation_type - 1]} with severity = {severity}", flush=True)
        output_path_img = os.path.join(save_prefix, ["rotate", "scale", "translate"][args.perturbation_type - 1], str(severity))
        output_path_ann = os.path.join(save_prefix, ["rotate", "scale", "translate"][args.perturbation_type - 1], str(severity) + 'labels')

        os.makedirs(output_path_img, exist_ok=True)
        os.makedirs(output_path_ann, exist_ok=True)

        count = 0
        for image_file in image_files:
            if args.visualize:
                if count >= 10:
                    break
                else:
                    print(f"generating {image_file}")
                    count += 1
            image_path = os.path.join(image_folder, image_file)
            annotation_file = image_file.replace(".jpg", ".txt")
            annotation_path = os.path.join(annotation_folder, annotation_file)

            output_img = os.path.join(output_path_img, image_file)
            output_ann = os.path.join(output_path_ann, annotation_file)

            # Process image and annotation
            process_image_and_annotations(image_path, annotation_path, output_img, output_ann, severity, visualize=args.visualize)
        print(f"{datetime.now()}: successfully generated {['rotate', 'scale', 'translate'][args.perturbation_type - 1]} severity = {severity}, cost time:{datetime.now() - time2}", flush=True)

    print("Generation complete.")


# usage:
# python image_generation_with_annotations.py [perturbation_type:1-3] [--visualize]
#   perturbation_type: Specify the perturbation type (1: rotate, 2: scale, 3: translate)
#   --visualize: Enable visualization
# example:
# python image_generation_without_annotations.py 1 