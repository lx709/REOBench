### 
# Image Generation Scripts

This repository includes two Python scripts for generating perturbed images, with or without annotations.

---

## Files Overview

- **`image_generation_with_annotations.py`**  
  Generates perturbed images **with annotations**, updating bounding boxes according to transformations.  
  The annotation processing is based on the **DIOR-R dataset** format, for other dataset formats, please modify the annotation handling part accordingly.

- **`image_generation_without_annotations.py`**  
  Generates perturbed images **without annotations**, applying various noise and distortion effects.

---



### usage

```
python image_generation_without_annotations.py [perturbation_type:1-9] [--fused] [--visualize]
````

  - perturbation_type: Specify the perturbation type (1-9 --> ["gaussian_noise", "salt_and_pepper_noise", "gaussian_blur", "motion_blur", "brightness_contrast", "clouds", "haze", "gaps", "compression_artifacts"])

  - --fused: Only use for fuse noise (Only implement for ```image_generation_without_annotations.py```)

  - --visualize: Enable visualization

### example:
```
python image_generation_without_annotations.py 1
```
