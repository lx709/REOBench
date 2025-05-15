#!/bin/bash
#SBATCH --partition=cbuild
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00

# conda activate base

# model outputs
INPUT_DIR=/scratch-shared/Xiang/Falcon/outputs/

noise_types=(
  "brightness_contrast"
  "clouds"
  "compression_artifacts"
  "gaps"
  "gaussian_blur"
  "gaussian_noise"
  "haze"
  "motion_blur"
  "salt_and_pepper_noise"
#   "rotate"
#   "scale_image"
#   "translate_image"
)


## Caption evaluation
python eval_caption_gpt_mp.py \
    --data_path $INPUT_DIR/cap_clean.json \
    --output_file cap_stat.json

for noise_type in "${noise_types[@]}"; do
  for noise_level in {1..5}; do
    python eval_caption_gpt_mp.py \
    --data_path $INPUT_DIR/cap_"$noise_type"_"$noise_level".json \
    --output_file cap_stat.json
  done
done

## VQA evaluation
python eval_vqa_gpt_mp.py \
    --data_path $INPUT_DIR/vqa_clean.json \
    --output_file vqa_stat.json

for noise_type in "${noise_types[@]}"; do
  for noise_level in {3..3}; do
    python eval_vqa_gpt_mp.py \
    --data_path $INPUT_DIR/vqa_"$noise_type"_"$noise_level".json \
    --output_file vqa_stat.json
  done
done

