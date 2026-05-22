source ~/miniconda3/bin/activate
# conda remove -n mmseg --all -y
conda remove -n mmseg2 --all -y
conda remove -n mmseg3 --all -y
# conda remove -n mmseg4 --all -y
conda remove -n mmseg5 --all -y

conda create --name mmseg python=3.10 -y
conda activate mmseg
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.4 torchvision torchaudio -c conda-forge -c nodefaults