conda deactivate
conda create -n Z123
conda activate Z123
mkdir zero123
cd zero123
git clone https://github.com/ptnv-s/Object_Pose_Change.git
cd /kaggle/working/zero123/Object_Pose_Change/src/zero123/zero123
pip install diffusers einops fire lovely-numpy lovely-tensors
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
wget -nv https://cv.cs.columbia.edu/zero123/assets/105000.ckpt    
pip install carvekit omegaconf pytorch_lightning==1.7.7 torchmetrics==0.11.4

