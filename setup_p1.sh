conda init
conda create -n SAM_Module
conda activate SAM_Module
pip install -r /kaggle/working/Object_Pose_Change/requirement.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install wget
cd /kaggle/working/Object_Pose_Change
mkdir models
