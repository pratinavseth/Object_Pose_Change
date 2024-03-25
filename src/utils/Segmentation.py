import wget
import os
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SAM_Model:
    def __init__(self, sam_model,image,device):
        self.device = device
        self.sam_model = sam_model
        self.image = image
        self.cwd = os.getcwd()
        self.initalize()
        self.initial_predictions()

    def model_check(self,model_name,model_checkpoint, model_url):
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        
        self.sam_path = os.path.join(self.cwd,self.sam_checkpoint)
        self.flag_model = os.path.isfile(self.sam_path)
        if self.flag_model:
            print("model exists")
        else:
            print("Downloading Model")
            self.sam_path = os.path.join(self.cwd,"models")
            wget.download(model_url, out=self.sam_path)

    def initalize(self):
        if self.sam_model == "vit_l":
            self.sam_checkpoint = "models/sam_vit_l_0b3195.pth"
            self.sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
            self.model_check(model_name=self.sam_model, model_checkpoint=self.sam_checkpoint,model_url=self.sam_url)
            
        elif self.sam_model == "vit_b":
            self.sam_checkpoint = "models/sam_vit_b_01ec64.pth"
            self.sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            self.model_check(model_name=self.sam_model, model_checkpoint=self.sam_checkpoint,model_url=self.sam_url)
        else:
            self.sam_checkpoint = "models/sam_vit_h_4b8939.pth"
            self.sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            self.model_check(model_name=self.sam_model, model_checkpoint=self.sam_checkpoint,model_url=self.sam_url)

        self.model_type = self.model_name
        self.model_checkpoint = os.path.join(self.cwd,self.sam_checkpoint)
        self.sam = sam_model_registry[self.model_type](checkpoint=self.model_checkpoint)
        self.sam.to(device=self.device)
    
        
    def initial_predictions(self):
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.image)
