import torch
from .bb_extractor import BB_Model
from .Segmentation import SAM_Model
from .show_mask import show_mask,show_box,show_points,show_mask_org,show_non_masked
import torch
import cv2
import matplotlib.pyplot as plt
import uuid
import os
from PIL import Image

def inference_run(configs,device):
    img_path = configs['img_path']
    output_path = configs['output_dir']
    bb_model = configs['bb_model']
    sam_model = configs['SAM_model']
    text_prompt = configs['Text_Prompt']
    device = device

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    model_bb = BB_Model(bb_model=bb_model,device=device)
    model_bb.get_boxes(text_prompt=text_prompt,image=image)    
    input_boxes = model_bb.final_input_boxes
    input_label = model_bb.final_labels
    num_box = model_bb.num_boxes
    del model_bb
    model_sam = SAM_Model(sam_model=sam_model,device=device,image=image)
    predictor = model_sam.predictor
    del model_sam



    if(num_box<1):
        print("\nPrompted Object Not Detected\n")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        filename = str("no-obj") + '.png'
        file_path = os.path.join(output_path,filename)
    else:
        print("\nPrompted Object Detected\n")
        input_boxes = torch.tensor(input_boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        plt.figure(figsize=(10, 10))
        for mask in masks:
            show_mask_org(mask.cpu().numpy(), Image.fromarray(image), plt.gca())
        plt.axis('off')
        filename = str("mask") + '.png'
        file_path = os.path.join(output_path,filename)
        plt.savefig(file_path)
        
        plt.figure(figsize=(10, 10))
        for mask in masks:
            show_non_masked(mask.cpu().numpy(), Image.fromarray(image), plt.gca())
        plt.axis('off')
        filename = str("image-wo-mask") + '.png'
        file_path = os.path.join(output_path,filename)
        plt.savefig(file_path)
        
        
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        filename = str("image-mask") + '.png'
        file_path = os.path.join(output_path,filename)

    plt.savefig(file_path)
    plt.show()




def inference_localization(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    inference_run(configs=configs, device=device)
