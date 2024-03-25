from PIL import Image
import sys
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class BB_Model:
    def __init__(self, bb_model,device):
        self.bb_model = bb_model
        self.device = device
        self.initalize()

    def initalize(self):
        if self.bb_model == 'owlvit':
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        else:
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-finetuned")
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-finetuned")    
    
    def get_boxes(self,text_prompt,image):
        self.text_prompt = str(text_prompt)
        self.image = Image.fromarray(image) 
        inputs = self.processor(text=self.text_prompt, images=self.image, return_tensors="pt")
        self.outputs = self.model(**inputs)
        # Target image sizes (height, width) for resizing box predictions are specified as [batch_size, 2].
        self.target_sizes = torch.Tensor([self.image.size[::-1]])
        # Converting bounding boxes and class logits outputs to (xmin, ymin, xmax, ymax) Pascal VOC format 
        self.results = self.processor.post_process_object_detection(outputs=self.outputs, target_sizes=self.target_sizes, threshold=0.1)
        
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = self.results[i]["boxes"], self.results[i]["scores"], self.results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text_prompt} with confidence {round(score.item(), 3)} at location {box}")

        boxes = boxes.tolist()
        box_number = len(boxes)
        if len(boxes) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            for box in boxes:
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            plt.title("Detected objects", fontsize=20)
            plt.show()
        
        self.input_boxes = torch.tensor(boxes,device=self.device)
        self.labels = [1]*len(self.input_boxes)
        self.final_input_boxes = self.input_boxes
        self.final_labels = self.labels
        self.num_boxes = box_number
