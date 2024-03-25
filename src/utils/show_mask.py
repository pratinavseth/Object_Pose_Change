import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask_image, axis, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 153/255, 10/255, 0.6])
    h, w = mask_image.shape[-2:]
    masked_image = mask_image.reshape(h, w, 1) * color.reshape(1, 1, -1)
    axis.imshow(masked_image)

def show_box(box_coords, axis):
    x0, y0 = box_coords[0], box_coords[1]
    w, h = box_coords[2] - box_coords[0], box_coords[3] - box_coords[1]
    axis.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))  

def show_points(coordinates, labels, axis, marker_size=375):
    pos_points = coordinates[labels==1]
    neg_points = coordinates[labels==0]
    axis.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    axis.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)  

def show_mask_org(mask_image, original_image, axis):
    mask_image = np.squeeze(mask_image)
    mask_image = np.expand_dims(mask_image, axis=-1)
    masked_image = np.where(mask_image > 0, original_image, 0)
    axis.imshow(masked_image)

def show_non_masked(mask, image, axis):
    mask_image = np.squeeze(mask)
    mask_image = np.expand_dims(mask_image, axis=-1)
    masked_image = np.where(mask_image > 0, 0, image)
    axis.imshow(masked_image)
