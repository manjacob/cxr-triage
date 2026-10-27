from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
 
def get_gradcam(model, image_tensor, target_layer, target_category=None):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=target_category)[0]
    return grayscale_cam  # overlay this on the original image for the heatmap
