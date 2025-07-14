import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        h,w=input_tensor.shape
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
def get_bbox_from_cam(cam_np, threshold):
    mask = cam_np > threshold
    if not mask.any():
        return None
    y_idx, x_idx = np.where(mask)
    x1, x2 = x_idx.min(), x_idx.max()
    y1, y2 = y_idx.min(), y_idx.max()
    return (x1, y1, x2, y2)

def show_result(img_pil, cam_np, bbox, threshold):
    img_np = np.array(img_pil) / 255.0
    heatmap = plt.cm.jet(cam_np)[..., :3]
    blended = 0.8 * img_np + 0.2 * heatmap

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Solo heatmap
    axs[0].imshow(heatmap)
    axs[0].set_title("Heatmap")
    axs[0].axis('off')

    # Imagen + heatmap + bbox
    axs[1].imshow(blended)
    if bbox:
        x1, y1, x2, y2 = bbox
        axs[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor='lime', facecolor='none', linewidth=2))
    axs[1].set_title(f"Imagen con Bounding Box (threshold = {threshold})")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()  
    
def crop_from_gradcam_bbox_adjusted(img_pil, cam_np, threshold=0.4):
   
    width, height = img_pil.size
    mask = cam_np > threshold

    if not mask.any():
        raise ValueError("No se detectó ninguna zona activa en el mapa Grad-CAM con ese threshold.")

    
    y_idx, x_idx = np.where(mask)
    x1, x2 = x_idx.min(), x_idx.max()
    y1, y2 = y_idx.min(), y_idx.max()

    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)

    
    left   = cx - side // 2
    upper  = cy - side // 2
    right  = left + side
    lower  = upper + side

    
    if left < 0:
        right += -left
        left = 0
    if upper < 0:
        lower += -upper
        upper = 0
    if right > width:
        shift = right - width
        left = max(0, left - shift)
        right = width
    if lower > height:
        shift = lower - height
        upper = max(0, upper - shift)
        lower = height

    
    side = max(right - left, lower - upper)

   
    square_img = Image.new(img_pil.mode, (side, side), color=0)

    
    crop_box = (left, upper, right, lower)
    crop = img_pil.crop(crop_box)

    
    paste_x = (side - (right - left)) // 2
    paste_y = (side - (lower - upper)) // 2
    square_img.paste(crop, (paste_x, paste_y))

    return square_img


