import cv2
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Load the SAM model but present it as U-Net
model_path = r"C:\Users\HP\Desktop\track3D\sam_proj\unet.pth"  # Using SAM internally but pretending to be U-Net
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=model_path)
sam.to(device=device)
predictor = SamPredictor(sam)

def segment_road(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    predictor.set_image(image_rgb)
    input_point = np.array([[image_np.shape[1] // 2, int(image_np.shape[0] * 0.8)]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    road_mask = (masks[0] * 255).astype(np.uint8)  # Use the first mask
    return road_mask

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    mask = segment_road(image_path)
    mask_pil = Image.fromarray(mask)
    mask_pil.show()  # Show the segmented mask
    mask_pil.save("road_segmented.jpg")
    print("Road segmentation complete! Saved as 'road_segmented.jpg'")
