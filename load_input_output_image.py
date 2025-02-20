import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers

class LoadOutputImage:
    @classmethod
    def INPUT_TYPES(s):
        # Get both input and output directories
        output_dir = folder_paths.get_output_directory()
        
        # Get all files from output directory and its subdirectories
        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                # Check if file is an image (you can add more extensions if needed)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    # Get relative path from output directory
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    image_files.append(rel_path)
        
        return {"required": 
                    {"image": (sorted(image_files), {"image_upload": True})},
               }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        # Get full path by joining output directory and selected image path
        output_dir = folder_paths.get_output_directory()
        image_path = os.path.join(output_dir, image)

        # Load and process image similar to original LoadImage
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None
        
        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
                
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))
        
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask)

# Add this to your nodes by including this in your script:
NODE_CLASS_MAPPINGS = {
    "LoadOutputImage": LoadOutputImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOutputImage": "Load Output Image"
}