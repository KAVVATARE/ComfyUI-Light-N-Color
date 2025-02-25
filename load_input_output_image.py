import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers
import random

class LoadInputOutputImage:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        # Get both input and output directories
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        # Get all files from both directories and their subdirectories
        image_files = []
        
        # Helper function to collect image files from a directory
        def collect_images(base_dir, prefix=""):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    # Check if file is an image
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
                        # Get relative path from base directory
                        rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                        # Add directory prefix to distinguish between input and output
                        prefixed_path = os.path.join(prefix, rel_path)
                        image_files.append(prefixed_path)
        
        # Collect images from input directory with "input/" prefix
        collect_images(input_dir, "input")
        # Collect images from output directory with "output/" prefix
        collect_images(output_dir, "output")
        
        return {"required": 
                    {"image": (sorted(image_files), {"image_upload": True})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
               }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image, prompt=None, extra_pnginfo=None):
        # Split the path to determine if it's from input or output directory
        parts = image.split(os.sep)
        source_dir = parts[0]  # Will be either "input" or "output"
        rel_path = os.path.sep.join(parts[1:])  # Rest of the path
        
        # Get the appropriate base directory
        if source_dir == "input":
            base_dir = folder_paths.get_input_directory()
        else:  # output
            base_dir = folder_paths.get_output_directory()
        
        # Construct full path
        image_path = os.path.join(base_dir, rel_path)

        # Load and process image
        img = node_helpers.pillow(Image.open, image_path)
        i = node_helpers.pillow(ImageOps.exif_transpose, img)
        
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        i = i.convert("RGB")
        image = i.convert("RGB")
        
        # プレビュー用の保存処理
        filename_prefix = "preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image.size[0], image.size[1])
        file = f"{filename}_{counter:05}_.png"
        image.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
        
        results = [{"filename": file, "subfolder": subfolder, "type": self.type}]
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        
        return (image, mask, { "ui": { "images": results } })

# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadInputOutputImage": LoadInputOutputImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadInputOutputImage": "Load Input/Output Image"
}
