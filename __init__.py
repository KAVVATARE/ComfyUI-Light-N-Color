from .nodes import (ControlNetSwitch, ImageSwitch, LatentSwitch, FluxSamplerPuLID, 
                   NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
from .Lighting_and_Color import FluxLightingAndColor
from .load_input_output_image import LoadInputOutputImage
from .clip_text_encode_io import CLIPTextEncodeWithPromptIO

# ノードマッピングに追加
NODE_CLASS_MAPPINGS.update({
    "LoadInputOutputImage": LoadInputOutputImage,
    "FluxLightingAndColor": FluxLightingAndColor,
    "CLIPTextEncodeWithPromptIO": CLIPTextEncodeWithPromptIO,
})

# 表示名マッピングに追加
NODE_DISPLAY_NAME_MAPPINGS.update({
    "LoadInputOutputImage": "Load Input/Output Image",
    "FluxLightingAndColor": "Flux Lighting & Color",
    "CLIPTextEncodeWithPromptIO": "CLIP Text Encode with Prompt I/O",
})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]