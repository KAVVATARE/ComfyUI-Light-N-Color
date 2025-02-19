from .nodes import (ControlNetSwitch, ImageSwitch, LatentSwitch, FluxSamplerPuLID, 
                   NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
from .Lighting_and_Color import FluxLightingAndColor
from .load_input_output_image import NODE_CLASS_MAPPINGS as LOAD_IMAGE_NODES
from .load_input_output_image import NODE_DISPLAY_NAME_MAPPINGS as LOAD_IMAGE_DISPLAY_NAMES

# ノードマッピングに追加
NODE_CLASS_MAPPINGS.update({
    "FluxLightingAndColor": FluxLightingAndColor,
})
NODE_CLASS_MAPPINGS.update(LOAD_IMAGE_NODES)

# 表示名マッピングに追加
NODE_DISPLAY_NAME_MAPPINGS.update({
    "FluxLightingAndColor": "Flux Lighting & Color",
})
NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_IMAGE_DISPLAY_NAMES)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]