from .nodes import (ControlNetSwitch, ImageSwitch, LatentSwitch, FluxSamplerPuLID, 
                   NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
from .Lighting_and_Color import FluxLightingAndColor
from .load_input_output_image import LoadInputOutputImage

# ノードマッピングに追加
NODE_CLASS_MAPPINGS.update({
    "LoadInputOutputImage": LoadInputOutputImage,
    "FluxLightingAndColor": FluxLightingAndColor,
})

# 表示名マッピングに追加
NODE_DISPLAY_NAME_MAPPINGS.update({
    "LoadInputOutputImage": "Load Input/Output Image",
    "FluxLightingAndColor": "Flux Lighting & Color",
})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]