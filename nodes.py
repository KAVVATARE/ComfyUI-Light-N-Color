# XLabsのControlNet条件を切り替えるカスタムノード
class ControlNetSwitch:
   # ノードの初期化
   def __init__(self):
       self.type = "ControlNetSwitch"
       print("ControlNetSwitch initialized")
       
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "controlnet_condition_1": ("ControlNetCondition",),
               "controlnet_condition_2": ("ControlNetCondition",),
               "use_first": ("BOOLEAN", {"default": True}),
           }
       }
   
   RETURN_TYPES = ("ControlNetCondition",)
   RETURN_NAMES = ("controlnet_condition",)
   FUNCTION = "switch"
   CATEGORY = "XLabsNodes"
   
   def switch(self, controlnet_condition_1, controlnet_condition_2, use_first):
       print(f"Switching ControlNet conditions. Using {'first' if use_first else 'second'} condition")
       return (controlnet_condition_1 if use_first else controlnet_condition_2,)

class ImageSwitch:
   def __init__(self):
       self.type = "ImageSwitch"
       print("ImageSwitch initialized")
       
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "image_1": ("IMAGE",),
               "image_2": ("IMAGE",),
               "use_first": ("BOOLEAN", {"default": True}),
           }
       }
   
   RETURN_TYPES = ("IMAGE",)
   RETURN_NAMES = ("image",)
   FUNCTION = "switch"
   CATEGORY = "image"
   
   def switch(self, image_1, image_2, use_first):
       print(f"Switching images. Using {'first' if use_first else 'second'} image")
       return (image_1 if use_first else image_2,)

class LatentSwitch:
   def __init__(self):
       self.type = "LatentSwitch"
       print("LatentSwitch initialized")
       
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "latent_1": ("LATENT",),
               "latent_2": ("LATENT",),
               "use_first": ("BOOLEAN", {"default": True}),
           }
       }
   
   RETURN_TYPES = ("LATENT",)
   RETURN_NAMES = ("latent",)
   FUNCTION = "switch"
   CATEGORY = "latent"
   
   def switch(self, latent_1, latent_2, use_first):
       print(f"Switching latents. Using {'first' if use_first else 'second'} latent")
       print(f"Latent 1 shape: {latent_1['samples'].shape}")
       print(f"Latent 2 shape: {latent_2['samples'].shape}")
       result = latent_1 if use_first else latent_2
       print(f"Result shape: {result['samples'].shape}")
       return (result,)

class FluxSamplerPuLID:
   @classmethod
   def INPUT_TYPES(s):
       return {
           "required": {
                   "model": ("MODEL",),
                   "conditioning": ("CONDITIONING",),
                   "neg_conditioning": ("CONDITIONING",),
                   "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                   "steps": ("INT",  {"default": 20, "min": 1, "max": 100}),
                   "timestep_to_start_cfg": ("INT",  {"default": 20, "min": 0, "max": 100}),
                   "true_gs": ("FLOAT",  {"default": 3, "min": 0, "max": 100}),
                   "image_to_image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                   "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                   "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                   "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
               },
           "optional": {
                   "latent_image": ("LATENT", {"default": None}),
                   "controlnet_condition": ("ControlNetCondition", {"default": None}),
               }
           }
   
   RETURN_TYPES = ("LATENT",)
   RETURN_NAMES = ("latent",)
   FUNCTION = "sampling"
   CATEGORY = "XLabsNodes"

   def sampling(self, model, conditioning, neg_conditioning,
            noise_seed, steps, timestep_to_start_cfg, true_gs,
            image_to_image_strength, denoise_strength,
            max_shift, base_shift,
            latent_image=None, controlnet_condition=None
            ):
       import torch
       import comfy.model_management as mm
       from comfy_extras.nodes_model_advanced import ModelSamplingFlux
       import latent_preview
       import importlib.util
       import os
       import sys

       # モジュールを動的にインポート
       def import_from_path(module_name, file_path):
           spec = importlib.util.spec_from_file_location(module_name, file_path)
           module = importlib.util.module_from_spec(spec)
           sys.modules[module_name] = module  # これを追加
           spec.loader.exec_module(module)
           return module

       root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       x_flux_path = os.path.join(root_path, "x-flux-comfyui")

       try:
           # 各モジュールを動的にインポート
           layers_module = import_from_path("layers", os.path.join(x_flux_path, "layers.py"))
           sampling_module = import_from_path("sampling", os.path.join(x_flux_path, "sampling.py"))
           utils_module = import_from_path("utils", os.path.join(x_flux_path, "utils.py"))

           # 必要な関数とクラスを取得
           get_noise = sampling_module.get_noise
           prepare = sampling_module.prepare
           get_schedule = sampling_module.get_schedule
           denoise = sampling_module.denoise
           denoise_controlnet = sampling_module.denoise_controlnet
           unpack = sampling_module.unpack
           LATENT_PROCESSOR_COMFY = utils_module.LATENT_PROCESSOR_COMFY
           ControlNetContainer = utils_module.ControlNetContainer
           DoubleStreamMixerProcessor = layers_module.DoubleStreamMixerProcessor
           timestep_embedding = layers_module.timestep_embedding

       except Exception as e:
           print(f"Error importing x-flux-comfyui modules: {str(e)}")
           print(f"Looking in path: {x_flux_path}")
           print(f"Available files: {os.listdir(x_flux_path)}")
           raise

       # PuLID Fluxのモデル処理を追加
       modelsamplingflux = ModelSamplingFlux()
       width = latent_image["samples"].shape[3]*8
       height = latent_image["samples"].shape[2]*8
       work_model = modelsamplingflux.patch(model, max_shift, base_shift, width, height)[0]

       additional_steps = 11 if controlnet_condition is None else 12
       mm.load_model_gpu(work_model)
       inmodel = work_model.model

       try:
           guidance = conditioning[0][1]['guidance']
       except:
           guidance = 1.0

       device = mm.get_torch_device()
       if torch.backends.mps.is_available():
           device = torch.device("mps")
       if torch.cuda.is_bf16_supported():
           dtype_model = torch.bfloat16
       else:
           dtype_model = torch.float16

       offload_device = mm.unet_offload_device()
       torch.manual_seed(noise_seed)

       bc, c, h, w = latent_image['samples'].shape
       height = (h//2) * 16
       width = (w//2) * 16

       x = get_noise(
           bc, height, width, device=device,
           dtype=dtype_model, seed=noise_seed
       )
       orig_x = None
       if c==16:
           orig_x = latent_image['samples']
           lat_processor2 = LATENT_PROCESSOR_COMFY()
           orig_x = lat_processor2.go_back(orig_x)
           orig_x = orig_x.to(device, dtype=dtype_model)

       timesteps = get_schedule(
           steps,
           (width // 8) * (height // 8) // 4,
           shift=True,
       )
       try:
           inmodel.to(device)
       except:
           pass
       x.to(device)
       
       inmodel.diffusion_model.to(device)
       inp_cond = prepare(conditioning[0][0], conditioning[0][1]['pooled_output'], img=x)
       neg_inp_cond = prepare(neg_conditioning[0][0], neg_conditioning[0][1]['pooled_output'], img=x)

       if denoise_strength <= 0.99:
           try:
               timesteps = timesteps[:int(len(timesteps)*denoise_strength)]
           except:
               pass

       x0_output = {}
       callback = latent_preview.prepare_callback(model, len(timesteps) - 1, x0_output)

       if controlnet_condition is None:
           x = denoise(
               inmodel.diffusion_model, **inp_cond, timesteps=timesteps, guidance=guidance,
               timestep_to_start_cfg=timestep_to_start_cfg,
               neg_txt=neg_inp_cond['txt'],
               neg_txt_ids=neg_inp_cond['txt_ids'],
               neg_vec=neg_inp_cond['vec'],
               true_gs=true_gs,
               image2image_strength=image_to_image_strength,
               orig_image=orig_x,
               callback=callback,
               width=width,
               height=height,
           )
       else:
           def prepare_controlnet_condition(controlnet_condition):
               controlnet = controlnet_condition['model']
               controlnet_image = controlnet_condition['img']
               controlnet_image = torch.nn.functional.interpolate(
                   controlnet_image, size=(height, width), scale_factor=None, mode='bicubic',)
               controlnet_strength = controlnet_condition['controlnet_strength']
               controlnet_start = controlnet_condition['start']
               controlnet_end = controlnet_condition['end']
               controlnet.to(device, dtype=dtype_model)
               controlnet_image = controlnet_image.to(device, dtype=dtype_model)
               return {
                   "img": controlnet_image,
                   "controlnet_strength": controlnet_strength,
                   "model": controlnet,
                   "start": controlnet_start,
                   "end": controlnet_end,
               }

           cnet_conditions = [prepare_controlnet_condition(el) for el in controlnet_condition]
           containers = []
           for el in cnet_conditions:
               start_step = int(el['start']*len(timesteps))
               end_step = int(el['end']*len(timesteps))
               container = ControlNetContainer(el['model'], el['img'], el['controlnet_strength'], start_step, end_step)
               containers.append(container)

           mm.load_models_gpu([work_model,])

           total_steps = len(timesteps)

           x = denoise_controlnet(
               inmodel.diffusion_model, **inp_cond, 
               controlnets_container=containers,
               timesteps=timesteps, guidance=guidance,
               timestep_to_start_cfg=timestep_to_start_cfg,
               neg_txt=neg_inp_cond['txt'],
               neg_txt_ids=neg_inp_cond['txt_ids'],
               neg_vec=neg_inp_cond['vec'],
               true_gs=true_gs,
               image2image_strength=image_to_image_strength,
               orig_image=orig_x,
               callback=callback,
               width=width,
               height=height,
           )

       x = unpack(x, height, width)
       lat_processor = LATENT_PROCESSOR_COMFY()
       x = lat_processor(x)
       lat_ret = {"samples": x}

       return (lat_ret,)

# ノードの登録
NODE_CLASS_MAPPINGS = {
   "ControlNetSwitch": ControlNetSwitch,
   "ImageSwitch": ImageSwitch,
   "LatentSwitch": LatentSwitch,
   "FluxSamplerPuLID": FluxSamplerPuLID
}

NODE_DISPLAY_NAME_MAPPINGS = {
   "ControlNetSwitch": "ControlNet Switcher",
   "ImageSwitch": "Image Switcher",
   "LatentSwitch": "Latent Switcher",
   "FluxSamplerPuLID": "Flux Sampler For PuLID"
}