import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import random

class LoadInputOutputImage:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        # デフォルト読み込みディレクトリの選択肢
        directory_options = ["output", "input"]  # outputをデフォルトに設定
        
        # 入力ディレクトリと出力ディレクトリのパスを取得
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # 全ファイルリストを取得
        image_files = []
        
        # ディレクトリからファイルを収集する関数
        def collect_images(base_dir, prefix=""):
            if os.path.exists(base_dir):
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
                            rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                            # ディレクトリプレフィックスを追加
                            prefixed_path = os.path.join(prefix, rel_path)
                            image_files.append(prefixed_path)
        
        # 入力ディレクトリと出力ディレクトリから画像を収集
        collect_images(input_dir, "input")
        collect_images(output_dir, "output")
        
        print(f"Total files found: {len(image_files)}")
        
        # ファイルリストが空の場合のデフォルト値
        if not image_files:
            image_files = ["none"]
        
        return {
            "required": {
                "image": (sorted(image_files), {"image_upload": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    OUTPUT_NODE = True

    def load_image(self, image, prompt=None, extra_pnginfo=None):
        print(f"Selected image: {image}")
        
        try:
            # パスからディレクトリ部分と相対パス部分を抽出
            parts = image.split(os.sep)
            if len(parts) >= 2:
                source_dir = parts[0]  # "input" または "output"
                rel_path = os.path.sep.join(parts[1:])  # 相対パス
            else:
                # パスにディレクトリ情報がない場合はinputとみなす
                source_dir = "input"
                rel_path = image
            
            print(f"Source directory: {source_dir}")
            print(f"Relative path: {rel_path}")
            
            # ベースディレクトリを決定
            if source_dir == "input":
                base_dir = folder_paths.get_input_directory()
            else:  # "output"
                base_dir = folder_paths.get_output_directory()
            
            print(f"Base directory: {base_dir}")
            
            # 画像ファイルのフルパス
            image_path = os.path.join(base_dir, rel_path)
            print(f"Full image path: {image_path}")
            
            # ファイルの存在チェック
            if not os.path.exists(image_path) or rel_path == "none":
                print(f"Error: Image file does not exist: {image_path}")
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            
            # 画像を読み込む
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            print(f"Image mode: {i.mode}")
            print(f"Image size: {i.size}")
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image = i.convert("RGB")
            
            # プレビュー用に一時ファイルを作成
            filename_prefix = "ComfyUI"
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, image.size[0], image.size[1])
            preview_file = f"{filename}_{counter:05}_.png"
            preview_path = os.path.join(full_output_folder, preview_file)
            image.save(preview_path, compress_level=self.compress_level)
            
            # UI用の結果を作成
            results = []
            results.append({
                "filename": preview_file,
                "subfolder": subfolder,
                "type": self.type
            })
            print(f"Preview saved to: {preview_path}")
            print(f"UI Results: {results}")
            
            # テンソル変換
            image_tensor = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_tensor)[None,]
            
            print(f"Image tensor shape: {image_tensor.shape}")
            
            # マスクの処理
            if 'A' in i.getbands():
                print("Alpha channel found, creating mask")
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                print("No alpha channel, creating empty mask")
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
            print(f"Mask shape: {mask.shape}")
            
            # マスクにバッチ次元を追加
            mask_with_batch = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
            
            return (image_tensor, mask_with_batch, {"ui": {"images": results}})
            
        except Exception as e:
            import traceback
            print(f"Error loading image: {str(e)}")
            print(f"Exception details: {traceback.format_exc()}")
            raise e

# ノード登録
NODE_CLASS_MAPPINGS = {
    "LoadInputOutputImage": LoadInputOutputImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadInputOutputImage": "Load Input/Output Image"
}