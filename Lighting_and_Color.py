from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
import cv2 as cv

def tensor2pil(tensor):
    """
    PyTorchテンソルをPIL画像に変換する。
    BHWC形式で値が[0, 1]の範囲のテンソルを想定。
    """
    # テンソルがCPUにあることを確認
    tensor = tensor.cpu()
    
    # numpy配列に変換し、0-255の範囲に正規化
    img_array = tensor.numpy()
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    
    # バッチ処理の場合は最初の画像を取得
    if len(img_array.shape) == 4:
        img_array = img_array[0]  # バッチ次元を削除
    
    return Image.fromarray(img_array, mode='RGB')

def pil2tensor(image, original_shape):
    """
    PIL画像をPyTorchテンソルに変換し、元の形状に合わせる。
    BHWC形式で値が[0, 1]の範囲のテンソルを返す。
    """
    # numpy配列に変換し、[0, 1]に正規化
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 元の形状に合わせてバッチ次元を追加
    img_array = img_array[np.newaxis, ...]
    
    # PyTorchテンソルに変換
    tensor = torch.from_numpy(img_array).float()
    
    return tensor

def medianFilter(image, radius, num_samples, threshold):
    """
    画像に高品質なメディアンフィルタを適用する
    """
    # PILからCV2形式に変換
    cv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    
    # エッジを保持しながらスムージングを適用
    blurred = cv.bilateralFilter(cv_image, radius, num_samples, threshold)
    
    # PIL形式に戻す
    return Image.fromarray(cv.cvtColor(blurred, cv.COLOR_BGR2RGB))

class FluxLightingAndColor:
    """
    FluxLightingAndColor
    画像の照明と色調を調整するためのノードクラス
    
    主な機能:
    - 彩度調整
    - 被写界深度(DoF)処理
    - 最適化された処理順序
    - デバッグ出力
    """
    
    @classmethod
    def INPUT_TYPES(s):
        """
        入力パラメータの定義
        required: 必須パラメータ
        - image: 入力画像
        - black/mid/white_level: レベル調整用パラメータ
        - red/green/blue_level: 各色チャンネルの強度
        - brightness: 明るさ
        - saturation: 彩度
        
        optional: オプションパラメータ
        - depth: 深度マップ画像
        - dof_mode: 被写界深度エフェクトモード
        - dof_radius: ぼかしの半径
        - dof_samples: サンプル数
        - debug_mode: デバッグ出力の有無
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "white_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "red_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "green_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blue_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "dof_mode": (["none", "mock", "gaussian", "box"],),
                "dof_radius": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "dof_samples": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lighting_and_color"
    CATEGORY = "image/adjustments"

    def apply_dof(self, img, depth_map=None, mode='none', radius=8, samples=1, debug=False):
        """
        被写界深度(DoF)エフェクトを適用する
        
        Parameters:
        - img: 元画像
        - depth_map: 深度マップ
        - mode: エフェクトモード (none/mock/gaussian/box)
        - radius: ぼかしの半径
        - samples: サンプル数
        - debug: デバッグ出力フラグ
        """
        if mode == 'none' or depth_map is None:
            if debug:
                print("DoF: Skipped (mode: none or no depth map)")
            return img
            
        if debug:
            print(f"DoF: Applying {mode} blur with radius {radius} and {samples} samples")
            
        # Resize depth map to match image size and convert to grayscale
        depth_map = depth_map.resize(img.size).convert('L')
        
        # Apply blur based on selected mode
        if mode == 'mock':
            blurred = medianFilter(img, radius, (radius * 1500), 75)
        elif mode == 'gaussian':
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif mode == 'box':
            blurred = img.filter(ImageFilter.BoxBlur(radius))
        else:
            return img
            
        blurred = blurred.convert(img.mode)
        
        # Apply multiple samples if requested
        if samples > 1:
            result = None
            for i in range(samples):
                if not result:
                    result = Image.composite(img, blurred, depth_map)
                else:
                    result = Image.composite(result, blurred, depth_map)
                if debug:
                    print(f"DoF: Applied sample {i+1}/{samples}")
        else:
            result = Image.composite(img, blurred, depth_map).convert('RGB')
            
        return result

    def adjust_levels(self, img_array, black_level, mid_level, white_level, debug=False):
        """
        画像のレベル調整を行う
        
        Parameters:
        - img_array: 画像配列
        - black_level: 黒レベル
        - mid_level: 中間トーン
        - white_level: 白レベル
        - debug: デバッグ出力フラグ
        """
        if debug:
            print(f"Levels: Adjusting (black: {black_level}, mid: {mid_level}, white: {white_level})")
        # Apply level adjustments
        img_array = (img_array - black_level) / (white_level - black_level)
        img_array = np.clip(img_array, 0, 1)
        return img_array

    def adjust_channels(self, img_array, red_level, green_level, blue_level, debug=False):
        """
        RGB各チャンネルの強度を調整する
        
        Parameters:
        - img_array: 画像配列
        - red_level: 赤チャンネルの強度
        - green_level: 緑チャンネルの強度
        - blue_level: 青チャンネルの強度
        - debug: デバッグ出力フラグ
        """
        if debug:
            print(f"Channels: Adjusting (R: {red_level}, G: {green_level}, B: {blue_level})")
        # Split and adjust each channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * red_level, 0, 1)
        img_array[:,:,1] = np.clip(img_array[:,:,1] * green_level, 0, 1)
        img_array[:,:,2] = np.clip(img_array[:,:,2] * blue_level, 0, 1)
        return img_array

    def apply_lighting_and_color(self, image, black_level, mid_level, white_level, 
                               red_level, green_level, blue_level, brightness, saturation,
                               depth=None, dof_mode="none", dof_radius=8, dof_samples=1,
                               debug_mode=False):
        """
        メインの処理関数。以下の順序で画像処理を実行:
        1. 入力テンソルをPIL画像に変換
        2. 被写界深度エフェクトの適用（深度マップがある場合）
        3. numpy配列に変換
        4. 明るさの調整
        5. レベル調整
        6. チャンネル調整
        7. PIL画像に再変換
        8. 彩度の調整
        9. 最終的なコントラスト調整
        10. テンソルに再変換して返却
        """
        try:
            if debug_mode:
                print("\n=== FluxLightingAndColor Starting ===")
                print(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
            
            # 1. 入力テンソルをPIL画像に変換
            img_pil = tensor2pil(image)
            if debug_mode:
                print("Step 1: Converted input tensor to PIL image")
            
            # 2. 深度マップがある場合はDoFを適用
            if depth is not None and dof_mode != "none":
                depth_pil = tensor2pil(depth)
                img_pil = self.apply_dof(img_pil, depth_pil, dof_mode, dof_radius, dof_samples, debug_mode)
                if debug_mode:
                    print("Step 2: Applied depth of field effect")
            
            # 3. 処理用にnumpy配列に変換
            img_array = np.array(img_pil).astype(float) / 255.0
            
            # 4. トーン調整を適用
            if debug_mode:
                print(f"Step 4: Applying brightness boost: {brightness}")
            img_array = np.power(img_array, 0.7) * brightness
            
            # 5. レベル調整を適用
            img_array = self.adjust_levels(img_array, black_level, mid_level, white_level, debug_mode)
            
            # 6. チャンネル調整を適用
            img_array = self.adjust_channels(img_array, red_level, green_level, blue_level, debug_mode)
            
            # 7. エンハンス処理用にPIL画像に再変換
            processed = Image.fromarray((np.clip(img_array * 255.0, 0, 255)).astype(np.uint8))
            
            # 8. 彩度を適用
            if debug_mode:
                print(f"Step 8: Applying saturation: {saturation}")
            enhancer = ImageEnhance.Color(processed)
            processed = enhancer.enhance(saturation)
            
            # 9. 最終的なコントラストを適用
            if debug_mode:
                print("Step 9: Applying final contrast boost (1.3)")
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.3)
            
            # 10. テンソルに再変換
            result = pil2tensor(processed, image.shape)
            
            if debug_mode:
                print(f"Output tensor shape: {result.shape}, dtype: {result.dtype}")
                print("=== Processing complete ===\n")
            
            return (result,)
            
        except Exception as e:
            print(f"Error in apply_lighting_and_color: {str(e)}")
            import traceback
            traceback.print_exc()
            return (image,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "FluxLightingAndColor": FluxLightingAndColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLightingAndColor": "Flux Lighting & Color"
}
