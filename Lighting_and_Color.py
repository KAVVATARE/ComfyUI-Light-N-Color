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
    BHWC形状で値が[0, 1]の範囲のテンソルを返す。
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
    FluxLightingAndColor - Enhanced Version
    画像の照明と色調を自動解析し調整するためのノードクラス
    
    新機能:
    - 画像解析による自動パラメーター補完
    - 色温度分析
    - 輝度分布解析
    - 彩度解析
    - カスタマイズ可能な自動調整
    """
    
    @classmethod
    def INPUT_TYPES(s):
        """
        入力パラメータの定義
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_analysis": ("BOOLEAN", {"default": True}),
                "analysis_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
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
                "auto_brightness": ("BOOLEAN", {"default": True}),
                "auto_color_balance": ("BOOLEAN", {"default": True}),
                "auto_saturation": ("BOOLEAN", {"default": True}),
                "auto_contrast": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "analysis_report")
    FUNCTION = "apply_lighting_and_color"
    CATEGORY = "image/adjustments"

    def analyze_image_characteristics(self, img_array, debug=False):
        """
        画像の特性を解析し、最適な調整パラメーターを算出する
        
        Returns:
        - dict: 解析結果と推奨パラメーター
        """
        analysis = {}
        
        # 輝度分析
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # 基本統計
        analysis['mean_luminance'] = float(np.mean(luminance))
        analysis['std_luminance'] = float(np.std(luminance))
        analysis['min_luminance'] = float(np.min(luminance))
        analysis['max_luminance'] = float(np.max(luminance))
        
        # ヒストグラム分析
        hist, bins = np.histogram(luminance.flatten(), bins=256, range=(0, 1))
        analysis['histogram'] = hist
        
        # 色温度分析（簡易版）
        mean_r = float(np.mean(img_array[:,:,0]))
        mean_g = float(np.mean(img_array[:,:,1]))
        mean_b = float(np.mean(img_array[:,:,2]))
        
        analysis['mean_rgb'] = [mean_r, mean_g, mean_b]
        
        # 色温度推定 (青が強い = 冷たい、赤が強い = 暖かい)
        if mean_b > mean_r:
            analysis['color_temperature'] = 'cool'
            analysis['temp_bias'] = mean_b - mean_r
        else:
            analysis['color_temperature'] = 'warm'
            analysis['temp_bias'] = mean_r - mean_b
            
        # 彩度分析
        hsv_array = cv.cvtColor((img_array * 255).astype(np.uint8), cv.COLOR_RGB2HSV)
        saturation_values = hsv_array[:,:,1] / 255.0
        analysis['mean_saturation'] = float(np.mean(saturation_values))
        analysis['std_saturation'] = float(np.std(saturation_values))
        
        # コントラスト分析
        analysis['contrast_ratio'] = analysis['max_luminance'] - analysis['min_luminance']
        
        # 推奨パラメーター計算
        recommendations = self.calculate_recommendations(analysis, debug)
        analysis['recommendations'] = recommendations
        
        if debug:
            print("\n=== Image Analysis Results ===")
            print(f"Mean Luminance: {analysis['mean_luminance']:.3f}")
            print(f"Luminance Range: {analysis['min_luminance']:.3f} - {analysis['max_luminance']:.3f}")
            print(f"Color Temperature: {analysis['color_temperature']} (bias: {analysis['temp_bias']:.3f})")
            print(f"Mean Saturation: {analysis['mean_saturation']:.3f}")
            print(f"Contrast Ratio: {analysis['contrast_ratio']:.3f}")
            print("Recommendations:", recommendations)
            print("===============================\n")
        
        return analysis

    def calculate_recommendations(self, analysis, debug=False):
        """
        解析結果に基づいて推奨パラメーターを計算する
        """
        recommendations = {}
        
        # 明るさ調整の推奨値
        target_luminance = 0.5
        current_luminance = analysis['mean_luminance']
        
        if current_luminance < 0.3:  # 暗い画像
            recommendations['brightness'] = 1.2 + (0.3 - current_luminance) * 2
        elif current_luminance > 0.7:  # 明るい画像
            recommendations['brightness'] = 0.8 + (0.7 - current_luminance) * 0.5
        else:
            recommendations['brightness'] = 1.0
            
        # コントラスト調整
        if analysis['contrast_ratio'] < 0.5:  # 低コントラスト
            recommendations['black_level'] = max(0, analysis['min_luminance'] - 0.05)
            recommendations['white_level'] = min(1, analysis['max_luminance'] + 0.05)
        else:
            recommendations['black_level'] = 0.0
            recommendations['white_level'] = 1.0
            
        recommendations['mid_level'] = 0.5
        
        # 色温度補正
        temp_bias = analysis['temp_bias']
        if analysis['color_temperature'] == 'cool' and temp_bias > 0.05:
            # 冷たすぎる場合、赤を強化、青を減少
            recommendations['red_level'] = 1.0 + min(temp_bias * 2, 0.3)
            recommendations['green_level'] = 1.0
            recommendations['blue_level'] = 1.0 - min(temp_bias * 1.5, 0.2)
        elif analysis['color_temperature'] == 'warm' and temp_bias > 0.05:
            # 暖かすぎる場合、青を強化、赤を減少
            recommendations['red_level'] = 1.0 - min(temp_bias * 1.5, 0.2)
            recommendations['green_level'] = 1.0
            recommendations['blue_level'] = 1.0 + min(temp_bias * 2, 0.3)
        else:
            recommendations['red_level'] = 1.0
            recommendations['green_level'] = 1.0
            recommendations['blue_level'] = 1.0
            
        # 彩度調整
        mean_sat = analysis['mean_saturation']
        if mean_sat < 0.4:  # 彩度が低い
            recommendations['saturation'] = 1.0 + (0.4 - mean_sat) * 2
        elif mean_sat > 0.8:  # 彩度が高い
            recommendations['saturation'] = 1.0 - (mean_sat - 0.8) * 1.5
        else:
            recommendations['saturation'] = 1.0
            
        # 値を適切な範囲にクリップ
        recommendations['brightness'] = np.clip(recommendations['brightness'], 0.0, 2.0)
        recommendations['saturation'] = np.clip(recommendations['saturation'], 0.0, 2.0)
        recommendations['red_level'] = np.clip(recommendations['red_level'], 0.0, 2.0)
        recommendations['green_level'] = np.clip(recommendations['green_level'], 0.0, 2.0)
        recommendations['blue_level'] = np.clip(recommendations['blue_level'], 0.0, 2.0)
        
        return recommendations

    def blend_parameters(self, user_params, recommendations, strength, auto_flags, debug=False):
        """
        ユーザー設定と推奨値をブレンドする
        
        Parameters:
        - user_params: ユーザーが設定したパラメーター
        - recommendations: 解析による推奨パラメーター
        - strength: 自動調整の強度 (0.0-1.0)
        - auto_flags: 各調整項目の自動有効フラグ
        """
        blended = {}
        
        param_mapping = {
            'brightness': 'auto_brightness',
            'saturation': 'auto_saturation', 
            'red_level': 'auto_color_balance',
            'green_level': 'auto_color_balance',
            'blue_level': 'auto_color_balance',
            'black_level': 'auto_contrast',
            'white_level': 'auto_contrast',
            'mid_level': 'auto_contrast'
        }
        
        for param_name in ['brightness', 'saturation', 'red_level', 'green_level', 'blue_level', 
                          'black_level', 'white_level', 'mid_level']:
            
            user_value = user_params.get(param_name, 1.0)
            recommended_value = recommendations.get(param_name, user_value)
            auto_flag_name = param_mapping.get(param_name, 'auto_brightness')
            
            if auto_flags.get(auto_flag_name, True):
                # 自動調整が有効な場合、ユーザー値と推奨値をブレンド
                blended[param_name] = user_value * (1 - strength) + recommended_value * strength
            else:
                # 自動調整が無効な場合、ユーザー値をそのまま使用
                blended[param_name] = user_value
                
        if debug:
            print("\n=== Parameter Blending ===")
            for param_name in blended:
                user_val = user_params.get(param_name, 1.0)
                rec_val = recommendations.get(param_name, user_val)
                final_val = blended[param_name]
                auto_flag = param_mapping.get(param_name, 'auto_brightness')
                enabled = auto_flags.get(auto_flag, True)
                print(f"{param_name}: User={user_val:.3f}, Rec={rec_val:.3f}, Final={final_val:.3f} (Auto: {enabled})")
            print("=========================\n")
                
        return blended

    def generate_analysis_report(self, analysis, blended_params, user_params):
        """
        解析結果のレポートを生成する
        """
        report = []
        report.append("=== Image Analysis Report ===")
        report.append(f"Mean Luminance: {analysis['mean_luminance']:.3f}")
        report.append(f"Luminance Range: {analysis['min_luminance']:.3f} - {analysis['max_luminance']:.3f}")
        report.append(f"Color Temperature: {analysis['color_temperature']} (bias: {analysis['temp_bias']:.3f})")
        report.append(f"Mean Saturation: {analysis['mean_saturation']:.3f}")
        report.append(f"Contrast Ratio: {analysis['contrast_ratio']:.3f}")
        report.append("")
        report.append("=== Applied Adjustments ===")
        
        for param_name in ['brightness', 'saturation', 'red_level', 'green_level', 'blue_level']:
            user_val = user_params.get(param_name, 1.0)
            final_val = blended_params.get(param_name, user_val)
            change = final_val - user_val
            if abs(change) > 0.01:
                sign = "+" if change > 0 else ""
                report.append(f"{param_name}: {user_val:.3f} → {final_val:.3f} ({sign}{change:.3f})")
            else:
                report.append(f"{param_name}: {final_val:.3f} (no change)")
                
        report.append("=============================")
        
        return "\n".join(report)

    def apply_dof(self, img, depth_map=None, mode='none', radius=8, samples=1, debug=False):
        """
        被写界深度(DoF)エフェクトを適用する
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
        """
        if debug:
            print(f"Levels: Adjusting (black: {black_level:.3f}, mid: {mid_level:.3f}, white: {white_level:.3f})")
        
        # Apply level adjustments
        img_array = (img_array - black_level) / (white_level - black_level)
        img_array = np.clip(img_array, 0, 1)
        
        # Apply gamma correction based on mid level
        if mid_level != 0.5:
            gamma = np.log(0.5) / np.log(mid_level + 1e-8)  # avoid division by zero
            img_array = np.power(img_array, gamma)
            
        return img_array

    def adjust_channels(self, img_array, red_level, green_level, blue_level, debug=False):
        """
        RGB各チャンネルの強度を調整する
        """
        if debug:
            print(f"Channels: Adjusting (R: {red_level:.3f}, G: {green_level:.3f}, B: {blue_level:.3f})")
        
        # Split and adjust each channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * red_level, 0, 1)
        img_array[:,:,1] = np.clip(img_array[:,:,1] * green_level, 0, 1)
        img_array[:,:,2] = np.clip(img_array[:,:,2] * blue_level, 0, 1)
        return img_array

    def apply_lighting_and_color(self, image, auto_analysis, analysis_strength,
                               black_level, mid_level, white_level, 
                               red_level, green_level, blue_level, brightness, saturation,
                               depth=None, dof_mode="none", dof_radius=8, dof_samples=1,
                               auto_brightness=True, auto_color_balance=True, 
                               auto_saturation=True, auto_contrast=True, debug_mode=False):
        """
        メインの処理関数。画像解析と自動調整機能を含む。
        """
        try:
            if debug_mode:
                print("\n=== FluxLightingAndColor Enhanced Starting ===")
                print(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
                print(f"Auto analysis: {auto_analysis}, Strength: {analysis_strength}")
            
            # 1. 入力テンソルをPIL画像に変換
            img_pil = tensor2pil(image)
            if debug_mode:
                print("Step 1: Converted input tensor to PIL image")
            
            # 2. 画像解析（自動調整が有効な場合）
            analysis_report = "No analysis performed"
            final_params = {
                'black_level': black_level,
                'mid_level': mid_level, 
                'white_level': white_level,
                'red_level': red_level,
                'green_level': green_level,
                'blue_level': blue_level,
                'brightness': brightness,
                'saturation': saturation
            }
            
            if auto_analysis:
                img_array_for_analysis = np.array(img_pil).astype(float) / 255.0
                analysis = self.analyze_image_characteristics(img_array_for_analysis, debug_mode)
                
                user_params = {
                    'black_level': black_level,
                    'mid_level': mid_level,
                    'white_level': white_level, 
                    'red_level': red_level,
                    'green_level': green_level,
                    'blue_level': blue_level,
                    'brightness': brightness,
                    'saturation': saturation
                }
                
                auto_flags = {
                    'auto_brightness': auto_brightness,
                    'auto_color_balance': auto_color_balance,
                    'auto_saturation': auto_saturation,
                    'auto_contrast': auto_contrast
                }
                
                final_params = self.blend_parameters(
                    user_params, analysis['recommendations'], 
                    analysis_strength, auto_flags, debug_mode
                )
                
                analysis_report = self.generate_analysis_report(analysis, final_params, user_params)
                
                if debug_mode:
                    print("Step 2: Completed image analysis and parameter blending")
            
            # 3. 深度マップがある場合はDoFを適用
            if depth is not None and dof_mode != "none":
                depth_pil = tensor2pil(depth)
                img_pil = self.apply_dof(img_pil, depth_pil, dof_mode, dof_radius, dof_samples, debug_mode)
                if debug_mode:
                    print("Step 3: Applied depth of field effect")
            
            # 4. 処理用にnumpy配列に変換
            img_array = np.array(img_pil).astype(float) / 255.0
            
            # 5. 明るさ調整を適用
            if debug_mode:
                print(f"Step 5: Applying brightness: {final_params['brightness']:.3f}")
            img_array = np.power(img_array, 0.7) * final_params['brightness']
            
            # 6. レベル調整を適用
            img_array = self.adjust_levels(
                img_array, 
                final_params['black_level'], 
                final_params['mid_level'], 
                final_params['white_level'], 
                debug_mode
            )
            
            # 7. チャンネル調整を適用
            img_array = self.adjust_channels(
                img_array, 
                final_params['red_level'], 
                final_params['green_level'], 
                final_params['blue_level'], 
                debug_mode
            )
            
            # 8. エンハンス処理用にPIL画像に再変換
            processed = Image.fromarray((np.clip(img_array * 255.0, 0, 255)).astype(np.uint8))
            
            # 9. 彩度を適用
            if debug_mode:
                print(f"Step 9: Applying saturation: {final_params['saturation']:.3f}")
            enhancer = ImageEnhance.Color(processed)
            processed = enhancer.enhance(final_params['saturation'])
            
            # 10. 最終的なコントラストを適用
            contrast_boost = 1.3
            if auto_analysis and auto_contrast:
                # 既にコントラストが高い場合は控えめに
                if analysis['contrast_ratio'] > 0.7:
                    contrast_boost = 1.1
                    
            if debug_mode:
                print(f"Step 10: Applying final contrast boost: {contrast_boost}")
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(contrast_boost)
            
            # 11. テンソルに再変換
            result = pil2tensor(processed, image.shape)
            
            if debug_mode:
                print(f"Output tensor shape: {result.shape}, dtype: {result.dtype}")
                print("=== Enhanced Processing Complete ===\n")
            
            return (result, analysis_report)
            
        except Exception as e:
            print(f"Error in apply_lighting_and_color: {str(e)}")
            import traceback
            traceback.print_exc()
            return (image, f"Error occurred: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "FluxLightingAndColor": FluxLightingAndColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLightingAndColor": "Flux Lighting & Color (Enhanced)"
}
