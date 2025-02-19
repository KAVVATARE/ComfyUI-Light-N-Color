# ComfyUI-Light-N-Color
ComfyUI node that adds Brightness, RGB channels, and Depth of Field to AI-generated image


# スクリプトの内容：
このスクリプトは、AI生成画像に対して明るさ、RGBチャンネル、被写界深度(DoF)を調整するためのComfyUIノードを提供します。
主に以下の機能を持っています：

テンソルとPIL画像の変換:

tensor2pil: PyTorchテンソルをPIL画像に変換
pil2tensor: PIL画像をPyTorchテンソルに変換

画像処理機能:
medianFilter: 高品質なメディアンフィルタを適用
apply_dof: 被写界深度エフェクトを適用
adjust_levels: 画像のレベル調整
adjust_channels: RGB各チャンネルの強度を調整
FluxLightingAndColorクラス:

パラメータの設定例
params = {
    "image": input_tensor,
    "black_level": 0.1,
    "mid_level": 0.5,
    "white_level": 1.0,
    "red_level": 1.2,
    "green_level": 0.9,
    "blue_level": 1.1,
    "brightness": 1.1,
    "saturation": 1.3,
    "depth": None,  # 深度マップがない場合
    "dof_mode": "none",
    "dof_radius": 8,
    "dof_samples": 1,
    "debug_mode": True  # デバッグ出力を有効にする
}
'''
