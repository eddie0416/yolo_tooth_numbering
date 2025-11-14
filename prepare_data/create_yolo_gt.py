import numpy as np
from PIL import Image
from collections import Counter
import os
from color_utils import FDI2color
import glob
from pathlib import Path

def create_yolo_annotation_from_mask(image_path, output_dir=None, pixel_threshold=100):
    """
    從色塊 mask 圖片生成 YOLO 格式的標註檔案
    
    Parameters:
    image_path: mask 圖片路徑
    output_dir: 輸出目錄（若為 None，則與圖片同目錄）
    pixel_threshold: 像素數量閾值，低於此值視為 outlier
    """
    # 讀取圖片
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # 定義要忽略的顏色
    ignore_colors = {
        (0, 0, 0),      # 背景
        (125, 125, 125) # 牙齦
    }
    
    # 統計所有顏色
    if len(img_array.shape) == 3:
        pixels = img_array.reshape(-1, img_array.shape[-1])
        pixel_tuples = [tuple(pixel) for pixel in pixels]
    else:
        pixels = img_array.reshape(-1)
        pixel_tuples = pixels.tolist()
    
    color_counter = Counter(pixel_tuples)
    
    # 過濾顏色：移除 outlier、背景和牙齦
    valid_colors = []
    for color, count in color_counter.items():
        if count >= pixel_threshold and color not in ignore_colors:
            valid_colors.append(color)
    
    print(f"總共找到 {len(valid_colors)} 個有效色塊（牙齒）")
    
    # 為每個有效顏色找出邊界框
    annotations = []
    
    RGB2FDI = {v[2]: k for k, v in FDI2color.items()} #原本是21: ("aaff7f", "UL1", (170, 255, 127)), 轉換成 (170, 255, 127):(21,UL1)

    for idx, color in enumerate(valid_colors):
        # 創建該顏色的 mask
        if len(img_array.shape) == 3:
            color_mask = np.all(img_array == color, axis=-1)
        else:
            color_mask = img_array == color
        
        # 找出該顏色的所有像素位置
        rows, cols = np.where(color_mask)
        
        if len(rows) == 0:
            continue
        
        # 計算邊界框
        x_min = int(np.min(cols))
        x_max = int(np.max(cols))
        y_min = int(np.min(rows))
        y_max = int(np.max(rows))
        
        # 轉換為 YOLO 格式 (class_id, x_center, y_center, width, height)
        # 所有值都需要歸一化到 [0, 1]
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height
        
        # class_id 使用 0（所有牙齒都是同一類別）
        class_id = RGB2FDI.get(color, -1)
        if class_id == -1:
            continue
        
        annotations.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': bbox_width,
            'height': bbox_height,
            'color': color,
            'pixel_count': color_counter[color],
            'bbox_pixel': (x_min, y_min, x_max, y_max)
        })
        
        print(f"色塊 {idx+1}: RGB{color} | 像素數: {color_counter[color]:>6} | 邊界: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
    
    # 決定輸出目錄
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # 生成輸出檔名（去掉 _label 後綴，如果有的話）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if base_name.endswith('_label'):
        base_name = base_name[:-6]  # 移除 '_label'
    
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    # 寫入 YOLO 格式標註檔
    with open(output_file, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
    
    print(f"\n標註檔案已儲存至: {output_file}")
    print(f"共標註 {len(annotations)} 個物件（牙齒）")
    
    return annotations, output_file
'''
image_path = r'ply\00OMSZGW_lower\00OMSZGW_lower_label.png'
output_dir = 'yolo_numbering_dataset/dataset/labels'

annotations, txt_file = create_yolo_annotation_from_mask(image_path, output_dir)
'''
mask_dir = Path('yolo_numbering_dataset/render_mask')
output_dir = 'yolo_numbering_dataset/dataset/labels'

# 找出所有 _label.png 檔案
label_images = list(mask_dir.glob('**/*_label.png'))

print(f"找到 {len(label_images)} 個 _label.png 檔案")

for image_path in label_images:
    print(f"處理: {image_path}")
    annotations, txt_file = create_yolo_annotation_from_mask(str(image_path), output_dir)
    print(f"已產生標註檔: {txt_file}")