import numpy as np
from PIL import Image
from collections import Counter
import os
from color_utils import FDI2color
import glob
from pathlib import Path


# 建立 FDI 到連續 class ID 的映射
def create_fdi_mapping():
    """
    建立 FDI 編號（11-48）到連續 class ID（0-31）的映射
    """
    fdi_numbers = []
    # FDI 編號順序：11-18, 21-28, 31-38, 41-48
    for quadrant in [1, 2, 3, 4]:
        for tooth in range(1, 9):
            fdi_numbers.append(quadrant * 10 + tooth)
    
    # 建立雙向映射
    fdi_to_classid = {fdi: idx for idx, fdi in enumerate(fdi_numbers)}
    classid_to_fdi = {idx: fdi for idx, fdi in enumerate(fdi_numbers)}
    
    return fdi_to_classid, classid_to_fdi

# 全局映射字典
FDI_TO_CLASSID, CLASSID_TO_FDI = create_fdi_mapping()


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
    
    RGB2FDI = {v[2]: k for k, v in FDI2color.items()}  # (R,G,B) -> FDI number

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
        
        # 從 RGB 取得 FDI 編號，再映射到連續的 class_id (0-31)
        fdi_number = RGB2FDI.get(color, -1)
        if fdi_number == -1:
            print(f"警告：找不到顏色 {color} 對應的 FDI 編號，跳過")
            continue
        
        class_id = FDI_TO_CLASSID.get(fdi_number, -1)
        if class_id == -1:
            print(f"警告：FDI 編號 {fdi_number} 無效，跳過")
            continue
        
        annotations.append({
            'class_id': class_id,
            'fdi_number': fdi_number,
            'x_center': x_center,
            'y_center': y_center,
            'width': bbox_width,
            'height': bbox_height,
            'color': color,
            'pixel_count': color_counter[color],
            'bbox_pixel': (x_min, y_min, x_max, y_max)
        })
        
        print(f"色塊 {idx+1}: RGB{color} | FDI={fdi_number} -> ClassID={class_id} | 像素數: {color_counter[color]:>6} | 邊界: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
    
    # 決定輸出目錄
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
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


# 生成 data.yaml 的輔助函數
def generate_data_yaml(output_path='yolo_numbering_dataset/dataset/data.yaml'):
    """
    生成 YOLO 訓練用的 data.yaml 檔案
    """
    # 按照 FDI 順序生成類別名稱
    class_names = [f'FDI_{CLASSID_TO_FDI[i]}' for i in range(32)]
    
    yaml_content = f"""# Tooth Numbering Dataset
path: ../dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 32  # number of classes
names: {class_names}

# FDI Mapping (for reference)
# ClassID 0-7:   FDI 11-18 (Upper Right)
# ClassID 8-15:  FDI 21-28 (Upper Left)
# ClassID 16-23: FDI 31-38 (Lower Left)
# ClassID 24-31: FDI 41-48 (Lower Right)
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"data.yaml 已生成至: {output_path}")
    print(f"nc=32，類別名稱: {class_names[:4]}...{class_names[-4:]}")


if __name__ == "__main__":
    mask_dir = Path('yolo_numbering_dataset/render_mask')
    output_dir = 'yolo_numbering_dataset/labels'
    
    # 找出所有 _label.png 檔案
    label_images = list(mask_dir.glob('**/*_label.png'))
    
    print(f"找到 {len(label_images)} 個 _label.png 檔案\n")
    print(f"FDI 到 ClassID 映射範例：")
    print(f"  FDI 11-18 -> ClassID 0-7")
    print(f"  FDI 21-28 -> ClassID 8-15")
    print(f"  FDI 31-38 -> ClassID 16-23")
    print(f"  FDI 41-48 -> ClassID 24-31\n")
    
    for image_path in label_images:
        print(f"\n{'='*60}")
        print(f"處理: {image_path.name}")
        print(f"{'='*60}")
        annotations, txt_file = create_yolo_annotation_from_mask(str(image_path), output_dir)
        print(f"已產生標註檔: {txt_file}")
    
    # 生成 data.yaml
    #print(f"\n{'='*60}")
    #generate_data_yaml()
