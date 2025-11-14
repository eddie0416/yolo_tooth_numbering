import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_yolo_annotations(image_path, annotation_path, output_path=None, show_labels=True):
    """
    將 YOLO 標註疊合到原圖上進行視覺化
    
    Parameters:
    image_path: 原始圖片路徑（可以是 label mask 或對應的原圖）
    annotation_path: YOLO 標註檔路徑 (.txt)
    output_path: 輸出圖片路徑（若為 None，則自動生成）
    show_labels: 是否顯示標籤編號
    """
    # 讀取圖片
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # 創建可繪圖的副本
    draw = ImageDraw.Draw(img)
    
    # 讀取 YOLO 標註
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                annotations.append((class_id, x_center, y_center, bbox_width, bbox_height))
    
    print(f"載入了 {len(annotations)} 個標註框")
    
    # 定義顏色（用於繪製不同的框）
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080',
        '#FF6347', '#4682B4', '#32CD32', '#FFD700'
    ]
    
    # 繪製每個標註框
    for idx, (class_id, x_center, y_center, bbox_width, bbox_height) in enumerate(annotations):
        # 將歸一化座標轉回像素座標
        x_center_px = x_center * width
        y_center_px = y_center * height
        bbox_width_px = bbox_width * width
        bbox_height_px = bbox_height * height
        
        # 計算邊界框的四個角
        x_min = int(x_center_px - bbox_width_px / 2)
        y_min = int(y_center_px - bbox_height_px / 2)
        x_max = int(x_center_px + bbox_width_px / 2)
        y_max = int(y_center_px + bbox_height_px / 2)
        
        # 選擇顏色
        color = colors[idx % len(colors)]
        
        # 繪製矩形框（線寬 3）
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        
        # 繪製標籤編號
        if show_labels:
            label_text = f"#{class_id}"
            # 使用默認字體
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 計算文字位置（放在框的左上角）
            text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 繪製文字背景
            draw.rectangle(
                [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
                fill=color
            )
            # 繪製文字
            draw.text((x_min + 2, y_min - text_height - 2), label_text, fill='white', font=font)
        
        print(f"框 #{idx+1}: class={class_id}, 座標=({x_min}, {y_min}, {x_max}, {y_max})")
    
    # 決定輸出路徑
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
    
    # 儲存結果
    img.save(output_path)
    print(f"\n視覺化結果已儲存至: {output_path}")
    
    return output_path


def visualize_with_transparency(image_path, annotation_path, output_path=None, box_alpha=0.3):
    """
    將 YOLO 標註以半透明填充的方式疊合到原圖上
    
    Parameters:
    image_path: 原始圖片路徑
    annotation_path: YOLO 標註檔路徑 (.txt)
    output_path: 輸出圖片路徑
    box_alpha: 填充透明度 (0-1)
    """
    # 讀取圖片
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    
    # 創建透明圖層
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_outline = ImageDraw.Draw(img)
    
    # 讀取標註
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                annotations.append((class_id, x_center, y_center, bbox_width, bbox_height))
    
    print(f"載入了 {len(annotations)} 個標註框")
    
    # 定義顏色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 165, 0), (128, 0, 128), (255, 192, 203), (165, 42, 42), (128, 128, 128)
    ]
    
    # 繪製標註
    for idx, (class_id, x_center, y_center, bbox_width, bbox_height) in enumerate(annotations):
        x_center_px = x_center * width
        y_center_px = y_center * height
        bbox_width_px = bbox_width * width
        bbox_height_px = bbox_height * height
        
        x_min = int(x_center_px - bbox_width_px / 2)
        y_min = int(y_center_px - bbox_height_px / 2)
        x_max = int(x_center_px + bbox_width_px / 2)
        y_max = int(y_center_px + bbox_height_px / 2)
        
        color = colors[idx % len(colors)]
        
        # 半透明填充
        fill_color = color + (int(255 * box_alpha),)
        draw_overlay.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)
        
        # 實心邊框
        draw_outline.rectangle([x_min, y_min, x_max, y_max], outline=color + (255,), width=2)
    
    # 合併圖層
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    
    # 決定輸出路徑
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_annotated_filled.png")
    
    img.save(output_path)
    print(f"\n視覺化結果已儲存至: {output_path}")
    
    return output_path

output = visualize_yolo_annotations(
    r'ply\00OMSZGW_lower\00OMSZGW_lower_neutral.png',
    r'ply\00OMSZGW_lower\00OMSZGW_lower.txt'
)