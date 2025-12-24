import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import yaml  # 新增

def visualize_yolo_annotations(image_path, annotation_path, data_yaml,
                               output_path=None, show_labels=True):
    """
    將 YOLO 標註疊合到原圖上進行視覺化 (顯示 class name)
    """
    # 讀 data.yaml 取得 class names
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", None)
    if class_names is None:
        raise ValueError(f"`names` not found in {data_yaml}")
    # names 可能是 dict 或 list，統一成 list
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    # 讀取圖片
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
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

    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080',
        '#FF6347', '#4682B4', '#32CD32', '#FFD700'
    ]

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
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        if show_labels:
            # 用 class name 取代 class_id
            if 0 <= class_id < len(class_names):
                label_text = str(class_names[class_id])
            else:
                label_text = f"unknown({class_id})"

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
            except Exception as e:
                print("字體載入失敗，改用預設字體:", e)
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.rectangle(
                [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
                fill=color
            )
            draw.text((x_min + 2, y_min - text_height - 2), label_text, fill='white', font=font)

        print(f"框 #{idx+1}: class={class_id}({label_text}), 座標=({x_min}, {y_min}, {x_max}, {y_max})")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_annotated.png")

    img.save(output_path)
    print(f"\n視覺化結果已儲存至: {output_path}")
    return output_path


output = visualize_yolo_annotations(
    'yolo_numbering_dataset/dataset_splited_augmented/images/val/Z5SBGG6H_upper.png',
    'yolo_numbering_dataset/dataset_splited_augmented/labels/val/Z5SBGG6H_upper.txt',
    'yolo_numbering_dataset/dataset_splited_augmented/data.yaml',
    'Z5SBGG6H_upper_annotated.png'
)
