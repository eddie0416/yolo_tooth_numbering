import os
import shutil
from sklearn.model_selection import train_test_split


def split_train_val(images_dir, labels_dir, output_dir, val_ratio=0.2):
    """
    將資料集切分為 train/val
    複製圖片時去掉 '_neutral' 後綴，使其與標註檔名稱匹配
    """
    # 取得所有圖片
    images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(f"找到 {len(images)} 張圖片")

    # 切分訓練集和驗證集
    train_images, val_images = train_test_split(images, test_size=val_ratio, random_state=42)

    print(f"訓練集: {len(train_images)} 張 ({len(train_images)/len(images)*100:.1f}%)")
    print(f"驗證集: {len(val_images)} 張 ({len(val_images)/len(images)*100:.1f}%)")

    # 建立目錄並複製檔案
    for split_name, split_images in [('train', train_images), ('val', val_images)]:
        img_dir = os.path.join(output_dir, 'images', split_name)
        lbl_dir = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        missing_labels = []
        copied_count = 0
        
        for img_name in split_images:
            # 處理圖片檔名：去掉 '_neutral'
            # 原始：7PQAZ8X1_lower_neutral.png
            # 目標：7PQAZ8X1_lower.png
            name_parts = os.path.splitext(img_name)  # ('7PQAZ8X1_lower_neutral', '.png')
            base_name = name_parts[0]  # '7PQAZ8X1_lower_neutral'  ← 修正：加 [0]
            extension = name_parts[1]  # '.png'  ← 修正：加 [1]
            
            new_base_name = base_name.replace('_neutral', '')  # '7PQAZ8X1_lower'
            new_img_name = new_base_name + extension  # '7PQAZ8X1_lower.png'
            
            # 複製圖片（重新命名）
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(img_dir, new_img_name)
            shutil.copy2(src_img, dst_img)
            copied_count += 1
            
            # 標註檔案名稱（與新的圖片名稱對應）
            label_name = new_base_name + '.txt'  # '7PQAZ8X1_lower.txt'
            
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(lbl_dir, label_name)
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                missing_labels.append((img_name, new_img_name, label_name))
                print(f"警告: 找不到標註檔案 {label_name} (原圖片 {img_name} -> 新圖片 {new_img_name})")
        
        print(f"完成 {split_name} 集的複製: {copied_count} 張圖片")
        if missing_labels:
            print(f"  警告: {split_name} 集中有 {len(missing_labels)} 個圖片缺少對應的標註檔")

    print(f"\n資料集切分完成！輸出目錄: {output_dir}")


if __name__ == '__main__':
    split_train_val(
        images_dir='/home/q56144107/yolo_tooth_numbering/yolo_numbering_dataset/images',
        labels_dir='/home/q56144107/yolo_tooth_numbering/yolo_numbering_dataset/labels',
        output_dir='/home/q56144107/yolo_tooth_numbering/yolo_numbering_dataset/dataset_splited',
        val_ratio=0.2
    )
