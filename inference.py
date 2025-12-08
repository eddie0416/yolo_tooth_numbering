from ultralytics import YOLO

# 載入模型
model = YOLO("runs/detect/s_using_m.autotune/weights/best.pt")

# 四張影像路徑

img_paths = [
    "uninference_tooth/00240433UpperJaw_neutral.png",
    "uninference_tooth/2025_1126_upper_neutral.png",
    "uninference_tooth/00233324_neutral.png",
    "uninference_tooth/00240433LowerJaw_neutral.png",
]
'''
img_paths = [
    "yolo_numbering_dataset/dataset_splited/images/val/Z5SBGG6H_upper.png",
    "yolo_numbering_dataset/dataset_splited/images/val/YN26T284_upper.png",
    "yolo_numbering_dataset/dataset_splited/images/val/YNKZHRP0_lower.png",
    "yolo_numbering_dataset/dataset_splited/images/val/OJKNS9DO_upper.png",
]
'''

# 逐張推論
for img_path in img_paths:
    print(f"\n=== Inference on: {img_path} ===")
    
    results = model(img_path)  # results 是 Results 物件列表
    
    for r in results:
        print("boxes:", r.boxes)   # 預測框資訊
        print("probs:", r.probs)   # 類別機率（如果你的模型有 classification head 才會有）
        
        # 存推論結果（預設會存到 runs/ 之類的資料夾）
        r.save()
