from ultralytics import YOLO

# 載入模型
model = YOLO("runs/detect/m.pt_new_autotune_closeflip/weights/best.pt")

# 指定要推論的圖片路徑
img_path = "00240433LowerJaw_neutral.png"

# 執行推論
results = model(img_path)  # results 是 Results 物件列表

# 可以列印結果資訊
for r in results:
    print(r.boxes)           # 預測框資訊
    print(r.probs)           # 類別機率
    r.save()                 # 顯示帶框圖片（可選）
