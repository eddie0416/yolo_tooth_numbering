from ultralytics import YOLO

model = YOLO("yolo11m.pt")
#model = YOLO("runs/detect/l.pt_epoch500_1024/weights/best.pt")

model.tune(
    data="yolo_numbering_dataset/dataset_splited/data.yaml",
    epochs=30,          # 每個 trial 訓練多少 epoch（通常 20~50 就夠）
    iterations=50,     # 要嘗試多少組超參數（越大越好，但越花時間）
    optimizer="AdamW",  # 可以試和預設不同的 optimizer
    plots=False,        # 調參階段通常關掉作圖加速
    save=False,         # 不必存下每個 trial 的權重
    val=False,           # 可選：只在最後幾個 epoch 或內部策略驗證，加速搜尋
    amp=True,
    cache=True,
    imgsz=1024,
    batch=-1,  
    workers=8
)