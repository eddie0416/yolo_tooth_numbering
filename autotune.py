from ultralytics import YOLO

model = YOLO("yolo11m.pt")
#model = YOLO("runs/detect/l.pt_epoch500_1024/weights/best.pt")

model.tune(
    data="yolo_numbering_dataset/dataset_splited/data.yaml",
    epochs=40,
    iterations=150,        # 再多跑 150 組
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
    amp=True,
    cache=True,
    imgsz=1024,
    batch=-1,
    workers=8
)