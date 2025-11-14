import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 載入預訓練模型
    model = YOLO('yolo11n.yaml')  # 使用 nano 版本，速度快
    
    # 開始訓練
    '''
    results = model.train(
        data='data.yaml',           # 你的資料集配置檔
        epochs=100,                 # 訓練 100 個週期
        imgsz=640,                  # 輸入圖片大小
        batch=8,                    # 批次大小（根據 GPU 記憶體調整）
        device=0,                   # 使用 GPU 0
        workers=0,                  # 資料載入執行緒數（Linux 遠端建議設 0）
        project='runs/train',       # 儲存結果的資料夾
        name='tooth_detection',     # 實驗名稱
        exist_ok=True,              # 允許覆蓋現有資料夾
        patience=50,                # 早停：50 個 epoch 沒改善就停止
        save=True,                  # 儲存檢查點
        plots=True,                 # 生成訓練圖表
        verbose=True,               # 顯示詳細資訊
        optimizer='SGD',            # 優化器（SGD 或 Adam）
        lr0=0.01,                   # 初始學習率
        weight_decay=0.0005,        # 權重衰減
        warmup_epochs=3,            # 熱身訓練週期數
        close_mosaic=10,            # 最後 10 個 epoch 關閉 mosaic 增強
    )
    '''
    results = model.train(
        data='yolo_numbering_dataset/dataset_splited/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=50
    )
    
    print("訓練完成！")
    print(f"最佳模型保存在: runs/train/tooth_detection/weights/best.pt")
