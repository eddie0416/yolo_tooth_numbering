'''
在val_set中找出異常sample

'''
import numpy as np
import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from post_process import run_viterbi_alignment

'''
def get_tooth_sequences(model, img_path, label_path=None):
    """
    輸入影像與 Label路徑，回傳經過「空間排序」且「轉換為FDI數值」的序列。
    
    Args:
        model: YOLO 模型物件
        img_path: 影像檔案路徑
        label_path: (Optional) GT label txt 的路徑
        
    Returns:
        dict: {
            "pred_seq": [38, 37, ...],      # 預測的 FDI 序列 (int)
            "gt_seq": [38, 37, ...],        # GT 的 FDI 序列 (int)
            "pred_conf": [0.95, 0.88...],   # 對應的信心度
            "jaw_type": "upper" or "lower", # 根據預測判斷
            "status": "ok" or "error"
        }
    """

    # 1. 定義類別映射 (根據你的 data.yaml)
    # 格式: class_id -> fdi_int
    # 0->11, ..., 7->18, 8->21, ..., 15->28 (上顎 0-15)
    # 16->31, ..., 23->38, 24->41, ..., 31->48 (下顎 16-31)
    
    FDI_MAP = {}
    # 構建 Mapping 表
    raw_names = {
        0: 'fdi_11', 1: 'fdi_12', 2: 'fdi_13', 3: 'fdi_14', 4: 'fdi_15', 5: 'fdi_16', 6: 'fdi_17', 7: 'fdi_18',
        8: 'fdi_21', 9: 'fdi_22', 10: 'fdi_23', 11: 'fdi_24', 12: 'fdi_25', 13: 'fdi_26', 14: 'fdi_27', 15: 'fdi_28',
        16: 'fdi_31', 17: 'fdi_32', 18: 'fdi_33', 19: 'fdi_34', 20: 'fdi_35', 21: 'fdi_36', 22: 'fdi_37', 23: 'fdi_38',
        24: 'fdi_41', 25: 'fdi_42', 26: 'fdi_43', 27: 'fdi_44', 28: 'fdi_45', 29: 'fdi_46', 30: 'fdi_47', 31: 'fdi_48'
    }
    
    for k, v in raw_names.items():
        FDI_MAP[k] = int(v.split('_')[1]) # 取出 'fdi_11' -> 11

    # --- 內部 Helper: 幾何排序邏輯 (Pred/GT 共用) ---
    def _sort_spatially(items):
        """
        items: list of [cx, cy, class_id, extra_data...]
        return: sorted list of [class_id, extra_data...]
        """
        if len(items) == 0: return []
        
        # 轉 numpy 方便計算
        points = np.array([x[:2] for x in items]) # 取 cx, cy
        centroid = np.mean(points, axis=0)
        
        # 計算角度
        angles = []
        for i, point in enumerate(points):
            angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
            angles.append((i, angle))
            
        # 按角度排序
        angles_sorted = sorted(angles, key=lambda x: x[1])
        
        # 找最大間隙 (斷點)
        max_gap = 0
        gap_idx = 0
        N = len(angles_sorted)
        if N > 1:
            for i in range(N):
                next_i = (i + 1) % N
                gap = angles_sorted[next_i][1] - angles_sorted[i][1]
                if gap < 0: gap += 2 * np.pi
                if gap > max_gap:
                    max_gap = gap
                    gap_idx = next_i
        
        # 重新排列並輸出
        sorted_result = []
        for i in range(N):
            idx = (gap_idx + i) % N
            original_idx = angles_sorted[idx][0]
            # items[original_idx][2:] 代表回傳 class_id 及其後的資料
            sorted_result.append(items[original_idx][2:]) 
            
        return sorted_result

    # ==========================================
    # A. 執行預測 (Prediction)
    # ==========================================
    # verbose=False 關閉沒必要的 print
    results = model.predict(img_path, classes="soft", agnostic_nms=True, verbose=False)
    
    # 取得原圖尺寸 (為了還原 GT 座標)
    orig_h, orig_w = results[0].orig_shape[:2]
    
    pred_items = [] # 格式: [cx, cy, class_id, conf]
    
    count_upper = 0
    count_lower = 0
    
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        #print(f"DEBUG: box.cls[0] = {box.cls[0]}, type = {type(box.cls[0])}")
        c_id = int(box.cls[0])
        conf = float(max(box.conf[0])) if hasattr(box.conf[0], '__iter__') else float(box.conf[0])
        #conf = float(box.conf)
        
        pred_items.append([cx, cy, c_id, conf])
        
        # 顎別統計 (根據 yaml: 0-15上, 16-31下)
        if c_id >= 16: count_lower += 1
        else: count_upper += 1

    jaw_type = 'upper' if count_upper > count_lower else 'lower'
    
    # 進行排序: sort之後的格式是 [[class_id, conf], [class_id, conf]...]
    sorted_pred_raw = _sort_spatially(pred_items)
    
    # 轉換成單純的 FDI List 和 Conf List
    pred_seq = [FDI_MAP[item[0]] for item in sorted_pred_raw]
    pred_conf = [item[1] for item in sorted_pred_raw]

    # ==========================================
    # B. 處理 GT (Ground Truth)
    # ==========================================
    gt_seq = []
    
    if label_path and os.path.exists(label_path):
        gt_items = [] # 格式: [cx, cy, class_id]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id = int(parts[0])
                    # YOLO 格式是 normalized (0~1)
                    n_cx = float(parts[1])
                    n_cy = float(parts[2])
                    
                    # 關鍵：為了跟 Pred 用同樣的邏輯排序，必須還原成像素座標
                    # (其實不還原只用 normalized 算角度也可以，但為了避免長寬比造成質心偏移，統一座標系最保險)
                    cx = n_cx * orig_w
                    cy = n_cy * orig_h
                    
                    gt_items.append([cx, cy, c_id])
        
        # 進行排序 (使用同一套 Helper)
        sorted_gt_raw = _sort_spatially(gt_items)
        
        # 轉換成 FDI List
        gt_seq = [FDI_MAP[item[0]] for item in sorted_gt_raw]

    return {
        "pred_seq": pred_seq,
        "gt_seq": gt_seq,
        "pred_conf": pred_conf,
        "jaw_type": jaw_type
    }
'''
import numpy as np
import os
from ultralytics import YOLO

def get_tooth_sequences(model, img_path, label_path=None):
    # Mapping 表 (照舊)
    FDI_MAP = {}
    raw_names = {
        0: 'fdi_11', 1: 'fdi_12', 2: 'fdi_13', 3: 'fdi_14', 4: 'fdi_15', 5: 'fdi_16', 6: 'fdi_17', 7: 'fdi_18',
        8: 'fdi_21', 9: 'fdi_22', 10: 'fdi_23', 11: 'fdi_24', 12: 'fdi_25', 13: 'fdi_26', 14: 'fdi_27', 15: 'fdi_28',
        16: 'fdi_31', 17: 'fdi_32', 18: 'fdi_33', 19: 'fdi_34', 20: 'fdi_35', 21: 'fdi_36', 22: 'fdi_37', 23: 'fdi_38',
        24: 'fdi_41', 25: 'fdi_42', 26: 'fdi_43', 27: 'fdi_44', 28: 'fdi_45', 29: 'fdi_46', 30: 'fdi_47', 31: 'fdi_48'
    }
    for k, v in raw_names.items():
        FDI_MAP[k] = int(v.split('_')[1])

    # 內部 Helper: 空間排序 (照舊)
    def _sort_spatially(items):
        if len(items) == 0: return []
        points = np.array([x[:2] for x in items])
        if len(points) == 0: return []
        
        centroid = np.mean(points, axis=0)
        angles = []
        for i, point in enumerate(points):
            angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
            angles.append((i, angle))
            
        angles_sorted = sorted(angles, key=lambda x: x[1])
        
        max_gap = 0
        gap_idx = 0
        N = len(angles_sorted)
        if N > 1:
            for i in range(N):
                next_i = (i + 1) % N
                gap = angles_sorted[next_i][1] - angles_sorted[i][1]
                if gap < 0: gap += 2 * np.pi
                if gap > max_gap:
                    max_gap = gap
                    gap_idx = next_i
        
        sorted_result = []
        for i in range(N):
            idx = (gap_idx + i) % N
            original_idx = angles_sorted[idx][0]
            sorted_result.append(items[original_idx][2:]) 
        return sorted_result

    # 1. 執行預測 (確保使用 soft 並且 agnostic_nms)
    results = model.predict(img_path, classes="soft", agnostic_nms=True, verbose=False) 
    
    orig_h, orig_w = results[0].orig_shape[:2]
    pred_items = []
    
    count_upper = 0
    count_lower = 0
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            
            c_id = int(box.cls[0])
            
            # 取得 32 維機率向量
            raw_conf_tensor = box.conf
            if raw_conf_tensor.numel() > 1:
                prob_vec = raw_conf_tensor.cpu().numpy().flatten()
                conf = float(prob_vec.max())
            else:
                conf = float(box.conf)
                prob_vec = np.zeros(32)
                prob_vec[c_id] = conf

            # 儲存: [cx, cy, class_id, conf, prob_vec]
            pred_items.append([cx, cy, c_id, conf, prob_vec])
            
            # 統計顎別
            if c_id >= 16: count_lower += 1
            else: count_upper += 1

    jaw_type = 'upper' if count_upper > count_lower else 'lower'
    
    # 2. 進行空間排序
    sorted_pred_raw = _sort_spatially(pred_items)
    
    pred_seq = [FDI_MAP[item[0]] for item in sorted_pred_raw]
    pred_conf = [item[1] for item in sorted_pred_raw]
    
    # ========================================================
    # 3. 建構解剖學順序的機率矩陣 (由你的程式碼邏輯整合)
    # ========================================================
    num_teeth = len(sorted_pred_raw)
    num_classes_per_jaw = 16
    prob_matrix = np.zeros((num_teeth, num_classes_per_jaw)) # Shape: (N, 16)
    
    if num_teeth > 0:
        for i, item in enumerate(sorted_pred_raw):
            all_conf = item[2] # 取出 32 維機率向量
            
            if jaw_type == 'upper':
                # 上顎邏輯: FDI 18->11 (Class 7->0) 接 FDI 21->28 (Class 8->15)
                # 你的原始註解寫反了，這裡是處理 Upper Jaw (Class 0-15)
                right_side = all_conf[7::-1]  # indices: 7,6,5,4,3,2,1,0
                left_side = all_conf[8:16]    # indices: 8,9,10,11,12,13,14,15
                prob_matrix[i, :] = np.concatenate([right_side, left_side])
                
            else:
                # 下顎邏輯: FDI 38->31 (Class 23->16) 接 FDI 41->48 (Class 24->31)
                # 你的原始註解寫反了，這裡是處理 Lower Jaw (Class 16-31)
                right_side = all_conf[23:15:-1] # indices: 23,22...16
                left_side = all_conf[24:32]     # indices: 24,25...31
                prob_matrix[i, :] = np.concatenate([right_side, left_side])

    # B. 處理 GT (保持不變)
    gt_seq = []
    if label_path and os.path.exists(label_path):
        gt_items = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id = int(parts[0])
                    n_cx = float(parts[1])
                    n_cy = float(parts[2])
                    cx = n_cx * orig_w
                    cy = n_cy * orig_h
                    gt_items.append([cx, cy, c_id])
        sorted_gt_raw = _sort_spatially(gt_items)
        gt_seq = [FDI_MAP[item[0]] for item in sorted_gt_raw]

    return {
        "pred_seq": pred_seq,
        "gt_seq": gt_seq,
        "prob_matrix": prob_matrix,
        "jaw_type": jaw_type
    }

'''
def get_tooth_sequences(model, img_path, label_path=None):
    return {
        "pred_seq": pred_seq,
        "gt_seq": gt_seq,
        "prob_matrix": prob_matrix,
        "jaw_type": jaw_type
    }
    print pred_seq：[16, 15, 14, 13, 12, 11, 21, 22, 26, 27]

def run_viterbi_alignment(prob_matrix, jaw_type='lower', print_dp_path=True):
    return final_fdi_labels, assigned_indices
    print(final_fdi_labels)：[38, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47]
'''

img_path = 'yolo_numbering_dataset/dataset_splited/images/val/0AAQ6BO3_upper.png'
label_path = 'yolo_numbering_dataset/dataset_splited/labels/val/0AAQ6BO3_upper.txt'
model_path = 'runs/detect/m_using_m.autotune_dataaugmented/weights/best.pt'
model = YOLO(model_path)
result = get_tooth_sequences(model=model, img_path=img_path, label_path=label_path)

final_fdi_labels, assigned_indices = run_viterbi_alignment(
    result['prob_matrix'],
    result['jaw_type'],
    print_dp_path=False
)

print("pred_seq:", result.get("pred_seq"))
print("gt_seq:", result.get("gt_seq"))
print("final_fdi_labels:", final_fdi_labels)