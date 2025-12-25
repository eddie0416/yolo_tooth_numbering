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
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

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

IMG_DIR = Path("yolo_numbering_dataset/dataset_splited/images/val")
LBL_DIR = Path("yolo_numbering_dataset/dataset_splited/labels/val")
MODEL_PATH = "runs/detect/m_using_m.autotune_dataaugmented/weights/best.pt"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def main():
    #model = YOLO(MODEL_PATH)

    img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    total_imgs = len(img_paths)

    pred_eq_gt = 0
    viterbi_eq_gt = 0
    missing_label = 0
    errors = 0

    # 若你也想記錄不相等的案例，解除註解這些
    # bad_pred_cases = []
    # bad_viterbi_cases = []

    for img_path in img_paths:
        label_path = LBL_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_label += 1
            continue

        try:
            temp_model = YOLO(MODEL_PATH)
            result = get_tooth_sequences(model=temp_model, img_path=str(img_path), label_path=str(label_path))
            del temp_model

            pred_seq = result.get("pred_seq")
            gt_seq = result.get("gt_seq")

            final_fdi_labels, assigned_indices = run_viterbi_alignment(
                result["prob_matrix"],
                result["jaw_type"],
                print_dp_path=False
            )

            if pred_seq == gt_seq:
                pred_eq_gt += 1
            # else:
            #     bad_pred_cases.append(img_path.name)

            if final_fdi_labels == gt_seq:
                viterbi_eq_gt += 1
            # else:
            #     bad_viterbi_cases.append(img_path.name)

        except Exception as e:
            errors += 1
            print(f"[ERROR] {img_path.name}: {e}")

    evaluated = total_imgs - missing_label

    print("\n===== Summary =====")
    print(f"Total images in val: {total_imgs}")
    print(f"Missing label files: {missing_label}")
    print(f"Errors during processing: {errors}")
    print(f"Evaluated (has label): {evaluated}")

    if evaluated > 0:
        print(f"\npred_seq == gt_seq: {pred_eq_gt} / {evaluated} ({pred_eq_gt / evaluated:.2%})")
        print(f"final_fdi_labels == gt_seq: {viterbi_eq_gt} / {evaluated} ({viterbi_eq_gt / evaluated:.2%})")
    else:
        print("\nNo images evaluated (check label paths).")

    # 若你想把失敗清單印出來：
    # print("\nBad pred cases:", bad_pred_cases[:20], "..." if len(bad_pred_cases) > 20 else "")
    # print("\nBad viterbi cases:", bad_viterbi_cases[:20], "..." if len(bad_viterbi_cases) > 20 else "")

if __name__ == "__main__":
    main()