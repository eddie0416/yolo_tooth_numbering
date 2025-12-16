from ultralytics import YOLO
import numpy as np

# 載入模型和推論
model = YOLO('runs/detect/m_using_m.autotune_dataaugmented/weights/best.pt')
img_path = 'uninference_tooth/00240433UpperJaw_neutral.png'
#results = model(img_path, agnostic_nms=True)
results = model.predict(img_path, classes="soft", agnostic_nms=True) # pip install git+https://github.com/ultralytics/ultralytics@exp-nms 可以顯示bbox每個class的conf
'''
results[0]  這張圖的 Results
results[0].names
'''

# 1. 提取每個bbox的中心點
boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
centers = []
for box in boxes:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    centers.append([cx, cy])

centers = np.array(centers)

# 2. 計算質心
centroid = np.mean(centers, axis=0)
print(f"質心位置: {centroid}")

# 3. 計算每個點相對於質心的角度
angles = []
for i, point in enumerate(centers):
    angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    angles.append((i, angle))  # (原始索引, 角度)

# 按角度排序
angles_sorted = sorted(angles, key=lambda x: x[1])

# 4. 找出相鄰點中角度差異最大的兩個點
max_gap = 0
gap_idx = 0

for i in range(len(angles_sorted)):
    next_i = (i + 1) % len(angles_sorted)
    gap = angles_sorted[next_i][1] - angles_sorted[i][1]
    
    # 處理跨越-π到π的情況
    if gap < 0:
        gap += 2 * np.pi
    
    if gap > max_gap:
        max_gap = gap
        gap_idx = next_i

print(f"最大角度間隙: {np.degrees(max_gap):.2f}度")
print(f"間隙位於索引 {angles_sorted[(gap_idx-1) % len(angles_sorted)][0]} 和 {angles_sorted[gap_idx][0]} 之間")

# 輸出環形順序的list（從最大間隙後的點開始）
circular_order = []
for i in range(len(angles_sorted)):
    idx = (gap_idx + i) % len(angles_sorted)
    original_idx = angles_sorted[idx][0]
    circular_order.append(original_idx)

print(f"\n環形順序list（從間隙開始）:")
print(circular_order)

class_names = results[0].names  # 取得類別名稱字典

jaw_type = '' #判斷這張影像是屬於上/下顎
count_upper_teeth = 0
count_lower_teeth = 0

for idx in circular_order:
    box = results[0].boxes[idx]
    class_id = int(box.cls[0])
    # confidence = float(box.conf[0]) #原本只有最高的那個conf
    confidence = float(max(box.conf[0])) #現在已經有all_class的conf 因此這邊要先取最大的那個出來
    class_name = class_names[class_id]  # 根據class_id取得名稱
    xyxy = box.xyxy[0].cpu().numpy()

    if class_id >= 16: # class_id=16以上的牙齒都屬於上顎
        count_lower_teeth += 1
    else:
        count_upper_teeth += 1

    print(f"環形位置 {circular_order.index(idx)+1}: bbox索引{idx}, "
          f"class={class_id}({class_name}), conf={confidence:.2f}")

if count_upper_teeth > count_lower_teeth:
    jaw_type = 'upper'
else:
    jaw_type = 'lower'
print(f"顎別：{jaw_type}")
'''
如果是upper，那麼接下來在做動態的時候，conf應該參考class_id>=16的那16個即可
'''
print(f"results[0].boxes.conf.shape(牙齒數量, 維度)：{results[0].boxes.conf.shape}") # torch.Size([12, 32])

# 建立機率矩陣
num_teeth = len(circular_order)
num_classes_per_jaw = 16
# 初始化矩陣
prob_matrix = np.zeros((num_teeth, num_classes_per_jaw))

for i, idx in enumerate(circular_order):
    box = results[0].boxes[idx]
    # 提取所有32個類別的信心分數
    all_conf = box.conf[0].cpu().numpy()
    
    if jaw_type == 'upper':
        # 下顎：先右側(8往回到0)，再左側(8往前到16)
        right_side = all_conf[7::-1]      # 從 class_id 7 倒數到 0，共8個
        left_side = all_conf[8:16]        # 從 class_id 8 到 15，共8個
        prob_matrix[i, :] = np.concatenate([right_side, left_side])
    else:
        # 上顎：先右側(24往回到16)，再左側(24往前到32)
        right_side = all_conf[23:15:-1]  # 從 class_id 23 倒數到 16，共8個
        left_side = all_conf[24:32]       # 從 class_id 24 到 31，共8個
        prob_matrix[i, :] = np.concatenate([right_side, left_side])



'''
############此處為prob_matrix視覺化############
'''
if jaw_type == 'upper':
    # 下顎：右側(7→0)倒序 + 左側(8→15)正序
    right_headers = [class_names[i] for i in range(7, -1, -1)]
    left_headers = [class_names[i] for i in range(8, 16)]
    class_headers = right_headers + left_headers
else:
    # 上顎：右側(23→16)倒序 + 左側(24→31)正序
    right_headers = [class_names[i] for i in range(23, 15, -1)]
    left_headers = [class_names[i] for i in range(24, 32)]
    class_headers = right_headers + left_headers


# 印出 header
header = f"{'idx':<6}" + "".join([f"{name:<10}" for name in class_headers])
print(header)
print("-" * len(header))

# 印出每一列資料
for i in range(prob_matrix.shape[0]):
    row_str = f"{i:<6}"
    for j in range(prob_matrix.shape[1]):
        row_str += f"{prob_matrix[i, j]:<10.4f}"
    print(row_str)
'''
############此處為prob_matrix視覺化############
'''


# 這裡建立一個對照表： Template Index -> FDI Name 這邊應該是機率矩陣的header
TEMPLATE = {
    'upper': [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    'lower': [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
}

def run_viterbi_alignment(prob_matrix, jaw_type='lower'):
    """
    prob_matrix: shape (N, 16), 你的機率矩陣
    jaw_type: 'upper' 或 'lower'，用來決定輸出的 FDI 編號
    """
    N, M = prob_matrix.shape
    
    template_fdi = TEMPLATE[jaw_type]

    # --- 參數設定 ---
    # 缺牙懲罰：如果從 index 2 跳到 index 4 (缺一顆)，扣一點分
    # 這能避免 DP 為了追逐微小的機率提升而亂跳牙位
    GAP_PENALTY = 0.05 

    # dp[i][j]: 第 i 個 bbox 匹配到第 j 個標準牙位的最大分數
    # 使用 log 機率可以避免數值下溢，但這裡直接用加法分數也很直觀
    dp = np.full((N, M), -1e9) 
    path = np.full((N, M), -1, dtype=int)
    
    # --- 1. 初始化 (Base Case) ---
    # 第一個 bbox 可以是前幾顆牙的任何一顆
    # 我們通常不希望第一個 bbox 直接對應到很後面的牙齒 (例如 index 0 對應到 48)
    # 所以可以給予一點位置懲罰，或者不限制
    for j in range(M):
        dp[0][j] = prob_matrix[0][j]
        
    # --- 2. 遞推 (Recursion) ---
    for i in range(1, N): # 遍歷每個 BBox
        for j in range(M): # 遍歷每個標準牙位 (Current Tooth)
            
            # 我們要找前一個 BBox (i-1) 對應的最佳牙位 k
            # 限制：k 必須小於 j (幾何順序)
            
            best_prev_score = -1e9
            best_k = -1
            
            # 優化：不需要從頭掃描 k，通常牙齒不會跳超過 5 顆
            # k 的範圍是 [0, j-1]
            search_start = max(0, j - 6)
            
            for k in range(search_start, j):
                # 計算跳躍步數 (缺牙數)
                # gap = 0 代表緊鄰 (e.g., 38->37) -> j-k=1
                # gap = 1 代表缺一顆 (e.g., 38->36) -> j-k=2
                gap_count = (j - k) - 1
                penalty = gap_count * GAP_PENALTY
                
                score = dp[i-1][k] - penalty
                
                if score > best_prev_score:
                    best_prev_score = score
                    best_k = k
            
            # 更新 DP 表
            if best_k != -1:
                # 當前狀態分數 = 前一狀態最佳分 + 當前機率
                dp[i][j] = best_prev_score + prob_matrix[i][j]
                path[i][j] = best_k

    # --- 3. 回溯 (Backtracking) ---
    # 找出最後一個 bbox 的最佳位置
    # 通常最後一個 bbox 會落在矩陣的後半段
    last_bbox_idx = N - 1
    best_last_col = np.argmax(dp[last_bbox_idx, :])
    
    # 開始回推
    final_fdi_labels = []
    curr_col = best_last_col
    
    # 記錄每個 bbox 分配到的 column index
    assigned_indices = []
    
    for i in range(N - 1, -1, -1):
        fdi = template_fdi[curr_col]
        final_fdi_labels.append(fdi)
        assigned_indices.append(curr_col)
        
        # 往回走
        prev_col = path[i][curr_col]
        
        # 安全檢查：如果路徑斷了 (should not happen usually)
        if prev_col == -1 and i > 0:
            print(f"Warning: Path broken at step {i}")
            break
        curr_col = prev_col
        
    final_fdi_labels.reverse()
    assigned_indices.reverse()
    
    return final_fdi_labels, assigned_indices

# --- 測試執行 ---
final_labels, assigned_cols = run_viterbi_alignment(prob_matrix, jaw_type=jaw_type)

print("\n=== DP 校正後結果 ===")
for i, fdi in enumerate(final_labels):
    # 原始最高分 label (只為了比較用)
    orig_max_idx = np.argmax(prob_matrix[i])
    orig_conf = prob_matrix[i][orig_max_idx]
    
    # DP 分配的機率
    dp_col = assigned_cols[i]
    dp_conf = prob_matrix[i][dp_col]
    
    print(f"BBox {i:2d}: Final Label = {fdi} (Conf: {dp_conf:.4f}) | Original Max was fdi {TEMPLATE[jaw_type][orig_max_idx]} ({orig_conf:.4f})")