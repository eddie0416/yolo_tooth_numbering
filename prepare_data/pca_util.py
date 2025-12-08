# pca_util.py (版本 4)
import trimesh
import numpy as np
import os

def align_mesh_with_pca(source_obj_path, output_obj_path):
    """
    使用 PCA 對齊 3D Mesh，並嘗試確定牙齒的上下方向。
    已修正鏡像翻轉問題。
    """

    # --- 讀取網格 ---
    try:
        mesh = trimesh.load_mesh(source_obj_path, process=False)
        print(f"[INFO] 成功讀取: {source_obj_path}")
    except Exception as e:
        print(f"[ERROR] 讀取失敗: {e}")
        return None

    # --- PCA 計算 ---
    points = mesh.vertices
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    cov = np.cov(points_centered, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eig(cov)
    sort_idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sort_idx]
    eigen_vectors = eigen_vectors[:, sort_idx]

    # ---------------------------------------------------------
    # [FIX] 修正鏡像翻轉問題
    # ---------------------------------------------------------
    # 檢查旋轉矩陣的行列式值
    # 如果是 -1 (或接近 -1)，代表座標系是左手系(有鏡像)，需要轉回右手系
    if np.linalg.det(eigen_vectors) < 0:
        print("[INFO] 偵測到鏡像翻轉，正在修正 PCA 矩陣...")
        # 將最後一個特徵向量(對應最小特徵值)反向，
        # 這樣行列式值就會變成正的，變回純旋轉。
        eigen_vectors[:, 2] *= -1 
    # ---------------------------------------------------------

    # 建立旋轉矩陣
    R = eigen_vectors
    rotation_matrix = R.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transform)

    # --- 額外：判斷 Z 軸正反 (牙齒開口方向判斷) ---
    try:
        # 新版 trimesh
        boundary_edges = aligned_mesh.edges_boundary
    except AttributeError:
        # 舊版手動找出「只出現一次的邊」
        unique_edge_ids = trimesh.grouping.group_rows(
            aligned_mesh.edges_sorted, require_count=1
        )
        boundary_edges = aligned_mesh.edges_sorted[unique_edge_ids]

    boundary_vertices = aligned_mesh.vertices[np.unique(boundary_edges)]

    # 防呆：如果網格是封閉的(沒有邊界)，boundary_vertices 可能為空
    if len(boundary_vertices) > 0:
        avg_boundary_z = np.mean(boundary_vertices[:, 2])
        avg_all_z = np.mean(aligned_mesh.vertices[:, 2])

        # 假設：邊界平均 z 比整體 z 小 → 底部朝下 (符合直覺)
        # 這裡單純翻轉 Z 軸，不會造成鏡像 (因為 flip 矩陣 determinant 是 -1)
        # 但要注意：如果前面已經是純旋轉，這裡再乘一個鏡像矩陣(Z scale -1)會變回鏡像
        # **修正建議**：如果要保持物體不被鏡像，這裡應該要繞 X 或 Y 軸旋轉 180 度，而不是 scale Z = -1
        
        if avg_boundary_z > avg_all_z:
            print("[INFO] Z 軸方向調整 (因為邊界在上方)")
            
            # --- 方法 A: 鏡像翻轉 Z (你原本的寫法) ---
            # 這會導致物體再次變成鏡像！除非你不在意左右相反。
            # flip = np.eye(4)
            # flip[2, 2] = -1 
            
            # --- 方法 B: 繞 X 軸旋轉 180 度 (推薦) ---
            # 這樣可以把上面轉到下面，且保持「右手座標系」(不鏡像)
            rotation_180 = np.eye(4)
            rotation_180[:3, :3] = [
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ]
            
            aligned_mesh.apply_transform(rotation_180)
            transform = rotation_180 @ transform
    else:
        print("[WARN] 找不到邊界邊，跳過 Z 軸方向判斷。")

    # --- 儲存 ---
    try:
        output_dir = os.path.dirname(output_obj_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        aligned_mesh.export(output_obj_path)
        print(f"[INFO] 已輸出: {output_obj_path}")
    except Exception as e:
        print(f"[ERROR] 儲存失敗: {e}")

    return aligned_mesh

if __name__ == "__main__":
    source_path = "uninference_tooth/00240433UpperJaw.stl"
    output_path = "uninference_tooth/00240433UpperJaw.ply"

    align_mesh_with_pca(source_path, output_path)