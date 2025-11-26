# pca_util.py (版本 3)

import trimesh
import numpy as np
import os


def align_mesh_with_pca(source_obj_path, output_obj_path):
    """
    使用 PCA 對齊 3D Mesh，並嘗試確定牙齒的上下方向。
    """

    # --- 讀取網格 ---
    try:
        mesh = trimesh.load_mesh(source_obj_path, process=False)
        print(f"[INFO] 成功讀取: {source_obj_path}")
    except Exception as e:
        print(f"[ERROR] 讀取失敗: {e}")
        return None, None

    # --- PCA 計算 ---
    points = mesh.vertices
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    cov = np.cov(points_centered, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eig(cov)
    sort_idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sort_idx]
    eigen_vectors = eigen_vectors[:, sort_idx]

    # 建立旋轉矩陣
    R = eigen_vectors
    rotation_matrix = R.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transform)

    # --- 額外：判斷 Z 軸正反 ---
    # 找出邊界邊 (只屬於一個面的邊)
    # 找出邊界邊
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


    avg_boundary_z = np.mean(boundary_vertices[:, 2])
    avg_all_z = np.mean(aligned_mesh.vertices[:, 2])

    # 假設：邊界平均 z 比整體 z 小 → 底部朝下 (符合直覺)
    if avg_boundary_z > avg_all_z:
        print("[INFO] Z 軸翻轉 (因為邊界在上方)")
        flip = np.eye(4)
        flip[2, 2] = -1  # Z 軸取反
        aligned_mesh.apply_transform(flip)
        transform = flip @ transform

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
    source_path = "2025_1126_upper.stl"
    output_path = "2025_1126_upper.ply"

    align_mesh_with_pca(source_path, output_path)