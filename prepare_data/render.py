import math
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import pyrender
import trimesh
import json
from PIL import Image
from color_utils import color2label
import glob
from pathlib import Path


def m3dLookAt(eye, target, up):
    def normalize(v):
        return v / np.sqrt(np.sum(v ** 2))

    mz = normalize(eye - target)
    mx = normalize(np.cross(up, mz))
    my = normalize(np.cross(mz, mx))

    return np.array([
        [mx[0], my[0], mz[0], eye[0]],
        [mx[1], my[1], mz[1], eye[1]],
        [mx[2], my[2], mz[2], eye[2]],
        [0, 0, 0, 1]
    ])


def calculate_optimal_distance(vertices, camera_direction, yfov, target_fill_ratio=0.85):
    """
    計算最佳相機距離，使模型在影像中佔據指定比例
    
    Args:
        vertices: 模型頂點 (N×3 numpy array)
        camera_direction: 相機朝向（單位向量）
        yfov: 垂直視場角（弧度）
        target_fill_ratio: 目標填充比例（0.85 表示佔據 85% 影像）
    
    Returns:
        optimal_distance: 最佳相機距離
    """
    center = np.mean(vertices, axis=0)
    
    # 構建相機座標系（假設相機朝 -Z 看，Y 軸朝上）
    cam_z = -camera_direction / np.linalg.norm(camera_direction)
    
    # 選擇合適的 up 向量（避免與 cam_z 平行）
    if abs(np.dot(cam_z, [0, 1, 0])) > 0.9:
        up_temp = [1, 0, 0]
    else:
        up_temp = [0, 1, 0]
    
    cam_x = np.cross(up_temp, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    
    # 將頂點轉換到相機座標系
    vertices_centered = vertices - center
    vertices_cam = np.stack([
        np.dot(vertices_centered, cam_x),
        np.dot(vertices_centered, cam_y),
        np.dot(vertices_centered, cam_z)
    ], axis=1)
    
    # 計算投影後的包圍盒
    x_range = np.max(vertices_cam[:, 0]) - np.min(vertices_cam[:, 0])
    y_range = np.max(vertices_cam[:, 1]) - np.min(vertices_cam[:, 1])
    z_depth = np.max(vertices_cam[:, 2]) - np.min(vertices_cam[:, 2])
    
    # 根據視場角計算所需距離
    # distance = (size / 2) / tan(fov / 2) / fill_ratio
    max_size = max(x_range, y_range)
    required_distance = (max_size / 2) / np.tan(yfov / 2) / target_fill_ratio
    
    # 加上模型深度的一半，確保前後都不被裁切
    safe_distance = required_distance + z_depth / 2
    
    return safe_distance


def render_top_view(ply_path, img_save_dir, mask_save_dir, camera_params_save_dir, rend_size=(1024, 1024), fill_ratio=0.85):
    """
    從單一 .ply 檔案渲染兩張俯視影像：
    1. 純打光影像（無顏色）
    2. 語義標籤影像（帶顏色）
    
    Args:
        ply_path: .ply 檔案路徑
        save_dir: 輸出目錄
        rend_size: 渲染影像尺寸 (高, 寬)
        fill_ratio: 模型填充比例 (0.0-1.0)，建議 0.85
    """
    base_name = os.path.basename(ply_path)[:-4]
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(camera_params_save_dir, exist_ok=True)

    # 載入 .ply 檔案
    label_trimesh = trimesh.load(ply_path)
    vertices = np.asarray(label_trimesh.vertices)
    
    # 計算模型的邊界和中心
    minCoord = np.min(vertices, axis=0)
    maxCoord = np.max(vertices, axis=0)
    meanCoord = np.mean(vertices, axis=0)
    
    # 使用新的最佳距離計算方法
    yfov = np.pi / 2
    camera_direction = np.array([0, 0, -1])  # 俯視朝下
    optimal_distance = calculate_optimal_distance(
        vertices, camera_direction, yfov, fill_ratio
    )
    
    print(f"Model center: {meanCoord}")
    print(f"Model bounds: {maxCoord - minCoord}")
    print(f"Optimal camera distance: {optimal_distance:.2f} (fill ratio: {fill_ratio})")

    # 設定俯視相機位置（從上往下看）
    camera_pos = meanCoord + np.asarray([0, 0, optimal_distance], dtype=float)
    camera_pose = m3dLookAt(camera_pos, meanCoord, np.asarray([0, 1, 0], dtype=float))

    # 計算相機參數（與原始程式碼相同的格式）
    Rt = np.eye(4)
    Rt[:3, :3] = camera_pose[:3, :3].T
    Rt[:3, 3] = -np.dot(camera_pose[:3, :3].T, camera_pose[:3, 3])
    
    # 清理浮點數誤差
    Rt[np.abs(Rt) < 1e-10] = 0
    
    f_y = (rend_size[0] / 2) / math.tan(np.pi / 2 / 2)
    f_x = f_y * 1.0
    cx, cy = rend_size[1] / 2.0, rend_size[0] / 2.0
    K = np.array([[f_x, 0, cx], [0, f_y, cy], [0, 0, 1]])
    
    camera_params = {
        "frame": 0,
        "theta": math.pi / 2,  # 俯視角為 90 度
        "beta": 0.0,
        "Rt": Rt.tolist(),
        "K": K.tolist()
    }

    # === 場景 1：純打光渲染 ===
    neutral_mesh = trimesh.Trimesh(
        vertices=label_trimesh.vertices,
        faces=label_trimesh.faces
    )
    pyrender_neutral = pyrender.Mesh.from_trimesh(neutral_mesh)
    scene_neutral = pyrender.Scene()
    scene_neutral.add(pyrender_neutral)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene_neutral.add(light)

    # === 場景 2：語義標籤渲染 ===
    label_scene = pyrender.Scene()
    seg_node_map = {}
    label_color_map = {}
    face_instances = {}

    # 解析面顏色
    if 'face' in label_trimesh.metadata['_ply_raw']:
        face_meta = label_trimesh.metadata['_ply_raw']['face']
        if 'red' in face_meta['data'] and 'green' in face_meta['data'] and 'blue' in face_meta['data']:
            face_colors = np.stack([
                face_meta['data']['red'],
                face_meta['data']['green'],
                face_meta['data']['blue']
            ], axis=-1).squeeze(1)

            for i, face in enumerate(label_trimesh.faces):
                face_color = tuple(face_colors[i])
                if face_color in color2label:
                    label = color2label[face_color][2]
                    if label not in face_instances:
                        face_instances[label] = []
                    face_instances[label].append(face)
                    if label not in label_color_map:
                        label_color_map[label] = face_color

    # 為每個類別創建分離的 mesh
    for label, faces in face_instances.items():
        vertex_indices = set(v for f in faces for v in f)
        vertex_idx_map = {v: i for i, v in enumerate(vertex_indices)}
        
        vertice_node = np.array([label_trimesh.vertices[v] for v in vertex_indices])
        face_node = np.array([[vertex_idx_map[v] for v in f] for f in faces])

        label_color = label_color_map[label] + (255,)
        mesh_node = trimesh.Trimesh(vertices=vertice_node, faces=face_node, face_colors=label_color)
        
        node = label_scene.add(pyrender.Mesh.from_trimesh(mesh_node, smooth=False))
        seg_node_map[node] = label_color[:3]

    label_scene.add(light)

    # === 設定相機和渲染器 ===
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1], viewport_height=rend_size[0])

    # 渲染純打光影像
    camera_node = scene_neutral.add(camera, pose=camera_pose)
    color_neutral, _ = r.render(scene_neutral)
    scene_neutral.remove_node(camera_node)
    neutral_path = os.path.join(img_save_dir, f'{base_name}_neutral.png')
    Image.fromarray(color_neutral).save(neutral_path)
    print(f"Saved neutral image: {neutral_path}")

    # 渲染語義標籤影像
    camera_node = label_scene.add(camera, pose=camera_pose)
    color_label, _ = r.render(label_scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map)
    label_scene.remove_node(camera_node)
    label_path = os.path.join(mask_save_dir, f'{base_name}_label.png')
    Image.fromarray(color_label).save(label_path)
    print(f"Saved label image: {label_path}")

    # 儲存相機參數
    json_path = os.path.join(camera_params_save_dir, f'{base_name}_view.json')
    with open(json_path, 'w') as f:
        json.dump(camera_params, f, indent=4)
    print(f"Saved camera parameters: {json_path}")

    r.delete()

'''
if __name__ == '__main__':
    ply_path = 'ply/00OMSZGW_lower.ply'
    mask_save_dir = 'yolo_numbering_dataset/render_mask'
    img_save_dir = 'yolo_numbering_dataset/dataset/images'
    camera_params_save_dir = 'yolo_numbering_dataset/camera_params'
    
    # 預設使用 85% 填充比例（推薦）
    render_top_view(ply_path, img_save_dir, mask_save_dir, camera_params_save_dir, rend_size=(1024, 1024), fill_ratio=0.95)
'''
if __name__ == '__main__':
    ply_dir = 'ply'
    mask_save_dir = 'yolo_numbering_dataset/render_mask'
    img_save_dir = 'yolo_numbering_dataset/dataset/images'
    camera_params_save_dir = 'yolo_numbering_dataset/camera_params'
    
    # 找出所有 .ply 檔案
    ply_files = glob.glob(f'{ply_dir}/*.ply')
    
    print(f"找到 {len(ply_files)} 個 .ply 檔案")
    
    for ply_path in ply_files:
        print(f"處理: {ply_path}")
        render_top_view(ply_path, img_save_dir, mask_save_dir, 
                       camera_params_save_dir, rend_size=(1024, 1024), 
                       fill_ratio=0.95)