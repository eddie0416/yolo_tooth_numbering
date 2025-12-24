from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/m_using_m.autotune_dataaugmented/weights/best.pt")

img_path = "uninference_tooth/00240433LowerJaw_neutral.png"
print(f"\n=== Inference on: {img_path} ===")

results = model(img_path, agnostic_nms=True)

for i, r in enumerate(results):
    im = r.plot(labels=False, conf=False)

    centers = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()

        # 收集每個 bbox 中心點
        for (x1, y1, x2, y2) in xyxy:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])

        centers = np.array(centers, dtype=np.float64)  # (N, 2)

        # 算術平均中心點（質心）
        ax = float(np.mean(centers[:, 0]))
        ay = float(np.mean(centers[:, 1]))
        centroid_xy = np.array([ax, ay], dtype=np.float64)
        centroid_pt = (int(ax), int(ay))

        # 先畫「中心點 -> 質心」連線 + 中心點 + 角度標註
        for (cx, cy) in centers:
            p = (int(cx), int(cy))

            # 連線
            cv2.line(im, p, centroid_pt, color=(0, 255, 255), thickness=2)

            # 中心點
            cv2.circle(im, p, radius=4, color=(0, 0, 255), thickness=-1)

            # 角度（弧度->度）
            dx = cx - centroid_xy[0]
            dy = cy - centroid_xy[1]

            # 影像座標版本（y往下）：直接用 dy
            angle_deg = np.degrees(np.arctan2(dy, dx))

            # 如果你想用「數學平面直覺（y往上）」：改用下面這行取代上面那行
            # angle_deg = np.degrees(np.arctan2(-dy, dx))

            # 標註文字（放在點右上方一點點）
            text = f"degree {angle_deg:.1f}"
            cv2.putText(im, text, (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(im, text, (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 最後畫質心（大點）
        cv2.circle(im, centroid_pt, radius=5, color=(0, 0, 0), thickness=-1)

        print(f"Arithmetic-mean center: ({ax:.2f}, {ay:.2f})")

    out_path = f"pred_with_centers_{i}.jpg"
    cv2.imwrite(out_path, im)
    print("saved:", out_path)
