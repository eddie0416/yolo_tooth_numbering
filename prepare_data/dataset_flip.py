# 這邊是針對原始的影像複製一張，並且對img/labels都做左右翻轉，以此增加資料數量為兩倍
from pathlib import Path
from PIL import Image
import shutil

# === 你的 class 映射表 ===
MAPPING = {
    0:8, 1:9, 2:10, 3:11, 4:12, 5:13, 6:14, 7:15,
    8:0, 9:1, 10:2, 11:3, 12:4, 13:5, 14:6, 15:7,
    16:24, 17:25, 18:26, 19:27, 20:28, 21:29, 22:30, 23:31,
    24:16, 25:17, 26:18, 27:19, 28:20, 29:21, 30:22, 31:23
}

SRC_ROOT = Path("yolo_numbering_dataset/dataset_splited")
DST_ROOT = Path("yolo_numbering_dataset/dataset_splited_augmented")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def flip_label_file(src_txt: Path, dst_txt: Path):
    """
    讀 YOLO label txt，做:
      1) class id 映射
      2) x_center 左右翻轉: x -> 1-x
    """
    lines_out = []
    if not src_txt.exists():
        # 沒 label 就建立空檔
        dst_txt.write_text("")
        return

    for line in src_txt.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label format error in {src_txt}: {line}")

        cls = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # class mapping
        if cls not in MAPPING:
            raise KeyError(f"class id {cls} not in mapping")
        cls_new = MAPPING[cls]

        # fliplr on x_center
        x_new = 1.0 - x

        lines_out.append(f"{cls_new} {x_new:.6f} {y:.6f} {w:.6f} {h:.6f}")

    dst_txt.write_text("\n".join(lines_out) + "\n")

def process_split(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split

    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    # 支援常見影像副檔名
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() not in exts:
            continue

        stem = img_path.stem  # 不含副檔名
        lbl_path = src_lbl_dir / f"{stem}.txt"

        # === 1) 複製原圖 ===
        dst_img_path = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        # === 2) 複製原 label ===
        dst_lbl_path = dst_lbl_dir / lbl_path.name
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_path)
        else:
            dst_lbl_path.write_text("")

        # === 3) 產生 fliplr 圖 ===
        img = Image.open(img_path)
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img_name = f"{stem}_fliplr{img_path.suffix}"
        flip_img_path = dst_img_dir / flip_img_name
        img_flip.save(flip_img_path)

        # === 4) 產生 fliplr label ===
        flip_lbl_name = f"{stem}_fliplr.txt"
        flip_lbl_path = dst_lbl_dir / flip_lbl_name
        flip_label_file(lbl_path, flip_lbl_path)

        print(f"[{split}] done: {img_path.name} -> {flip_img_name}")

def main():
    for split in ["train", "val"]:
        process_split(split)

    print("All done! Augmented dataset created at:")
    print(DST_ROOT)

if __name__ == "__main__":
    main()
