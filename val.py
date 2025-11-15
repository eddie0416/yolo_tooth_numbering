import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument('--m', required=True, help='model folder name, e.g. l.pt or m.yaml')
args = p.parse_args()

model = YOLO(f"runs/detect/{args.m}/weights/best.pt")
model.val(data='yolo_numbering_dataset/dataset_splited/data.yaml', save=False)
