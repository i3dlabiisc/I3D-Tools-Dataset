from ultralytics import YOLO

# Load model
model = YOLO('../data/yolo_runs/yolo11n.pt')  # or a path to your custom model

# Train with augmentations enabled
model.train(
    data='../data/datasets/custom/may1/yolo_dataset_v8_MI_and_SI_real/data.yaml',
    imgsz=1024,
    epochs=50,
    patience=10,
    batch=16,
    name='yolov11n_training_v5',
    project='../data/yolo_runs/',
    device=0,
    workers=8,
    augment=True  # ✅ explicitly enable augmentation
)

# === Evaluate on Test Set ===
metrics = model.val(
    data='../data/datasets/custom/may1/yolo_dataset_v5/data.yaml',
    split='test',
    imgsz=1024,
    batch=16,
    device=0
)