from ultralytics import YOLO

model = YOLO('runs/detect/runs/train/yolov26_license_plate/weights/best.pt')
model.export(format='coreml', nms=True)
# model.export(format='mlpackage', imgsz=640, batch=1, device='cpu')